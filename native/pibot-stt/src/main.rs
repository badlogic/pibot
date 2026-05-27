use std::collections::VecDeque;
use std::env;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Instant;

use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use serde_json::json;
use voice_activity_detector::VoiceActivityDetector;

const SAMPLE_RATE: usize = 16_000;
const VAD_CHUNK_FRAMES: usize = 512;
const VAD_CHUNK_MS: usize = 32;

struct Config {
    model_dir: PathBuf,
    vad_threshold: f32,
    min_silence_ms: usize,
    speech_pad_ms: usize,
    preroll_ms: usize,
    min_utterance_ms: usize,
    interim_interval_ms: usize,
    interim_min_audio_ms: usize,
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn config() -> Result<Config, String> {
    let model_dir = env::args()
        .nth(1)
        .or_else(|| env::var("PARAKEET_TDT_MODEL_DIR").ok())
        .ok_or_else(|| {
            "missing Parakeet TDT model dir argument or PARAKEET_TDT_MODEL_DIR".to_string()
        })?;
    Ok(Config {
        model_dir: PathBuf::from(model_dir),
        vad_threshold: env_f32("PARAKEET_VAD_THRESHOLD", 0.45),
        min_silence_ms: env_usize("PARAKEET_MIN_SILENCE_MS", 800),
        speech_pad_ms: env_usize("PARAKEET_SPEECH_PAD_MS", 250),
        preroll_ms: env_usize("PARAKEET_PREROLL_MS", 1800),
        min_utterance_ms: env_usize("PARAKEET_MIN_UTTERANCE_MS", 450),
        interim_interval_ms: env_usize("PARAKEET_INTERIM_INTERVAL_MS", 600),
        interim_min_audio_ms: env_usize("PARAKEET_INTERIM_MIN_AUDIO_MS", 600),
    })
}

fn emit(value: serde_json::Value) {
    println!("{value}");
}

fn log(message: impl AsRef<str>) {
    eprintln!("{}", message.as_ref());
}

fn read_exact_or_eof(reader: &mut impl Read, buffer: &mut [u8]) -> io::Result<bool> {
    let mut offset = 0;
    while offset < buffer.len() {
        match reader.read(&mut buffer[offset..])? {
            0 if offset == 0 => return Ok(false),
            0 => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected EOF in frame",
                ))
            }
            read => offset += read,
        }
    }
    Ok(true)
}

fn read_frame(reader: &mut impl Read) -> io::Result<Option<Vec<f32>>> {
    let mut header = [0_u8; 4];
    if !read_exact_or_eof(reader, &mut header)? {
        return Ok(None);
    }
    let length = u32::from_le_bytes(header) as usize;
    if length == 0 {
        return Ok(Some(Vec::new()));
    }
    if length % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PCM frame byte length is not even",
        ));
    }
    let mut bytes = vec![0_u8; length];
    read_exact_or_eof(reader, &mut bytes)?;
    let samples = bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0)
        .collect();
    Ok(Some(samples))
}

fn transcribe(
    model: &mut ParakeetTDT,
    audio: &[f32],
) -> Result<String, Box<dyn std::error::Error>> {
    if audio.len() < VAD_CHUNK_FRAMES {
        return Ok(String::new());
    }
    let result = model.transcribe_samples(
        audio.to_vec(),
        SAMPLE_RATE as u32,
        1,
        Some(TimestampMode::Sentences),
    )?;
    Ok(result.text.trim().to_string())
}

fn concat_chunks(chunks: &[Vec<f32>]) -> Vec<f32> {
    let len = chunks.iter().map(Vec::len).sum();
    let mut audio = Vec::with_capacity(len);
    for chunk in chunks {
        audio.extend_from_slice(chunk);
    }
    audio
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = config().map_err(io::Error::other)?;
    log(format!(
        "loading Parakeet TDT model: {}",
        cfg.model_dir.display()
    ));
    let load_started = Instant::now();
    let mut model = ParakeetTDT::from_pretrained(&cfg.model_dir, None)?;
    log(format!(
        "loaded Parakeet TDT in {:.3}s",
        load_started.elapsed().as_secs_f32()
    ));

    log("loading Silero VAD");
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(SAMPLE_RATE as i64)
        .chunk_size(VAD_CHUNK_FRAMES)
        .build()?;

    emit(json!({
        "type": "ready",
        "sampleRate": SAMPLE_RATE,
        "vadChunkMs": VAD_CHUNK_MS,
        "vadThreshold": cfg.vad_threshold,
        "minSilenceMs": cfg.min_silence_ms,
        "speechPadMs": cfg.speech_pad_ms,
        "prerollMs": cfg.preroll_ms,
        "interimIntervalMs": cfg.interim_interval_ms,
    }));

    let preroll_chunks = (cfg.preroll_ms / VAD_CHUNK_MS).max(1);
    let min_utterance_frames = (SAMPLE_RATE * cfg.min_utterance_ms / 1000).max(1);
    let interim_min_frames = (SAMPLE_RATE * cfg.interim_min_audio_ms / 1000).max(1);

    let mut stdin = io::stdin().lock();
    let mut pending = Vec::<f32>::new();
    let mut preroll = VecDeque::<Vec<f32>>::with_capacity(preroll_chunks + 1);
    let mut in_utterance = false;
    let mut utterance_chunks = Vec::<Vec<f32>>::new();
    let mut utterance_index = 0_usize;
    let mut silence_ms = 0_usize;
    let mut last_interim_ms = 0_usize;
    let started_at = Instant::now();

    while let Some(frame) = read_frame(&mut stdin)? {
        if frame.is_empty() {
            continue;
        }
        pending.extend(frame);

        while pending.len() >= VAD_CHUNK_FRAMES {
            let chunk: Vec<f32> = pending.drain(..VAD_CHUNK_FRAMES).collect();
            let speech_probability = vad.predict(chunk.iter().copied());
            let is_speech = speech_probability >= cfg.vad_threshold;

            preroll.push_back(chunk.clone());
            while preroll.len() > preroll_chunks {
                preroll.pop_front();
            }

            if !in_utterance && is_speech {
                in_utterance = true;
                utterance_index += 1;
                utterance_chunks = preroll.iter().cloned().collect();
                silence_ms = 0;
                last_interim_ms = 0;
                emit(json!({
                    "type": "speech_start",
                    "index": utterance_index,
                    "time": started_at.elapsed().as_secs_f32(),
                }));
                continue;
            }

            if in_utterance {
                utterance_chunks.push(chunk);
                if is_speech {
                    silence_ms = 0;
                } else {
                    silence_ms += VAD_CHUNK_MS;
                }

                let audio_frames: usize = utterance_chunks.iter().map(Vec::len).sum();
                let audio_ms = audio_frames * 1000 / SAMPLE_RATE;
                if cfg.interim_interval_ms > 0
                    && audio_frames >= interim_min_frames
                    && audio_ms.saturating_sub(last_interim_ms) >= cfg.interim_interval_ms
                {
                    last_interim_ms = audio_ms;
                    let audio = concat_chunks(&utterance_chunks);
                    let decode_started = Instant::now();
                    match transcribe(&mut model, &audio) {
                        Ok(text) => emit(json!({
                            "type": "interim",
                            "index": utterance_index,
                            "text": text,
                            "audioMs": audio_ms,
                            "decodeMs": decode_started.elapsed().as_millis(),
                        })),
                        Err(error) => {
                            emit(json!({ "type": "error", "message": error.to_string() }))
                        }
                    }
                }

                if silence_ms >= cfg.min_silence_ms {
                    let audio = concat_chunks(&utterance_chunks);
                    let duration = audio.len() as f64 / SAMPLE_RATE as f64;
                    if audio.len() < min_utterance_frames {
                        emit(json!({
                            "type": "speech_drop",
                            "index": utterance_index,
                            "duration": duration,
                            "reason": "too_short",
                        }));
                    } else {
                        emit(json!({
                            "type": "speech_end",
                            "index": utterance_index,
                            "duration": duration,
                        }));
                        let decode_started = Instant::now();
                        match transcribe(&mut model, &audio) {
                            Ok(text) => emit(json!({
                                "type": "final",
                                "index": utterance_index,
                                "text": text,
                                "duration": duration,
                                "decodeMs": decode_started.elapsed().as_millis(),
                            })),
                            Err(error) => {
                                emit(json!({ "type": "error", "message": error.to_string() }))
                            }
                        }
                    }

                    in_utterance = false;
                    utterance_chunks.clear();
                    preroll.clear();
                    vad.reset();
                }
            }
        }
    }

    Ok(())
}

fn main() {
    if let Err(error) = run() {
        emit(json!({ "type": "error", "message": error.to_string() }));
        std::process::exit(1);
    }
}
