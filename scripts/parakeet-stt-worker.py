#!/usr/bin/env python3
import collections
import json
import os
import struct
import sys
import time

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import get_logmel
from silero_vad import VADIterator, load_silero_vad

SAMPLE_RATE = 16000
MODEL_NAME = os.environ.get("PARAKEET_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
VAD_THRESHOLD = float(os.environ.get("PARAKEET_VAD_THRESHOLD", "0.45"))
VAD_CHUNK_MS = int(os.environ.get("PARAKEET_VAD_CHUNK_MS", "32"))
MIN_SILENCE_MS = int(os.environ.get("PARAKEET_MIN_SILENCE_MS", "800"))
SPEECH_PAD_MS = int(os.environ.get("PARAKEET_SPEECH_PAD_MS", "250"))
PREROLL_MS = int(os.environ.get("PARAKEET_PREROLL_MS", "1800"))
MIN_UTTERANCE_MS = int(os.environ.get("PARAKEET_MIN_UTTERANCE_MS", "450"))
INTERIM_INTERVAL_MS = int(os.environ.get("PARAKEET_INTERIM_INTERVAL_MS", "600"))
INTERIM_MIN_AUDIO_MS = int(os.environ.get("PARAKEET_INTERIM_MIN_AUDIO_MS", "600"))


def emit(message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def log(message):
    print(message, file=sys.stderr, flush=True)


def read_exact(size):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame():
    header = read_exact(4)
    if header is None:
        return None
    (length,) = struct.unpack("<I", header)
    if length == 0:
        return np.empty(0, dtype=np.float32)
    data = read_exact(length)
    if data is None:
        return None
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def batch_transcribe(model, audio):
    if len(audio) < model.preprocessor_config.hop_length * 2:
        return ""
    mel = get_logmel(mx.array(audio.astype(np.float32, copy=False)), model.preprocessor_config)
    return model.generate(mel)[0].text.strip()


def main():
    log(f"loading Parakeet model: {MODEL_NAME}")
    model = from_pretrained(MODEL_NAME)
    sample_rate = int(model.preprocessor_config.sample_rate)
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError(f"expected Parakeet sample rate {SAMPLE_RATE}, got {sample_rate}")

    log("loading Silero VAD")
    vad_model = load_silero_vad()
    vad = VADIterator(
        vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    vad_chunk_frames = max(1, SAMPLE_RATE * VAD_CHUNK_MS // 1000)
    preroll_chunks = max(1, PREROLL_MS // VAD_CHUNK_MS)
    min_utterance_frames = max(1, SAMPLE_RATE * MIN_UTTERANCE_MS // 1000)

    emit({
        "type": "ready",
        "sampleRate": SAMPLE_RATE,
        "vadChunkMs": VAD_CHUNK_MS,
        "vadThreshold": VAD_THRESHOLD,
        "minSilenceMs": MIN_SILENCE_MS,
        "speechPadMs": SPEECH_PAD_MS,
        "prerollMs": PREROLL_MS,
        "interimIntervalMs": INTERIM_INTERVAL_MS,
    })

    interim_min_frames = max(1, SAMPLE_RATE * INTERIM_MIN_AUDIO_MS // 1000)

    pending = np.empty(0, dtype=np.float32)
    preroll = collections.deque(maxlen=preroll_chunks)
    in_utterance = False
    utterance_chunks = []
    utterance_index = 0
    started_at = time.monotonic()

    while True:
        frame = read_frame()
        if frame is None:
            return
        if len(frame) == 0:
            continue
        pending = np.concatenate((pending, frame))

        while len(pending) >= vad_chunk_frames:
            chunk = pending[:vad_chunk_frames]
            pending = pending[vad_chunk_frames:]
            vad_event = vad(chunk)
            preroll.append(chunk)
            elapsed = time.monotonic() - started_at

            if vad_event and "start" in vad_event and not in_utterance:
                in_utterance = True
                utterance_index += 1
                utterance_chunks = list(preroll)
                last_interim_ms = 0
                emit({"type": "speech_start", "index": utterance_index, "time": elapsed})
                continue

            if in_utterance:
                utterance_chunks.append(chunk)
                if INTERIM_INTERVAL_MS > 0 and utterance_chunks:
                    audio_so_far = np.concatenate(utterance_chunks)
                    audio_ms = int(len(audio_so_far) * 1000 / SAMPLE_RATE)
                    if (
                        len(audio_so_far) >= interim_min_frames
                        and audio_ms - last_interim_ms >= INTERIM_INTERVAL_MS
                    ):
                        last_interim_ms = audio_ms
                        decode_started = time.monotonic()
                        try:
                            text = batch_transcribe(model, audio_so_far)
                            emit({
                                "type": "interim",
                                "index": utterance_index,
                                "text": text,
                                "audioMs": audio_ms,
                                "decodeMs": round((time.monotonic() - decode_started) * 1000),
                            })
                        except Exception as exc:
                            emit({"type": "error", "message": str(exc)})

            if vad_event and "end" in vad_event and in_utterance:
                audio = np.concatenate(utterance_chunks) if utterance_chunks else np.empty(0, dtype=np.float32)
                duration = len(audio) / SAMPLE_RATE
                if len(audio) < min_utterance_frames:
                    emit({"type": "speech_drop", "index": utterance_index, "duration": duration, "reason": "too_short"})
                else:
                    emit({"type": "speech_end", "index": utterance_index, "duration": duration})
                    decode_started = time.monotonic()
                    try:
                        text = batch_transcribe(model, audio)
                        emit({
                            "type": "final",
                            "index": utterance_index,
                            "text": text,
                            "duration": duration,
                            "decodeMs": round((time.monotonic() - decode_started) * 1000),
                        })
                    except Exception as exc:
                        emit({"type": "error", "message": str(exc)})

                in_utterance = False
                utterance_chunks = []
                preroll.clear()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        raise
