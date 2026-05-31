#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import queue
import re
import struct
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from collections.abc import Callable, Iterator
from contextlib import redirect_stdout
from typing import Any, Iterable

import numpy as np

DEFAULT_OUTPUT_SR = 16000
MLX_STREAMING_TOKENS_PER_SECOND = 12.5
DEFAULT_FASTER_STREAMING_CHUNK_SIZE = 8
DEFAULT_MLX_STREAMING_CHUNK_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 1536
MIN_UTTERANCE_TOKENS = 360
ESTIMATED_WORDS_PER_SECOND = 2.6
ESTIMATED_CHARS_PER_SECOND = 14.0
TOKEN_SAFETY_MARGIN = 1.35
BASE_PROMPT_SECONDS = 1.0
PUNCTUATION_PAUSE_SECONDS = 0.5
VALID_MLX_QUANTIZATION_SUFFIXES = ("bf16", "4bit", "6bit", "8bit")

DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit"
DEFAULT_REF_TEXT = "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach of training on verifiable outcomes."
NORMALIZED_REF_AUDIO_CACHE: dict[str, str] = {}
BINARY_MODE = False

WORKER_INPUT_SPEAK = 1
WORKER_INPUT_CANCEL = 2
WORKER_INPUT_SHUTDOWN = 3
WORKER_OUTPUT_READY = 1
WORKER_OUTPUT_AUDIO_START = 2
WORKER_OUTPUT_AUDIO_CHUNK = 3
WORKER_OUTPUT_AUDIO_DONE = 4
WORKER_OUTPUT_ERROR = 5
FRAME_HEADER = struct.Struct("<BII")
STDOUT_BUFFER = sys.stdout.buffer


class RequestCancelled(Exception):
    pass


def log_json(message: dict[str, Any]) -> None:
    line = json.dumps(message, ensure_ascii=False)
    if BINARY_MODE:
        print(line, file=sys.stderr, flush=True)
    else:
        print(line, flush=True)


def read_exact(stream: Any, length: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = stream.read(length - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def read_binary_frame(stream: Any) -> tuple[int, int, bytes] | None:
    header = read_exact(stream, FRAME_HEADER.size)
    if header is None:
        return None
    frame_type, request_id, payload_len = FRAME_HEADER.unpack(header)
    payload = read_exact(stream, payload_len)
    if payload is None:
        return None
    return frame_type, request_id, payload


def write_binary_frame(frame_type: int, request_id: int, payload: bytes = b"") -> None:
    STDOUT_BUFFER.write(FRAME_HEADER.pack(frame_type, request_id & 0xFFFFFFFF, len(payload)))
    STDOUT_BUFFER.write(payload)
    STDOUT_BUFFER.flush()


def normalize_mlx_quantization(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("", "none", "default"):
        return None
    if normalized not in VALID_MLX_QUANTIZATION_SUFFIXES:
        supported = ", ".join(VALID_MLX_QUANTIZATION_SUFFIXES)
        raise ValueError(f"unsupported --mlx-quantization {value!r}; supported: {supported}")
    return normalized


def normalize_language(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "": "auto",
        "auto": "auto",
        "de": "german",
        "de-de": "german",
        "en": "english",
        "en-us": "english",
        "en-gb": "english",
        "fr": "french",
        "fr-fr": "french",
        "es": "spanish",
        "es-es": "spanish",
        "it": "italian",
        "it-it": "italian",
        "pt": "portuguese",
        "pt-br": "portuguese",
        "pt-pt": "portuguese",
        "ja": "japanese",
        "ja-jp": "japanese",
        "ko": "korean",
        "ko-kr": "korean",
        "zh": "chinese",
        "zh-cn": "chinese",
        "zh-tw": "chinese",
        "ru": "russian",
        "ru-ru": "russian",
    }
    return aliases.get(normalized, normalized)


def model_quantization_suffix(model_name: str) -> str | None:
    for suffix in VALID_MLX_QUANTIZATION_SUFFIXES:
        if model_name.endswith(f"-{suffix}"):
            return suffix
    return None


def apply_mlx_quantization_suffix(model_name: str, quantization: str | None) -> str:
    if quantization is None:
        return model_name
    desired_suffix = f"-{quantization}"
    for suffix in VALID_MLX_QUANTIZATION_SUFFIXES:
        current_suffix = f"-{suffix}"
        if model_name.endswith(current_suffix):
            return model_name[: -len(current_suffix)] + desired_suffix
    return f"{model_name}{desired_suffix}"


def resolve_mlx_model_name(model_name: str, quantization: str | None) -> str:
    if model_name.startswith("mlx-community/"):
        if quantization is None and model_quantization_suffix(model_name) is None:
            return f"{model_name}-6bit"
        return apply_mlx_quantization_suffix(model_name, quantization)
    if model_name.startswith("Qwen/"):
        mapped = model_name.replace("Qwen/", "mlx-community/", 1)
        if model_quantization_suffix(mapped) is None:
            mapped = f"{mapped}-6bit" if quantization is None else f"{mapped}-bf16"
        return apply_mlx_quantization_suffix(mapped, quantization)
    return model_name


def infer_model_type(model: Any, model_name: str, backend: str) -> str:
    if backend == "mlx":
        config = getattr(model, "config", None)
        model_type = getattr(config, "tts_model_type", None)
    else:
        inner = getattr(getattr(model, "model", None), "model", None)
        model_type = getattr(inner, "tts_model_type", None)
    if model_type:
        return str(model_type)
    name = model_name.lower()
    if "voicedesign" in name:
        return "voice_design"
    if "customvoice" in name:
        return "custom_voice"
    return "base"


def supported_speakers(model: Any) -> list[str] | None:
    for candidate in (model, getattr(model, "model", None), getattr(getattr(model, "model", None), "model", None)):
        get_speakers = getattr(candidate, "get_supported_speakers", None)
        if callable(get_speakers):
            speakers = get_speakers()
            if speakers is None:
                return None
            return [str(speaker) for speaker in speakers if speaker]
    return None


def resolve_speaker(model: Any, speaker: str | None) -> str | None:
    if speaker:
        return speaker
    speakers = supported_speakers(model)
    if speakers:
        return speakers[0]
    return None


def estimate_max_new_tokens(text: str, streaming_chunk_size: int, configured_cap: int) -> int:
    text = text.strip()
    chunk_size = max(1, streaming_chunk_size)
    cap = max(1, configured_cap)
    if not text:
        return min(cap, MIN_UTTERANCE_TOKENS)

    word_count = len(re.findall(r"\w+", text, flags=re.UNICODE))
    char_count = len(re.sub(r"\s+", "", text))
    word_seconds = word_count / ESTIMATED_WORDS_PER_SECOND if word_count else 0.0
    char_seconds = char_count / ESTIMATED_CHARS_PER_SECOND if char_count else 0.0
    punctuation_count = sum(unicodedata.category(ch).startswith("P") for ch in text)
    punctuation_seconds = punctuation_count * PUNCTUATION_PAUSE_SECONDS
    estimated_seconds = max(word_seconds, char_seconds) + punctuation_seconds + BASE_PROMPT_SECONDS
    estimated_tokens = math.ceil(estimated_seconds * MLX_STREAMING_TOKENS_PER_SECOND * TOKEN_SAFETY_MARGIN)
    aligned_tokens = max(chunk_size, math.ceil(estimated_tokens / chunk_size) * chunk_size)
    requested_tokens = max(MIN_UTTERANCE_TOKENS, aligned_tokens)
    return min(cap, requested_tokens)


def load_text(value: str | None, file: str | None, fallback: str | None = None) -> str:
    if file:
        return Path(file).expanduser().read_text(encoding="utf8").strip()
    if value is not None:
        return value.strip()
    return (fallback or "").strip()


def resample_to_output_sr(audio: np.ndarray, sample_rate: int, output_sample_rate: int) -> np.ndarray:
    if sample_rate == output_sample_rate:
        return audio
    from scipy.signal import resample_poly

    gcd = np.gcd(output_sample_rate, sample_rate)
    return resample_poly(audio, up=output_sample_rate // gcd, down=sample_rate // gcd)


def to_int16(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio * 32768, -32768, 32767).astype(np.int16)


def prepare_audio_chunk(item: Any) -> tuple[np.ndarray | None, int | None]:
    if isinstance(item, tuple):
        audio_chunk, sample_rate, *_rest = item
        return np.asarray(audio_chunk, dtype=np.float32).squeeze(), int(sample_rate)

    audio = getattr(item, "audio", None)
    if audio is None:
        return None, None
    sample_rate = getattr(item, "sample_rate", None) or DEFAULT_OUTPUT_SR
    return np.asarray(audio, dtype=np.float32).squeeze(), int(sample_rate)


def debug_generation_enabled() -> bool:
    return os.environ.get("QWEN3_TTS_DEBUG_GENERATION", "").lower() in {"1", "true", "yes", "on"}


def iter_int16_chunks(items: Iterable[Any], blocksize: int, output_sample_rate: int, label: str) -> Iterator[np.ndarray]:
    started_at = time.perf_counter()
    total_samples = 0
    first_chunk = True
    found_speech = False
    leftover = np.array([], dtype=np.int16)
    debug_generation = debug_generation_enabled()
    stream_chunk_index = 0

    for item in items:
        stream_chunk_index += 1
        audio_chunk, sample_rate = prepare_audio_chunk(item)
        if audio_chunk is None or sample_rate is None or audio_chunk.size == 0:
            continue

        if debug_generation:
            log_json(
                {
                    "type": "generation_debug",
                    "backend": "python_mlx",
                    "chunk": stream_chunk_index,
                    "tokenCount": getattr(item, "token_count", None),
                    "samples": int(audio_chunk.size),
                    "rms": round(float(np.sqrt(np.mean(np.square(audio_chunk)))), 6),
                    "final": bool(getattr(item, "is_final_chunk", False)),
                }
            )

        if first_chunk:
            log_json({"type": "ttfa", "seconds": round(time.perf_counter() - started_at, 3), "label": label})
            first_chunk = False

        audio_chunk = to_int16(resample_to_output_sr(audio_chunk, sample_rate, output_sample_rate))

        if not found_speech:
            threshold = int(32768 * 0.01)
            above = np.abs(audio_chunk) > threshold
            if not np.any(above):
                continue
            start_index = max(0, int(np.argmax(above)) - int(output_sample_rate * 0.040))
            audio_chunk = audio_chunk[start_index:]
            found_speech = True

        audio_chunk = np.concatenate([leftover, audio_chunk])
        n = (len(audio_chunk) // blocksize) * blocksize
        for i in range(0, n, blocksize):
            chunk = audio_chunk[i : i + blocksize]
            total_samples += len(chunk)
            yield chunk
        leftover = audio_chunk[n:]

    if len(leftover) > 0:
        chunk = np.pad(leftover, (0, blocksize - len(leftover)))
        total_samples += len(leftover)
        yield chunk

    generation_time = time.perf_counter() - started_at
    audio_duration = total_samples / output_sample_rate
    rtf = audio_duration / generation_time if generation_time > 0 else 0
    log_json(
        {
            "type": "generated",
            "seconds": round(generation_time, 3),
            "audioSeconds": round(audio_duration, 3),
            "rtf": round(rtf, 3),
            "label": label,
        }
    )



def normalize_ref_audio_for_mlx(model: Any, ref_audio: str) -> str:
    import soundfile as sf
    from scipy.signal import resample_poly

    source_path = Path(ref_audio).expanduser().resolve()
    target_sample_rate = int(getattr(model, "sample_rate", 24000))
    cache_key = f"{source_path}:{target_sample_rate}"
    cached_path = NORMALIZED_REF_AUDIO_CACHE.get(cache_key)
    if cached_path and Path(cached_path).exists():
        return cached_path

    waveform, sample_rate = sf.read(str(source_path), always_2d=False, dtype="float32")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sample_rate != target_sample_rate:
        gcd = np.gcd(int(sample_rate), target_sample_rate)
        waveform = resample_poly(waveform, up=target_sample_rate // gcd, down=int(sample_rate) // gcd)
        sample_rate = target_sample_rate

    temp = tempfile.NamedTemporaryFile(prefix="qwen3_ref_", suffix=".wav", delete=False)
    temp.close()
    sf.write(temp.name, waveform, sample_rate, format="WAV", subtype="PCM_16")
    NORMALIZED_REF_AUDIO_CACHE[cache_key] = temp.name
    return temp.name


def collect_chunks(chunks: Iterable[np.ndarray], on_chunk: Callable[[np.ndarray], None] | None) -> list[np.ndarray]:
    if on_chunk is not None:
        for chunk in chunks:
            on_chunk(chunk)
        return []

    collected: list[np.ndarray] = []
    for chunk in chunks:
        collected.append(chunk)
    return collected


def write_wav(path: str, chunks: list[np.ndarray], sample_rate: int) -> None:
    import soundfile as sf

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
    sf.write(str(output_path), audio, sample_rate, format="WAV", subtype="PCM_16")


def set_generation_seed(seed: int | None) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    try:
        import mlx.core as mx

        mx.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def mlx_generation_kwargs(args: argparse.Namespace, max_tokens: int, chunk_size: int) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "verbose": False,
        "stream": True,
        "streaming_interval": chunk_size / MLX_STREAMING_TOKENS_PER_SECOND,
    }


def load_backend(args: argparse.Namespace) -> tuple[str, Any, str]:
    if platform.system().lower() == "darwin":
        from mlx_audio.tts.utils import load_model

        model_name = resolve_mlx_model_name(args.model_name, normalize_mlx_quantization(args.mlx_quantization))
        log_json({"type": "loading", "backend": "mlx", "model": model_name})
        return "mlx", load_model(model_name), model_name

    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    dtype = args.dtype
    if dtype == "auto":
        resolved_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    else:
        resolved_dtype = getattr(torch, dtype)
    log_json({"type": "loading", "backend": "faster_qwen3_tts", "model": args.model_name, "device": args.device})
    return (
        "faster_qwen3_tts",
        FasterQwen3TTS.from_pretrained(
            args.model_name,
            device=args.device,
            dtype=resolved_dtype,
            attn_implementation=args.attn_implementation,
        ),
        args.model_name,
    )


def synthesize_chunks(
    args: argparse.Namespace,
    backend: str,
    model: Any,
    model_name: str,
    text: str,
    on_chunk: Callable[[np.ndarray], None] | None = None,
) -> list[np.ndarray]:
    set_generation_seed(args.seed)
    ref_text = load_text(args.ref_text, args.ref_text_file, DEFAULT_REF_TEXT)
    model_type = infer_model_type(model, model_name, backend)
    chunk_size = args.streaming_chunk_size or (
        DEFAULT_MLX_STREAMING_CHUNK_SIZE if backend == "mlx" else DEFAULT_FASTER_STREAMING_CHUNK_SIZE
    )
    max_new_tokens = estimate_max_new_tokens(text, chunk_size, args.max_new_tokens)
    log_json(
        {
            "type": "ready",
            "backend": backend,
            "model": model_name,
            "modelType": model_type,
            "chunkSize": chunk_size,
            "maxNewTokens": max_new_tokens,
            "temperature": args.temperature,
            "topK": args.top_k,
            "topP": args.top_p,
            "repetitionPenalty": args.repetition_penalty,
            "seed": args.seed,
        }
    )

    if args.ref_audio:
        if backend == "mlx":
            ref_audio = normalize_ref_audio_for_mlx(model, args.ref_audio)
            items = model.generate(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                lang_code=args.language,
                speed=args.speed,
                **mlx_generation_kwargs(args, max_new_tokens, chunk_size),
            )
            chunks = collect_chunks(iter_int16_chunks(items, args.blocksize, args.output_sample_rate, "voice_clone_mlx"), on_chunk)
        else:
            items = model.generate_voice_clone_streaming(
                text=text,
                language=args.language,
                ref_audio=args.ref_audio,
                ref_text=ref_text,
                xvec_only=args.xvec_only,
                chunk_size=chunk_size,
                max_new_tokens=max_new_tokens,
                parity_mode=args.parity_mode,
                non_streaming_mode=args.non_streaming_mode,
            )
            chunks = collect_chunks(iter_int16_chunks(items, args.blocksize, args.output_sample_rate, "voice_clone"), on_chunk)
    elif model_type == "custom_voice":
        speaker = resolve_speaker(model, args.speaker)
        if not speaker:
            raise ValueError("custom voice generation requires --speaker or model-supported speakers")
        if backend == "mlx":
            items = model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=args.language,
                instruct=args.instruct,
                **mlx_generation_kwargs(args, max_new_tokens, chunk_size),
            )
        else:
            items = model.generate_custom_voice_streaming(
                text=text,
                speaker=speaker,
                language=args.language,
                instruct=args.instruct,
                chunk_size=chunk_size,
                max_new_tokens=max_new_tokens,
                non_streaming_mode=args.non_streaming_mode,
            )
        chunks = collect_chunks(iter_int16_chunks(items, args.blocksize, args.output_sample_rate, "custom_voice"), on_chunk)
    elif model_type == "voice_design":
        if backend == "mlx":
            items = model.generate_voice_design(
                text=text,
                instruct=args.instruct,
                language=args.language,
                **mlx_generation_kwargs(args, max_new_tokens, chunk_size),
            )
        else:
            items = model.generate_voice_design_streaming(
                text=text,
                instruct=args.instruct,
                language=args.language,
                chunk_size=chunk_size,
                max_new_tokens=max_new_tokens,
                non_streaming_mode=args.non_streaming_mode,
            )
        chunks = collect_chunks(iter_int16_chunks(items, args.blocksize, args.output_sample_rate, "voice_design"), on_chunk)
    else:
        raise ValueError("base Qwen3-TTS models require --ref-audio and matching --ref-text/--ref-text-file")

    return chunks


def generate_once(args: argparse.Namespace) -> None:
    text = load_text(args.text, args.text_file)
    if not text:
        raise ValueError("provide --text or --text-file")
    backend, model, model_name = load_backend(args)
    chunks = synthesize_chunks(args, backend, model, model_name, text)
    write_wav(args.output, chunks, args.output_sample_rate)
    log_json({"type": "output", "path": str(Path(args.output).expanduser().resolve())})


def serve(args: argparse.Namespace) -> None:
    global BINARY_MODE
    BINARY_MODE = True
    with redirect_stdout(sys.stderr):
        backend, model, model_name = load_backend(args)
    incoming: queue.Queue[tuple[int, int, bytes]] = queue.Queue()
    cancelled: set[int] = set()
    cancelled_lock = threading.Lock()
    shutdown = threading.Event()

    def reader() -> None:
        while not shutdown.is_set():
            frame = read_binary_frame(sys.stdin.buffer)
            if frame is None:
                shutdown.set()
                incoming.put((WORKER_INPUT_SHUTDOWN, 0, b""))
                return
            frame_type, request_id, payload = frame
            if frame_type == WORKER_INPUT_CANCEL:
                with cancelled_lock:
                    cancelled.add(request_id)
                continue
            incoming.put((frame_type, request_id, payload))

    def is_cancelled(request_id: int) -> bool:
        with cancelled_lock:
            return request_id in cancelled

    def clear_cancelled(request_id: int) -> None:
        with cancelled_lock:
            cancelled.discard(request_id)

    threading.Thread(target=reader, daemon=True).start()
    write_binary_frame(WORKER_OUTPUT_READY, 0)
    log_json({"type": "server_ready", "backend": backend, "model": model_name})

    while not shutdown.is_set():
        frame_type, request_id, payload = incoming.get()
        if frame_type == WORKER_INPUT_SHUTDOWN:
            shutdown.set()
            break
        if frame_type != WORKER_INPUT_SPEAK:
            write_binary_frame(WORKER_OUTPUT_ERROR, request_id, f"unknown frame type {frame_type}".encode("utf8"))
            continue
        try:
            if is_cancelled(request_id):
                clear_cancelled(request_id)
                write_binary_frame(WORKER_OUTPUT_AUDIO_DONE, request_id)
                continue
            text = payload.decode("utf8").strip()
            if not text:
                raise ValueError("empty text")
            request_args = argparse.Namespace(**vars(args))
            write_binary_frame(WORKER_OUTPUT_AUDIO_START, request_id, struct.pack("<I", request_args.output_sample_rate))

            def on_chunk(chunk: np.ndarray) -> None:
                if is_cancelled(request_id):
                    raise RequestCancelled()
                write_binary_frame(WORKER_OUTPUT_AUDIO_CHUNK, request_id, chunk.tobytes())

            try:
                with redirect_stdout(sys.stderr):
                    synthesize_chunks(request_args, backend, model, model_name, text, on_chunk)
            except RequestCancelled:
                log_json({"type": "request_cancelled", "id": request_id})
            write_binary_frame(WORKER_OUTPUT_AUDIO_DONE, request_id)
            clear_cancelled(request_id)
        except Exception as exc:
            write_binary_frame(WORKER_OUTPUT_ERROR, request_id, str(exc).encode("utf8"))
            clear_cancelled(request_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Qwen3-TTS worker. Writes 16/24 kHz mono PCM WAV output.")
    parser.add_argument("--serve", action="store_true", help="Run as a persistent binary-framed worker on stdin/stdout")
    parser.add_argument("--text", help="Target text to synthesize")
    parser.add_argument("--text-file", help="File containing target text to synthesize")
    parser.add_argument("--output", default="data/voices/qwen3-test.wav", help="Output WAV path")
    parser.add_argument("--output-dir", default=tempfile.gettempdir(), help="Output directory for --serve requests")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Qwen3-TTS model name or local path")
    parser.add_argument("--ref-audio", help="Reference audio path for voice cloning")
    parser.add_argument("--ref-text", help="Exact transcript of --ref-audio")
    parser.add_argument("--ref-text-file", help="File containing exact transcript of --ref-audio")
    parser.add_argument("--speaker", default="Aiden", help="Speaker for CustomVoice models")
    parser.add_argument("--instruct", help="VoiceDesign/custom voice instruction")
    parser.add_argument("--language", default="auto", help="Target language, e.g. auto, en, de")
    parser.add_argument("--mlx-quantization", default="6bit", help="Apple Silicon MLX quantization: bf16, 4bit, 6bit, 8bit")
    parser.add_argument("--streaming-chunk-size", type=int, help="Codec steps per streaming chunk")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Upper cap for codec tokens")
    parser.add_argument("--blocksize", type=int, default=512, help="Output chunk size in samples")
    parser.add_argument("--output-sample-rate", type=int, default=DEFAULT_OUTPUT_SR, help="Output WAV sample rate. Use 24000 for less muffled quality tests; 16000 matches the realtime pipeline.")
    parser.add_argument("--seed", type=int, help="Optional RNG seed for more repeatable generation")
    parser.add_argument("--temperature", type=float, default=0.9, help="MLX sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="MLX top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="MLX top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="MLX repetition penalty")
    parser.add_argument("--speed", type=float, default=1.0, help="MLX voice-clone speech speed")
    parser.add_argument("--device", default="cuda", help="Non-macOS faster-qwen3-tts device")
    parser.add_argument("--dtype", default="auto", help="Non-macOS dtype: auto, float16, bfloat16, float32")
    parser.add_argument("--attn-implementation", default="eager", help="Non-macOS attention implementation")
    parser.add_argument("--xvec-only", action="store_true", help="Use x-vector-only voice cloning on faster-qwen3-tts")
    parser.add_argument("--parity-mode", action="store_true", help="Use parity mode on faster-qwen3-tts")
    parser.add_argument("--non-streaming-mode", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.language = normalize_language(args.language)
    if args.serve:
        serve(args)
    else:
        generate_once(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log_json({"type": "error", "message": str(exc)})
        raise
