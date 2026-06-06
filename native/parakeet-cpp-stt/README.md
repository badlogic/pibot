# parakeet.cpp STT worker

Native C++ STT worker for pibot using `mudler/parakeet.cpp` GGUF models plus whisper.cpp's GGML Silero VAD.

It reads multiplexed binary input frames from stdin and emits newline-delimited JSON events. Every user-specific event includes `userId`.

Input frame format:

```text
u8 type                  # 1 audio_frame, 2 close_user
u32le userIdByteLength
userIdUtf8
u32le payloadByteLength
payloadBytes             # PCM16LE mono audio for audio_frame, empty for close_user
```

## Build

```bash
npm run build:stt-parakeet-cpp
```

The CMake project fetches `mudler/parakeet.cpp` and `ggml-org/whisper.cpp`, then builds:

```text
native/parakeet-cpp-stt/build/parakeet-cpp-stt-worker
native/parakeet-cpp-stt/build/parakeet-cpp-transcribe
```

## Transcribe a WAV file

```bash
native/parakeet-cpp-stt/build/parakeet-cpp-transcribe /path/to/audio.wav
```

By default, the CLI uses and downloads this model if missing:

```text
~/models/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf
```

Override with `PARAKEET_CPP_MODEL_PATH`, `PARAKEET_CPP_MODEL_FILE`, or pass the model path as the second argument:

```bash
PARAKEET_CPP_MODEL_PATH=/path/to/model.gguf native/parakeet-cpp-stt/build/parakeet-cpp-transcribe /path/to/audio.wav
native/parakeet-cpp-stt/build/parakeet-cpp-transcribe /path/to/audio.wav /path/to/model.gguf
```

The utility emits JSON with the full transcript and word timestamps/confidence:

```json
{"text":"...","words":[{"w":"hello","start":0.48,"end":0.72,"conf":0.91}]}
```

It accepts PCM 16/24/32-bit and float32 WAV files, mixes multi-channel audio to mono, resamples to 16 kHz, and transcribes in chunks. Word timestamps are offset by each chunk start. Override the chunk length with `PARAKEET_CLI_CHUNK_SECONDS`.

## Run from the app

```bash
npm run dev
```

By default the TypeScript server downloads:

```text
~/models/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf
~/models/whisper-vad/ggml-silero-v6.2.0.bin
```

Override with:

```bash
PARAKEET_CPP_MODEL_PATH=/path/to/model.gguf SILERO_VAD_GGML_MODEL_PATH=/path/to/ggml-silero.bin npm run dev
```

or:

```bash
PARAKEET_CPP_MODEL_FILE=realtime_eou_120m-v1-q8_0.gguf npm run dev
```
