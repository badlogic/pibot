# Pipi

Pipi is a smartphone robot that can talk, remember things, take photos, and drive around when mounted on an [Octobot](https://robo.silverlit.com/products/octobot/).

## Requirements

- Apple Silicon Mac with at least 32 GB unified memory.
- The default local model set needs about 8-10 GB of unified memory at runtime.
- Node.js 22+, Rust, Xcode command line tools, Xcode Metal Toolchain, CMake, pkg-config, Opus, and `tar`.

Install native build prerequisites:

```bash
brew install cmake pkg-config opus
xcodebuild -downloadComponent MetalToolchain
```

## Setup

```bash
npm install --ignore-scripts
npm run submodules
npm run build:native
npm run dev
```

Open:

```text
http://localhost:8010
```

For phone access, expose port `8010` over HTTPS, for example with ngrok.

## Models

Pipi runs local LLM, STT, and TTS models. Missing default models are downloaded automatically on startup.

- LLM: Gemma 4 26B A4B MoE Q4 via llama.cpp.
  - Model: `ggml-org/gemma-4-26B-A4B-it-GGUF`
  - Downloaded into: `~/models/gemma-4-26b-a4b-it`
  - Pipi also downloads a pinned llama.cpp release into `~/.cache/pibot/llama.cpp`.

- STT: Parakeet TDT 0.6B int8 ONNX with Silero VAD.
  - Model: `istupakov/parakeet-tdt-0.6b-v3-onnx`
  - Downloaded into: `~/models/parakeet-tdt-0.6b-v3-onnx-int8`

- TTS: Qwen3-TTS 0.6B Base 6-bit MLX.
  - Model: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit`
  - Downloaded into: `~/models/qwen3-tts-12hz-0.6b-base-6bit`

## Commands

```bash
npm run dev             # start the development server
npm run build:native    # build STT and TTS native workers
npm run build:stt-rust  # build only the Rust STT worker
npm run build:tts-rust  # build only the Rust Qwen3-TTS worker
npm run check           # format/lint/typecheck/build client
npm run bench:stt       # benchmark STT worker
npm run bench:tts       # benchmark TTS worker
npm run bench:llm       # benchmark local LLM server
```
