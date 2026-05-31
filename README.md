# Pipi

Pipi is a smartphone robot that can talk, remember things, take photos, and drive around when mounted on an [Octobot](https://robo.silverlit.com/products/octobot/).

## Requirements

- Node.js 22+
- Rust toolchain for native STT/TTS workers
- Apple Silicon macOS for the Rust MLX Qwen3-TTS backend
- Xcode command line tools, Xcode Metal Toolchain, CMake, pkg-config, and Opus
- `uv` on `PATH` for the optional Python/MLX Qwen3-TTS worker
- `tar` on `PATH` for llama.cpp release extraction

Install native build prerequisites on Apple Silicon macOS:

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

## Scripts

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

## Local LLM

Pipi uses a local OpenAI-compatible llama.cpp server. On startup it connects to `http://127.0.0.1:8080/v1` if one is already running. Otherwise it downloads a pinned llama.cpp release into `~/.cache/pibot/llama.cpp`, downloads missing GGUF model files, and starts `llama-server` automatically.

The default local LLM is Gemma 4 26B A4B MoE Q4. Set `LOCAL_LLM=qwen` to use Qwen3.6 35B A3B instead.

Useful overrides:

```bash
LOCAL_LLM=gemma # or qwen
PIBOT_CACHE_DIR=/path/to/cache
LLAMA_BASE_URL=http://127.0.0.1:8080/v1
LLAMA_HOST=127.0.0.1
LLAMA_PORT=8080
LLAMA_CONTEXT_WINDOW=131072
LLAMA_MODEL_DIR=/path/to/model-dir
```

## STT

Default STT uses the Rust Parakeet/Silero worker:

```text
native/pibot-stt/target/release/pibot-stt-worker
```

The default model is the Parakeet TDT 0.6B int8 ONNX model:

```text
~/models/parakeet-tdt-0.6b-v3-onnx-int8
istupakov/parakeet-tdt-0.6b-v3-onnx
```

Pipi provisions missing model files on startup. Override the model path with:

```bash
PARAKEET_TDT_MODEL_DIR=/path/to/parakeet-model
```

## TTS

Default TTS uses the Rust MLX Qwen3-TTS worker:

```text
native/qwen3_tts_rs/target/release/pibot-tts-worker
```

The default model is the Qwen3-TTS 0.6B 6-bit MLX model:

```text
~/models/qwen3-tts-12hz-0.6b-base-6bit
mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit
```

Pipi provisions missing model files on startup. Override the worker or model path with:

```bash
QWEN3_TTS_RUST_WORKER_PATH=/path/to/pibot-tts-worker
QWEN3_TTS_RUST_MODEL_PATH=/path/to/qwen3-tts-model
QWEN3_TTS_RUST_MODEL_REPO=mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit
```

To test the 1.7B 6-bit MLX model:

```bash
QWEN3_TTS_RUST_MODEL_PATH=~/models/qwen3-tts-12hz-1.7b-base-6bit \
QWEN3_TTS_RUST_MODEL_REPO=mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit \
npm run dev
```

To run the optional Python/MLX worker:

```bash
QWEN3_TTS_WORKER=python npm run dev
```
