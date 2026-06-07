# Pipi

Pipi is a smartphone robot that can talk, remember things, take photos, and drive around when mounted on an [Octobot](https://robo.silverlit.com/products/octobot/).

## Requirements

- macOS/Apple Silicon or Linux x86_64 for local Qwen3-TTS speech synthesis.
- Linux uses Vulkan for native STT/TTS acceleration.
- The default local model set needs about 8-10 GB of unified memory at runtime.
- Node.js 22+, CMake, pkg-config, C/C++ build tools, and `tar`.
- Xcode command line tools, Xcode Metal Toolchain, and Opus are required for the macOS Qwen3-TTS worker. Rust is only needed for the optional Rust/MLX worker (`QWEN3_TTS_WORKER=rust`).

macOS native build prerequisites:

```bash
brew install cmake pkg-config opus
xcodebuild -downloadComponent MetalToolchain
```

Ubuntu 26.04 native build prerequisites:

```bash
sudo apt install -y build-essential cmake pkg-config git curl tar \
  mesa-vulkan-drivers vulkan-tools libvulkan-dev glslc libshaderc-dev spirv-tools
```

For AMD Strix Halo/Radeon 8060S, install ROCm 7.2.2 from AMD's noble repository if you want HIP/ROCm experiments. The default Linux STT build uses Vulkan.

## Setup

Fast path after cloning:

```bash
./setup.sh
npm run dev
```

`setup.sh` installs npm dependencies, initializes submodules, and compiles native STT/TTS workers for the active platform.

Manual equivalent:

```bash
npm ci --ignore-scripts
git submodule update --init --recursive
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

- LLM default: Gemma 4 26B A4B MoE Q4 via llama.cpp.
  - Model: `ggml-org/gemma-4-26B-A4B-it-GGUF`
  - Downloaded into: `~/models/gemma-4-26b-a4b-it`
  - Pipi also downloads a pinned llama.cpp release into `~/.cache/pibot/llama.cpp`.
  - Linux downloads the pinned Vulkan-enabled llama.cpp release.
  - Override the llama.cpp server binary with `LLAMA_CPP_BINARY_PATH=/path/to/llama-server`.
  - Use `LOCAL_LLM=gemma12b npm run dev` for Gemma 4 12B IT Q4 from `unsloth/gemma-4-12b-it-GGUF`, downloaded into `~/models/gemma-4-12b-it`.

- STT default: native `parakeet.cpp` GGUF worker with whisper.cpp GGML Silero VAD.
  - Build with `npm run build:stt-parakeet-cpp`.
  - Uses Metal on Apple platforms and Vulkan on Linux by default.
  - Prebuilt STT worker archives are published from the `parakeet-cpp-stt-v*` GitHub release workflow for Linux x64 Vulkan, macOS arm64 Metal, and Windows x64 Vulkan.
  - Model: `mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf`.
  - Downloaded into: `~/models/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf`.
  - VAD model: `ggml-org/whisper-vad/ggml-silero-v6.2.0.bin`.
  - Downloaded into: `~/models/whisper-vad/ggml-silero-v6.2.0.bin`.
  - Override with `PARAKEET_CPP_MODEL_PATH`/`PARAKEET_CPP_MODEL_FILE` and `SILERO_VAD_GGML_MODEL_PATH`/`SILERO_VAD_GGML_MODEL_FILE`.

- TTS default: native C++/GGML Qwen3-TTS worker (`QWEN3_TTS_WORKER=cpp`).
  - Uses Metal on macOS and Vulkan on Linux/Windows.
  - Built with `npm run build:tts-cpp` (also part of `npm run build:native` and `./setup.sh`).
  - Model: `badlogicgames/qwen3-tts-0.6b-q8_0-gguf` (Q8_0 GGUF + tokenizer).
  - Downloaded into: `~/models/qwen3-tts-0.6b-q8_0-gguf`.
  - Override with `QWEN3_TTS_CPP_WORKER_PATH` / `QWEN3_TTS_CPP_MODEL_PATH` / `QWEN3_TTS_CPP_MODEL_REPO`.
  - Uses x-vector voice cloning.
- TTS alternative: Rust Qwen3-TTS 0.6B Base 6-bit MLX (`QWEN3_TTS_WORKER=rust`).
  - Better voice cloning (ICL), but requires Rust + Apple MLX; build with `npm run build:tts-rust`.
  - Model: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit`, downloaded into `~/models/qwen3-tts-12hz-0.6b-base-6bit`.
- Select the TTS engine with `QWEN3_TTS_WORKER` (`cpp` default, `rust`, `python`, or `disabled`).

## Commands

```bash
./setup.sh              # install deps, initialize submodules, build native workers
npm run dev             # start the development server
npm run build:native    # build STT and TTS native workers
npm run build:stt-parakeet-cpp # build the native parakeet.cpp STT worker
npm run build:tts-cpp   # build only the native C++/Metal Qwen3-TTS worker (default)
npm run build:tts-rust  # build only the optional Rust/MLX Qwen3-TTS worker
npm run check           # format/lint/typecheck/build client
npm run bench:stt       # benchmark STT worker
npm run bench:tts       # benchmark TTS worker
npm run bench:llm       # benchmark local LLM server
```
