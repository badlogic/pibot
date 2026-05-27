# Pipi

A cute little smartphone robot that can talk to you, store memories about you, take photos, and drive around (provided you give it the legs of a [Octobot](https://robo.silverlit.com/products/octobot/))

## Run web demo

Prerequisites:

- Node.js 22+
- Rust toolchain for the native Parakeet/Silero STT worker
- `uv` available on `PATH` if you use the optional Qwen3-TTS local cloned-voice sidecar

```bash
npm install
npm run dev
```

Open:

```text
http://localhost:8010
```

For phone access, expose port `8010` via ngrok HTTPS.

## Native Rust STT worker

Build the STT worker:

```bash
npm run build:stt-rust
```

The server uses this worker automatically. On startup, Pipi downloads missing Parakeet TDT int8 ONNX files from Hugging Face into `~/models/parakeet-tdt-0.6b-v3-onnx-int8`. Set `PARAKEET_TDT_MODEL_DIR` to use a different location.

Model source: `istupakov/parakeet-tdt-0.6b-v3-onnx` on Hugging Face. Required files: `encoder-model.int8.onnx`, `decoder_joint-model.int8.onnx`, and `vocab.txt`.