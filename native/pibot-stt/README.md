# pibot STT worker

Native STT worker for pibot.

It reads length-prefixed 16-bit little-endian mono PCM frames from stdin and emits the same newline-delimited JSON events as the Python worker.

## Build

```bash
npm run build:stt-rust
```

## Run from the app

The server starts this worker automatically.

By default the worker expects the Parakeet TDT int8 ONNX files in:

```text
~/models/parakeet-tdt-0.6b-v3-onnx-int8/
  encoder-model.int8.onnx
  decoder_joint-model.int8.onnx
  vocab.txt
```

Override with:

```bash
PARAKEET_TDT_MODEL_DIR=/path/to/model npm run dev
```

## Packaging notes

The worker is a native executable, but ONNX Runtime is still a platform-specific shared library managed by the `ort` crate. For a polished distribution, ship the worker with the matching ONNX Runtime dylib/dll/so and the Parakeet model directory.

The embedded VAD model and local `voice_activity_detector` crate are adapted from `nkeenan38/voice_activity_detector` to use the same `ort` version as `parakeet-rs`.
