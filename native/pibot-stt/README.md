# pibot-stt

Local STT helper binary for the robot. It links against the vendored `vendor/whisper.cpp` and communicates with Node over a small binary frame protocol on stdin/stdout.

## Build

```bash
npm run build:stt
```

## Download German model

```bash
npm run download:stt-model
```

This downloads `cstr/whisper-large-v3-turbo-german-ggml` q5_0 to:

```text
models/ggml-whisper-large-v3-turbo-german-q5_0.bin
```

## Test from macOS microphone

```bash
npm run stt:test
```

The test CLI uses `ffmpeg -f avfoundation -i :0` and streams raw 16 kHz mono s16le PCM to the C++ process.

## Test with raw PCM stdin

```bash
ffmpeg -i some-audio.wav -ac 1 -ar 16000 -f s16le - | npm run stt:test -- --stdin
```

## Binary protocol

Every frame is:

```c
struct Header {
  uint32_t type;   // little endian
  uint32_t flags;  // currently 0
  uint32_t length; // payload bytes
};
// followed by `length` payload bytes
```

### Node -> C++

- `1 START`
- `2 AUDIO`: payload is raw PCM s16le, 16 kHz, mono
- `3 FLUSH`
- `4 STOP`

### C++ -> Node

- `101 READY`: UTF-8 message
- `102 PARTIAL`: UTF-8 partial transcript
- `103 FINAL`: UTF-8 final transcript
- `104 SPEECH_START`
- `105 SPEECH_END`
- `199 ERROR`: UTF-8 error message

## Current streaming policy

This first implementation is intentionally simple:

1. Use RMS VAD to detect speech.
2. Buffer one utterance as 16 kHz float PCM.
3. Optionally run partial Whisper passes during speech.
4. On silence, run Whisper and emit a final transcript.

This proves the process/protocol and gives us a place to improve toward LocalAgreement / WhisperLiveKit-style stabilization later.
