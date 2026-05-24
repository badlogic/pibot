# Phone Robot Agent Demo

Goal: prove continuous-ish STT + `pi-agent-core` harness + simulated robot movement tools.

## STT choice for the proof-out

Fastest demo: browser Web Speech API (`SpeechRecognition` / `webkitSpeechRecognition`) in Android Chrome.

Pros: no server audio plumbing, easy permission flow.  
Cons: browser/vendor dependent, often cloud-backed, not truly forever-continuous; it may stop and need restart logic.

Later options:

- Android native SpeechRecognizer in a small wrapper app/WebView
- streaming Whisper/Groq/Deepgram server-side
- local Vosk/Sherpa ONNX if offline matters

## Run

```bash
cd robot-llm
npm install
OPENAI_API_KEY=... npm run dev
```

Open locally:

```text
http://localhost:8010
```

On phone, for STT only HTTP LAN is usually OK. For WebUSB/camera later, use HTTPS or Chrome's insecure-origin-as-secure flag.

## What it does

- Browser page has setup screen and robot face screen.
- Continuous STT sends final utterances to the backend.
- Backend uses `AgentHarness` from `@earendil-works/pi-agent-core`.
- Tools are simulated and broadcast back to the UI:
  - `move_forward`
  - `move_backward`
  - `turn_left`
  - `turn_right`
  - `stop`
  - `set_expression`
  - `take_photo` stub

Motor/WebUSB integration is intentionally not connected here yet, so debugging the agent cannot move real hardware.
