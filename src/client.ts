type ServerMessage =
	| { type: "hello"; motorLog: Array<{ t: number; command: string; durationMs: number }> }
	| { type: "sim_motor"; command: string; durationMs: number }
	| { type: "take_photo_request"; id: string }
	| { type: "error"; message: string }
	| { type: "speak_request"; id: string; text: string }
	| { type: "agent_event"; event: AgentEvent };

interface AgentMessageLike {
	role: string;
	content?: unknown;
}

type AgentEvent =
	| { type: "message_start"; message: AgentMessageLike }
	| { type: "message_update"; assistantMessageEvent?: { type: string; delta?: string } }
	| { type: "message_end"; message: AgentMessageLike }
	| { type: "tool_execution_start"; toolName: string; args: unknown }
	| { type: "other"; eventType: string };

type ConversationPhase = "idle" | "listening" | "thinking" | "speaking";
type RobotFaceState = "idle" | "listening" | "hearing" | "thinking" | "speaking" | "tool" | "error";
type SpeechRecognitionConstructor = new () => SpeechRecognition;

interface SpeechRecognitionAlternative {
	transcript: string;
}

interface SpeechRecognitionResult {
	readonly isFinal: boolean;
	readonly length: number;
	item(index: number): SpeechRecognitionAlternative;
	[index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionResultList {
	readonly length: number;
	item(index: number): SpeechRecognitionResult;
	[index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionEvent extends Event {
	readonly resultIndex: number;
	readonly results: SpeechRecognitionResultList;
}

interface SpeechRecognitionErrorEvent extends Event {
	readonly error: string;
}

interface SpeechRecognition extends EventTarget {
	continuous: boolean;
	interimResults: boolean;
	lang: string;
	onstart: (() => void) | null;
	onend: (() => void) | null;
	onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
	onresult: ((event: SpeechRecognitionEvent) => void) | null;
	start(): void;
	stop(): void;
}

declare global {
	interface Window {
		SpeechRecognition?: SpeechRecognitionConstructor;
		webkitSpeechRecognition?: SpeechRecognitionConstructor;
	}
}

const setup = document.querySelector<HTMLElement>("#setup");
const robot = document.querySelector<HTMLElement>("#robot");
const logEl = document.querySelector<HTMLElement>("#log");
const face = document.querySelector<HTMLElement>("#face");
const promptInput = document.querySelector<HTMLInputElement>("#prompt");
const sendButton = document.querySelector<HTMLButtonElement>("#send");
const robotModeButton = document.querySelector<HTMLButtonElement>("#robotMode");
const backButton = document.querySelector<HTMLButtonElement>("#back");
const micButton = document.querySelector<HTMLButtonElement>("#mic");
const stopMicButton = document.querySelector<HTMLButtonElement>("#stopMic");
const testTtsButton = document.querySelector<HTMLButtonElement>("#testTts");
const enableCameraButton = document.querySelector<HTMLButtonElement>("#enableCamera");

if (
	!setup ||
	!robot ||
	!logEl ||
	!face ||
	!promptInput ||
	!sendButton ||
	!robotModeButton ||
	!backButton ||
	!micButton ||
	!stopMicButton ||
	!testTtsButton ||
	!enableCameraButton
) {
	throw new Error("Missing required DOM elements");
}

const logOutput = logEl;
const robotFace = face;
const wsProtocol = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${wsProtocol}://${location.host}`);
const ttsEnabledKey = "robot-tts-enabled";
const isAndroid = /Android/i.test(navigator.userAgent);
const recognitionRestartDelayMs = isAndroid ? 1800 : 300;
const androidMaxFastRecognitionEnds = 2;

let recognition: SpeechRecognition | undefined;
let recognitionWanted = false;
let recognitionRestartTimer: ReturnType<typeof setTimeout> | undefined;
let recognitionStartedAt = 0;
let recognitionHadFinalResult = false;
let androidFastRecognitionEnds = 0;
let assistantSpeechBuffer = "";
let ttsEnabled = localStorage.getItem(ttsEnabledKey) === "true";
let phase: ConversationPhase = "idle";
let ignoreMicUntil = 0;
let currentTtsAudio: HTMLAudioElement | undefined;
let audioContext: AudioContext | undefined;
let activeRobotEffectNodes: AudioNode[] = [];
let ttsGeneration = 0;
let activeSpeakRequestId: string | undefined;
let cameraStream: MediaStream | undefined;
let cameraVideo: HTMLVideoElement | undefined;
const cameraEnabledKey = "robot-camera-enabled";
let cameraEnabled = localStorage.getItem(cameraEnabledKey) === "true";

function stringifyLogValue(value: unknown): string {
	if (value instanceof Error) return `${value.name}: ${value.message}\n${value.stack ?? ""}`.trim();
	if (typeof value === "string") return value;
	try {
		return JSON.stringify(value);
	} catch {
		return String(value);
	}
}

function sendClientLog(level: "log" | "info" | "warn" | "error" | "debug" | "app", message: string): void {
	const payload = {
		type: "client_log",
		level,
		message: message.slice(0, 4000),
		url: location.href,
		userAgent: navigator.userAgent,
		time: Date.now(),
	};
	const body = JSON.stringify(payload);
	if (ws.readyState === WebSocket.OPEN) {
		ws.send(body);
		return;
	}
	if (navigator.sendBeacon) {
		navigator.sendBeacon("/api/client-log", new Blob([body], { type: "application/json" }));
		return;
	}
	void fetch("/api/client-log", {
		method: "POST",
		body,
		headers: { "content-type": "application/json" },
		keepalive: true,
	});
}

function installClientLogForwarding(): void {
	for (const level of ["log", "info", "warn", "error", "debug"] as const) {
		const original = console[level].bind(console);
		console[level] = (...args: unknown[]) => {
			original(...args);
			sendClientLog(level, args.map(stringifyLogValue).join(" "));
		};
	}
	window.addEventListener("error", (event) => {
		sendClientLog("error", `window error: ${event.message} at ${event.filename}:${event.lineno}:${event.colno}`);
	});
	window.addEventListener("unhandledrejection", (event) => {
		sendClientLog("error", `unhandled rejection: ${stringifyLogValue(event.reason)}`);
	});
}

installClientLogForwarding();

function log(text: string, className = ""): void {
	const line = document.createElement("div");
	line.textContent = `${new Date().toLocaleTimeString()} ${text}`;
	if (className) line.className = className;
	logOutput.append(line);
	logOutput.scrollTop = logOutput.scrollHeight;
	console.info(`[app] ${text}`);
}

function send(data: unknown): void {
	if (ws.readyState !== WebSocket.OPEN) {
		sendClientLog("warn", `WebSocket not open; dropping message ${JSON.stringify(data).slice(0, 500)}`);
		return;
	}
	ws.send(JSON.stringify(data));
}

function setRobotFaceState(state: RobotFaceState): void {
	robotFace.className = `face ${state}`;
}

async function ensureCameraStream(): Promise<MediaStream> {
	if (cameraStream?.getVideoTracks().every((track) => track.readyState === "live")) return cameraStream;
	if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API unavailable");
	const stream = await navigator.mediaDevices.getUserMedia({
		video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
		audio: false,
	});
	cameraStream = stream;
	localStorage.setItem(cameraEnabledKey, "true");
	cameraEnabled = true;
	if (!cameraVideo) {
		cameraVideo = document.createElement("video");
		cameraVideo.muted = true;
		cameraVideo.playsInline = true;
		cameraVideo.autoplay = true;
		cameraVideo.style.position = "fixed";
		cameraVideo.style.width = "1px";
		cameraVideo.style.height = "1px";
		cameraVideo.style.opacity = "0";
		cameraVideo.style.pointerEvents = "none";
		document.body.append(cameraVideo);
	}
	cameraVideo.srcObject = stream;
	await cameraVideo.play().catch(() => undefined);
	return stream;
}

async function capturePhotoDataUrl(): Promise<string> {
	await ensureCameraStream();
	const video = cameraVideo;
	if (!video) throw new Error("Camera video element missing");
	if (video.readyState < 2) {
		await new Promise<void>((resolve) => {
			const handler = () => {
				video.removeEventListener("loadeddata", handler);
				resolve();
			};
			video.addEventListener("loadeddata", handler);
		});
	}
	const width = video.videoWidth || 640;
	const height = video.videoHeight || 480;
	const canvas = document.createElement("canvas");
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext("2d");
	if (!ctx) throw new Error("Canvas 2d context unavailable");
	ctx.drawImage(video, 0, 0, width, height);
	return canvas.toDataURL("image/jpeg", 0.82);
}

async function handlePhotoRequest(id: string): Promise<void> {
	try {
		const dataUrl = await capturePhotoDataUrl();
		send({ type: "photo_result", id, dataUrl });
		log(`Captured photo for tool request ${id} (${dataUrl.length} chars)`, "agent");
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error);
		send({ type: "photo_result", id, error: message });
		log(`Photo capture failed: ${message}`, "agent");
	}
}

function setPhase(nextPhase: ConversationPhase): void {
	phase = nextPhase;
	if (nextPhase === "idle") setRobotFaceState("idle");
	if (nextPhase === "listening") setRobotFaceState("listening");
	if (nextPhase === "thinking") setRobotFaceState("thinking");
	if (nextPhase === "speaking") setRobotFaceState("speaking");
	log(`phase: ${phase}`);
}

function assistantMessageHasToolCall(message: AgentMessageLike): boolean {
	if (!Array.isArray(message.content)) return false;
	return message.content.some(
		(content) => typeof content === "object" && content !== null && "type" in content && content.type === "toolCall",
	);
}

function normalizeCommandText(text: string): string {
	return text
		.toLowerCase()
		.normalize("NFKD")
		.replace(/[.,!?;:()[\]{}"'`´]/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function containsStopCommand(text: string): boolean {
	const normalized = normalizeCommandText(text);
	if (!normalized) return false;
	const stopPhrases = [
		"stop",
		"stopp",
		"halt",
		"anhalten",
		"abbrechen",
		"schluss",
		"ruhe",
		"sei still",
		"hör auf",
		"hoer auf",
	];
	return stopPhrases.some(
		(phrase) =>
			normalized === phrase ||
			normalized.includes(` ${phrase} `) ||
			normalized.startsWith(`${phrase} `) ||
			normalized.endsWith(` ${phrase}`),
	);
}

function clearCurrentTtsAudio(): void {
	if (activeRobotEffectNodes.length > 0)
		log(`Clearing ${activeRobotEffectNodes.length} robot voice effect nodes`, "stt");
	if (!currentTtsAudio) return;
	currentTtsAudio.pause();
	currentTtsAudio.src = "";
	currentTtsAudio = undefined;
	activeRobotEffectNodes = [];
}

function createRobotEffect(audio: HTMLAudioElement): void {
	try {
		audioContext ??= new AudioContext();
		void audioContext.resume();
		const source = audioContext.createMediaElementSource(audio);

		const highpass = audioContext.createBiquadFilter();
		highpass.type = "highpass";
		highpass.frequency.value = 130;

		const lowpass = audioContext.createBiquadFilter();
		lowpass.type = "lowpass";
		lowpass.frequency.value = 6500;

		const presence = audioContext.createBiquadFilter();
		presence.type = "peaking";
		presence.frequency.value = 2400;
		presence.Q.value = 1;
		presence.gain.value = 4;

		const dryGain = audioContext.createGain();
		dryGain.gain.value = 0.82;

		const ringWetGain = audioContext.createGain();
		ringWetGain.gain.value = 0.22;
		const ringModulator = audioContext.createGain();
		ringModulator.gain.value = 0;
		const ringOsc = audioContext.createOscillator();
		ringOsc.type = "sine";
		ringOsc.frequency.value = 55;
		ringOsc.connect(ringModulator.gain);
		ringOsc.start();

		const slap = audioContext.createDelay(0.3);
		slap.delayTime.value = 0.085;
		const slapFeedback = audioContext.createGain();
		slapFeedback.gain.value = 0.18;
		const slapWet = audioContext.createGain();
		slapWet.gain.value = 0.12;

		const flutterLfo = audioContext.createOscillator();
		flutterLfo.type = "sine";
		flutterLfo.frequency.value = 6.5;
		const flutterDepth = audioContext.createGain();
		flutterDepth.gain.value = 0.04;
		const flutter = audioContext.createGain();
		flutter.gain.value = 0.96;
		flutterLfo.connect(flutterDepth);
		flutterDepth.connect(flutter.gain);
		flutterLfo.start();

		const output = audioContext.createGain();
		output.gain.value = 0.95;

		source.connect(highpass);
		highpass.connect(lowpass);
		lowpass.connect(presence);

		presence.connect(dryGain);
		presence.connect(ringModulator);
		ringModulator.connect(ringWetGain);

		dryGain.connect(flutter);
		ringWetGain.connect(flutter);

		flutter.connect(output);
		flutter.connect(slap);
		slap.connect(slapFeedback);
		slapFeedback.connect(slap);
		slap.connect(slapWet);
		slapWet.connect(output);
		output.connect(audioContext.destination);
		activeRobotEffectNodes = [
			source,
			highpass,
			lowpass,
			presence,
			dryGain,
			ringModulator,
			ringWetGain,
			ringOsc,
			flutter,
			flutterLfo,
			flutterDepth,
			slap,
			slapFeedback,
			slapWet,
			output,
		];
		log("Robot voice effect enabled (glados-ish)", "stt");
	} catch (error) {
		activeRobotEffectNodes = [];
		log(`Robot voice effect unavailable: ${error instanceof Error ? error.message : String(error)}`, "stt");
	}
}

function ttsOutputActive(): boolean {
	return phase === "speaking" || (currentTtsAudio !== undefined && !currentTtsAudio.paused && !currentTtsAudio.ended);
}

function micInputBlocked(): boolean {
	return Date.now() < ignoreMicUntil || ttsOutputActive();
}

function scheduleRecognitionRestart(delayMs: number): void {
	if (recognitionRestartTimer) clearTimeout(recognitionRestartTimer);
	recognitionRestartTimer = setTimeout(() => {
		recognitionRestartTimer = undefined;
		if (recognitionWanted) startRecognition();
	}, delayMs);
}

function startRecognition(): void {
	if (recognition) return;
	const SpeechRecognitionApi = window.SpeechRecognition ?? window.webkitSpeechRecognition;
	if (!SpeechRecognitionApi) {
		log("SpeechRecognition unavailable. Android Chrome usually has it; desktop support varies.");
		return;
	}

	recognitionStartedAt = Date.now();
	recognitionHadFinalResult = false;
	recognition = new SpeechRecognitionApi();
	recognition.continuous = true;
	recognition.interimResults = true;
	recognition.lang = "de-DE";
	recognition.onstart = () => {
		if (phase === "idle") setPhase("listening");
		log("continuous STT started: de-DE", "stt");
	};
	recognition.onerror = (event) => log(`stt error ${event.error}`, "stt");
	recognition.onend = () => {
		recognition = undefined;
		const lifetimeMs = Date.now() - recognitionStartedAt;
		log(`stt ended after ${lifetimeMs}ms`, "stt");
		if (isAndroid && !recognitionHadFinalResult && lifetimeMs < 3000) {
			androidFastRecognitionEnds++;
			log(`Android STT fast-end ${androidFastRecognitionEnds}/${androidMaxFastRecognitionEnds}`, "stt");
		} else {
			androidFastRecognitionEnds = 0;
		}
		if (isAndroid && androidFastRecognitionEnds > androidMaxFastRecognitionEnds) {
			recognitionWanted = false;
			setPhase("idle");
			log("Android Web Speech is ending immediately; stopped auto-restart to avoid mic chime loop.", "stt");
			return;
		}
		if (recognitionWanted) scheduleRecognitionRestart(recognitionRestartDelayMs);
	};
	recognition.onresult = handleRecognitionResult;

	try {
		recognition.start();
	} catch (error) {
		recognition = undefined;
		log(`stt start failed: ${error instanceof Error ? error.message : String(error)}`, "stt");
		if (recognitionWanted) scheduleRecognitionRestart(Math.max(750, recognitionRestartDelayMs));
	}
}

function stopRecognition(): void {
	recognitionWanted = false;
	if (recognitionRestartTimer) clearTimeout(recognitionRestartTimer);
	recognitionRestartTimer = undefined;
	recognition?.stop();
	recognition = undefined;
	setPhase("idle");
}

function resetRecognitionAfterTts(): void {
	ignoreMicUntil = Date.now() + 1500;
	if (recognitionWanted && !recognition) scheduleRecognitionRestart(recognitionRestartDelayMs);
}

function interruptTtsOnly(): void {
	ttsGeneration++;
	clearCurrentTtsAudio();
	if ("speechSynthesis" in window) window.speechSynthesis.cancel();
}

function interruptSpeech(): void {
	const requestId = activeSpeakRequestId;
	interruptTtsOnly();
	if (requestId) send({ type: "speak_cancelled", id: requestId });
	activeSpeakRequestId = undefined;
	ignoreMicUntil = Date.now() + 500;
	setRobotFaceState("error");
	send({ type: "abort" });
	setPhase(recognitionWanted ? "listening" : "idle");
	resetRecognitionAfterTts();
	log("TTS stopped, agent aborted", "stt");
}

function enableTts(): void {
	ttsEnabled = true;
	localStorage.setItem(ttsEnabledKey, "true");
}

function finishTts(message: string): void {
	clearCurrentTtsAudio();
	const requestId = activeSpeakRequestId;
	activeSpeakRequestId = undefined;
	if (requestId) send({ type: "speak_done", id: requestId });
	ignoreMicUntil = Date.now() + 500;
	setPhase(recognitionWanted ? "listening" : "idle");
	resetRecognitionAfterTts();
	log(message, "stt");
}

function speakGerman(text: string, requestId?: string): void {
	const trimmed = text.trim();
	if (!trimmed) {
		finishTts("TTS skipped: empty text");
		return;
	}

	const generation = ++ttsGeneration;
	activeSpeakRequestId = requestId;
	clearCurrentTtsAudio();
	setPhase("speaking");
	ignoreMicUntil = Number.POSITIVE_INFINITY;

	const audio = new Audio(`/api/tts?text=${encodeURIComponent(trimmed)}`);
	currentTtsAudio = audio;
	createRobotEffect(audio);
	audio.onplay = () => log(`ElevenLabs TTS playing full response ${trimmed.length} chars`, "stt");
	audio.onended = () => {
		if (generation !== ttsGeneration) return;
		finishTts("ElevenLabs TTS finished, resetting STT");
	};
	audio.onerror = () => {
		if (generation !== ttsGeneration) return;
		finishTts("ElevenLabs TTS failed, resetting STT");
	};
	audio.play().catch((error: unknown) => {
		if (generation !== ttsGeneration) return;
		finishTts(`ElevenLabs TTS play failed, resetting STT: ${error instanceof Error ? error.message : String(error)}`);
	});
}

function handleRecognitionResult(event: SpeechRecognitionEvent): void {
	let interim = "";
	let final = "";
	for (let i = event.resultIndex; i < event.results.length; i++) {
		const text = event.results[i]?.[0]?.transcript ?? "";
		if (event.results[i]?.isFinal) final += text;
		else interim += text;
	}

	const heardText = `${interim} ${final}`;
	if (micInputBlocked()) {
		if (containsStopCommand(heardText)) {
			log(`Stop detected during TTS/block: ${normalizeCommandText(heardText)}`, "stt");
			interruptSpeech();
		} else if (heardText.trim()) {
			log(`Mic blocked, ignored: ${heardText.trim()}`, "stt");
		}
		return;
	}

	if (phase === "thinking") {
		if (containsStopCommand(heardText)) {
			log(`Stop detected while thinking: ${normalizeCommandText(heardText)}`, "stt");
			interruptSpeech();
		}
		return;
	}

	if (interim) setRobotFaceState("hearing");
	if (!final.trim()) return;

	recognitionHadFinalResult = true;
	log(`STT final: ${final}`, "stt");
	setPhase("thinking");
	send({ type: "stt", text: final, final: true });
}

let reloadVersion: string | undefined;

async function pollReloadVersion(): Promise<void> {
	try {
		const response = await fetch("/__version", { cache: "no-store" });
		const data = (await response.json()) as { version?: string };
		if (!data.version) return;
		if (!reloadVersion) {
			reloadVersion = data.version;
			return;
		}
		if (data.version !== reloadVersion) location.reload();
	} catch (error) {
		sendClientLog("debug", `reload poll failed: ${error instanceof Error ? error.message : String(error)}`);
	}
}

function connectReloadSocket(reloadOnOpen = false): void {
	const reloadWs = new WebSocket(`${wsProtocol}://${location.host}/__reload`);
	reloadWs.onopen = () => {
		if (reloadOnOpen) location.reload();
	};
	reloadWs.onmessage = () => location.reload();
	reloadWs.onclose = () => {
		setTimeout(() => connectReloadSocket(true), 500);
	};
	reloadWs.onerror = () => reloadWs.close();
}

connectReloadSocket();
void pollReloadVersion();
setInterval(() => void pollReloadVersion(), 2000);

ws.onopen = () => log("ws connected");
ws.onclose = (event) => log(`ws closed code=${event.code} reason=${event.reason || "none"}`);
ws.onerror = () => log("ws error", "agent");
ws.onmessage = (event) => {
	const message = JSON.parse(String(event.data)) as ServerMessage;
	if (message.type === "sim_motor") {
		log(`SIM MOTOR ${message.command} ${message.durationMs}ms`, "sim");
		setRobotFaceState(message.command === "stop" ? "listening" : "tool");
	}
	if (message.type === "take_photo_request") {
		log(`photo requested ${message.id}`, "agent");
		void handlePhotoRequest(message.id);
	}
	if (message.type === "error") {
		setPhase(recognitionWanted ? "listening" : "idle");
		log(`ERROR ${message.message}`);
	}
	if (message.type === "speak_request") {
		if (ttsEnabled) speakGerman(message.text, message.id);
		else send({ type: "speak_done", id: message.id });
	}
	if (message.type === "agent_event") {
		const agentEvent = message.event;
		if (agentEvent.type === "message_start" && agentEvent.message.role === "assistant") {
			setPhase("thinking");
			assistantSpeechBuffer = "";
		}
		if (agentEvent.type === "message_update" && agentEvent.assistantMessageEvent?.type === "text_delta") {
			const delta = agentEvent.assistantMessageEvent.delta ?? "";
			assistantSpeechBuffer += delta;
		}
		if (agentEvent.type === "message_end" && agentEvent.message.role === "assistant") {
			log(`LLM: ${assistantSpeechBuffer.trim()}`, "agent");
			if (assistantMessageHasToolCall(agentEvent.message)) log("LLM message contains tool call", "agent");
			if (!ttsEnabled) setPhase(recognitionWanted ? "listening" : "idle");
		}
		if (agentEvent.type === "tool_execution_start") {
			log(`tool ${agentEvent.toolName} ${JSON.stringify(agentEvent.args)}`, "agent");
		}
	}
};

sendButton.onclick = () => {
	const text = promptInput.value;
	setPhase("thinking");
	send({ type: "prompt", text });
	log(`typed: ${text}`);
};

robotModeButton.onclick = async () => {
	setup.hidden = true;
	robot.hidden = false;
	try {
		if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
	} catch (error) {
		log(`Fullscreen request failed: ${error instanceof Error ? error.message : String(error)}`);
	}
};

backButton.onclick = async () => {
	robot.hidden = true;
	setup.hidden = false;
	try {
		if (document.fullscreenElement) await document.exitFullscreen();
	} catch (error) {
		log(`Fullscreen exit failed: ${error instanceof Error ? error.message : String(error)}`);
	}
};

document.addEventListener("fullscreenchange", () => {
	if (!document.fullscreenElement && !robot.hidden) {
		robot.hidden = true;
		setup.hidden = false;
	}
});

robotFace.onclick = () => {
	if (ttsOutputActive()) interruptSpeech();
};

micButton.onclick = () => {
	recognitionWanted = true;
	androidFastRecognitionEnds = 0;
	log(`STT start requested; restartDelay=${recognitionRestartDelayMs}ms`, "stt");
	startRecognition();
};

stopMicButton.onclick = stopRecognition;

enableCameraButton.onclick = async () => {
	try {
		await ensureCameraStream();
		log("Camera enabled", "agent");
	} catch (error) {
		log(`Camera enable failed: ${error instanceof Error ? error.message : String(error)}`, "agent");
	}
};

if (cameraEnabled) void ensureCameraStream().catch(() => undefined);

testTtsButton.onclick = () => {
	enableTts();
	speakGerman("Hallo, ich bin dein kleiner Roboter. Die Sprachausgabe ist bereit.");
	log("TTS enabled", "stt");
};

log(
	"STT options: Web Speech API is quickest, but browser/cloud-backed and may stop after pauses. Later: native Android STT, Whisper streaming, or Vosk/Sherpa ONNX.",
);
