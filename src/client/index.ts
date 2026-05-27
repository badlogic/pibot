import type { ClientMessage, RobotState } from "../types.js";
import "./components/robot-face.js";
import type { RobotFaceElement, RobotFaceState } from "./components/robot-face.js";
import "./components/robot-log.js";
import type { RobotLogElement } from "./components/robot-log.js";
import "./components/setup-panel.js";
import type { RobotSetupPanelElement } from "./components/setup-panel.js";

import { BrowserClientLogger } from "./logger.js";
import { RobotServer } from "./robot-server.js";
import { createRobotTools } from "./tools/index.js";
import type { ConversationPhase } from "./tools/speech.js";

const setup = document.querySelector<HTMLElement>("#setup");
const robot = document.querySelector<HTMLElement>("#robot");
const logEl = document.querySelector<RobotLogElement>("#log");
const face = document.querySelector<RobotFaceElement>("#face");
const setupFaceHost = document.querySelector<HTMLElement>("#setupFaceHost");
const setupPanel = document.querySelector<RobotSetupPanelElement>("#setupPanel");
const backButton = document.querySelector<HTMLButtonElement>("#back");

if (!setup || !robot || !logEl || !face || !setupFaceHost || !setupPanel || !backButton) {
	throw new Error("Missing required DOM elements");
}

const robotLog = logEl;
const robotFace = face;
const setupPanelElement = setupPanel;
const setupSection = setup;
const robotSection = robot;
const setupFaceHostElement = setupFaceHost;

function moveFaceToSetup(): void {
	if (robotFace.parentElement === setupFaceHostElement) return;
	robotFace.classList.add("in-setup");
	setupFaceHostElement.append(robotFace);
}

function moveFaceToRobotMode(): void {
	if (robotFace.parentElement === robotSection) return;
	robotFace.classList.remove("in-setup");
	robotSection.prepend(robotFace);
}

moveFaceToSetup();
const wsProtocol = location.protocol === "https:" ? "wss" : "ws";
const targetSttSampleRate = 16000;
const clientLogger = new BrowserClientLogger();

let robotServer: RobotServer;
let recognitionWanted = false;
let micStream: MediaStream | undefined;
let micAudioContext: AudioContext | undefined;
let micSource: MediaStreamAudioSourceNode | undefined;
let micProcessor: ScriptProcessorNode | undefined;
let phase: ConversationPhase = "inactive";
let serverRobotState: RobotState = { phase: "inactive" };
let currentFaceState: RobotFaceState = "inactive";
let ignoreMicUntil = 0;
let ttsSpeaking = false;
let errorUntil = 0;
let robotStarted = false;

function stringifyLogValue(value: unknown): string {
	if (value instanceof Error) return `${value.name}: ${value.message}\n${value.stack ?? ""}`.trim();
	if (typeof value === "string") return value;
	try {
		return JSON.stringify(value);
	} catch {
		return String(value);
	}
}

window.addEventListener("error", (event) => {
	clientLogger.tag("error").log(`window error: ${event.message} at ${event.filename}:${event.lineno}:${event.colno}`);
});
window.addEventListener("unhandledrejection", (event) => {
	clientLogger.tag("error").log(`unhandled rejection: ${stringifyLogValue(event.reason)}`);
});

function log(text: string, tag = ""): void {
	(tag ? clientLogger.tag(tag) : clientLogger).log(text);
}

function send(data: ClientMessage): void {
	robotServer.send(data);
}

function deriveRobotFaceState(): RobotFaceState {
	if (Date.now() < errorUntil) return "error";
	if (serverRobotState.phase === "speaking" || ttsSpeaking) return "speaking";
	return serverRobotState.phase;
}

function renderRobotFace(): void {
	const next = deriveRobotFaceState();
	if (next === currentFaceState) return;
	currentFaceState = next;
	robotFace.state = next;
}

function showErrorFor(durationMs: number): void {
	errorUntil = Date.now() + durationMs;
	renderRobotFace();
	setTimeout(renderRobotFace, durationMs + 20);
}

function setPhase(nextPhase: ConversationPhase): void {
	if (phase !== nextPhase) {
		phase = nextPhase;
		log(`phase: ${phase}`);
	}
	renderRobotFace();
}

function setTtsSpeaking(active: boolean): void {
	ttsSpeaking = active;
	renderRobotFace();
}

function resetToListeningOrIdle(): void {
	setPhase(recognitionWanted ? "listening" : "inactive");
}

function resetRecognitionAfterTts(): void {
	ignoreMicUntil = Date.now() + 1500;
	if (recognitionWanted) setPhase("listening");
}

function setMicInputBlockedUntil(time: number): void {
	ignoreMicUntil = time;
}

const tools = createRobotTools({
	logger: clientLogger,
	ttsProviderControl: setupPanelElement.ttsProviderControl,
	face: robotFace,
	setPhase,
	resetToListeningOrIdle,
	resetRecognitionAfterTts,
	setMicInputBlockedUntil,
	onSpeakingChange: setTtsSpeaking,
});

robotServer = new RobotServer({
	url: `${wsProtocol}://${location.host}`,
	logger: clientLogger,
	tools: tools.handlers,
	events: {
		onState: (state) => {
			serverRobotState = state;
			renderRobotFace();
		},
		onLog: (entry) => robotLog.appendLine(entry.origin, entry.tags, entry.message),
		onRejected: (reason) => {
			showErrorFor(5000);
			robotLog.appendLine("client", ["error"], `connection rejected: ${reason}`);
		},
	},
});
clientLogger.setSender((message) => robotServer.send(message));

function micInputBlocked(): boolean {
	return Date.now() < ignoreMicUntil;
}

function resampleToPcm16(input: Float32Array, inputSampleRate: number): Int16Array {
	const ratio = inputSampleRate / targetSttSampleRate;
	const outputLength = Math.max(1, Math.floor(input.length / ratio));
	const output = new Int16Array(outputLength);
	for (let i = 0; i < outputLength; i++) {
		const start = Math.floor(i * ratio);
		const end = Math.min(input.length, Math.floor((i + 1) * ratio));
		let sum = 0;
		for (let j = start; j < end; j++) sum += input[j] ?? 0;
		const sample = Math.max(-1, Math.min(1, sum / Math.max(1, end - start)));
		output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
	}
	return output;
}

function sendMicAudio(input: Float32Array, sampleRate: number): void {
	if (!recognitionWanted || !robotServer.isOpen() || micInputBlocked()) return;
	const pcm = resampleToPcm16(input, sampleRate);
	const buffer = new ArrayBuffer(pcm.byteLength);
	new Uint8Array(buffer).set(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength));
	robotServer.sendBinary(buffer);
}

async function startRecognition(): Promise<void> {
	if (micStream || micAudioContext) return;
	recognitionWanted = true;
	try {
		micStream = await navigator.mediaDevices.getUserMedia({
			audio: {
				echoCancellation: true,
				noiseSuppression: true,
				autoGainControl: true,
			},
			video: false,
		});
		micAudioContext = new AudioContext();
		micSource = micAudioContext.createMediaStreamSource(micStream);
		micProcessor = micAudioContext.createScriptProcessor(4096, 1, 1);
		micProcessor.onaudioprocess = (event) => {
			const input = event.inputBuffer.getChannelData(0);
			sendMicAudio(input, event.inputBuffer.sampleRate);
		};
		micSource.connect(micProcessor);
		micProcessor.connect(micAudioContext.destination);
		setPhase("listening");
		log(`local STT started: phone PCM -> Parakeet/Silero server, browserRate=${micAudioContext.sampleRate}`, "stt");
	} catch (error) {
		recognitionWanted = false;
		setPhase("inactive");
		log(`local STT start failed: ${error instanceof Error ? error.message : String(error)}`, "stt");
	}
}

function abortCurrentAgentTurn(): void {
	tools.speech.cancelSpeech("agent turn aborted by touch");
	tools.motor.stopLocalMotorsNow();
	showErrorFor(700);
	send({ type: "abort" });
	resetToListeningOrIdle();
	resetRecognitionAfterTts();
	log("agent turn aborted by touch", "stt");
}

let reloadVersion: string | undefined;
let autoReloadPending = false;

function robotModeActive(): boolean {
	return !robotSection.hidden;
}

function requestAutoReload(reason: string): void {
	if (robotModeActive()) {
		if (!autoReloadPending) log(`Auto-reload deferred while robot mode is active: ${reason}`, "reload");
		autoReloadPending = true;
		return;
	}
	location.reload();
}

function reloadIfPending(): void {
	if (!autoReloadPending || robotModeActive()) return;
	location.reload();
}

async function pollReloadVersion(): Promise<void> {
	try {
		const response = await fetch("/__version", { cache: "no-store" });
		const data = (await response.json()) as { version?: string };
		if (!data.version) return;
		if (!reloadVersion) {
			reloadVersion = data.version;
			return;
		}
		if (data.version !== reloadVersion) requestAutoReload("server version changed");
	} catch (error) {
		clientLogger.tag("debug").log(`reload poll failed: ${error instanceof Error ? error.message : String(error)}`);
	}
}

function connectReloadSocket(reloadOnOpen = false): void {
	const reloadWs = new WebSocket(`${wsProtocol}://${location.host}/__reload`);
	reloadWs.onopen = () => {
		if (reloadOnOpen) requestAutoReload("reload socket reconnected");
	};
	reloadWs.onmessage = () => requestAutoReload("reload socket message");
	reloadWs.onclose = () => {
		setTimeout(() => connectReloadSocket(true), 500);
	};
	reloadWs.onerror = () => reloadWs.close();
}

connectReloadSocket();
void pollReloadVersion();
setInterval(() => void pollReloadVersion(), 2000);

async function enterRobotMode(): Promise<void> {
	moveFaceToRobotMode();
	setupSection.hidden = true;
	robotSection.hidden = false;
	try {
		if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
	} catch (error) {
		log(`Fullscreen request failed: ${error instanceof Error ? error.message : String(error)}`);
	}
}

async function startRobot(): Promise<void> {
	if (robotStarted) {
		void enterRobotMode();
		return;
	}
	setupPanelElement.mode = "starting";
	try {
		tools.speech.enableTts();
		log(`TTS enabled: ${tools.speech.ttsProviderLabel(tools.speech.selectedTtsProvider())}`, "stt");
		const usbOk = await tools.motor.connectFt232h(true);
		if (!usbOk) log("FT232H not connected; motor tools will report errors", "hardware");
		await tools.motor
			.startOrientationTracking()
			.catch((error) =>
				log(
					`Orientation tracking failed: ${error instanceof Error ? error.message : String(error)}`,
					"orientation",
				),
			);
		log("Camera will open on demand for photos", "camera");
		await startRecognition();
		robotStarted = true;
		setupPanelElement.mode = "started";
		await enterRobotMode();
	} catch (error) {
		robotStarted = false;
		setupPanelElement.mode = "idle";
		throw error;
	}
}

setupPanelElement.addEventListener("start-robot", () => void startRobot());

backButton.onclick = async () => {
	robot.hidden = true;
	setup.hidden = false;
	moveFaceToSetup();
	reloadIfPending();
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
		moveFaceToSetup();
		reloadIfPending();
	}
});

function handleRobotTouch(event: PointerEvent): void {
	if (event.target === backButton) return;
	event.preventDefault();
	abortCurrentAgentTurn();
}

robotSection.addEventListener("pointerdown", handleRobotTouch);

setupPanelElement.addEventListener("reset-session", () => {
	if (!confirm("Reset session? All context messages will be lost.")) return;
	send({ type: "reset_session" });
	log("session reset requested", "ui");
});

setupPanelElement.addEventListener("tts-provider-change", () => tools.speech.handleProviderChange());
