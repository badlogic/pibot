import { type ChildProcess, spawn } from "node:child_process";
import type { Logger } from "./logger.js";

const workerInputSpeak = 1;
const workerInputCancel = 2;
const workerInputShutdown = 3;
const workerOutputReady = 1;
const workerOutputAudioStart = 2;
const workerOutputAudioChunk = 3;
const workerOutputAudioDone = 4;
const workerOutputError = 5;
const frameHeaderBytes = 9;

type TtsWorkerKind = "python" | "rust";

export interface TtsServiceDeps {
	workerKind: string;
	pythonWorkerPath: string;
	rustWorkerPath: string | undefined;
	rustModelPath: string | undefined;
	logger: Logger;
}

export interface TtsCallbacks {
	onStart: (sampleRate: number) => void;
	onAudio: (pcm: Uint8Array) => void;
	onDone: () => void;
	onError: (message: string) => void;
}

export interface TtsService {
	ready: Promise<void>;
	start: (callbacks: TtsCallbacks) => void;
	pushText: (text: string) => void;
	end: () => void;
	cancel: (reason: string) => void;
	stop: () => void;
}

interface QueuedRequest {
	id: number;
	text: string;
}

function envNumber(name: string, fallback: number): number {
	const value = process.env[name];
	if (value === undefined || value.trim() === "") return fallback;
	const parsed = Number(value);
	if (!Number.isFinite(parsed)) throw new Error(`${name} must be a number`);
	return parsed;
}

function makeFrame(type: number, id: number, payload: Uint8Array = new Uint8Array()): Buffer {
	const frame = Buffer.allocUnsafe(frameHeaderBytes + payload.byteLength);
	frame.writeUInt8(type, 0);
	frame.writeUInt32LE(id >>> 0, 1);
	frame.writeUInt32LE(payload.byteLength, 5);
	Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength).copy(frame, frameHeaderBytes);
	return frame;
}

function decodeUtf8(payload: Uint8Array): string {
	return Buffer.from(payload.buffer, payload.byteOffset, payload.byteLength).toString("utf8");
}

function parseWorkerKind(value: string): TtsWorkerKind {
	if (value === "python" || value === "rust") return value;
	throw new Error(`QWEN3_TTS_WORKER must be python or rust, got ${value}`);
}

export function createTtsService(deps: TtsServiceDeps): TtsService {
	const workerKind = parseWorkerKind(deps.workerKind);
	const qwen3ModelName = process.env.QWEN3_TTS_MODEL_NAME ?? "Qwen/Qwen3-TTS-12Hz-1.7B-Base";
	const qwen3RefAudio = process.env.QWEN3_TTS_REF_AUDIO ?? "data/voices/elevenlabs-pibot-reference-de.wav";
	const qwen3RefTextFile = process.env.QWEN3_TTS_REF_TEXT_FILE ?? "data/voices/elevenlabs-pibot-reference-de.txt";
	const qwen3Language = process.env.QWEN3_TTS_LANGUAGE ?? "de";
	const qwen3OutputSampleRate = envNumber("QWEN3_TTS_OUTPUT_SAMPLE_RATE", 24000);
	const qwen3Temperature = envNumber("QWEN3_TTS_TEMPERATURE", 0.7);
	const qwen3TopK = envNumber("QWEN3_TTS_TOP_K", 30);
	const qwen3Seed = process.env.QWEN3_TTS_SEED ?? "1234";

	const logger = deps.logger.tag("tts");
	const qwen3Logger = logger.tag("qwen3");
	const textEncoder = new TextEncoder();
	const queue: QueuedRequest[] = [];
	let worker: ChildProcess | undefined;
	let stdoutBuffer = Buffer.alloc(0);
	let callbacks: TtsCallbacks | undefined;
	let nextRequestId = 1;
	let activeRequestId: number | undefined;
	let turnEnded = false;
	let cancelled = false;
	let streamStarted = false;
	let resolveReady: (() => void) | undefined;
	let rejectReady: ((error: Error) => void) | undefined;

	const ready = new Promise<void>((resolve, reject) => {
		resolveReady = resolve;
		rejectReady = reject;
	});

	function sendFrame(type: number, id: number, payload?: Uint8Array): void {
		if (!worker?.stdin || worker.stdin.destroyed) throw new Error("Qwen3 TTS worker is not available");
		worker.stdin.write(makeFrame(type, id, payload));
	}

	function finishIfIdle(): void {
		if (!callbacks || activeRequestId !== undefined || queue.length > 0 || !turnEnded) return;
		const done = callbacks.onDone;
		callbacks = undefined;
		done();
	}

	function pump(): void {
		if (!callbacks || activeRequestId !== undefined) return;
		const request = queue.shift();
		if (!request) {
			finishIfIdle();
			return;
		}
		activeRequestId = request.id;
		try {
			sendFrame(workerInputSpeak, request.id, textEncoder.encode(request.text));
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			callbacks.onError(message);
			callbacks = undefined;
		}
	}

	function handleWorkerFrame(type: number, id: number, payload: Uint8Array): void {
		if (type === workerOutputReady) {
			qwen3Logger.log("ready");
			resolveReady?.();
			resolveReady = undefined;
			return;
		}
		if (!callbacks || id !== activeRequestId || cancelled) return;
		if (type === workerOutputAudioStart) {
			if (!streamStarted) {
				streamStarted = true;
				const sampleRate = payload.byteLength >= 4 ? Buffer.from(payload).readUInt32LE(0) : qwen3OutputSampleRate;
				callbacks.onStart(sampleRate);
			}
			return;
		}
		if (type === workerOutputAudioChunk) {
			callbacks.onAudio(payload);
			return;
		}
		if (type === workerOutputAudioDone) {
			activeRequestId = undefined;
			pump();
			return;
		}
		if (type === workerOutputError) {
			activeRequestId = undefined;
			callbacks.onError(decodeUtf8(payload));
			callbacks = undefined;
		}
	}

	function handleStdoutData(chunk: Buffer): void {
		stdoutBuffer = Buffer.concat([stdoutBuffer, chunk]);
		while (stdoutBuffer.byteLength >= frameHeaderBytes) {
			const type = stdoutBuffer.readUInt8(0);
			const id = stdoutBuffer.readUInt32LE(1);
			const payloadLength = stdoutBuffer.readUInt32LE(5);
			const frameLength = frameHeaderBytes + payloadLength;
			if (stdoutBuffer.byteLength < frameLength) return;
			const payload = stdoutBuffer.subarray(frameHeaderBytes, frameLength);
			stdoutBuffer = stdoutBuffer.subarray(frameLength);
			handleWorkerFrame(type, id, payload);
		}
	}

	function workerCommand(): { command: string; args: string[]; label: string } {
		const commonArgs = [
			"--serve",
			"--ref-audio",
			qwen3RefAudio,
			"--ref-text-file",
			qwen3RefTextFile,
			"--language",
			qwen3Language,
			"--output-sample-rate",
			String(qwen3OutputSampleRate),
			"--temperature",
			String(qwen3Temperature),
			"--top-k",
			String(qwen3TopK),
		];
		if (qwen3Seed.trim()) commonArgs.push("--seed", qwen3Seed);
		if (workerKind === "python") {
			const args = [
				"run",
				"--no-project",
				"--with",
				"speech-to-speech==0.2.9",
				"python",
				deps.pythonWorkerPath,
				...commonArgs,
				"--model-name",
				qwen3ModelName,
			];
			return { command: "uv", args, label: `uv ${args.join(" ")}` };
		}
		if (!deps.rustWorkerPath) throw new Error("QWEN3_TTS_RUST_WORKER_PATH is required when QWEN3_TTS_WORKER=rust");
		const args = [...commonArgs];
		const rustModelPath = deps.rustModelPath ?? process.env.QWEN3_TTS_MODEL_NAME;
		if (rustModelPath) args.push("--model-name", rustModelPath);
		return { command: deps.rustWorkerPath, args, label: `${deps.rustWorkerPath} ${args.join(" ")}` };
	}

	function startWorker(): void {
		const { command, args, label } = workerCommand();
		logger.log(`starting Qwen3 TTS ${workerKind} worker: ${label}`);
		const child = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"] });
		worker = child;
		child.stdout?.on("data", (chunk: Buffer) => handleStdoutData(chunk));
		child.stderr?.on("data", (chunk: Buffer) => {
			for (const line of chunk.toString("utf8").split(/\r?\n/)) if (line.trim()) qwen3Logger.log(line.trim());
		});
		child.once("error", (error) => {
			rejectReady?.(error);
			callbacks?.onError(error.message);
			callbacks = undefined;
		});
		child.once("exit", (code, signal) => {
			if (worker === child) worker = undefined;
			const error = new Error(`Qwen3 TTS worker exited code=${code ?? "none"} signal=${signal ?? "none"}`);
			rejectReady?.(error);
			callbacks?.onError(error.message);
			callbacks = undefined;
			if (code !== 0) logger.log(error.message);
		});
	}

	function start(nextCallbacks: TtsCallbacks): void {
		if (callbacks) cancel("new TTS stream");
		callbacks = nextCallbacks;
		queue.length = 0;
		activeRequestId = undefined;
		turnEnded = false;
		cancelled = false;
		streamStarted = false;
	}

	function pushText(text: string): void {
		const trimmed = text.trim();
		if (!callbacks || !trimmed) return;
		queue.push({ id: nextRequestId++, text: trimmed });
		pump();
	}

	function end(): void {
		turnEnded = true;
		if (!streamStarted && activeRequestId === undefined && queue.length === 0)
			callbacks?.onStart(qwen3OutputSampleRate);
		finishIfIdle();
	}

	function cancel(reason: string): void {
		qwen3Logger.log(`cancel: ${reason}`);
		cancelled = true;
		try {
			for (const request of queue) sendFrame(workerInputCancel, request.id);
			if (activeRequestId !== undefined) sendFrame(workerInputCancel, activeRequestId);
		} catch {
			// process may already be gone
		}
		queue.length = 0;
		activeRequestId = undefined;
		callbacks = undefined;
	}

	function stop(): void {
		try {
			sendFrame(workerInputShutdown, 0);
		} catch {
			// process may already be gone
		}
		worker?.kill();
	}

	startWorker();

	return { ready, start, pushText, end, cancel, stop };
}
