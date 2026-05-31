import { type ChildProcess, spawn } from "node:child_process";
import { once } from "node:events";
import { createWriteStream, type Dirent, existsSync } from "node:fs";
import { mkdir, readdir, rename, stat, unlink } from "node:fs/promises";
import { dirname, join } from "node:path";
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
const defaultRustTtsModelRepo = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit";
const ignoredHuggingFaceFiles = new Set([".gitattributes", "README.md"]);
const requiredRustTtsModelFiles = [
	"config.json",
	"model.safetensors",
	"vocab.json",
	"merges.txt",
	"speech_tokenizer/model.safetensors",
] as const;

type TtsWorkerKind = "python" | "rust";

export interface TtsServiceDeps {
	workerKind: string;
	pythonCommand: string;
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

interface DownloadFile {
	url: string;
	path: string;
	label: string;
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

function shouldLogQwen3Line(line: string): boolean {
	if (line.startsWith("{")) return true;
	if (/^(ready|cancel:|error|failed|traceback)/i.test(line)) return true;
	if (
		/^(ICL voice clone|Reference text:|Synthesis text:|Reference codec frames:|ref_text tokens:|Building ICL|Built ICL|Generating audio codes|Generated \d+ code frames|Streaming decode|Streaming codes tensor shape:|Streaming vocoder output shape:|EOS detected)/.test(
			line,
		)
	) {
		return false;
	}
	if (/^(Loaded|Loading|Found |Audio encoder input:|After |Before |Encoded codes:|Encoded \d+ frames)/.test(line))
		return false;
	return true;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function huggingFaceFileUrl(repo: string, file: string): string {
	return `https://huggingface.co/${repo}/resolve/main/${file.split("/").map(encodeURIComponent).join("/")}`;
}

async function hasUsableFile(path: string): Promise<boolean> {
	try {
		return (await stat(path)).size > 0;
	} catch {
		return false;
	}
}

async function hasRequiredRustTtsModelFiles(modelDir: string): Promise<boolean> {
	for (const file of requiredRustTtsModelFiles) {
		if (!(await hasUsableFile(join(modelDir, file)))) return false;
	}
	return true;
}

async function listHuggingFaceFiles(repo: string): Promise<string[]> {
	const response = await fetch(`https://huggingface.co/api/models/${repo}`);
	if (!response.ok) throw new Error(`failed to list ${repo}: HTTP ${response.status}`);
	const value: unknown = await response.json();
	if (!isRecord(value) || !Array.isArray(value.siblings))
		throw new Error(`invalid Hugging Face model API response for ${repo}`);
	const files: string[] = [];
	for (const sibling of value.siblings) {
		if (!isRecord(sibling) || typeof sibling.rfilename !== "string") continue;
		files.push(sibling.rfilename);
	}
	return files;
}

async function downloadFile(file: DownloadFile, logger: Logger): Promise<void> {
	const tmpPath = `${file.path}.tmp-${process.pid}`;
	await unlink(tmpPath).catch(() => undefined);
	await mkdir(dirname(file.path), { recursive: true });
	logger.log(`downloading ${file.label}`);
	const response = await fetch(file.url);
	if (!response.ok || !response.body) throw new Error(`failed to download ${file.label}: HTTP ${response.status}`);
	const total = Number(response.headers.get("content-length") ?? "0");
	const reader = response.body.getReader();
	const output = createWriteStream(tmpPath, { flags: "wx" });
	let received = 0;
	let lastLog = Date.now();
	try {
		while (true) {
			const chunk = await reader.read();
			if (chunk.done) break;
			received += chunk.value.byteLength;
			if (!output.write(chunk.value)) await once(output, "drain");
			if (Date.now() - lastLog > 5000) {
				lastLog = Date.now();
				const suffix = total > 0 ? ` / ${(total / 1024 / 1024).toFixed(1)} MiB` : "";
				logger.log(`downloading ${file.label}: ${(received / 1024 / 1024).toFixed(1)} MiB${suffix}`);
			}
		}
		output.end();
		await once(output, "finish");
		await rename(tmpPath, file.path);
		const suffix = total > 0 ? ` / ${(total / 1024 / 1024).toFixed(1)} MiB` : "";
		logger.log(`downloaded ${file.label}: ${(received / 1024 / 1024).toFixed(1)} MiB${suffix}`);
	} catch (error) {
		output.destroy();
		await unlink(tmpPath).catch(() => undefined);
		throw error;
	}
}

async function cleanupStaleDownloads(dir: string): Promise<void> {
	let entries: Dirent[];
	try {
		entries = await readdir(dir, { withFileTypes: true });
	} catch {
		return;
	}
	for (const entry of entries) {
		const path = join(dir, entry.name);
		if (entry.isDirectory()) {
			await cleanupStaleDownloads(path);
			continue;
		}
		if (entry.isFile() && entry.name.includes(".tmp-")) await unlink(path).catch(() => undefined);
	}
}

async function ensureRustTtsModel(modelDir: string, logger: Logger): Promise<void> {
	await mkdir(modelDir, { recursive: true });
	await cleanupStaleDownloads(modelDir);
	if (await hasRequiredRustTtsModelFiles(modelDir)) return;

	const repo = process.env.QWEN3_TTS_RUST_MODEL_REPO ?? defaultRustTtsModelRepo;
	logger.log(`provisioning Rust Qwen3 TTS model ${repo} into ${modelDir}`);
	const files = (await listHuggingFaceFiles(repo)).filter((file) => !ignoredHuggingFaceFiles.has(file));
	for (const file of files) {
		const path = join(modelDir, file);
		if (await hasUsableFile(path)) continue;
		await downloadFile({ url: huggingFaceFileUrl(repo, file), path, label: `Qwen3 TTS model file ${file}` }, logger);
	}

	if (!(await hasRequiredRustTtsModelFiles(modelDir))) {
		throw new Error(`Rust Qwen3 TTS model is incomplete after download: ${modelDir}`);
	}
}

export function createTtsService(deps: TtsServiceDeps): TtsService {
	const workerKind = parseWorkerKind(deps.workerKind);
	const qwen3ModelName = process.env.QWEN3_TTS_MODEL_NAME ?? "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit";
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
			const directArgs = [deps.pythonWorkerPath, ...commonArgs, "--model-name", qwen3ModelName];
			if (deps.pythonCommand === "uv") {
				const args = ["run", "--no-project", "--with", "speech-to-speech==0.2.9", "python", ...directArgs];
				return { command: deps.pythonCommand, args, label: `${deps.pythonCommand} ${args.join(" ")}` };
			}
			return {
				command: deps.pythonCommand,
				args: directArgs,
				label: `${deps.pythonCommand} ${directArgs.join(" ")}`,
			};
		}
		if (!deps.rustWorkerPath) throw new Error("QWEN3_TTS_RUST_WORKER_PATH is required when QWEN3_TTS_WORKER=rust");
		if (!existsSync(deps.rustWorkerPath)) {
			throw new Error(`Rust Qwen3 TTS worker binary missing: ${deps.rustWorkerPath}. Run npm run build:tts-rust.`);
		}
		const args = [...commonArgs];
		const rustModelPath = deps.rustModelPath ?? process.env.QWEN3_TTS_MODEL_NAME;
		if (!rustModelPath) throw new Error("QWEN3_TTS_RUST_MODEL_PATH is required when QWEN3_TTS_WORKER=rust");
		args.push("--model-name", rustModelPath);
		return { command: deps.rustWorkerPath, args, label: `${deps.rustWorkerPath} ${args.join(" ")}` };
	}

	async function startWorkerAsync(): Promise<void> {
		if (workerKind === "rust") {
			const rustModelPath = deps.rustModelPath ?? process.env.QWEN3_TTS_MODEL_NAME;
			if (!rustModelPath) throw new Error("QWEN3_TTS_RUST_MODEL_PATH is required when QWEN3_TTS_WORKER=rust");
			await ensureRustTtsModel(rustModelPath, logger);
		}
		const { command, args, label } = workerCommand();
		logger.log(`starting Qwen3 TTS ${workerKind} worker: ${label}`);
		const child = spawn(command, args, { stdio: ["pipe", "pipe", "pipe"] });
		worker = child;
		child.stdout?.on("data", (chunk: Buffer) => handleStdoutData(chunk));
		child.stderr?.on("data", (chunk: Buffer) => {
			for (const line of chunk.toString("utf8").split(/\r?\n/)) {
				const trimmed = line.trim();
				if (trimmed && shouldLogQwen3Line(trimmed)) qwen3Logger.log(trimmed);
			}
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

	function startWorker(): void {
		void startWorkerAsync().catch((error) => {
			const normalized = error instanceof Error ? error : new Error(String(error));
			logger.log(normalized.message);
			rejectReady?.(normalized);
			callbacks?.onError(normalized.message);
			callbacks = undefined;
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
