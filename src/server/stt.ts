import { type ChildProcess, spawn } from "node:child_process";
import { once } from "node:events";
import { createWriteStream, existsSync } from "node:fs";
import { mkdir, rename, stat, unlink } from "node:fs/promises";
import { join } from "node:path";
import type { Logger } from "./logger.js";

export interface SttServiceDeps {
	workerBinaryPath: string;
	modelDir: string;
	logger: Logger;
	onEvent: (event: SttEvent) => void;
}

export interface SttService {
	handleAudioFrame: (data: Buffer) => void;
	stopChildProcess: () => void;
}

export type SttEvent =
	| {
			type: "ready";
			sampleRate: number;
			vadChunkMs: number;
			vadThreshold: number;
			minSilenceMs: number;
			prerollMs: number;
			interimIntervalMs?: number;
	  }
	| { type: "speech_start"; index: number }
	| { type: "speech_end"; index: number; duration: number }
	| { type: "speech_drop"; index: number; duration: number; reason: string }
	| { type: "interim"; index: number; text: string; audioMs: number; decodeMs: number }
	| { type: "final"; index: number; text: string; decodeMs: number }
	| { type: "error"; message: string };

const PARAKEET_TDT_REPO = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main";
const PARAKEET_TDT_FILES = ["encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", "vocab.txt"] as const;

type SttWorkerMsg =
	| {
			type: "ready";
			sampleRate: number;
			vadChunkMs: number;
			vadThreshold: number;
			minSilenceMs: number;
			speechPadMs: number;
			prerollMs: number;
			interimIntervalMs?: number;
	  }
	| { type: "speech_start"; index: number; time: number }
	| { type: "speech_end"; index: number; duration: number }
	| { type: "speech_drop"; index: number; duration: number; reason: string }
	| { type: "interim"; index: number; text: string; audioMs: number; decodeMs: number }
	| { type: "final"; index: number; text: string; duration: number; decodeMs: number }
	| { type: "error"; message: string };

function streamLines(stream: NodeJS.ReadableStream | null | undefined, onLine: (line: string) => void): void {
	if (!stream) return;
	let buffered = "";
	stream.on("data", (chunk: Buffer | string) => {
		buffered += chunk.toString();
		while (true) {
			const newline = buffered.indexOf("\n");
			if (newline < 0) return;
			const line = buffered.slice(0, newline).trim();
			buffered = buffered.slice(newline + 1);
			if (line) onLine(line);
		}
	});
	stream.on("end", () => {
		const line = buffered.trim();
		buffered = "";
		if (line) onLine(line);
	});
}

export function createSttService(deps: SttServiceDeps): SttService {
	let childProcess: ChildProcess | undefined;
	let stdout = "";
	const logger = deps.logger.tag("stt");
	const emit = deps.onEvent;

	function startWorker(): void {
		void startWorkerAsync().catch((error) => {
			emit({ type: "error", message: error instanceof Error ? error.message : String(error) });
		});
	}

	async function startWorkerAsync(): Promise<void> {
		if (childProcess && !childProcess.killed) return;
		stdout = "";
		const command = workerCommand();
		await ensureModelFiles();
		logger.log("loading Rust Parakeet/Silero worker");
		const child = spawn(command.file, command.args, {
			env: { ...process.env, PARAKEET_TDT_MODEL_DIR: deps.modelDir },
			stdio: ["pipe", "pipe", "pipe"],
		});
		childProcess = child;
		child.stdout?.on("data", (data: Buffer) => handleStdout(data));
		streamLines(child.stderr, (line) => logger.log(line));
		child.once("error", (error) => emit({ type: "error", message: error.message }));
		child.once("exit", (code, signal) => {
			if (childProcess === child) childProcess = undefined;
			logger.log(`Rust Parakeet worker exited code=${code ?? "none"} signal=${signal ?? "none"}`);
		});
	}

	function workerCommand(): { file: string; args: string[] } {
		if (!existsSync(deps.workerBinaryPath)) {
			throw new Error(`Rust STT worker binary missing: ${deps.workerBinaryPath}. Run npm run build:stt-rust.`);
		}
		return { file: deps.workerBinaryPath, args: [deps.modelDir] };
	}

	async function ensureModelFiles(): Promise<void> {
		await mkdir(deps.modelDir, { recursive: true });
		for (const file of PARAKEET_TDT_FILES) {
			const path = join(deps.modelDir, file);
			if (await hasUsableFile(path)) continue;
			await downloadModelFile(file, path);
		}
	}

	async function hasUsableFile(path: string): Promise<boolean> {
		try {
			return (await stat(path)).size > 0;
		} catch {
			return false;
		}
	}

	async function downloadModelFile(file: string, path: string): Promise<void> {
		const tmpPath = `${path}.tmp-${process.pid}`;
		await unlink(tmpPath).catch(() => undefined);
		logger.log(`downloading STT model file ${file}`);
		const response = await fetch(`${PARAKEET_TDT_REPO}/${file}`);
		if (!response.ok || !response.body) throw new Error(`failed to download ${file}: HTTP ${response.status}`);
		const total = Number(response.headers.get("content-length") ?? "0");
		const reader = response.body.getReader();
		const output = createWriteStream(tmpPath, { flags: "wx" });
		let received = 0;
		try {
			while (true) {
				const chunk = await reader.read();
				if (chunk.done) break;
				received += chunk.value.byteLength;
				if (!output.write(chunk.value)) await once(output, "drain");
			}
			output.end();
			await once(output, "finish");
			await rename(tmpPath, path);
			const suffix = total > 0 ? ` / ${(total / 1024 / 1024).toFixed(1)} MiB` : "";
			logger.log(`downloaded ${file} (${(received / 1024 / 1024).toFixed(1)} MiB${suffix})`);
		} catch (error) {
			output.destroy();
			await unlink(tmpPath).catch(() => undefined);
			throw error;
		}
	}

	function handleStdout(data: Buffer): void {
		stdout += data.toString("utf8");
		while (true) {
			const newline = stdout.indexOf("\n");
			if (newline < 0) return;
			const line = stdout.slice(0, newline).trim();
			stdout = stdout.slice(newline + 1);
			if (!line) continue;
			try {
				handleMessage(JSON.parse(line) as SttWorkerMsg);
			} catch (error) {
				logger.log(
					`failed to parse worker line: ${line}; ${error instanceof Error ? error.message : String(error)}`,
				);
			}
		}
	}

	function handleMessage(message: SttWorkerMsg): void {
		if (message.type === "ready") {
			emit({
				type: "ready",
				sampleRate: message.sampleRate,
				vadChunkMs: message.vadChunkMs,
				vadThreshold: message.vadThreshold,
				minSilenceMs: message.minSilenceMs,
				prerollMs: message.prerollMs,
				interimIntervalMs: message.interimIntervalMs,
			});
			return;
		}
		if (message.type === "speech_start") {
			emit({ type: "speech_start", index: message.index });
			return;
		}
		if (message.type === "speech_end") {
			emit({ type: "speech_end", index: message.index, duration: message.duration });
			return;
		}
		if (message.type === "speech_drop") {
			emit({ type: "speech_drop", index: message.index, duration: message.duration, reason: message.reason });
			return;
		}
		if (message.type === "interim") {
			emit({
				type: "interim",
				index: message.index,
				text: message.text.trim(),
				audioMs: message.audioMs,
				decodeMs: message.decodeMs,
			});
			return;
		}
		if (message.type === "final") {
			emit({ type: "final", index: message.index, text: message.text.trim(), decodeMs: message.decodeMs });
			return;
		}
		emit({ type: "error", message: message.message });
	}

	function handleAudioFrame(data: Buffer): void {
		if (!childProcess?.stdin || childProcess.stdin.destroyed) return;
		const header = Buffer.allocUnsafe(4);
		header.writeUInt32LE(data.byteLength, 0);
		childProcess.stdin.write(header);
		childProcess.stdin.write(data);
	}

	function stopChildProcess(): void {
		childProcess?.kill();
	}

	startWorker();
	return { handleAudioFrame, stopChildProcess };
}
