import { type ChildProcess, spawn } from "node:child_process";
import { logStreamLines } from "./log-stream.js";
import type { Logger } from "./logger.js";

export interface SttServiceDeps {
	workerPath: string;
	logger: Logger;
	broadcast: (data: object) => void;
	enqueuePrompt: (text: string) => void;
	performAbort: (reason: string, sttIndex?: number) => Promise<void>;
	shouldIgnoreNonStopFinal: () => boolean;
}

export interface SttService {
	handleAudioFrame: (data: Buffer) => void;
	stopChildProcess: () => void;
}

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

function normalizeForStopMatch(text: string): string {
	return text
		.toLowerCase()
		.normalize("NFKD")
		.replace(/[.,!?;:()[\]{}"'`´]/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

const stopWordPhrases = [
	"stop",
	"stopp",
	"halt",
	"anhalten",
	"abbrechen",
	"schluss",
	"ruhe",
	"sei still",
	"sei ruhig",
	"hör auf",
	"hoer auf",
];

function looksLikeStopCommand(text: string): boolean {
	const normalized = normalizeForStopMatch(text);
	if (!normalized) return false;
	for (const phrase of stopWordPhrases) {
		if (normalized === phrase) return true;
		if (normalized.startsWith(`${phrase} `)) return true;
		if (normalized.endsWith(` ${phrase}`)) return true;
		if (normalized.includes(` ${phrase} `)) return true;
	}
	return false;
}

export function createSttService(deps: SttServiceDeps): SttService {
	let process: ChildProcess | undefined;
	let ready = false;
	let loadingAnnounced = false;
	let stdout = "";
	let stoppedUtteranceIndex: number | undefined;
	const logger = deps.logger.tag("stt");

	function startWorker(): void {
		if (process && !process.killed) return;
		ready = false;
		loadingAnnounced = false;
		stdout = "";
		logger.log("starting Parakeet STT worker via uvx");
		const child = spawn("uvx", ["--with", "parakeet-mlx", "--with", "silero-vad", "python", deps.workerPath], {
			stdio: ["pipe", "pipe", "pipe"],
		});
		process = child;
		child.stdout?.on("data", (data: Buffer) => handleStdout(data));
		logStreamLines(child.stderr, logger);
		child.once("error", (error) => {
			deps.broadcast({ type: "stt_event", event: "error", message: error.message });
			logger.log(`Parakeet worker failed to start: ${error.message}`);
		});
		child.once("exit", (code, signal) => {
			if (process === child) process = undefined;
			ready = false;
			logger.log(`Parakeet worker exited code=${code ?? "none"} signal=${signal ?? "none"}`);
		});
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
			ready = true;
			logger.log(
				`Parakeet ready sampleRate=${message.sampleRate} vadChunkMs=${message.vadChunkMs} threshold=${message.vadThreshold} minSilenceMs=${message.minSilenceMs} prerollMs=${message.prerollMs} interimIntervalMs=${message.interimIntervalMs ?? "off"}`,
			);
			deps.broadcast({ type: "stt_event", event: "ready" });
			return;
		}
		if (message.type === "speech_start") {
			stoppedUtteranceIndex = undefined;
			logger.log(`speech_start #${message.index}`);
			deps.broadcast({ type: "stt_event", event: "speech_start", index: message.index });
			return;
		}
		if (message.type === "speech_end") {
			logger.log(`speech_end #${message.index} duration=${message.duration.toFixed(2)}s`);
			deps.broadcast({ type: "stt_event", event: "speech_end", index: message.index });
			return;
		}
		if (message.type === "speech_drop") {
			logger.log(`speech_drop #${message.index} reason=${message.reason} duration=${message.duration.toFixed(2)}s`);
			deps.broadcast({ type: "stt_event", event: "speech_drop", index: message.index });
			return;
		}
		if (message.type === "interim") {
			const text = message.text.trim();
			logger.log(
				`interim #${message.index} audioMs=${message.audioMs} decodeMs=${message.decodeMs} text=${JSON.stringify(text)}`,
			);
			deps.broadcast({ type: "stt_interim", index: message.index, text });
			if (text && looksLikeStopCommand(text) && stoppedUtteranceIndex !== message.index) {
				stoppedUtteranceIndex = message.index;
				logger.log("stop-word detected in interim, aborting current turn");
				void deps.performAbort(`interim stop word: ${text}`, message.index);
			}
			return;
		}
		if (message.type === "final") {
			const text = message.text.trim();
			logger.log(`final #${message.index} decodeMs=${message.decodeMs} text=${JSON.stringify(text)}`);
			if (!text) {
				deps.broadcast({ type: "stt_final", index: message.index, text, accepted: false, ignoredReason: "empty" });
				return;
			}
			if (stoppedUtteranceIndex === message.index) {
				logger.log(`ignoring final #${message.index}; utterance already handled as stop`);
				deps.broadcast({ type: "stt_final", index: message.index, text, accepted: false, ignoredReason: "stop" });
				return;
			}
			if (looksLikeStopCommand(text)) {
				stoppedUtteranceIndex = message.index;
				logger.log("stop-word detected in final, aborting current turn");
				deps.broadcast({ type: "stt_final", index: message.index, text, accepted: false, ignoredReason: "stop" });
				void deps.performAbort(`stop word: ${text}`, message.index);
				return;
			}
			if (deps.shouldIgnoreNonStopFinal()) {
				logger.log(`ignoring non-stop final during/recently-after TTS: ${JSON.stringify(text)}`);
				deps.broadcast({
					type: "stt_final",
					index: message.index,
					text,
					accepted: false,
					ignoredReason: "tts_bleed",
				});
				return;
			}
			deps.broadcast({ type: "stt_final", index: message.index, text, accepted: true });
			deps.enqueuePrompt(text);
			return;
		}
		logger.log(`worker error: ${message.message}`);
		deps.broadcast({ type: "stt_event", event: "error", message: message.message });
	}

	function handleAudioFrame(data: Buffer): void {
		startWorker();
		if (!process?.stdin || process.stdin.destroyed) return;
		const header = Buffer.allocUnsafe(4);
		header.writeUInt32LE(data.byteLength, 0);
		process.stdin.write(header);
		process.stdin.write(data);
		if (!ready && !loadingAnnounced) {
			loadingAnnounced = true;
			deps.broadcast({ type: "stt_event", event: "loading" });
		}
	}

	function stopChildProcess(): void {
		process?.kill();
	}

	return { handleAudioFrame, stopChildProcess };
}
