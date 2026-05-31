#!/usr/bin/env node

import { spawn } from "node:child_process";
import { once } from "node:events";

const FRAME_HEADER_BYTES = 9;
const INPUT_SPEAK = 1;
const INPUT_SHUTDOWN = 3;
const OUTPUT_READY = 1;
const OUTPUT_AUDIO_START = 2;
const OUTPUT_AUDIO_CHUNK = 3;
const OUTPUT_AUDIO_DONE = 4;
const OUTPUT_ERROR = 5;
const KID_TEXTS = [
	"I found a shiny beetle under a rock. Let's talk about what it might be and how to look at it safely.",
	"Here is a short moon adventure: we pack tools, help a lost rover, and use science to get home.",
	"Airplanes fly because moving air can push wings upward. Try this with a strip of paper and your breath.",
	"If your friend took your robot, you can say: I felt upset when you took it. Please ask me next time.",
];

function parseArgs(argv) {
	const args = {
		worker: process.env.QWEN3_TTS_RUST_WORKER_PATH ?? "native/qwen3_tts_rs/target/release/pibot-tts-worker",
		model: process.env.QWEN3_TTS_RUST_MODEL_PATH ?? `${process.env.HOME}/models/qwen3-tts-12hz-1.7b-base-6bit`,
		refAudio: process.env.QWEN3_TTS_REF_AUDIO ?? "data/voices/elevenlabs-pibot-reference.wav",
		refTextFile: process.env.QWEN3_TTS_REF_TEXT_FILE ?? "data/voices/elevenlabs-pibot-reference.txt",
		language: process.env.QWEN3_TTS_LANGUAGE ?? "english",
		maxConcurrency: 4,
		runs: 1,
		text: "",
	};
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		const next = argv[i + 1];
		if (arg === "--worker" && next) {
			args.worker = next;
			i++;
			continue;
		}
		if (arg === "--model" && next) {
			args.model = next;
			i++;
			continue;
		}
		if (arg === "--ref-audio" && next) {
			args.refAudio = next;
			i++;
			continue;
		}
		if (arg === "--ref-text-file" && next) {
			args.refTextFile = next;
			i++;
			continue;
		}
		if (arg === "--language" && next) {
			args.language = next;
			i++;
			continue;
		}
		if (arg === "--max-concurrency" && next) {
			args.maxConcurrency = Number(next);
			i++;
			continue;
		}
		if (arg === "--runs" && next) {
			args.runs = Number(next);
			i++;
			continue;
		}
		if (arg === "--text" && next) {
			args.text = next;
			i++;
			continue;
		}
		if (arg === "--help") {
			printHelp();
			process.exit(0);
		}
		throw new Error(`unknown or incomplete argument: ${arg}`);
	}
	if (!Number.isInteger(args.maxConcurrency) || args.maxConcurrency < 1) throw new Error("--max-concurrency must be an integer >= 1");
	if (!Number.isInteger(args.runs) || args.runs < 1) throw new Error("--runs must be an integer >= 1");
	return args;
}

function printHelp() {
	console.log(`Usage: node scripts/benchmark-tts-concurrency.mjs [options]

Options:
  --worker PATH           Rust TTS worker path
  --model PATH            Rust TTS model directory
  --ref-audio PATH        reference WAV
  --ref-text-file PATH    reference transcript
  --language NAME         language (default: english)
  --max-concurrency N     enqueue concurrency 1..N (default: 4)
  --runs N                runs per concurrency level (default: 1)
  --text TEXT             use one text for every request
`);
}

function makeFrame(type, id, payload = Buffer.alloc(0)) {
	const frame = Buffer.allocUnsafe(FRAME_HEADER_BYTES + payload.byteLength);
	frame.writeUInt8(type, 0);
	frame.writeUInt32LE(id >>> 0, 1);
	frame.writeUInt32LE(payload.byteLength, 5);
	payload.copy(frame, FRAME_HEADER_BYTES);
	return frame;
}

function readFrames(stream, onFrame) {
	let buffer = Buffer.alloc(0);
	stream.on("data", (chunk) => {
		buffer = Buffer.concat([buffer, chunk]);
		while (buffer.byteLength >= FRAME_HEADER_BYTES) {
			const type = buffer.readUInt8(0);
			const id = buffer.readUInt32LE(1);
			const payloadLength = buffer.readUInt32LE(5);
			const frameLength = FRAME_HEADER_BYTES + payloadLength;
			if (buffer.byteLength < frameLength) return;
			const payload = buffer.subarray(FRAME_HEADER_BYTES, frameLength);
			buffer = buffer.subarray(frameLength);
			onFrame(type, id, payload);
		}
	});
}

async function startWorker(args) {
	const child = spawn(args.worker, [
		"--serve",
		"--model-name",
		args.model,
		"--ref-audio",
		args.refAudio,
		"--ref-text-file",
		args.refTextFile,
		"--language",
		args.language,
		"--output-sample-rate",
		"24000",
		"--temperature",
		"0.7",
		"--top-k",
		"30",
	], { stdio: ["pipe", "pipe", "pipe"] });
	child.stderr.setEncoding("utf8");
	child.stderr.on("data", (chunk) => {
		for (const line of chunk.split(/\r?\n/)) if (line.trim()) console.error(`[worker] ${line.trim()}`);
	});
	const pending = new Map();
	let readyResolve;
	let readyReject;
	const ready = new Promise((resolve, reject) => {
		readyResolve = resolve;
		readyReject = reject;
	});
	readFrames(child.stdout, (type, id, payload) => {
		if (type === OUTPUT_READY) {
			readyResolve();
			return;
		}
		const request = pending.get(id);
		if (!request) return;
		if (type === OUTPUT_AUDIO_START) {
			request.sampleRate = payload.byteLength >= 4 ? payload.readUInt32LE(0) : 24000;
			return;
		}
		if (type === OUTPUT_AUDIO_CHUNK) {
			request.audioBytes += payload.byteLength;
			return;
		}
		if (type === OUTPUT_AUDIO_DONE) {
			pending.delete(id);
			request.resolve({ elapsedMs: performance.now() - request.started, audioSeconds: request.audioBytes / 2 / request.sampleRate });
			return;
		}
		if (type === OUTPUT_ERROR) {
			pending.delete(id);
			request.reject(new Error(payload.toString("utf8")));
		}
	});
	child.once("error", readyReject);
	child.once("exit", (code, signal) => {
		const error = new Error(`TTS worker exited code=${code ?? "none"} signal=${signal ?? "none"}`);
		readyReject(error);
		for (const request of pending.values()) request.reject(error);
		pending.clear();
	});
	await ready;
	return { child, pending };
}

function speak(worker, id, text) {
	const payload = Buffer.from(text, "utf8");
	const promise = new Promise((resolve, reject) => {
		worker.pending.set(id, { started: performance.now(), sampleRate: 24000, audioBytes: 0, resolve, reject });
	});
	worker.child.stdin.write(makeFrame(INPUT_SPEAK, id, payload));
	return promise;
}

function summarize(samples) {
	const wallMs = Math.max(...samples.map((sample) => sample.elapsedMs));
	const audioSeconds = samples.reduce((sum, sample) => sum + sample.audioSeconds, 0);
	return { wallMs, audioSeconds, rtf: audioSeconds / (wallMs / 1000) };
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	console.log(`worker=${args.worker}`);
	console.log(`model=${args.model}`);
	console.log(`prompt=${args.text ? "custom" : "rotating kid-style texts"}`);
	console.log("loading worker...");
	const worker = await startWorker(args);
	console.log("ready");
	console.log("conc\trun\twall_s\taudio_s\trtf");
	let nextId = 1;
	for (let concurrency = 1; concurrency <= args.maxConcurrency; concurrency++) {
		for (let run = 1; run <= args.runs; run++) {
			const samples = await Promise.all(
				Array.from({ length: concurrency }, (_, index) => {
					const promptIndex = (run + index + KID_TEXTS.length - 2) % KID_TEXTS.length;
					const text = args.text || KID_TEXTS[promptIndex];
					return speak(worker, nextId++, text);
				}),
			);
			const summary = summarize(samples);
			console.log(`${concurrency}\t${run}\t${(summary.wallMs / 1000).toFixed(2)}\t${summary.audioSeconds.toFixed(2)}\t${summary.rtf.toFixed(2)}`);
		}
	}
	worker.child.stdin.write(makeFrame(INPUT_SHUTDOWN, 0));
	worker.child.stdin.end();
	await once(worker.child, "exit").catch(() => undefined);
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
