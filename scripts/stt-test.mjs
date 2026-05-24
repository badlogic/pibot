import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const MSG_START = 1;
const MSG_AUDIO = 2;
const MSG_FLUSH = 3;
const MSG_STOP = 4;

const MSG_READY = 101;
const MSG_PARTIAL = 102;
const MSG_FINAL = 103;
const MSG_SPEECH_START = 104;
const MSG_SPEECH_END = 105;
const MSG_ERROR = 199;

function arg(name, fallback) {
	const index = process.argv.indexOf(name);
	return index >= 0 ? process.argv[index + 1] : fallback;
}

const useStdin = process.argv.includes("--stdin");
const model = resolve(arg("--model", "models/ggml-whisper-large-v3-turbo-german-q5_0.bin"));
const binary = resolve(arg("--binary", "native/pibot-stt/build/pibot-stt"));
const language = arg("--language", "de");

if (!existsSync(binary)) {
	console.error(`missing binary: ${binary}`);
	console.error("run: npm run build:stt");
	process.exit(1);
}
if (!existsSync(model)) {
	console.error(`missing model: ${model}`);
	console.error("run: npm run download:stt-model");
	process.exit(1);
}

function frame(type, payload = Buffer.alloc(0)) {
	const header = Buffer.alloc(12);
	header.writeUInt32LE(type, 0);
	header.writeUInt32LE(0, 4);
	header.writeUInt32LE(payload.length, 8);
	return Buffer.concat([header, payload]);
}

const stt = spawn(binary, ["--model", model, "--language", language], { stdio: ["pipe", "pipe", "inherit"] });
stt.on("exit", (code, signal) => {
	console.error(`pibot-stt exited code=${code ?? "null"} signal=${signal ?? "null"}`);
	process.exit(code ?? 1);
});

let output = Buffer.alloc(0);
stt.stdout.on("data", (chunk) => {
	output = Buffer.concat([output, chunk]);
	while (output.length >= 12) {
		const type = output.readUInt32LE(0);
		const length = output.readUInt32LE(8);
		if (output.length < 12 + length) return;
		const payload = output.subarray(12, 12 + length).toString("utf8");
		output = output.subarray(12 + length);
		if (type === MSG_READY) console.error(`[ready] ${payload}`);
		else if (type === MSG_SPEECH_START) console.error("[speech_start]");
		else if (type === MSG_SPEECH_END) console.error("[speech_end]");
		else if (type === MSG_PARTIAL) console.log(`[partial] ${payload}`);
		else if (type === MSG_FINAL) console.log(`[final] ${payload}`);
		else if (type === MSG_ERROR) console.error(`[error] ${payload}`);
		else console.error(`[msg ${type}] ${payload}`);
	}
});

stt.stdin.write(frame(MSG_START));

const audioInput = useStdin
	? process.stdin
	: spawn(
			"ffmpeg",
			["-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-i", ":0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
			{ stdio: ["ignore", "pipe", "inherit"] },
		).stdout;

console.error(useStdin ? "reading raw s16le 16kHz mono PCM from stdin" : "recording default macOS microphone via ffmpeg avfoundation :0");
console.error("press Ctrl+C to stop");

audioInput.on("data", (chunk) => {
	stt.stdin.write(frame(MSG_AUDIO, chunk));
});

audioInput.on("end", () => {
	stt.stdin.write(frame(MSG_FLUSH));
	stt.stdin.write(frame(MSG_STOP));
});

process.on("SIGINT", () => {
	stt.stdin.write(frame(MSG_FLUSH));
	stt.stdin.write(frame(MSG_STOP));
	setTimeout(() => process.exit(130), 200);
});
