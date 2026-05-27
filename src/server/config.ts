import { homedir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const serverDir = fileURLToPath(new URL(".", import.meta.url));

export const serverConfig = {
	publicDir: resolve(serverDir, "../../public"),
	port: Number(process.env.PORT ?? 8010),
	host: process.env.HOST ?? "127.0.0.1",
	sttWorkerBinaryPath: resolve(
		serverDir,
		`../../native/pibot-stt/target/release/pibot-stt-worker${process.platform === "win32" ? ".exe" : ""}`,
	),
	parakeetTdtModelDir:
		process.env.PARAKEET_TDT_MODEL_DIR ?? resolve(homedir(), "models/parakeet-tdt-0.6b-v3-onnx-int8"),
	qwen3TtsWorkerPath: resolve(serverDir, "../../scripts/qwen3-tts-worker.py"),
	version: String(Date.now()),
	maxContextImages: Number(process.env.MAX_CONTEXT_IMAGES ?? 4),
	memoryFile: process.env.MEMORY_FILE ?? "data/memories.json",
};
