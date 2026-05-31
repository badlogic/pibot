import { homedir } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { localLlmConfigs, parseLocalLlmId } from "./llama.js";

const serverDir = fileURLToPath(new URL(".", import.meta.url));
const localLlm = parseLocalLlmId(process.env.LOCAL_LLM);

export const serverConfig = {
	publicDir: resolve(serverDir, "../../public"),
	pibotCacheDir: process.env.PIBOT_CACHE_DIR ?? resolve(homedir(), ".cache/pibot"),
	port: Number(process.env.PORT ?? 8010),
	host: process.env.HOST ?? "127.0.0.1",
	sttWorkerBinaryPath: resolve(
		serverDir,
		`../../native/pibot-stt/target/release/pibot-stt-worker${process.platform === "win32" ? ".exe" : ""}`,
	),
	parakeetTdtModelDir:
		process.env.PARAKEET_TDT_MODEL_DIR ?? resolve(homedir(), "models/parakeet-tdt-0.6b-v3-onnx-int8"),
	llamaBaseUrl: process.env.LLAMA_BASE_URL ?? "http://127.0.0.1:8080/v1",
	llamaHost: process.env.LLAMA_HOST ?? "127.0.0.1",
	llamaPort: Number(process.env.LLAMA_PORT ?? 8080),
	localLlm,
	llamaContextWindow: Number(process.env.LLAMA_CONTEXT_WINDOW ?? localLlmConfigs[localLlm].contextWindow),
	llamaModelDir:
		process.env.LLAMA_MODEL_DIR ?? resolve(homedir(), "models", localLlmConfigs[localLlm].defaultModelDirName),
	qwen3TtsWorker: process.env.QWEN3_TTS_WORKER ?? "rust",
	qwen3TtsPythonCommand: process.env.QWEN3_TTS_PYTHON_COMMAND ?? "uv",
	qwen3TtsPythonWorkerPath:
		process.env.QWEN3_TTS_PYTHON_WORKER_PATH ?? resolve(serverDir, "../../scripts/qwen3-tts-worker.py"),
	qwen3TtsRustWorkerPath:
		process.env.QWEN3_TTS_RUST_WORKER_PATH ??
		resolve(
			serverDir,
			`../../native/qwen3_tts_rs/target/release/pibot-tts-worker${process.platform === "win32" ? ".exe" : ""}`,
		),
	qwen3TtsRustModelPath:
		process.env.QWEN3_TTS_RUST_MODEL_PATH ?? resolve(homedir(), "models/qwen3-tts-12hz-0.6b-base-6bit"),
	version: String(Date.now()),
	maxContextImages: Number(process.env.MAX_CONTEXT_IMAGES ?? 4),
	memoryFile: process.env.MEMORY_FILE ?? "data/memories.json",
};
