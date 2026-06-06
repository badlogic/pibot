import { spawn } from "node:child_process";
import { once } from "node:events";
import { createWriteStream } from "node:fs";
import { chmod, mkdir, rename, stat, unlink } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { Logger } from "./logger.js";

const LLAMA_CPP_RELEASE = "b9518";
const LLAMA_CPP_BASE_URL = `https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_RELEASE}`;

export interface LocalLlmConfig {
	name: string;
	modelFile: string;
	mmprojFile: string;
	downloadBaseUrl: string;
	defaultModelDirName: string;
	contextWindow: number;
	maxTokens: number;
	input: ("text" | "image")[];
	chatTemplateKwargs?: string;
}

export const localLlmConfigs = {
	qwen: {
		name: "Qwen3.6 35B A3B Q5 llama.cpp Local",
		modelFile: "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
		mmprojFile: "mmproj-F16.gguf",
		downloadBaseUrl: "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main",
		defaultModelDirName: "qwen3.6-35b-a3b",
		contextWindow: 131072,
		maxTokens: 16384,
		input: ["text", "image"],
		chatTemplateKwargs: '{"enable_thinking":false}',
	},
	gemma: {
		name: "Gemma 4 26B A4B MoE Q4 llama.cpp Local",
		modelFile: "gemma-4-26B-A4B-it-Q4_K_M.gguf",
		mmprojFile: "mmproj-gemma-4-26B-A4B-it-Q8_0.gguf",
		downloadBaseUrl: "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main",
		defaultModelDirName: "gemma-4-26b-a4b-it",
		contextWindow: 131072,
		maxTokens: 16384,
		input: ["text", "image"],
		chatTemplateKwargs: '{"enable_thinking":false}',
	},
	gemma12b: {
		name: "Gemma 4 12B IT Q4 llama.cpp Local",
		modelFile: "gemma-4-12b-it-Q4_K_M.gguf",
		mmprojFile: "mmproj-F16.gguf",
		downloadBaseUrl: "https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main",
		defaultModelDirName: "gemma-4-12b-it",
		contextWindow: 131072,
		maxTokens: 16384,
		input: ["text", "image"],
		chatTemplateKwargs: '{"enable_thinking":false}',
	},
} satisfies Record<string, LocalLlmConfig>;

export type LocalLlmId = keyof typeof localLlmConfigs;

export function parseLocalLlmId(value: string | undefined): LocalLlmId {
	const normalized = value?.toLowerCase();
	if (!normalized || normalized === "gemma") return "gemma";
	if (normalized === "gemma12b" || normalized === "gemma-12b") return "gemma12b";
	if (normalized === "qwen") return "qwen";
	throw new Error(`Unknown LOCAL_LLM: ${value}. Expected qwen, gemma, or gemma12b.`);
}

export interface LlamaServiceDeps {
	cacheDir: string;
	modelDir: string;
	modelFile: string;
	mmprojFile: string;
	modelDownloadBaseUrl: string;
	modelLabel: string;
	baseUrl: string;
	host: string;
	port: number;
	contextWindow: number;
	chatTemplateKwargs?: string;
	logger: Logger;
}

export interface LlamaService {
	stop: () => void;
}

interface DownloadFile {
	url: string;
	path: string;
	label: string;
}

function llamaAssetName(): string {
	if (process.platform === "darwin") {
		if (process.arch === "arm64") return `llama-${LLAMA_CPP_RELEASE}-bin-macos-arm64.tar.gz`;
		if (process.arch === "x64") return `llama-${LLAMA_CPP_RELEASE}-bin-macos-x64.tar.gz`;
	}
	if (process.platform === "linux") {
		if (process.arch === "arm64") return `llama-${LLAMA_CPP_RELEASE}-bin-ubuntu-arm64.tar.gz`;
		if (process.arch === "x64") return `llama-${LLAMA_CPP_RELEASE}-bin-ubuntu-x64.tar.gz`;
	}
	throw new Error(`Unsupported llama.cpp release platform: ${process.platform}/${process.arch}`);
}

function llamaBinaryPath(cacheDir: string): string {
	return join(cacheDir, "llama.cpp", LLAMA_CPP_RELEASE, `llama-${LLAMA_CPP_RELEASE}`, "llama-server");
}

async function hasUsableFile(path: string): Promise<boolean> {
	try {
		return (await stat(path)).size > 0;
	} catch {
		return false;
	}
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

async function runCommand(file: string, args: string[], cwd: string, logger: Logger): Promise<void> {
	const child = spawn(file, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
	child.stdout?.on("data", (data: Buffer) => logger.log(data.toString().trim()));
	child.stderr?.on("data", (data: Buffer) => logger.log(data.toString().trim()));
	const [code, signal] = (await once(child, "exit")) as [number | null, NodeJS.Signals | null];
	if (code !== 0) throw new Error(`${file} exited code=${code ?? "none"} signal=${signal ?? "none"}`);
}

async function ensureLlamaBinary(cacheDir: string, logger: Logger): Promise<string> {
	const overridePath = process.env.LLAMA_CPP_BINARY_PATH;
	if (overridePath) {
		if (!(await hasUsableFile(overridePath)))
			throw new Error(`LLAMA_CPP_BINARY_PATH is not a usable file: ${overridePath}`);
		logger.log(`using llama.cpp binary override: ${overridePath}`);
		return overridePath;
	}

	const binaryPath = llamaBinaryPath(cacheDir);
	if (await hasUsableFile(binaryPath)) return binaryPath;

	const assetName = llamaAssetName();
	const archiveDir = join(cacheDir, "llama.cpp", LLAMA_CPP_RELEASE);
	const archivePath = join(archiveDir, assetName);
	await mkdir(archiveDir, { recursive: true });
	if (!(await hasUsableFile(archivePath))) {
		await downloadFile(
			{ url: `${LLAMA_CPP_BASE_URL}/${assetName}`, path: archivePath, label: `llama.cpp ${assetName}` },
			logger,
		);
	}
	logger.log(`extracting llama.cpp ${LLAMA_CPP_RELEASE}`);
	await runCommand("tar", ["-xzf", archivePath, "-C", archiveDir], archiveDir, logger);
	await chmod(binaryPath, 0o755).catch(() => undefined);
	return binaryPath;
}

async function ensureLlamaModel(
	modelDir: string,
	modelFile: string,
	mmprojFile: string,
	modelDownloadBaseUrl: string,
	modelLabel: string,
	logger: Logger,
): Promise<void> {
	await mkdir(modelDir, { recursive: true });
	for (const file of [modelFile, mmprojFile]) {
		const path = join(modelDir, file);
		if (await hasUsableFile(path)) continue;
		await downloadFile({ url: `${modelDownloadBaseUrl}/${file}`, path, label: `${modelLabel} file ${file}` }, logger);
	}
}

async function serverModelInfo(baseUrl: string): Promise<string | undefined> {
	try {
		const response = await fetch(`${baseUrl}/models`, { signal: AbortSignal.timeout(1000) });
		if (!response.ok) return undefined;
		return await response.text();
	} catch {
		return undefined;
	}
}

function modelInfoMatches(info: string, modelFile: string, contextWindow: number): boolean {
	if (!info.includes(modelFile)) return false;
	const contextMatch = /"n_ctx"\s*:\s*(\d+)/.exec(info);
	if (!contextMatch) return true;
	return Number(contextMatch[1]) >= contextWindow;
}

async function waitForServer(baseUrl: string, modelFile: string, contextWindow: number, logger: Logger): Promise<void> {
	const started = Date.now();
	while (Date.now() - started < 180_000) {
		const info = await serverModelInfo(baseUrl);
		if (info && modelInfoMatches(info, modelFile, contextWindow)) return;
		await new Promise((resolve) => setTimeout(resolve, 1000));
	}
	logger.log("llama.cpp server did not become ready within 180s");
	throw new Error("llama.cpp server startup timed out");
}

export async function createLlamaService(deps: LlamaServiceDeps): Promise<LlamaService> {
	const logger = deps.logger.tag("llama");
	const existingServerInfo = await serverModelInfo(deps.baseUrl);
	if (existingServerInfo) {
		if (!modelInfoMatches(existingServerInfo, deps.modelFile, deps.contextWindow)) {
			throw new Error(`Found an incompatible server at ${deps.baseUrl}; stop it or set LLAMA_PORT/LLAMA_BASE_URL.`);
		}
		logger.log(`using existing llama.cpp server at ${deps.baseUrl}`);
		return { stop: () => undefined };
	}

	const binaryPath = await ensureLlamaBinary(deps.cacheDir, logger);
	await ensureLlamaModel(
		deps.modelDir,
		deps.modelFile,
		deps.mmprojFile,
		deps.modelDownloadBaseUrl,
		deps.modelLabel,
		logger,
	);

	const binaryDir = dirname(binaryPath);
	const args = [
		"-m",
		join(deps.modelDir, deps.modelFile),
		"--mmproj",
		join(deps.modelDir, deps.mmprojFile),
		"-ngl",
		"999",
		"-c",
		String(deps.contextWindow),
		"--jinja",
	];
	if (deps.chatTemplateKwargs) args.push("--chat-template-kwargs", deps.chatTemplateKwargs);
	args.push("--host", deps.host, "--port", String(deps.port));
	logger.log(`starting llama.cpp server: ${binaryPath} ${args.join(" ")}`);
	const child = spawn(binaryPath, args, { cwd: binaryDir, stdio: ["ignore", "pipe", "pipe"] });
	child.stdout?.on("data", (data: Buffer) => logger.log(data.toString().trim()));
	child.stderr?.on("data", (data: Buffer) => logger.log(data.toString().trim()));
	child.once("exit", (code, signal) =>
		logger.log(`llama.cpp server exited code=${code ?? "none"} signal=${signal ?? "none"}`),
	);
	await waitForServer(deps.baseUrl, deps.modelFile, deps.contextWindow, logger);
	logger.log(`llama.cpp server ready at ${deps.baseUrl}`);
	return { stop: () => child.kill() };
}
