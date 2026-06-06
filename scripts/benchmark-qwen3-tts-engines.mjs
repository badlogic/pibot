#!/usr/bin/env node

import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";

function parseArgs(argv) {
	const args = {
		text: "",
		textFile: "data/voices/elevenlabs-pibot-reference-de.txt",
		refAudio: "data/voices/elevenlabs-pibot-reference-de.wav",
		refTextFile: "data/voices/elevenlabs-pibot-reference-de.txt",
		language: "de",
		outDir: "/tmp/qwen3tts-bench",
		mlxBin: "native/qwen3_tts_rs/target/release/qwen3-tts",
		mlxModel: `${process.env.HOME}/models/qwen3-tts-12hz-0.6b-base-6bit`,
		ggmlBin: "native/qwen3-tts.cpp/build-metal-coreml/qwen3-tts-cli",
		ggmlModel: "native/qwen3-tts.cpp/models",
		temperature: "0",
		topK: "50",
		maxTokens: "512",
		threads: "4",
		runMlx: true,
		runGgml: true,
		ggmlCoreml: false,
		quiet: false,
	};
	for (let i = 0; i < argv.length; i++) {
		const arg = argv[i];
		const next = argv[i + 1];
		if (arg === "--help") {
			printHelp();
			process.exit(0);
		}
		if (arg === "--skip-mlx") {
			args.runMlx = false;
			continue;
		}
		if (arg === "--skip-ggml") {
			args.runGgml = false;
			continue;
		}
		if (arg === "--ggml-coreml") {
			args.ggmlCoreml = true;
			continue;
		}
		if (arg === "--quiet") {
			args.quiet = true;
			continue;
		}
		const stringOptions = {
			"--text": "text",
			"--text-file": "textFile",
			"--ref-audio": "refAudio",
			"--ref-text-file": "refTextFile",
			"--language": "language",
			"--out-dir": "outDir",
			"--mlx-bin": "mlxBin",
			"--mlx-model": "mlxModel",
			"--ggml-bin": "ggmlBin",
			"--ggml-model": "ggmlModel",
			"--temperature": "temperature",
			"--top-k": "topK",
			"--max-tokens": "maxTokens",
			"--threads": "threads",
		};
		const key = stringOptions[arg];
		if (key && next) {
			args[key] = next;
			i++;
			continue;
		}
		throw new Error(`unknown or incomplete argument: ${arg}`);
	}
	if (!args.runMlx && !args.runGgml) throw new Error("nothing to run");
	return args;
}

function printHelp() {
	console.log(`Usage: node scripts/benchmark-qwen3-tts-engines.mjs [options]

Runs the MLX and GGML Qwen3-TTS CLIs with the same text/reference inputs and
prints comparable timing JSON/table output.

Options:
  --text TEXT              synth text; overrides --text-file
  --text-file PATH         synth text file (default: German reference text)
  --ref-audio PATH         reference WAV
  --ref-text-file PATH     reference transcript for MLX ICL mode
  --language LANG          language (default: de)
  --out-dir PATH           output directory (default: /tmp/qwen3tts-bench)
  --mlx-bin PATH           MLX qwen3-tts binary
  --mlx-model PATH         MLX model directory
  --ggml-bin PATH          GGML qwen3-tts-cli binary
  --ggml-model PATH        GGML model directory
  --temperature VALUE      default: 0
  --top-k N                default: 50
  --max-tokens N           default: 512
  --threads N              GGML -j value (default: 4)
  --ggml-coreml            allow GGML CoreML path; default disables CoreML
  --skip-mlx               only run GGML
  --skip-ggml              only run MLX
  --quiet                  suppress child logs
`);
}

async function resolveText(args) {
	if (args.text.trim()) return args.text.trim();
	return (await readFile(args.textFile, "utf8")).trim();
}

function commandLine(command, args) {
	return [command, ...args.map((arg) => (arg.includes(" ") ? JSON.stringify(arg) : arg))].join(" ");
}

async function runCommand(label, command, args, options) {
	console.log(`\n== ${label} ==`);
	console.log(commandLine(command, args));
	const child = spawn(command, args, {
		cwd: options.cwd,
		env: options.env,
		stdio: ["ignore", "pipe", "pipe"],
	});
	let combined = "";
	for (const [name, stream] of [
		["stdout", child.stdout],
		["stderr", child.stderr],
	]) {
		stream.setEncoding("utf8");
		stream.on("data", (chunk) => {
			combined += chunk;
			if (!options.quiet) process[name].write(chunk);
		});
	}
	const exit = await new Promise((resolveExit) => {
		child.on("error", (error) => resolveExit({ code: null, signal: null, error }));
		child.on("exit", (code, signal) => resolveExit({ code, signal, error: null }));
	});
	if (exit.error) throw exit.error;
	if (exit.code !== 0) throw new Error(`${label} exited with code=${exit.code ?? "none"} signal=${exit.signal ?? "none"}`);
	const timing = parseTiming(combined);
	if (!timing) throw new Error(`${label} did not emit timing JSON`);
	return timing;
}

function parseTiming(output) {
	for (const line of output.split(/\r?\n/)) {
		const trimmed = line.trim();
		if (!trimmed.startsWith("{")) continue;
		try {
			const value = JSON.parse(trimmed);
			if (value.type === "timing") return value;
		} catch {
			// ignore non-JSON log lines
		}
	}
	return undefined;
}

function formatMs(value) {
	return `${(Number(value ?? 0) / 1000).toFixed(2)}s`;
}

function formatNumber(value) {
	return Number(value ?? 0).toFixed(2);
}

function printSummary(results) {
	console.log("\nSummary");
	console.log("engine\tmode\taudio\tgenerate\tdecode\ttotal\tthroughput\toutput");
	for (const result of results) {
		console.log(
			[
				result.engine,
				result.mode,
				`${formatNumber(result.timing.audioSeconds)}s`,
				formatMs(result.timing.generateMs),
				formatMs(result.timing.decodeMs),
				formatMs(result.timing.totalMs),
				`${formatNumber(result.timing.throughput)}x`,
				result.timing.output,
			].join("\t"),
		);
	}
}

async function main() {
	const args = parseArgs(process.argv.slice(2));
	await mkdir(args.outDir, { recursive: true });
	const tempDir = await mkdtemp(join(tmpdir(), "qwen3tts-bench-"));
	const textFile = join(tempDir, "text.txt");
	const text = await resolveText(args);
	await writeFile(textFile, text, "utf8");

	const results = [];
	try {
		if (args.runMlx) {
			const output = resolve(args.outDir, "mlx.wav");
			const timing = await runCommand(
				"MLX",
				args.mlxBin,
				[
					"-m",
					args.mlxModel,
					"--text-file",
					textFile,
					"-r",
					args.refAudio,
					"--ref-text-file",
					args.refTextFile,
					"-l",
					args.language,
					"-o",
					output,
					"--temperature",
					args.temperature,
					"--top-k",
					args.topK,
					"--max-tokens",
					args.maxTokens,
				],
				{ cwd: process.cwd(), env: process.env, quiet: args.quiet },
			);
			results.push({ engine: "MLX", mode: "ICL", timing });
		}

		if (args.runGgml) {
			const output = resolve(args.outDir, "ggml.wav");
			const env = { ...process.env };
			if (!args.ggmlCoreml) env.QWEN3_TTS_USE_COREML = "0";
			const timing = await runCommand(
				"GGML",
				args.ggmlBin,
				[
					"-m",
					args.ggmlModel,
					"--text-file",
					textFile,
					"-r",
					args.refAudio,
					"-l",
					args.language,
					"-o",
					output,
					"--temperature",
					args.temperature,
					"--top-k",
					args.topK,
					"--max-tokens",
					args.maxTokens,
					"-j",
					args.threads,
					"--json-timing",
				],
				{ cwd: process.cwd(), env, quiet: args.quiet },
			);
			results.push({ engine: "GGML", mode: "speaker-embedding", timing });
		}

		printSummary(results);
		console.log("\nNote: current GGML CLI uses reference-audio speaker embedding only; MLX CLI uses reference text/codes ICL.");
	} finally {
		await rm(tempDir, { recursive: true, force: true });
	}
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
});
