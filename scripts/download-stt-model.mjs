import { createWriteStream, existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pipeline } from "node:stream/promises";

const modelUrl =
	process.argv[2] ??
	"https://huggingface.co/cstr/whisper-large-v3-turbo-german-ggml/resolve/main/ggml-model-q5_0.bin?download=true";
const outputPath = resolve(process.argv[3] ?? "models/ggml-whisper-large-v3-turbo-german-q5_0.bin");

if (existsSync(outputPath)) {
	console.log(`model already exists: ${outputPath}`);
	process.exit(0);
}

mkdirSync(dirname(outputPath), { recursive: true });
console.log(`downloading ${modelUrl}`);
console.log(`to ${outputPath}`);
const response = await fetch(modelUrl);
if (!response.ok || !response.body) throw new Error(`download failed: ${response.status} ${await response.text()}`);
await pipeline(response.body, createWriteStream(outputPath));
console.log("done");
