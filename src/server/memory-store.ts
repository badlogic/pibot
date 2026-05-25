import type { ExecutionEnv } from "@earendil-works/pi-agent-core";
import type { MemoryStore } from "./tools/memory.js";

export interface EnvMemoryStoreOptions {
	path: string;
}

function fileErrorMessage(error: unknown): string {
	if (error instanceof Error) return error.message;
	return JSON.stringify(error);
}

export function createEnvMemoryStore(env: ExecutionEnv, options: EnvMemoryStoreOptions): MemoryStore {
	let memories: string[] | undefined;

	async function load(): Promise<string[]> {
		if (memories) return memories;
		const exists = await env.exists(options.path);
		if (!exists.ok) throw new Error(`Failed to check memory file: ${fileErrorMessage(exists.error)}`);
		if (!exists.value) {
			memories = [];
			return memories;
		}
		const content = await env.readTextFile(options.path);
		if (!content.ok) throw new Error(`Failed to read memory file: ${fileErrorMessage(content.error)}`);
		const parsed = JSON.parse(content.value) as unknown;
		memories = Array.isArray(parsed) ? parsed.filter((entry): entry is string => typeof entry === "string") : [];
		return memories;
	}

	async function save(): Promise<void> {
		const result = await env.writeFile(options.path, `${JSON.stringify(await load(), null, "\t")}\n`);
		if (!result.ok) throw new Error(`Failed to write memory file: ${fileErrorMessage(result.error)}`);
	}

	return {
		list: async () => [...(await load())],
		append: async (text) => {
			const current = await load();
			current.push(text);
			await save();
			return [...current];
		},
		remove: async (index) => {
			const current = await load();
			if (index < 0 || index >= current.length) throw new Error(`Memory index out of range: ${index}`);
			const removed = current.splice(index, 1)[0]!;
			await save();
			return { memories: [...current], removed };
		},
	};
}
