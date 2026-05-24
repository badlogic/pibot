import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";

const memoryFile = resolve(process.cwd(), "data/memories.json");

const memoryParameters = Type.Object({
	action: Type.String({
		description:
			"Memory operation. Must be one of: read, append, remove. Use read to list memories, append to store a new memory, remove to delete by index.",
	}),
	text: Type.Optional(
		Type.String({
			description: "Required when action is append. The exact memory text to store.",
		}),
	),
	index: Type.Optional(
		Type.Number({
			description: "Required when action is remove. Zero-based memory index to remove.",
		}),
	),
});

type MemoryParameters = { action: "read" } | { action: "append"; text: string } | { action: "remove"; index: number };

function parseMemoryParameters(params: unknown): MemoryParameters {
	if (typeof params !== "object" || params === null) throw new Error("Memory tool parameters must be an object");
	const record = params as Record<string, unknown>;
	if (record.action === "read") return { action: "read" };
	if (record.action === "append") {
		if (typeof record.text !== "string") throw new Error('Memory action "append" requires string field "text"');
		return { action: "append", text: record.text };
	}
	if (record.action === "remove") {
		if (typeof record.index !== "number") throw new Error('Memory action "remove" requires number field "index"');
		return { action: "remove", index: record.index };
	}
	throw new Error('Memory field "action" must be one of: read, append, remove');
}

async function loadMemories(): Promise<string[]> {
	try {
		const parsed = JSON.parse(await readFile(memoryFile, "utf8")) as unknown;
		return Array.isArray(parsed) ? parsed.filter((entry): entry is string => typeof entry === "string") : [];
	} catch (error) {
		if (error instanceof Error && "code" in error && error.code === "ENOENT") return [];
		throw error;
	}
}

async function saveMemories(memories: string[]): Promise<void> {
	await mkdir(dirname(memoryFile), { recursive: true });
	await writeFile(memoryFile, `${JSON.stringify(memories, null, "\t")}\n`, "utf8");
}

export async function formatMemoriesForSystemPrompt(): Promise<string> {
	const memories = await loadMemories();
	if (memories.length === 0) return "No stored memories yet.";
	return memories.map((memory, index) => `${index}: ${memory}`).join("\n");
}

export const memoryTool = {
	name: "memory",
	label: "Memory",
	description:
		'Persistent robot memory. Call with {"action":"read"} to read all memories, {"action":"append","text":"..."} to store a new memory, or {"action":"remove","index":0} to remove one.',
	parameters: memoryParameters,
	executionMode: "sequential",
	execute: async (_id, rawParams) => {
		const params = parseMemoryParameters(rawParams);
		const memories = await loadMemories();
		if (params.action === "read") {
			return {
				content: [
					{
						type: "text",
						text:
							memories.length === 0
								? "No stored memories."
								: memories.map((memory, index) => `${index}: ${memory}`).join("\n"),
					},
				],
				details: { memories },
			};
		}
		if (params.action === "append") {
			const text = params.text.trim();
			if (!text) throw new Error("Memory text must not be empty");
			memories.push(text);
			await saveMemories(memories);
			return {
				content: [{ type: "text", text: `Stored memory ${memories.length - 1}: ${text}` }],
				details: { memories },
			};
		}
		if (params.index < 0 || params.index >= memories.length)
			throw new Error(`Memory index out of range: ${params.index}`);
		const removed = memories.splice(params.index, 1)[0]!;
		await saveMemories(memories);
		return {
			content: [{ type: "text", text: `Removed memory ${params.index}: ${removed}` }],
			details: { memories, removed },
		};
	},
} satisfies AgentTool;
