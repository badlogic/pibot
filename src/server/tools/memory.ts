import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";

const memoryParameters = Type.Union([
	Type.Object({ action: Type.Literal("read") }),
	Type.Object({
		action: Type.Literal("append"),
		text: Type.String({ description: "The exact memory text to store." }),
	}),
	Type.Object({
		action: Type.Literal("remove"),
		index: Type.Number({ description: "Zero-based memory index to remove." }),
	}),
]);

interface MemoryToolDetails {
	memories: string[];
	removed?: string;
}

export interface MemoryStore {
	list: () => Promise<string[]>;
	append: (text: string) => Promise<string[]>;
	remove: (index: number) => Promise<{ memories: string[]; removed: string }>;
}

export function createMemoryTool(store: MemoryStore): AgentTool<typeof memoryParameters, MemoryToolDetails> {
	return {
		name: "memory",
		label: "Memory",
		description:
			'Persistent robot memory. Call with {"action":"read"} to read all memories, {"action":"append","text":"..."} to store a new memory, or {"action":"remove","index":0} to remove one.',
		parameters: memoryParameters,
		executionMode: "sequential",
		execute: async (_id, params) => {
			if (params.action === "read") {
				const memories = await store.list();
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
				const memories = await store.append(text);
				return {
					content: [{ type: "text", text: `Stored memory ${memories.length - 1}: ${text}` }],
					details: { memories },
				};
			}
			const { memories, removed } = await store.remove(params.index);
			return {
				content: [{ type: "text", text: `Removed memory ${params.index}: ${removed}` }],
				details: { memories, removed },
			};
		},
	};
}
