import { AgentHarness, InMemorySessionRepo } from "@earendil-works/pi-agent-core";
import type { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
import {
	type Api,
	type AssistantMessage,
	getModel,
	getModels,
	getProviders,
	type KnownProvider,
	type Model,
	type TextContent,
} from "@earendil-works/pi-ai";
import type { LocalLlmConfig, LocalLlmId } from "./llama.js";
import { localLlmConfigs } from "./llama.js";
import type { Logger } from "./logger.js";
import type { RobotClient } from "./robot-client.js";
import { createRobotTools, pruneImagesForContext } from "./tools/index.js";
import type { MemoryStore } from "./tools/memory.js";

const LOCAL_PROVIDER = "llama-cpp-local";
const LOCAL_API_KEY = "EMPTY";

function createLocalModel(config: LocalLlmConfig, baseUrl: string, contextWindow: number): Model<"openai-completions"> {
	return {
		id: config.modelFile,
		name: config.name,
		api: "openai-completions",
		provider: LOCAL_PROVIDER,
		baseUrl,
		reasoning: false,
		input: [...config.input],
		cost: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
		},
		contextWindow,
		maxTokens: config.maxTokens,
		compat: {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			supportsUsageInStreaming: false,
			supportsStrictMode: false,
			maxTokensField: "max_tokens",
		},
	};
}

function selectModel(localLlm: LocalLlmId, localBaseUrl: string, localContextWindow: number): Model<Api> {
	const provider = process.env.PI_PROVIDER ?? LOCAL_PROVIDER;
	const modelId = process.env.PI_MODEL ?? localLlmConfigs[localLlm].modelFile;
	if (provider === LOCAL_PROVIDER) {
		const config = localLlmConfigs[localLlm];
		if (modelId !== config.modelFile) throw new Error(`Unknown PI_MODEL for ${provider}/${localLlm}: ${modelId}`);
		return createLocalModel(config, localBaseUrl, localContextWindow);
	}
	if (!getProviders().includes(provider as KnownProvider)) throw new Error(`Unknown PI_PROVIDER: ${provider}`);
	const models = getModels(provider as KnownProvider);
	if (!models.some((model) => model.id === modelId)) throw new Error(`Unknown PI_MODEL for ${provider}: ${modelId}`);
	return getModel(provider as KnownProvider, modelId as never) as Model<Api>;
}

function formatMemories(memories: string[]): string {
	if (memories.length === 0) return "No stored memories yet.";
	return memories.map((memory, index) => `${index}: ${memory}`).join("\n");
}

async function buildSystemPrompt(memoryStore: MemoryStore): Promise<string> {
	return `Du bist das Gehirn eines kleinen Roboters mit Smartphone. Antworte immer auf Deutsch. Sei verspielt, freundlich und sicher. Dein Text wird direkt an eine Plaintext-Sprachausgabe gesendet: Verwende kein Markdown, keine Listen, keine Codeblöcke, keine Überschriften und keine Emojis. Schreibe Zahlen so, dass eine Sprachausgabe sie natürlich vorliest: vermeide Ziffern mit Tausender- oder Dezimaltrennzeichen wie 6.400 oder 1,23; schreibe stattdessen ausgeschriebene oder eindeutig sprechbare Formen wie sechstausendvierhundert, eins Komma zwei drei oder one point two three, passend zur Antwortsprache. Nutze Bewegungswerkzeuge nur für kurze Dauer. Die Bewegungswerkzeuge stoppen automatisch nach ihrer Dauer. Die Hardware kann nur vorwärts fahren und sich gegen den Uhrzeigersinn drehen; rückwärts und rechts gibt es nicht. Für ungefähre Drehwinkel nutze turn_left_degrees. Wenn eine Aufgabe ein Werkzeug erfordert, rufe es sofort per Tool-Call auf, bevor du antwortest; kündige es nicht nur an. Nutze spotify_search für Musik, Kinderlieder, Playlists, Podcasts oder Hörbücher, wenn du die exakte Spotify-URI noch nicht kennst; spiele danach die gewünschte URI mit spotify_play ab. Nutze spotify_control zum Pausieren, Fortsetzen, Überspringen oder Prüfen der aktuellen Wiedergabe. Wenn du aktuelle Fakten oder Internet-Informationen brauchst, nutze web_search. Wenn du Details aus einem gefundenen Treffer brauchst, nutze fetch_page_content mit der URL.

Persistente Erinnerungen:
${formatMemories(await memoryStore.list())}

Memory-Werkzeug:
- Nutze das Memory-Werkzeug über die Tool-Calling-Schnittstelle, nicht als Text in deiner Antwort.
- Schreibe niemals Tool-Aufrufe wie memory(...) oder JSON für Werkzeuge in den normalen Antworttext.
- Wenn du Erinnerungen lesen sollst, rufe memory mit action read auf oder ohne Argumente.
- Wenn du etwas speichern sollst, rufe memory mit action append und text auf. Behaupte erst danach, dass es gespeichert wurde.
- Wenn du eine Erinnerung löschen sollst, rufe memory mit action remove und index auf.`;
}

function extractAssistantText(message: AssistantMessage): string {
	return message.content
		.filter((entry): entry is TextContent => entry.type === "text")
		.map((entry) => entry.text)
		.join("")
		.trim();
}

export type RobotHarnessEvent =
	| { type: "assistant_start" }
	| { type: "assistant_delta"; text: string }
	| { type: "tool_start"; name: string; args: unknown }
	| { type: "tool_end"; name: string; isError: boolean }
	| { type: "assistant_end"; text: string }
	| { type: "session_reset"; reason: string };

export interface RobotHarness {
	current: () => AgentHarness;
	model: () => Model<Api>;
	rebuildSession: (reason: string) => Promise<void>;
}

export async function createRobotHarness(deps: {
	env: NodeExecutionEnv;
	logger: Logger;
	memoryStore: MemoryStore;
	localLlm: LocalLlmId;
	localBaseUrl: string;
	localContextWindow: number;
	maxContextImages: number;
	robot: RobotClient;
	onEvent: (event: RobotHarnessEvent) => void | Promise<void>;
	beforeTool: (name: string, args: unknown) => void | Promise<void>;
}): Promise<RobotHarness> {
	const sessionRepo = new InMemorySessionRepo();
	const tools = createRobotTools(deps.robot, deps.memoryStore);
	const contextLogger = deps.logger.tag("context");
	const emit = async (event: RobotHarnessEvent): Promise<void> => {
		try {
			await deps.onEvent(event);
		} catch (error) {
			console.error(`[harness] event handler failed: ${error instanceof Error ? error.message : String(error)}`);
		}
	};
	const selectedModel = selectModel(deps.localLlm, deps.localBaseUrl, deps.localContextWindow);
	let harness = await buildHarness();

	async function buildHarness(): Promise<AgentHarness> {
		const session = await sessionRepo.create({ id: `robot-demo-${Date.now()}` });
		const newHarness = new AgentHarness({
			env: deps.env,
			session,
			model: selectedModel,
			getApiKeyAndHeaders: async (model) => {
				const envName = `${model.provider.toUpperCase()}_API_KEY`.replaceAll("-", "_");
				const apiKey = process.env[envName] ?? process.env.OPENAI_API_KEY ?? process.env.ANTHROPIC_API_KEY;
				if (apiKey) return { apiKey };
				if (model.provider === LOCAL_PROVIDER) return { apiKey: LOCAL_API_KEY };
				return undefined;
			},
			tools,
			systemPrompt: async () => buildSystemPrompt(deps.memoryStore),
		});
		newHarness.on("tool_call", async (event) => {
			await deps.beforeTool(event.toolName, event.input);
			return undefined;
		});
		newHarness.on("context", async (event) => {
			const context = pruneImagesForContext(event.messages, deps.maxContextImages);
			if (context.removedImages > 0) {
				contextLogger.log(`removed ${context.removedImages} old image(s), kept ${deps.maxContextImages}`);
			}
			return { messages: context.messages };
		});
		newHarness.subscribe(async (event) => {
			if (event.type === "message_start" && event.message.role === "assistant")
				await emit({ type: "assistant_start" });
			if (
				event.type === "message_update" &&
				event.message.role === "assistant" &&
				event.assistantMessageEvent.type === "text_delta"
			) {
				await emit({ type: "assistant_delta", text: event.assistantMessageEvent.delta });
			}
			if (event.type === "tool_execution_start") {
				await emit({ type: "tool_start", name: event.toolName, args: event.args });
			}
			if (event.type === "tool_execution_end") {
				await emit({ type: "tool_end", name: event.toolName, isError: event.isError });
			}
			if (event.type === "message_end" && event.message.role === "assistant") {
				await emit({ type: "assistant_end", text: extractAssistantText(event.message) });
			}
		});
		return newHarness;
	}

	return {
		current: () => harness,
		model: () => selectedModel,
		rebuildSession: async (reason) => {
			harness = await buildHarness();
			await emit({ type: "session_reset", reason });
		},
	};
}
