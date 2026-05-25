import { readFile } from "node:fs/promises";
import { createServer, type IncomingMessage } from "node:http";
import { extname, join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { AgentHarness, InMemorySessionRepo } from "@earendil-works/pi-agent-core";
import { NodeExecutionEnv } from "@earendil-works/pi-agent-core/node";
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
import { WebSocket, WebSocketServer } from "ws";
import type { ClientLogMsg, ClientMessage } from "../types.js";
import { createLogger } from "./logger.js";
import { createEnvMemoryStore } from "./memory-store.js";
import { RobotClient } from "./robot-client.js";
import { createSttService } from "./stt.js";
import { createRobotTools, pruneImagesForContext, stopMotorFireAndForget } from "./tools/index.js";
import { createTtsService } from "./tts.js";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const publicDir = resolve(__dirname, "../../public");
const port = Number(process.env.PORT ?? 8010);
const parakeetSttWorkerPath = resolve(__dirname, "../../scripts/parakeet-stt-worker.py");
const serverVersion = String(Date.now());
const maxContextImages = Number(process.env.MAX_CONTEXT_IMAGES ?? 4);

const logger = createLogger();
const serverLogger = logger.tag("server");
const clientLogger = logger.tag("client");
const agentLogger = logger.tag("agent");
const contextLogger = logger.tag("context");
const clients = new Set<WebSocket>();
const robot = new RobotClient(logger);
const executionEnv = new NodeExecutionEnv({ cwd: process.cwd() });
const memoryStore = createEnvMemoryStore(executionEnv, { path: process.env.MEMORY_FILE ?? "data/memories.json" });
let sttPromptQueue: Promise<void> = Promise.resolve();
let harness: AgentHarness;

async function readRequestJson<T>(req: AsyncIterable<Uint8Array>): Promise<T> {
	const chunks: Uint8Array[] = [];
	for await (const chunk of req) chunks.push(chunk);
	return JSON.parse(Buffer.concat(chunks).toString("utf8")) as T;
}

function logClientMessage(msg: ClientLogMsg): void {
	clientLogger.tag(msg.level).log(msg.message);
}

function broadcast(data: object): void {
	const msg = JSON.stringify(data);
	for (const client of clients) {
		if (client.readyState === WebSocket.OPEN) client.send(msg);
	}
}

function extractAssistantText(message: AssistantMessage): string {
	return message.content
		.filter((entry): entry is TextContent => entry.type === "text")
		.map((entry) => entry.text)
		.join("")
		.trim();
}

function formatMemoriesForSystemPrompt(memories: string[]): string {
	if (memories.length === 0) return "No stored memories yet.";
	return memories.map((memory, index) => `${index}: ${memory}`).join("\n");
}

function selectModel(): Model<Api> {
	const provider = process.env.PI_PROVIDER ?? "anthropic";
	const modelId = process.env.PI_MODEL ?? "claude-haiku-4-5";
	if (!getProviders().includes(provider as KnownProvider)) {
		throw new Error(`Unknown PI_PROVIDER: ${provider}`);
	}
	const models = getModels(provider as KnownProvider);
	if (!models.some((model) => model.id === modelId)) {
		throw new Error(`Unknown PI_MODEL for ${provider}: ${modelId}`);
	}
	return getModel(provider as KnownProvider, modelId as never) as Model<Api>;
}

const ttsService = createTtsService({
	logger,
	getRobotClient: () => robot.currentClient(),
});

async function performHarnessAbort(reason: string, sttIndex?: number): Promise<void> {
	agentLogger.log(`abort: ${reason}`);
	ttsService.cancelSpeechOnClients(reason, sttIndex);
	ttsService.resolveAllSpeech();
	robot.rejectAll(reason);
	stopMotorFireAndForget(robot);
	try {
		await harness.abort();
	} catch (error) {
		agentLogger.log(`harness abort error: ${error instanceof Error ? error.message : String(error)}`);
	}
}

function enqueueSttPrompt(text: string): void {
	sttPromptQueue = sttPromptQueue.then(async () => {
		for (let attempt = 1; attempt <= 30; attempt++) {
			try {
				await harness.prompt(`${text}`);
				return;
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				if (!message.includes("busy") || attempt === 30) {
					logger.tag("stt").log(`prompt failed: ${message}`);
					broadcast({ type: "stt_event", event: "error", message });
					return;
				}
				logger.tag("stt").log(`harness busy; retrying prompt in 500ms attempt=${attempt}`);
				await sleep(500);
			}
		}
	});
}

const sttService = createSttService({
	workerPath: parakeetSttWorkerPath,
	logger,
	broadcast,
	enqueuePrompt: enqueueSttPrompt,
	performAbort: performHarnessAbort,
	shouldIgnoreNonStopFinal: ttsService.shouldIgnoreNonStopSttAsTtsBleed,
});

function stopChildProcesses(): void {
	ttsService.stopChildProcess();
	sttService.stopChildProcess();
	robot.stop();
}

process.once("exit", stopChildProcesses);
process.once("SIGINT", async () => {
	stopChildProcesses();
	await logger.flush();
	process.exit(130);
});
process.once("SIGTERM", async () => {
	stopChildProcesses();
	await logger.flush();
	process.exit(143);
});

const tools = createRobotTools(robot, memoryStore);
const sessionRepo = new InMemorySessionRepo();

async function buildHarness(): Promise<AgentHarness> {
	const session = await sessionRepo.create({ id: `robot-demo-${Date.now()}` });
	const newHarness = new AgentHarness({
		env: executionEnv,
		session,
		model: selectModel(),
		getApiKeyAndHeaders: async (model) => {
			const envName = `${model.provider.toUpperCase()}_API_KEY`.replaceAll("-", "_");
			const apiKey = process.env[envName] ?? process.env.OPENAI_API_KEY ?? process.env.ANTHROPIC_API_KEY;
			return apiKey ? { apiKey } : undefined;
		},
		tools,
		systemPrompt:
			async () => `Du bist das Gehirn eines kleinen Roboters mit Smartphone. Antworte immer auf Deutsch. Sei verspielt, freundlich und sicher. Verwende keine Emojis. Nutze Bewegungswerkzeuge nur für kurze Dauer. Die Bewegungswerkzeuge stoppen automatisch nach ihrer Dauer. Die Hardware kann nur vorwärts fahren und sich gegen den Uhrzeigersinn drehen; rückwärts und rechts gibt es nicht. Für ungefähre Drehwinkel nutze turn_left_degrees. Wenn du aktuelle Fakten oder Internet-Informationen brauchst, nutze web_search. Wenn du Details aus einem gefundenen Treffer brauchst, nutze fetch_page_content mit der URL.

Persistente Erinnerungen:
${formatMemoriesForSystemPrompt(await memoryStore.list())}

Memory-Tool-Aufrufschema:
- Alle Erinnerungen lesen: memory({"action":"read"})
- Neue Erinnerung speichern: memory({"action":"append","text":"Pipi ist der Name des Roboters"})
- Erinnerung löschen: memory({"action":"remove","index":0})`,
	});
	newHarness.on("context", (event) => ({
		messages: pruneImagesForContext(event.messages, maxContextImages, contextLogger),
	}));
	newHarness.subscribe(async (event) => {
		broadcast({ type: "agent_event", event });
		if (event.type === "message_end" && event.message.role === "assistant") {
			await ttsService.speakOnClient(extractAssistantText(event.message));
		}
	});
	return newHarness;
}

harness = await buildHarness();

async function resetHarnessSession(reason: string): Promise<void> {
	serverLogger.log(`session reset: ${reason}`);
	await performHarnessAbort(`reset: ${reason}`);
	harness = await buildHarness();
	broadcast({ type: "session_reset" });
}

async function handleClientMessage(msg: ClientMessage): Promise<void> {
	if (msg.type === "client_log") {
		logClientMessage(msg);
		return;
	}
	if (msg.type === "speak_done" || msg.type === "speak_cancelled") {
		ttsService.resolveSpeech(msg.id);
		return;
	}
	if (robot.handleMessage(msg)) return;
	if (msg.type === "prompt") await harness.prompt(msg.text);
	if (msg.type === "abort") await performHarnessAbort("client abort");
	if (msg.type === "reset_session") await resetHarnessSession("client request");
}

const server = createServer(async (req, res) => {
	const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
	if (url.pathname === "/__version" && req.method === "GET") {
		res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
		res.end(JSON.stringify({ version: serverVersion }));
		return;
	}
	if (url.pathname === "/api/client-log" && req.method === "POST") {
		try {
			logClientMessage(await readRequestJson<ClientLogMsg>(req));
			res.writeHead(204).end();
		} catch (error) {
			serverLogger.log(`client log parse failed: ${error instanceof Error ? error.message : String(error)}`);
			res.writeHead(400).end();
		}
		return;
	}
	if (url.pathname === "/api/tts" && req.method === "GET") {
		try {
			await ttsService.handleTtsRequest(
				url.searchParams.get("text") ?? "",
				url.searchParams.get("provider") ?? undefined,
				res,
			);
		} catch (error) {
			res.writeHead(500, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: error instanceof Error ? error.message : String(error) }));
		}
		return;
	}
	const path = url.pathname === "/" ? "/index.html" : url.pathname;
	const file = join(publicDir, path);
	if (!file.startsWith(publicDir)) {
		res.writeHead(403).end();
		return;
	}
	try {
		const data = await readFile(file);
		const extension = extname(file);
		const contentType =
			extension === ".js"
				? "text/javascript; charset=utf-8"
				: extension === ".css"
					? "text/css; charset=utf-8"
					: "text/html; charset=utf-8";
		res.writeHead(200, {
			"content-type": contentType,
			"cache-control": "no-store, no-cache, must-revalidate, proxy-revalidate",
			pragma: "no-cache",
			expires: "0",
		});
		if (extension === ".html") {
			res.end(
				data
					.toString("utf8")
					.replaceAll("style.css?v=dev", `style.css?v=${serverVersion}`)
					.replaceAll("app.js?v=dev", `app.js?v=${serverVersion}`),
			);
			return;
		}
		res.end(data);
	} catch {
		res.writeHead(404).end("not found");
	}
});

const wss = new WebSocketServer({ noServer: true });
const reloadWss = new WebSocketServer({ noServer: true });

server.on("upgrade", (req, socket, head) => {
	const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
	const target = url.pathname === "/__reload" ? reloadWss : wss;
	target.handleUpgrade(req, socket, head, (ws) => target.emit("connection", ws, req));
});

wss.on("connection", (ws, _req: IncomingMessage) => {
	clients.add(ws);
	serverLogger.log("ws client connected");
	robot.handleConnection(ws);
	ws.send(JSON.stringify({ type: "hello" }));
	ws.on("message", async (data, isBinary) => {
		try {
			if (isBinary) {
				sttService.handleAudioFrame(Buffer.isBuffer(data) ? data : Buffer.from(data as ArrayBuffer));
				return;
			}
			await handleClientMessage(JSON.parse(String(data)) as ClientMessage);
		} catch (error) {
			ws.send(JSON.stringify({ type: "error", message: error instanceof Error ? error.message : String(error) }));
		}
	});
	ws.on("close", () => {
		serverLogger.log("ws client disconnected");
		clients.delete(ws);
		robot.handleDisconnect(ws);
		robot.selectFallback(clients);
		ttsService.resolveSpeechForClient(ws);
	});
});

reloadWss.on("connection", () => {
	// The client reloads when this socket reconnects after the dev supervisor restarts the server.
});

server.listen(port, "0.0.0.0", () => serverLogger.log(`robot demo: http://localhost:${port}`));
