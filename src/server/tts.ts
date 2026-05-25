import { type ChildProcess, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import type { ServerResponse } from "node:http";
import { createConnection } from "node:net";
import { setTimeout as sleep } from "node:timers/promises";
import { WebSocket } from "ws";
import { logStreamLines } from "./log-stream.js";
import type { Logger } from "./logger.js";

type TtsProvider = "elevenlabs" | "pocket";

type ServerToClientMsg = { type: "cancel_speech"; reason: string; sttIndex?: number };

export interface TtsServiceDeps {
	logger: Logger;
	getRobotClient: () => WebSocket | undefined;
}

export interface TtsService {
	handleTtsRequest: (text: string, providerValue: string | undefined, res: ServerResponse) => Promise<void>;
	speakOnClient: (text: string) => Promise<void>;
	resolveSpeech: (id: string) => void;
	resolveSpeechForClient: (client: WebSocket) => void;
	resolveAllSpeech: () => void;
	cancelSpeechOnClients: (reason: string, sttIndex?: number) => void;
	shouldIgnoreNonStopSttAsTtsBleed: () => boolean;
	stopChildProcess: () => void;
}

export function createTtsService(deps: TtsServiceDeps): TtsService {
	const pocketTtsPort = Number(process.env.POCKET_TTS_PORT ?? 8020);
	const pocketTtsBindHost = process.env.POCKET_TTS_BIND_HOST ?? "127.0.0.1";
	const pocketTtsLanguage = process.env.POCKET_TTS_LANGUAGE ?? "german";
	const pocketTtsVoice = process.env.POCKET_TTS_VOICE ?? "eve";
	const pocketTtsUrl = process.env.POCKET_TTS_URL ?? `http://127.0.0.1:${pocketTtsPort}/tts`;
	const elevenLabsApiKey = process.env.ELEVENLABS_API_KEY;
	const elevenLabsVoiceId = process.env.ELEVENLABS_VOICE_ID ?? "r1pUec9VJPfpUaMUuRX2";
	const elevenLabsVoiceName = process.env.ELEVENLABS_VOICE_NAME ?? "pibot";
	const elevenLabsModelId = process.env.ELEVENLABS_MODEL_ID ?? "eleven_v3";
	const defaultTtsProvider = process.env.TTS_PROVIDER ?? "elevenlabs";

	const pendingSpeech = new Map<string, { client: WebSocket; resolve: () => void; timeout: NodeJS.Timeout }>();
	let pocketTtsProcess: ChildProcess | undefined;
	let pocketTtsStartPromise: Promise<void> | undefined;
	let pocketTtsLastError: Error | undefined;
	let lastSpeechResolvedAt = 0;
	const logger = deps.logger.tag("tts");

	function sendToClient(client: WebSocket | undefined, data: ServerToClientMsg): void {
		if (!client || client.readyState !== WebSocket.OPEN) return;
		client.send(JSON.stringify(data));
	}

	function cancelSpeechOnClients(reason: string, sttIndex?: number): void {
		const targets = new Set<WebSocket>();
		const robotClient = deps.getRobotClient();
		if (robotClient) targets.add(robotClient);
		for (const pending of pendingSpeech.values()) targets.add(pending.client);
		for (const client of targets) sendToClient(client, { type: "cancel_speech", reason, sttIndex });
	}

	function pocketTtsEndpoint(): { host: string; port: number } {
		const url = new URL(pocketTtsUrl);
		return { host: url.hostname, port: Number(url.port || (url.protocol === "https:" ? 443 : 80)) };
	}

	async function canConnectToPocketTts(): Promise<boolean> {
		const endpoint = pocketTtsEndpoint();
		return await new Promise<boolean>((resolve) => {
			const socket = createConnection({ host: endpoint.host, port: endpoint.port });
			const done = (connected: boolean) => {
				socket.removeAllListeners();
				socket.destroy();
				resolve(connected);
			};
			socket.setTimeout(500);
			socket.once("connect", () => done(true));
			socket.once("error", () => done(false));
			socket.once("timeout", () => done(false));
		});
	}

	function startPocketTtsProcess(): void {
		if (pocketTtsProcess && !pocketTtsProcess.killed) return;
		pocketTtsLastError = undefined;
		logger.log(
			`starting Pocket TTS: uvx pocket-tts serve --language ${pocketTtsLanguage} --host ${pocketTtsBindHost} --port ${pocketTtsPort}`,
		);
		const child = spawn(
			"uvx",
			[
				"pocket-tts",
				"serve",
				"--language",
				pocketTtsLanguage,
				"--host",
				pocketTtsBindHost,
				"--port",
				String(pocketTtsPort),
			],
			{ stdio: ["ignore", "pipe", "pipe"] },
		);
		pocketTtsProcess = child;
		logStreamLines(child.stdout, logger.tag("pocket"));
		logStreamLines(child.stderr, logger.tag("pocket"));
		child.once("error", (error) => {
			pocketTtsLastError = error;
			logger.log(`Pocket TTS failed to start: ${error.message}`);
		});
		child.once("exit", (code, signal) => {
			if (pocketTtsProcess === child) pocketTtsProcess = undefined;
			if (code !== 0) logger.log(`Pocket TTS exited code=${code ?? "none"} signal=${signal ?? "none"}`);
		});
	}

	async function ensurePocketTtsStarted(): Promise<void> {
		if (await canConnectToPocketTts()) return;
		pocketTtsStartPromise ??= (async () => {
			if (!(await canConnectToPocketTts())) startPocketTtsProcess();
			const deadline = Date.now() + 60000;
			while (Date.now() < deadline) {
				if (pocketTtsLastError) throw pocketTtsLastError;
				if (await canConnectToPocketTts()) {
					logger.log(`Pocket TTS ready at ${pocketTtsUrl}`);
					return;
				}
				await sleep(500);
			}
			throw new Error("Pocket TTS did not become ready within 60s. Install uv and ensure uvx pocket-tts works.");
		})();
		try {
			await pocketTtsStartPromise;
		} finally {
			pocketTtsStartPromise = undefined;
		}
	}

	function normalizeTtsProvider(value: string | undefined): TtsProvider {
		if (value === "pocket" || value === "kyutai") return "pocket";
		return "elevenlabs";
	}

	async function resolveElevenLabsVoiceId(): Promise<string> {
		if (!elevenLabsApiKey || process.env.ELEVENLABS_VOICE_ID) return elevenLabsVoiceId;
		try {
			const response = await fetch("https://api.elevenlabs.io/v1/voices", {
				headers: { "xi-api-key": elevenLabsApiKey },
			});
			if (!response.ok) return elevenLabsVoiceId;
			const data = (await response.json()) as { voices?: Array<{ name?: string; voice_id?: string }> };
			const voice = data.voices?.find((entry) => entry.name === elevenLabsVoiceName);
			return voice?.voice_id ?? elevenLabsVoiceId;
		} catch (error) {
			logger.log(`ElevenLabs voice lookup failed: ${error instanceof Error ? error.message : String(error)}`);
			return elevenLabsVoiceId;
		}
	}

	async function proxyAudioResponse(
		response: Response,
		res: ServerResponse,
		fallbackContentType: string,
	): Promise<void> {
		if (!response.ok || !response.body) {
			res.writeHead(response.status || 502, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: await response.text() }));
			return;
		}
		res.writeHead(200, {
			"content-type": response.headers.get("content-type") ?? fallbackContentType,
			"cache-control": "no-store",
		});
		for await (const chunk of response.body as AsyncIterable<Uint8Array>) {
			res.write(chunk);
		}
		res.end();
	}

	async function handlePocketTtsRequest(text: string, res: ServerResponse): Promise<void> {
		await ensurePocketTtsStarted();
		const form = new FormData();
		form.set("text", text);
		form.set("voice_url", pocketTtsVoice);
		await proxyAudioResponse(await fetch(pocketTtsUrl, { method: "POST", body: form }), res, "audio/wav");
	}

	async function handleElevenLabsTtsRequest(text: string, res: ServerResponse): Promise<void> {
		if (!elevenLabsApiKey) {
			res.writeHead(503, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: "ELEVENLABS_API_KEY missing" }));
			return;
		}
		const voiceId = await resolveElevenLabsVoiceId();
		const response = await fetch(
			`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}/stream?output_format=mp3_44100_128`,
			{
				method: "POST",
				headers: {
					accept: "audio/mpeg",
					"content-type": "application/json",
					"xi-api-key": elevenLabsApiKey,
				},
				body: JSON.stringify({ text, model_id: elevenLabsModelId }),
			},
		);
		await proxyAudioResponse(response, res, "audio/mpeg");
	}

	async function handleTtsRequest(
		text: string,
		providerValue: string | undefined,
		res: ServerResponse,
	): Promise<void> {
		const trimmed = text.trim();
		if (!trimmed) {
			res.writeHead(400, { "content-type": "application/json" });
			res.end(JSON.stringify({ error: "text required" }));
			return;
		}
		const provider = normalizeTtsProvider(providerValue ?? defaultTtsProvider);
		res.setHeader("x-pibot-tts-provider", provider);
		if (provider === "pocket") await handlePocketTtsRequest(trimmed, res);
		else await handleElevenLabsTtsRequest(trimmed, res);
	}

	function resolveSpeech(id: string): void {
		const pending = pendingSpeech.get(id);
		if (!pending) return;
		logger.log(`speech resolved id=${id}`);
		lastSpeechResolvedAt = Date.now();
		clearTimeout(pending.timeout);
		pendingSpeech.delete(id);
		pending.resolve();
	}

	function resolveSpeechForClient(client: WebSocket): void {
		for (const [id, pending] of pendingSpeech) {
			if (pending.client === client) resolveSpeech(id);
		}
	}

	function resolveAllSpeech(): void {
		for (const id of pendingSpeech.keys()) resolveSpeech(id);
	}

	async function speakOnClient(text: string): Promise<void> {
		const client = deps.getRobotClient();
		if (!client || client.readyState !== WebSocket.OPEN) {
			logger.log("no robot client connected for speech");
			return;
		}
		const trimmed = text.trim();
		if (!trimmed) return;
		const id = randomUUID();
		logger.log(`speak_request id=${id} chars=${trimmed.length}`);
		client.send(JSON.stringify({ type: "speak_request", id, text: trimmed }));
		await new Promise<void>((resolve) => {
			const timeout = setTimeout(() => resolveSpeech(id), 30000);
			pendingSpeech.set(id, { client, resolve, timeout });
		});
	}

	function shouldIgnoreNonStopSttAsTtsBleed(): boolean {
		return pendingSpeech.size > 0 || Date.now() - lastSpeechResolvedAt < 1500;
	}

	function stopChildProcess(): void {
		pocketTtsProcess?.kill();
	}

	return {
		handleTtsRequest,
		speakOnClient,
		resolveSpeech,
		resolveSpeechForClient,
		resolveAllSpeech,
		cancelSpeechOnClients,
		shouldIgnoreNonStopSttAsTtsBleed,
		stopChildProcess,
	};
}
