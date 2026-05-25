import { randomUUID } from "node:crypto";
import { WebSocket } from "ws";
import type { ClientMessage, RobotRpcMap, RobotRpcType, RobotWireResponse } from "../types.js";
import type { Logger } from "./logger.js";

type RobotRpcResponse = RobotRpcMap[RobotRpcType]["response"];

interface PendingRobotRequest {
	client: WebSocket;
	requestType: RobotRpcType;
	resolve: (value: RobotRpcResponse) => void;
	reject: (error: Error) => void;
	timeout: NodeJS.Timeout;
}

export class RobotClient {
	private current: WebSocket | undefined;
	private readonly pending = new Map<string, PendingRobotRequest>();
	private readonly heartbeat: NodeJS.Timeout;
	private readonly logger: Logger;

	constructor(logger: Logger) {
		this.logger = logger.tag("robot");
		this.heartbeat = setInterval(() => this.checkConnection(), 5000);
	}

	currentClient(): WebSocket | undefined {
		return this.current;
	}

	handleConnection(client: WebSocket): void {
		if (!this.current || this.current.readyState !== WebSocket.OPEN) {
			this.current = client;
			this.logger.log("selected client");
		}
	}

	handleDisconnect(client: WebSocket): void {
		this.rejectPendingForClient(client, new Error("Robot client disconnected"));
		if (this.current === client) this.current = undefined;
	}

	handleMessage(msg: ClientMessage): boolean {
		if (msg.type !== "robot_response") return false;
		this.resolveResponse(msg);
		return true;
	}

	async execute<const T extends RobotRpcType>(request: {
		type: T;
		payload: RobotRpcMap[T]["request"];
		timeoutMs?: number;
	}): Promise<RobotRpcMap[T]["response"]> {
		const client = this.current;
		if (!client || client.readyState !== WebSocket.OPEN) throw new Error("Robot client not connected");
		const id = randomUUID();
		const timeoutMs = request.timeoutMs ?? 15000;
		client.send(JSON.stringify({ type: "robot_request", id, request }));
		return await new Promise<RobotRpcMap[T]["response"]>((resolve, reject) => {
			const timeout = setTimeout(() => {
				this.rejectPending(id, new Error(`Robot request timed out: ${request.type}`));
			}, timeoutMs);
			this.pending.set(id, {
				client,
				requestType: request.type,
				resolve: (value) => resolve(value as RobotRpcMap[T]["response"]),
				reject,
				timeout,
			});
		});
	}

	rejectAll(reason: string): void {
		for (const id of this.pending.keys()) this.rejectPending(id, new Error(reason));
	}

	stop(): void {
		clearInterval(this.heartbeat);
		this.rejectAll("Robot client stopped");
	}

	selectFallback(candidates: Iterable<WebSocket>): void {
		this.current = [...candidates][0];
		if (this.current) this.logger.log("selected fallback client");
	}

	private resolveResponse(response: RobotWireResponse): void {
		const pending = this.pending.get(response.id);
		if (!pending) return;
		if (pending.requestType !== response.requestType) {
			this.rejectPending(
				response.id,
				new Error(`Robot response type mismatch: expected ${pending.requestType}, got ${response.requestType}`),
			);
			return;
		}
		clearTimeout(pending.timeout);
		this.pending.delete(response.id);
		if (response.error) {
			pending.reject(new Error(response.error));
			return;
		}
		if (response.payload === undefined) {
			pending.reject(new Error(`Robot response missing payload: ${response.requestType}`));
			return;
		}
		pending.resolve(response.payload);
	}

	private rejectPending(id: string, error: Error): void {
		const pending = this.pending.get(id);
		if (!pending) return;
		clearTimeout(pending.timeout);
		this.pending.delete(id);
		pending.reject(error);
	}

	private rejectPendingForClient(client: WebSocket, error: Error): void {
		for (const [id, pending] of this.pending) {
			if (pending.client === client) this.rejectPending(id, error);
		}
	}

	private checkConnection(): void {
		const client = this.current;
		if (!client) return;
		if (client.readyState !== WebSocket.OPEN) {
			this.handleDisconnect(client);
			return;
		}
		try {
			client.ping();
		} catch {
			this.handleDisconnect(client);
		}
	}
}
