export type MotorCommand = "forward" | "turn_left" | "turn_left_degrees" | "stop";

export type ClientLogLevel = "log" | "info" | "warn" | "error" | "debug" | "app";

export interface ClientLogMsg {
	type: "client_log";
	level: ClientLogLevel;
	message: string;
	time: number;
}

export type AgentMessageLike = {
	role: string;
	content?: unknown;
};

export type AgentEvent =
	| { type: "message_start"; message: AgentMessageLike }
	| { type: "message_update"; assistantMessageEvent?: { type: string; delta?: string } }
	| { type: "message_end"; message: AgentMessageLike }
	| { type: "tool_execution_start"; toolName: string; args: unknown }
	| { type: "other"; eventType: string };

export type SttEventName = "loading" | "ready" | "speech_start" | "speech_end" | "speech_drop" | "error";

export interface RobotRpcMap {
	take_photo_request: {
		request: Record<string, never>;
		response: { dataUrl: string };
	};
	motor_request: {
		request: { command: MotorCommand; durationMs: number; degrees?: number };
		response: { ok: true } | { ok: false; error: string };
	};
}

export type RobotRpcType = keyof RobotRpcMap;

export type RobotExecuteRequest<T extends RobotRpcType = RobotRpcType> = {
	[K in RobotRpcType]: {
		type: K;
		payload: RobotRpcMap[K]["request"];
		timeoutMs?: number;
	};
}[T];

export type RobotWireRequest<T extends RobotRpcType = RobotRpcType> = {
	[K in RobotRpcType]: {
		type: "robot_request";
		id: string;
		request: {
			type: K;
			payload: RobotRpcMap[K]["request"];
		};
	};
}[T];

export type RobotWireResponse<T extends RobotRpcType = RobotRpcType> = {
	[K in RobotRpcType]: {
		type: "robot_response";
		id: string;
		requestType: K;
		payload?: RobotRpcMap[K]["response"];
		error?: string;
	};
}[T];

export type ServerMessage =
	| { type: "hello" }
	| { type: "error"; message: string }
	| { type: "speak_request"; id: string; text: string }
	| { type: "cancel_speech"; reason: string; sttIndex?: number }
	| { type: "stt_event"; event: SttEventName; index?: number; message?: string }
	| { type: "stt_interim"; index: number; text: string }
	| { type: "stt_final"; index: number; text: string; accepted: boolean; ignoredReason?: string }
	| { type: "session_reset" }
	| { type: "agent_event"; event: AgentEvent }
	| RobotWireRequest;

export type ClientMessage =
	| { type: "prompt"; text: string }
	| { type: "speak_done"; id: string }
	| { type: "speak_cancelled"; id: string }
	| ClientLogMsg
	| { type: "abort" }
	| { type: "reset_session" }
	| RobotWireResponse;
