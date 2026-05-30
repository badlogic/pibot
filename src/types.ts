export type MotorCommand = "forward" | "turn_left" | "turn_left_degrees" | "stop";

export type SpotifyItemType = "track" | "album" | "playlist" | "audiobook" | "show" | "episode";
export type SpotifyAction = "search" | "play" | "pause" | "resume" | "next" | "current";
export type SpotifyControlAction = "pause" | "resume" | "next" | "current";

export interface SpotifyNowPlaying {
	title?: string;
	subtitle?: string;
	uri?: string;
	coverUrl?: string;
	isPlaying?: boolean;
}

export interface SpotifySearchResult extends SpotifyNowPlaying {
	type: SpotifyItemType;
	uri: string;
}

export interface SpotifyDeviceInfo {
	id: string;
	name: string;
	type: string;
	isActive: boolean;
}

export type SpotifyRpcRequest =
	| { action: "search"; query: string; itemType?: SpotifyItemType; limit?: number }
	| { action: "play"; uri: string }
	| { action: SpotifyControlAction };

export type SpotifyRpcResponse =
	| { ok: true; action: "search"; results: SpotifySearchResult[] }
	| ({ ok: true; action: Exclude<SpotifyAction, "search"> } & SpotifyNowPlaying)
	| { ok: false; error: string };

export type LogOrigin = "server" | "client";

export interface LogEntry {
	origin: LogOrigin;
	sequence: number;
	time: number;
	tags: string[];
	message: string;
	formatted: string;
}

export interface ClientLogMsg {
	type: "client_log";
	tags: string[];
	message: string;
	time: number;
}

export type RobotState =
	| { phase: "inactive" }
	| { phase: "listening" }
	| { phase: "hearing" }
	| { phase: "thinking"; heardText?: string; assistantText: string }
	| { phase: "tool"; name: string; args: unknown; assistantText: string }
	| { phase: "speaking"; assistantText: string }
	| { phase: "error"; message: string };

export interface RobotRpcMap {
	take_photo: {
		request: Record<string, never>;
		response: { dataUrl: string };
	};
	motor: {
		request: { command: MotorCommand; durationMs: number; degrees?: number };
		response: { ok: true } | { ok: false; error: string };
	};
	spotify: {
		request: SpotifyRpcRequest;
		response: SpotifyRpcResponse;
	};
	speak: {
		request: { url: string; text: string };
		response: { ok: true } | { ok: false; error: string };
	};
	cancel_speech: {
		request: { reason: string };
		response: { ok: true };
	};
}

export type RobotRpcType = keyof RobotRpcMap;

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

export interface RobotWireCancel {
	type: "robot_cancel";
	id: string;
	reason: string;
}

export type ServerMessage =
	| { type: "state"; state: RobotState }
	| { type: "log"; entry: LogEntry }
	| RobotWireRequest
	| RobotWireCancel;

export type ClientMessage =
	| ClientLogMsg
	| { type: "abort" }
	| { type: "barge_in" }
	| { type: "reset_session" }
	| { type: "tts_playback_done" }
	| { type: "tts_playback_error"; message: string }
	| RobotWireResponse;
