import type { RobotRpcMap, RobotRpcType, SpotifyNowPlaying } from "../../types.js";
import type { ClientLogger } from "../logger.js";
import { createMotorTool, type MotorTool } from "./motor.js";
import { createPhotoTool, type PhotoTool } from "./photo.js";
import { createSpeechTool, type SpeechTool } from "./speech.js";
import { createSpotifyTool, type SpotifyTool } from "./spotify.js";

export type RobotToolHandler<T extends RobotRpcType> = (
	payload: RobotRpcMap[T]["request"],
	signal: AbortSignal,
) => Promise<RobotRpcMap[T]["response"]> | RobotRpcMap[T]["response"];

export interface RobotToolHandlers {
	take_photo: RobotToolHandler<"take_photo">;
	motor: RobotToolHandler<"motor">;
	spotify: RobotToolHandler<"spotify">;
	speak: RobotToolHandler<"speak">;
	cancel_speech: RobotToolHandler<"cancel_speech">;
}

export interface RobotTools {
	handlers: RobotToolHandlers;
	photo: PhotoTool;
	motor: MotorTool;
	spotify: SpotifyTool;
	speech: SpeechTool;
}

export function createRobotTools(deps: {
	logger: ClientLogger;
	face: HTMLElement;
	setMicInputBlockedUntil: (time: number) => void;
	onSpeakingChange: (speaking: boolean) => void;
	onPlaybackAudio: (samples: Float32Array, sampleRate: number) => void;
	onSpotifyPlaybackChange: (playback: SpotifyNowPlaying | undefined) => void;
	onSpotifyStatusChange: () => void;
}): RobotTools {
	const photo = createPhotoTool({ logger: deps.logger });
	const motor = createMotorTool({ logger: deps.logger });
	const spotify = createSpotifyTool({
		logger: deps.logger,
		onPlaybackChange: deps.onSpotifyPlaybackChange,
		onStatusChange: deps.onSpotifyStatusChange,
	});
	const speech = createSpeechTool({
		logger: deps.logger,
		face: deps.face,
		setMicInputBlockedUntil: deps.setMicInputBlockedUntil,
		onSpeakingChange: deps.onSpeakingChange,
		onPlaybackAudio: deps.onPlaybackAudio,
	});
	return {
		handlers: {
			take_photo: photo.handle,
			motor: motor.handle,
			spotify: spotify.handle,
			speak: speech.handleSpeak,
			cancel_speech: speech.handleCancelSpeech,
		},
		photo,
		motor,
		spotify,
		speech,
	};
}
