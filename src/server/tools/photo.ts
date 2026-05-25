import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";
import type { RobotClient } from "../robot-client.js";

interface PhotoToolDetails {
	mimeType: string;
	bytes: number;
}

const emptyParameters = Type.Object({});

function parsePhotoDataUrl(dataUrl: string): { base64: string; mimeType: string } | undefined {
	const match = /^data:([^;,]+);base64,(.+)$/.exec(dataUrl);
	if (!match) return undefined;
	return { mimeType: match[1] ?? "image/jpeg", base64: match[2] ?? "" };
}

export function createPhotoTool(robot: RobotClient): AgentTool<typeof emptyParameters, PhotoToolDetails> {
	return {
		name: "take_photo",
		label: "Take Photo",
		description: "Take a photo of your surroundings using the phone front-facing camera.",
		parameters: emptyParameters,
		execute: async () => {
			const result = await robot.execute({ type: "take_photo_request", payload: {}, timeoutMs: 15000 });
			const capture = parsePhotoDataUrl(result.dataUrl);
			if (!capture) throw new Error("Invalid photo data URL");
			return {
				content: [{ type: "image", data: capture.base64, mimeType: capture.mimeType }],
				details: { mimeType: capture.mimeType, bytes: capture.base64.length },
			};
		},
	};
}
