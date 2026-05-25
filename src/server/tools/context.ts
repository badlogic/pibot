import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Logger } from "../logger.js";

export function pruneImagesForContext(messages: AgentMessage[], maxImages: number, logger: Logger): AgentMessage[] {
	const prunedMessages = [...messages];
	let imageCount = 0;
	let removedImages = 0;

	for (let i = prunedMessages.length - 1; i >= 0; --i) {
		const message = prunedMessages[i]!;
		if ((message.role !== "user" && message.role !== "toolResult") || typeof message.content === "string") continue;

		const content = [...message.content];
		let changed = false;
		for (let ii = content.length - 1; ii >= 0; --ii) {
			if (content[ii]!.type !== "image") continue;
			imageCount++;
			if (imageCount <= maxImages) continue;
			content[ii] = { type: "text", text: "image removed" };
			changed = true;
			removedImages++;
		}
		if (changed) prunedMessages[i] = { ...message, content };
	}

	if (removedImages > 0) logger.log(`removed ${removedImages} old image(s), kept ${maxImages}`);
	return prunedMessages;
}
