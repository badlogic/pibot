import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { RobotClient } from "../robot-client.js";
import type { MemoryStore } from "./memory.js";
import { createMemoryTool } from "./memory.js";
import { createMotorTools } from "./motor.js";
import { createPhotoTool } from "./photo.js";
import { pageContentTool, webSearchTool } from "./websearch.js";

export { pruneImagesForContext } from "./context.js";
export { stopMotorFireAndForget } from "./motor.js";

export function createRobotTools(robot: RobotClient, memoryStore: MemoryStore): AgentTool[] {
	return [
		...createMotorTools(robot),
		createPhotoTool(robot),
		webSearchTool,
		pageContentTool,
		createMemoryTool(memoryStore),
	];
}
