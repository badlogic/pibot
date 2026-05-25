import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";
import type { MotorCommand } from "../../types.js";
import type { RobotClient } from "../robot-client.js";

interface MotorToolDetails {
	command: string;
	durationMs: number;
	error?: string;
}

interface TurnDegreesDetails extends MotorToolDetails {
	degrees: number;
}

const motorParameters = Type.Object({
	durationMs: Type.Number({ description: "Duration in milliseconds. Required. No default is assumed." }),
});

const turnDegreesParameters = Type.Object({
	degrees: Type.Optional(
		Type.Number({ description: "Counter-clockwise turn amount in degrees. Max 359. Defaults to 45." }),
	),
});

function motorTool(
	name: "move_forward" | "turn_left",
	command: "forward" | "turn_left",
	description: string,
	robot: RobotClient,
): AgentTool<typeof motorParameters, MotorToolDetails> {
	return {
		name,
		label: name,
		description,
		executionMode: "sequential",
		parameters: motorParameters,
		execute: async (_id, params) => {
			const durationMs = Math.max(0, params.durationMs);
			try {
				const result = await robot.execute({
					type: "motor_request",
					payload: { command, durationMs },
					timeoutMs: durationMs + 6000,
				});
				if (!result.ok) throw new Error(result.error);
				return {
					content: [{ type: "text", text: `Executed ${name} for ${durationMs}ms.` }],
					details: { command: name, durationMs },
				};
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				return {
					content: [{ type: "text", text: `Motor ${name} failed: ${message}` }],
					details: { command: name, durationMs, error: message },
				};
			}
		},
	};
}

function turnLeftDegreesTool(robot: RobotClient): AgentTool<typeof turnDegreesParameters, TurnDegreesDetails> {
	return {
		name: "turn_left_degrees",
		label: "Turn Left Degrees",
		description:
			"Rotate counter-clockwise by an approximate number of degrees using the phone orientation sensor. Use this when the user asks for a specific angle.",
		executionMode: "sequential",
		parameters: turnDegreesParameters,
		execute: async (_id, params) => {
			const degrees = Math.max(1, Math.min(359, params.degrees ?? 45));
			const durationMs = Math.max(1200, Math.min(18000, Math.round(degrees * 65)));
			try {
				const result = await robot.execute({
					type: "motor_request",
					payload: { command: "turn_left_degrees", durationMs, degrees },
					timeoutMs: durationMs + 6000,
				});
				if (!result.ok) throw new Error(result.error);
				return {
					content: [{ type: "text", text: `Executed approximate left turn by ${degrees} degrees.` }],
					details: { command: "turn_left_degrees", degrees, durationMs },
				};
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error);
				return {
					content: [{ type: "text", text: `Motor turn_left_degrees failed: ${message}` }],
					details: { command: "turn_left_degrees", degrees, durationMs, error: message },
				};
			}
		},
	};
}

export function createMotorTools(robot: RobotClient): AgentTool[] {
	return [
		motorTool(
			"move_forward",
			"forward",
			"Drive forward for the requested duration in milliseconds. Hardware supports forward motion only.",
			robot,
		),
		motorTool(
			"turn_left",
			"turn_left",
			"Rotate counter-clockwise (left) in place for the requested duration in milliseconds. Hardware supports rotation in this direction only.",
			robot,
		),
		turnLeftDegreesTool(robot),
	];
}

export function stopMotorFireAndForget(robot: RobotClient): void {
	void robot
		.execute({
			type: "motor_request",
			payload: { command: "stop" satisfies MotorCommand, durationMs: 0 },
			timeoutMs: 1000,
		})
		.catch(() => undefined);
}
