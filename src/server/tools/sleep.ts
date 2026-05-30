import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "@earendil-works/pi-ai";

interface SleepDetails {
	sleepDuration: number;
}

const sleepParameters = Type.Object({
	sleepDuration: Type.Number({ description: "Sleep duration in milliseconds." }),
});

export const sleepTool: AgentTool<typeof sleepParameters, SleepDetails> = {
	name: "sleep",
	label: "Sleep",
	description: "Wait for the requested duration in milliseconds. Use for testing cancellation or intentional pauses.",
	executionMode: "sequential",
	parameters: sleepParameters,
	execute: async (_id, params, signal) => {
		const sleepDuration = Math.max(0, Math.min(120000, Math.round(params.sleepDuration)));
		await new Promise<void>((resolve, reject) => {
			const timeout = setTimeout(resolve, sleepDuration);
			const onAbort = () => {
				clearTimeout(timeout);
				reject(new Error(`sleep aborted after ${sleepDuration}ms request`));
			};
			signal?.addEventListener("abort", onAbort, { once: true });
			if (signal?.aborted) onAbort();
		});
		return {
			content: [{ type: "text", text: `Slept for ${sleepDuration}ms.` }],
			details: { sleepDuration },
		};
	},
};
