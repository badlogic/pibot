import type { RobotRpcMap } from "../../types.js";
import type { ClientLogger } from "../logger.js";
import { throwIfAborted } from "./common.js";

type VideoFrameCallbackVideo = HTMLVideoElement & {
	requestVideoFrameCallback?: (callback: () => void) => number;
};

export interface PhotoTool {
	handle: (
		payload: RobotRpcMap["take_photo"]["request"],
		signal: AbortSignal,
	) => Promise<RobotRpcMap["take_photo"]["response"]>;
}

export function createPhotoTool(deps: { logger: ClientLogger }): PhotoTool {
	const logger = deps.logger.tag("camera");
	async function openCamera(): Promise<{ stream: MediaStream; video: HTMLVideoElement }> {
		if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera API unavailable");
		const stream = await navigator.mediaDevices.getUserMedia({
			video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
			audio: false,
		});
		const video = document.createElement("video");
		video.muted = true;
		video.playsInline = true;
		video.autoplay = true;
		video.style.position = "fixed";
		video.style.left = "-10000px";
		video.style.top = "0";
		video.style.width = "320px";
		video.style.height = "240px";
		video.style.pointerEvents = "none";
		document.body.append(video);
		video.srcObject = stream;
		await video.play().catch(() => undefined);
		return { stream, video };
	}

	function closeCamera(stream: MediaStream, video: HTMLVideoElement): void {
		for (const track of stream.getTracks()) track.stop();
		video.pause();
		video.srcObject = null;
		video.remove();
	}

	async function delay(ms: number): Promise<void> {
		await new Promise<void>((resolve) => setTimeout(resolve, ms));
	}

	async function waitForVideoReady(video: HTMLVideoElement): Promise<void> {
		if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) return;
		await new Promise<void>((resolve) => {
			const cleanup = () => {
				video.removeEventListener("loadedmetadata", cleanup);
				video.removeEventListener("loadeddata", cleanup);
				video.removeEventListener("canplay", cleanup);
				resolve();
			};
			video.addEventListener("loadedmetadata", cleanup, { once: true });
			video.addEventListener("loadeddata", cleanup, { once: true });
			video.addEventListener("canplay", cleanup, { once: true });
		});
	}

	async function waitForVideoFrames(video: HTMLVideoElement, frameCount: number): Promise<void> {
		const videoWithFrames = video as VideoFrameCallbackVideo;
		for (let index = 0; index < frameCount; index++) {
			await new Promise<void>((resolve) => {
				if (videoWithFrames.requestVideoFrameCallback) {
					videoWithFrames.requestVideoFrameCallback(() => resolve());
					return;
				}
				requestAnimationFrame(() => resolve());
			});
		}
	}

	async function capturePhotoDataUrl(signal: AbortSignal): Promise<string> {
		const { stream, video } = await openCamera();
		try {
			throwIfAborted(signal);
			await waitForVideoReady(video);
			await waitForVideoFrames(video, 3);
			await delay(350);
			throwIfAborted(signal);
			const width = video.videoWidth || 640;
			const height = video.videoHeight || 480;
			const canvas = document.createElement("canvas");
			canvas.width = width;
			canvas.height = height;
			const ctx = canvas.getContext("2d");
			if (!ctx) throw new Error("Canvas 2d context unavailable");
			ctx.drawImage(video, 0, 0, width, height);
			return canvas.toDataURL("image/jpeg", 0.82);
		} finally {
			closeCamera(stream, video);
		}
	}

	async function handle(
		_payload: RobotRpcMap["take_photo"]["request"],
		signal: AbortSignal,
	): Promise<RobotRpcMap["take_photo"]["response"]> {
		throwIfAborted(signal);
		const dataUrl = await capturePhotoDataUrl(signal);
		throwIfAborted(signal);
		logger.log(`Captured photo (${dataUrl.length} chars)`);
		return { dataUrl };
	}

	return { handle };
}
