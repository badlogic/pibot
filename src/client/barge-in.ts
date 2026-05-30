import type { RobotState } from "../types.js";

export interface BargeInResult {
	triggered: boolean;
	preroll?: Int16Array;
}

export class BargeInDetector {
	private streaming = false;
	private consecutiveFrames = 0;
	private playbackReferenceSampleRate = 0;
	private playbackReferenceWrite = 0;
	private playbackReferenceSamples = 0;
	private playbackReferenceRing = new Float32Array(48000 * 8);
	private micBufferWrite = 0;
	private micBufferSamples = 0;
	private readonly micBufferRing: Int16Array;

	constructor(
		targetSampleRate: number,
		private readonly micThreshold = 0.012,
		private readonly residualThreshold = 0.45,
		private readonly triggerFrames = 3,
		private readonly residualRmsThreshold = 0.008,
	) {
		this.micBufferRing = new Int16Array(targetSampleRate);
	}

	resetStreaming(): void {
		this.streaming = false;
		this.consecutiveFrames = 0;
	}

	isStreaming(): boolean {
		return this.streaming;
	}

	handlePlaybackAudio(samples: Float32Array, sampleRate: number): void {
		if (this.playbackReferenceSampleRate !== sampleRate) {
			this.playbackReferenceSampleRate = sampleRate;
			this.playbackReferenceRing = new Float32Array(sampleRate * 8);
			this.playbackReferenceWrite = 0;
			this.playbackReferenceSamples = 0;
		}
		for (const sample of samples) {
			this.playbackReferenceRing[this.playbackReferenceWrite] = sample;
			this.playbackReferenceWrite = (this.playbackReferenceWrite + 1) % this.playbackReferenceRing.length;
			this.playbackReferenceSamples += 1;
		}
	}

	observeMic(
		input: Float32Array,
		sampleRate: number,
		pcm: Int16Array,
		state: RobotState,
		ttsSpeaking: boolean,
	): BargeInResult {
		this.appendMicBuffer(pcm);
		if (!this.shouldBufferForBargeIn(state) || this.streaming) return { triggered: false };
		const rms = this.micRms(input);
		const ratio = state.phase === "tool" && !ttsSpeaking ? 1 : this.bargeResidualRatio(input, sampleRate);
		const residualRms = rms * Math.sqrt(ratio);
		const triggered =
			(rms >= this.micThreshold && ratio >= this.residualThreshold) || residualRms >= this.residualRmsThreshold;
		this.consecutiveFrames = triggered ? this.consecutiveFrames + 1 : Math.max(0, this.consecutiveFrames - 1);
		if (this.consecutiveFrames < this.triggerFrames) return { triggered: false };
		this.streaming = true;
		this.consecutiveFrames = 0;
		return { triggered: true, preroll: this.bufferedMicPcm() };
	}

	shouldStreamMicNormally(state: RobotState): boolean {
		return state.phase === "listening" || state.phase === "hearing";
	}

	private shouldBufferForBargeIn(state: RobotState): boolean {
		return state.phase === "thinking" || state.phase === "tool" || state.phase === "speaking";
	}

	private appendMicBuffer(pcm: Int16Array): void {
		for (const sample of pcm) {
			this.micBufferRing[this.micBufferWrite] = sample;
			this.micBufferWrite = (this.micBufferWrite + 1) % this.micBufferRing.length;
			this.micBufferSamples = Math.min(this.micBufferSamples + 1, this.micBufferRing.length);
		}
	}

	private bufferedMicPcm(): Int16Array {
		const output = new Int16Array(this.micBufferSamples);
		const start =
			(this.micBufferWrite - this.micBufferSamples + this.micBufferRing.length) % this.micBufferRing.length;
		for (let index = 0; index < this.micBufferSamples; index++) {
			output[index] = this.micBufferRing[(start + index) % this.micBufferRing.length] ?? 0;
		}
		return output;
	}

	private micRms(input: Float32Array): number {
		let energy = 0;
		for (const sample of input) energy += sample * sample;
		return Math.sqrt(energy / Math.max(1, input.length));
	}

	private bargeResidualRatio(input: Float32Array, sampleRate: number): number {
		if (this.playbackReferenceSampleRate !== sampleRate || this.playbackReferenceSamples < input.length) return 1;
		let micEnergy = 0;
		for (const sample of input) micEnergy += sample * sample;
		micEnergy /= Math.max(1, input.length);
		if (micEnergy < 1e-7) return 0;
		let bestCorrelation = 0;
		let bestRatio = 1;
		for (let delayMs = 20; delayMs <= 420; delayMs += 10) {
			const delaySamples = Math.round((delayMs / 1000) * sampleRate);
			const start = this.playbackReferenceSamples - delaySamples - input.length;
			let referenceEnergy = 0;
			let dot = 0;
			for (let index = 0; index < input.length; index++) {
				const reference = this.readPlaybackReference(start + index);
				referenceEnergy += reference * reference;
				dot += input[index] * reference;
			}
			referenceEnergy /= Math.max(1, input.length);
			dot /= Math.max(1, input.length);
			if (referenceEnergy < 1e-7) continue;
			const correlation = Math.abs(dot) / Math.sqrt(referenceEnergy * micEnergy);
			const explained = (dot * dot) / referenceEnergy;
			const ratio = Math.max(0, micEnergy - explained) / micEnergy;
			if (correlation > bestCorrelation) {
				bestCorrelation = correlation;
				bestRatio = ratio;
			}
		}
		return bestRatio;
	}

	private readPlaybackReference(totalIndex: number): number {
		if (totalIndex < 0 || totalIndex >= this.playbackReferenceSamples) return 0;
		return this.playbackReferenceRing[totalIndex % this.playbackReferenceRing.length] ?? 0;
	}
}
