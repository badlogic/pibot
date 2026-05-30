import type { RobotRpcMap } from "../../types.js";
import type { ClientLogger } from "../logger.js";

interface ActiveSpeech {
	generation: number;
	resolve: (response: RobotRpcMap["speak"]["response"]) => void;
	signal: AbortSignal;
	onAbort: () => void;
	finished: boolean;
}

interface ActivePcmStream {
	generation: number;
	sampleRate: number;
	nextPlayTime: number;
	pendingSources: number;
	doneRequested: boolean;
	finishResolve: (() => void) | undefined;
	nodes: AudioNode[];
	cleanup: () => void;
}

export interface SpeechTool {
	enableTts: () => void;
	cancelSpeech: (reason: string) => void;
	cancelSpeechForBargeIn: () => void;
	handleSpeak: (
		payload: RobotRpcMap["speak"]["request"],
		signal: AbortSignal,
	) => Promise<RobotRpcMap["speak"]["response"]>;
	handleCancelSpeech: (
		payload: RobotRpcMap["cancel_speech"]["request"],
		signal: AbortSignal,
	) => RobotRpcMap["cancel_speech"]["response"];
	startPcmStream: (sampleRate: number) => void;
	pushPcmAudio: (pcm: Uint8Array) => void;
	finishPcmStream: () => Promise<void>;
	failPcmStream: (message: string) => void;
}

export function createSpeechTool(deps: {
	logger: ClientLogger;
	face: HTMLElement;
	setMicInputBlockedUntil: (time: number) => void;
	onSpeakingChange: (speaking: boolean) => void;
	onPlaybackAudio: (samples: Float32Array, sampleRate: number) => void;
}): SpeechTool {
	const logger = deps.logger.tag("stt");
	const ttsEnabledKey = "robot-tts-enabled";
	let ttsEnabled = localStorage.getItem(ttsEnabledKey) === "true";
	let currentTtsAudio: HTMLAudioElement | undefined;
	let robotVoiceEffectCleanup: (() => void) | undefined;
	let audioContext: AudioContext | undefined;
	let ttsGeneration = 0;
	let activeSpeech: ActiveSpeech | undefined;
	let activePcmStream: ActivePcmStream | undefined;
	function startFaceAmpLoop(analyser: AnalyserNode): () => void {
		const data = new Uint8Array(analyser.fftSize);
		let smoothed = 0;
		let frameHandle = 0;
		let stopped = false;
		const tick = () => {
			if (stopped) return;
			analyser.getByteTimeDomainData(data);
			let sum = 0;
			for (const sample of data) {
				const centered = (sample - 128) / 128;
				sum += centered * centered;
			}
			const rms = Math.sqrt(sum / data.length);
			const amp = Math.min(1, rms * 3.4);
			smoothed = smoothed * 0.55 + amp * 0.45;
			deps.face.style.setProperty("--amp", smoothed.toFixed(3));
			frameHandle = requestAnimationFrame(tick);
		};
		frameHandle = requestAnimationFrame(tick);
		return () => {
			stopped = true;
			cancelAnimationFrame(frameHandle);
			deps.face.style.setProperty("--amp", "0");
		};
	}

	function clearCurrentTtsAudio(): void {
		robotVoiceEffectCleanup?.();
		robotVoiceEffectCleanup = undefined;
		if (!currentTtsAudio) return;
		currentTtsAudio.onplay = null;
		currentTtsAudio.onended = null;
		currentTtsAudio.onerror = null;
		currentTtsAudio.pause();
		currentTtsAudio.removeAttribute("src");
		currentTtsAudio.load();
		currentTtsAudio = undefined;
	}

	function createRobotVoiceEffect(audio: HTMLAudioElement): void {
		try {
			audioContext ??= new AudioContext();
			void audioContext.resume();
			const source = audioContext.createMediaElementSource(audio);
			const highpass = audioContext.createBiquadFilter();
			highpass.type = "highpass";
			highpass.frequency.value = 150;
			const lowpass = audioContext.createBiquadFilter();
			lowpass.type = "lowpass";
			lowpass.frequency.value = 7200;
			const presence = audioContext.createBiquadFilter();
			presence.type = "peaking";
			presence.frequency.value = 2600;
			presence.Q.value = 0.9;
			presence.gain.value = 3.5;
			const compressor = audioContext.createDynamicsCompressor();
			compressor.threshold.value = -24;
			compressor.knee.value = 18;
			compressor.ratio.value = 3;
			compressor.attack.value = 0.006;
			compressor.release.value = 0.12;
			const dry = audioContext.createGain();
			dry.gain.value = 0.9;
			const ringModulator = audioContext.createGain();
			ringModulator.gain.value = 0;
			const ringWet = audioContext.createGain();
			ringWet.gain.value = 0.09;
			const ringOsc = audioContext.createOscillator();
			ringOsc.type = "sine";
			ringOsc.frequency.value = 42;
			ringOsc.connect(ringModulator.gain);
			ringOsc.start();
			const slap = audioContext.createDelay(0.25);
			slap.delayTime.value = 0.075;
			const slapWet = audioContext.createGain();
			slapWet.gain.value = 0.045;
			const output = audioContext.createGain();
			output.gain.value = 0.98;
			const analyser = audioContext.createAnalyser();
			analyser.fftSize = 512;
			analyser.smoothingTimeConstant = 0.55;

			source.connect(highpass);
			highpass.connect(lowpass);
			lowpass.connect(presence);
			presence.connect(compressor);
			compressor.connect(dry);
			compressor.connect(ringModulator);
			ringModulator.connect(ringWet);
			dry.connect(output);
			ringWet.connect(output);
			dry.connect(slap);
			slap.connect(slapWet);
			slapWet.connect(output);
			output.connect(audioContext.destination);
			output.connect(analyser);

			const stopAmpLoop = startFaceAmpLoop(analyser);

			robotVoiceEffectCleanup = () => {
				stopAmpLoop();
				try {
					ringOsc.stop();
				} catch {
					// already stopped
				}
				for (const node of [
					source,
					highpass,
					lowpass,
					presence,
					compressor,
					dry,
					ringModulator,
					ringWet,
					ringOsc,
					slap,
					slapWet,
					output,
					analyser,
				]) {
					node.disconnect();
				}
			};
			logger.log("Robot voice effect enabled");
		} catch (error) {
			robotVoiceEffectCleanup = undefined;
			logger.log(`Robot voice effect unavailable: ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	function clearActivePcmStream(resolveFinished: boolean): void {
		const stream = activePcmStream;
		activePcmStream = undefined;
		if (!stream) return;
		stream.cleanup();
		for (const node of stream.nodes) node.disconnect();
		if (resolveFinished) stream.finishResolve?.();
	}

	function maybeFinishPcmStream(stream: ActivePcmStream): void {
		if (activePcmStream !== stream || !stream.doneRequested || stream.pendingSources > 0) return;
		clearActivePcmStream(true);
		deps.onSpeakingChange(false);
		deps.setMicInputBlockedUntil(Date.now() + 500);
		logger.log("Qwen3 PCM stream finished");
	}

	function interruptTtsOnly(resolvePcmFinished = true): void {
		ttsGeneration++;
		clearCurrentTtsAudio();
		clearActivePcmStream(resolvePcmFinished);
		if ("speechSynthesis" in window) window.speechSynthesis.cancel();
	}

	function completeActiveSpeech(response: RobotRpcMap["speak"]["response"]): void {
		const active = activeSpeech;
		if (!active || active.finished) return;
		active.finished = true;
		active.signal.removeEventListener("abort", active.onAbort);
		activeSpeech = undefined;
		active.resolve(response);
	}

	function cancelSpeech(reason: string): void {
		interruptTtsOnly();
		deps.onSpeakingChange(false);
		completeActiveSpeech({ ok: true });
		deps.setMicInputBlockedUntil(Date.now() + 500);
		logger.log(`TTS cancelled: ${reason}`);
	}

	function cancelSpeechForBargeIn(): void {
		interruptTtsOnly(false);
		deps.onSpeakingChange(false);
		completeActiveSpeech({ ok: true });
		deps.setMicInputBlockedUntil(0);
		logger.log("TTS cancelled: barge-in");
	}

	function enableTts(): void {
		ttsEnabled = true;
		localStorage.setItem(ttsEnabledKey, "true");
	}

	function finishTts(message: string): void {
		clearCurrentTtsAudio();
		deps.onSpeakingChange(false);
		completeActiveSpeech({ ok: true });
		deps.setMicInputBlockedUntil(Date.now() + 500);
		logger.log(message);
	}

	function startSpeech(url: string, text: string, generation: number): void {
		const trimmed = text.trim();
		if (!trimmed) {
			finishTts("TTS skipped: empty text");
			return;
		}

		const providerLabel = "Qwen3 local clone";
		clearCurrentTtsAudio();
		deps.onSpeakingChange(true);
		deps.setMicInputBlockedUntil(0);

		const audio = new Audio(url);
		currentTtsAudio = audio;
		createRobotVoiceEffect(audio);
		audio.onplay = () => logger.log(`${providerLabel} playing streamed response ${trimmed.length} chars`);
		audio.onended = () => {
			if (generation !== ttsGeneration) return;
			finishTts(`${providerLabel} finished, resetting STT`);
		};
		audio.onerror = () => {
			if (generation !== ttsGeneration) return;
			finishTts(`${providerLabel} failed, resetting STT`);
		};
		audio.play().catch((error: unknown) => {
			if (generation !== ttsGeneration) return;
			finishTts(
				`${providerLabel} play failed, resetting STT: ${error instanceof Error ? error.message : String(error)}`,
			);
		});
	}

	async function handleSpeak(
		payload: RobotRpcMap["speak"]["request"],
		signal: AbortSignal,
	): Promise<RobotRpcMap["speak"]["response"]> {
		if (!ttsEnabled) return { ok: true };
		if (activeSpeech) cancelSpeech("new speech request");
		return await new Promise<RobotRpcMap["speak"]["response"]>((resolve) => {
			const generation = ++ttsGeneration;
			const onAbort = () => cancelSpeech(String(signal.reason ?? "aborted"));
			activeSpeech = { generation, resolve, signal, onAbort, finished: false };
			signal.addEventListener("abort", onAbort, { once: true });
			if (signal.aborted) {
				onAbort();
				return;
			}
			startSpeech(payload.url, payload.text, generation);
		});
	}

	function handleCancelSpeech(
		payload: RobotRpcMap["cancel_speech"]["request"],
		_signal: AbortSignal,
	): RobotRpcMap["cancel_speech"]["response"] {
		cancelSpeech(payload.reason);
		return { ok: true };
	}

	function startPcmStream(sampleRate: number): void {
		interruptTtsOnly();
		if (!audioContext) audioContext = new AudioContext();
		const context = audioContext;
		void context.resume();
		const highpass = context.createBiquadFilter();
		highpass.type = "highpass";
		highpass.frequency.value = 150;
		const lowpass = context.createBiquadFilter();
		lowpass.type = "lowpass";
		lowpass.frequency.value = 7200;
		const presence = context.createBiquadFilter();
		presence.type = "peaking";
		presence.frequency.value = 2600;
		presence.Q.value = 0.9;
		presence.gain.value = 3.5;
		const compressor = context.createDynamicsCompressor();
		compressor.threshold.value = -24;
		compressor.knee.value = 18;
		compressor.ratio.value = 3;
		compressor.attack.value = 0.006;
		compressor.release.value = 0.12;
		const output = context.createGain();
		output.gain.value = 0.98;
		const analyser = context.createAnalyser();
		analyser.fftSize = 512;
		analyser.smoothingTimeConstant = 0.55;
		const playbackTap = context.createScriptProcessor(1024, 1, 1);
		const silentTapOutput = context.createGain();
		silentTapOutput.gain.value = 0;
		playbackTap.onaudioprocess = (event) => {
			const input = event.inputBuffer.getChannelData(0);
			deps.onPlaybackAudio(input, event.inputBuffer.sampleRate);
		};
		highpass.connect(lowpass);
		lowpass.connect(presence);
		presence.connect(compressor);
		compressor.connect(output);
		output.connect(context.destination);
		output.connect(playbackTap);
		playbackTap.connect(silentTapOutput);
		silentTapOutput.connect(context.destination);
		output.connect(analyser);
		activePcmStream = {
			generation: ++ttsGeneration,
			sampleRate,
			nextPlayTime: context.currentTime + 0.05,
			pendingSources: 0,
			doneRequested: false,
			finishResolve: undefined,
			nodes: [highpass, lowpass, presence, compressor, output, playbackTap, silentTapOutput, analyser],
			cleanup: startFaceAmpLoop(analyser),
		};
		deps.onSpeakingChange(true);
		deps.setMicInputBlockedUntil(0);
		logger.log(`Qwen3 PCM stream started sampleRate=${sampleRate}`);
	}

	function pushPcmAudio(pcm: Uint8Array): void {
		const stream = activePcmStream;
		if (!stream || !audioContext || pcm.byteLength < 2) return;
		const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
		const sampleCount = Math.floor(pcm.byteLength / 2);
		const audioBuffer = audioContext.createBuffer(1, sampleCount, stream.sampleRate);
		const channel = audioBuffer.getChannelData(0);
		for (let index = 0; index < sampleCount; index++) channel[index] = view.getInt16(index * 2, true) / 32768;
		const source = audioContext.createBufferSource();
		source.buffer = audioBuffer;
		source.connect(stream.nodes[0] ?? audioContext.destination);
		stream.pendingSources++;
		source.onended = () => {
			source.disconnect();
			stream.pendingSources--;
			maybeFinishPcmStream(stream);
		};
		const startAt = Math.max(stream.nextPlayTime, audioContext.currentTime + 0.01);
		source.start(startAt);
		stream.nextPlayTime = startAt + audioBuffer.duration;
	}

	async function finishPcmStream(): Promise<void> {
		const stream = activePcmStream;
		if (!stream) return;
		stream.doneRequested = true;
		if (stream.pendingSources === 0) {
			maybeFinishPcmStream(stream);
			return;
		}
		await new Promise<void>((resolve) => {
			stream.finishResolve = resolve;
		});
	}

	function failPcmStream(message: string): void {
		clearActivePcmStream(true);
		deps.onSpeakingChange(false);
		deps.setMicInputBlockedUntil(Date.now() + 500);
		logger.log(`Qwen3 PCM stream failed: ${message}`);
	}

	return {
		enableTts,
		cancelSpeech,
		cancelSpeechForBargeIn,
		handleSpeak,
		handleCancelSpeech,
		startPcmStream,
		pushPcmAudio,
		finishPcmStream,
		failPcmStream,
	};
}
