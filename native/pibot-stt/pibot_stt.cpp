#include "whisper.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t MSG_START = 1;
constexpr uint32_t MSG_AUDIO = 2;
constexpr uint32_t MSG_FLUSH = 3;
constexpr uint32_t MSG_STOP = 4;

constexpr uint32_t MSG_READY = 101;
constexpr uint32_t MSG_PARTIAL = 102;
constexpr uint32_t MSG_FINAL = 103;
constexpr uint32_t MSG_SPEECH_START = 104;
constexpr uint32_t MSG_SPEECH_END = 105;
constexpr uint32_t MSG_ERROR = 199;

struct Header {
	uint32_t type;
	uint32_t flags;
	uint32_t length;
};

struct Options {
	std::string model = "models/ggml-model-q5_0.bin";
	std::string language = "de";
	int threads = std::max(1u, std::thread::hardware_concurrency());
	float vad_threshold = 0.012f;
	int silence_ms = 900;
	int min_speech_ms = 350;
	int partial_ms = 1600;
	bool partials = true;
	bool use_gpu = true;
};

bool read_exact(void * ptr, size_t size) {
	auto * p = static_cast<uint8_t *>(ptr);
	while (size > 0) {
		std::cin.read(reinterpret_cast<char *>(p), static_cast<std::streamsize>(size));
		const auto got = static_cast<size_t>(std::cin.gcount());
		if (got == 0) return false;
		p += got;
		size -= got;
	}
	return true;
}

void write_frame(uint32_t type, const std::string & payload = "") {
	Header header{type, 0, static_cast<uint32_t>(payload.size())};
	std::cout.write(reinterpret_cast<const char *>(&header), sizeof(header));
	if (!payload.empty()) std::cout.write(payload.data(), static_cast<std::streamsize>(payload.size()));
	std::cout.flush();
}

void write_error(const std::string & message) {
	write_frame(MSG_ERROR, message);
}

std::string trim(std::string text) {
	const auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
	text.erase(text.begin(), std::find_if(text.begin(), text.end(), [&](char c) { return !is_space(c); }));
	text.erase(std::find_if(text.rbegin(), text.rend(), [&](char c) { return !is_space(c); }).base(), text.end());
	return text;
}

float rms_i16le(const std::vector<uint8_t> & bytes) {
	const size_t n = bytes.size() / 2;
	if (n == 0) return 0.0f;
	double sum = 0.0;
	for (size_t i = 0; i < n; i++) {
		const int16_t sample = static_cast<int16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8));
		const double v = static_cast<double>(sample) / 32768.0;
		sum += v * v;
	}
	return static_cast<float>(std::sqrt(sum / static_cast<double>(n)));
}

void append_i16le_as_f32(const std::vector<uint8_t> & bytes, std::vector<float> & pcm) {
	const size_t n = bytes.size() / 2;
	pcm.reserve(pcm.size() + n);
	for (size_t i = 0; i < n; i++) {
		const int16_t sample = static_cast<int16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8));
		pcm.push_back(static_cast<float>(sample) / 32768.0f);
	}
}

std::string transcribe(whisper_context * ctx, const Options & options, const std::vector<float> & pcm, const std::string & prompt) {
	if (pcm.empty()) return "";
	auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
	params.n_threads = options.threads;
	params.language = options.language.c_str();
	params.translate = false;
	params.no_context = prompt.empty();
	params.no_timestamps = true;
	params.single_segment = true;
	params.print_special = false;
	params.print_progress = false;
	params.print_realtime = false;
	params.print_timestamps = false;
	params.suppress_blank = true;
	params.suppress_nst = true;
	params.temperature = 0.0f;
	params.max_tokens = 96;
	params.initial_prompt = prompt.empty() ? nullptr : prompt.c_str();

	if (whisper_full(ctx, params, pcm.data(), static_cast<int>(pcm.size())) != 0) {
		return "";
	}

	std::string text;
	const int n_segments = whisper_full_n_segments(ctx);
	for (int i = 0; i < n_segments; i++) {
		const char * segment = whisper_full_get_segment_text(ctx, i);
		if (segment) text += segment;
	}
	return trim(text);
}

Options parse_options(int argc, char ** argv) {
	Options options;
	for (int i = 1; i < argc; i++) {
		const std::string arg = argv[i];
		if ((arg == "-m" || arg == "--model") && i + 1 < argc) options.model = argv[++i];
		else if ((arg == "-l" || arg == "--language") && i + 1 < argc) options.language = argv[++i];
		else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) options.threads = std::stoi(argv[++i]);
		else if (arg == "--vad" && i + 1 < argc) options.vad_threshold = std::stof(argv[++i]);
		else if (arg == "--silence-ms" && i + 1 < argc) options.silence_ms = std::stoi(argv[++i]);
		else if (arg == "--min-speech-ms" && i + 1 < argc) options.min_speech_ms = std::stoi(argv[++i]);
		else if (arg == "--partial-ms" && i + 1 < argc) options.partial_ms = std::stoi(argv[++i]);
		else if (arg == "--no-partials") options.partials = false;
		else if (arg == "--no-gpu") options.use_gpu = false;
		else if (arg == "-h" || arg == "--help") {
			std::cerr << "usage: pibot-stt --model ggml-model.bin [--language de] [--vad 0.012]\n";
			std::exit(0);
		}
	}
	return options;
}

} // namespace

int main(int argc, char ** argv) {
	const Options options = parse_options(argc, argv);

	ggml_backend_load_all();

	whisper_context_params cparams = whisper_context_default_params();
	cparams.use_gpu = options.use_gpu;

	whisper_context * ctx = whisper_init_from_file_with_params(options.model.c_str(), cparams);
	if (!ctx) {
		write_error("failed to initialize whisper model: " + options.model);
		return 2;
	}
	if (options.language != "auto" && whisper_lang_id(options.language.c_str()) < 0) {
		write_error("unknown language: " + options.language);
		whisper_free(ctx);
		return 2;
	}

	write_frame(MSG_READY, "pibot-stt ready");

	std::vector<float> utterance;
	std::string context_prompt;
	bool in_speech = false;
	int64_t utterance_samples = 0;
	int64_t silence_samples = 0;
	int64_t samples_since_partial = 0;
	std::string last_partial;

	const int sample_rate = 16000;
	const int silence_samples_limit = options.silence_ms * sample_rate / 1000;
	const int min_speech_samples = options.min_speech_ms * sample_rate / 1000;
	const int partial_samples = options.partial_ms * sample_rate / 1000;

	while (true) {
		Header header{};
		if (!read_exact(&header, sizeof(header))) break;
		if (header.length > 16 * 1024 * 1024) {
			write_error("frame too large");
			break;
		}
		std::vector<uint8_t> payload(header.length);
		if (header.length > 0 && !read_exact(payload.data(), payload.size())) break;

		if (header.type == MSG_STOP) break;
		if (header.type == MSG_START) continue;
		if (header.type == MSG_AUDIO) {
			const float energy = rms_i16le(payload);
			const size_t n_samples = payload.size() / 2;
			const bool voice = energy >= options.vad_threshold;

			if (voice) {
				if (!in_speech) {
					in_speech = true;
					utterance.clear();
					utterance_samples = 0;
					silence_samples = 0;
					samples_since_partial = 0;
					last_partial.clear();
					write_frame(MSG_SPEECH_START);
				}
				silence_samples = 0;
			} else if (in_speech) {
				silence_samples += static_cast<int64_t>(n_samples);
			}

			if (in_speech) {
				append_i16le_as_f32(payload, utterance);
				utterance_samples += static_cast<int64_t>(n_samples);
				samples_since_partial += static_cast<int64_t>(n_samples);
				const bool enough_for_partial = options.partials && samples_since_partial >= partial_samples && utterance_samples >= min_speech_samples;
				if (enough_for_partial) {
					samples_since_partial = 0;
					std::string partial = transcribe(ctx, options, utterance, context_prompt);
					if (!partial.empty() && partial != last_partial) {
						last_partial = partial;
						write_frame(MSG_PARTIAL, partial);
					}
				}
				if (silence_samples >= silence_samples_limit) {
					write_frame(MSG_SPEECH_END);
					if (utterance_samples >= min_speech_samples) {
						std::string final_text = transcribe(ctx, options, utterance, context_prompt);
						if (!final_text.empty()) {
							write_frame(MSG_FINAL, final_text);
							context_prompt = final_text;
							if (context_prompt.size() > 512) context_prompt.erase(0, context_prompt.size() - 512);
						}
					}
					in_speech = false;
					utterance.clear();
					utterance_samples = 0;
					silence_samples = 0;
				}
			}
		} else if (header.type == MSG_FLUSH) {
			if (!utterance.empty()) {
				std::string final_text = transcribe(ctx, options, utterance, context_prompt);
				if (!final_text.empty()) write_frame(MSG_FINAL, final_text);
			}
			utterance.clear();
			in_speech = false;
		} else {
			write_error("unknown frame type: " + std::to_string(header.type));
		}
	}

	whisper_free(ctx);
	return 0;
}
