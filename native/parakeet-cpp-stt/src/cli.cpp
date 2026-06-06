#include "ggml_graph.hpp"
#include "model.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int TARGET_SAMPLE_RATE = 16000;
static constexpr size_t DEFAULT_CHUNK_SECONDS = 30;
static constexpr size_t TEXT_CHUNK_SECONDS = 15;
static constexpr const char* DEFAULT_MODEL_FILE = "tdt-0.6b-v3-q8_0.gguf";
static constexpr const char* PARAKEET_CPP_REPO = "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main";

struct WavAudio {
    int sample_rate = 0;
    std::vector<float> mono;
};

static uint16_t read_u16_le(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<uint16_t>(data[offset]) | (static_cast<uint16_t>(data[offset + 1]) << 8);
}

static uint32_t read_u32_le(const std::vector<uint8_t>& data, size_t offset) {
    return static_cast<uint32_t>(data[offset]) |
           (static_cast<uint32_t>(data[offset + 1]) << 8) |
           (static_cast<uint32_t>(data[offset + 2]) << 16) |
           (static_cast<uint32_t>(data[offset + 3]) << 24);
}

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("failed to open WAV file: " + path);
    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();
    if (size < 0) throw std::runtime_error("failed to stat WAV file: " + path);
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!data.empty() && !file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("failed to read WAV file: " + path);
    }
    return data;
}

static WavAudio load_wav(const std::string& path) {
    const std::vector<uint8_t> file = read_file(path);
    if (file.size() < 44 || std::string(reinterpret_cast<const char*>(file.data()), 4) != "RIFF" ||
        std::string(reinterpret_cast<const char*>(file.data() + 8), 4) != "WAVE") {
        throw std::runtime_error("not a RIFF/WAVE file: " + path);
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    size_t data_offset = 0;
    size_t data_size = 0;

    size_t offset = 12;
    while (offset + 8 <= file.size()) {
        const std::string id(reinterpret_cast<const char*>(file.data() + offset), 4);
        const uint32_t chunk_size = read_u32_le(file, offset + 4);
        const size_t chunk_data = offset + 8;
        if (chunk_data + chunk_size > file.size()) throw std::runtime_error("truncated WAV chunk: " + id);
        if (id == "fmt ") {
            if (chunk_size < 16) throw std::runtime_error("invalid WAV fmt chunk");
            format = read_u16_le(file, chunk_data);
            channels = read_u16_le(file, chunk_data + 2);
            sample_rate = read_u32_le(file, chunk_data + 4);
            bits_per_sample = read_u16_le(file, chunk_data + 14);
        } else if (id == "data") {
            data_offset = chunk_data;
            data_size = chunk_size;
        }
        offset = chunk_data + chunk_size + (chunk_size % 2);
    }

    if (channels == 0 || sample_rate == 0 || data_size == 0) throw std::runtime_error("missing WAV fmt or data chunk");
    if (!((format == 1 && (bits_per_sample == 16 || bits_per_sample == 24 || bits_per_sample == 32)) || (format == 3 && bits_per_sample == 32))) {
        throw std::runtime_error("unsupported WAV format; expected PCM 16/24/32-bit or float32");
    }

    const size_t bytes_per_sample = bits_per_sample / 8;
    const size_t frame_bytes = bytes_per_sample * channels;
    const size_t frames = data_size / frame_bytes;
    std::vector<float> mono;
    mono.reserve(frames);
    for (size_t frame = 0; frame < frames; ++frame) {
        float sum = 0.0f;
        for (uint16_t channel = 0; channel < channels; ++channel) {
            const size_t pos = data_offset + frame * frame_bytes + channel * bytes_per_sample;
            if (format == 3) {
                float sample;
                std::memcpy(&sample, file.data() + pos, sizeof(sample));
                sum += sample;
            } else if (bits_per_sample == 16) {
                const int16_t sample = static_cast<int16_t>(read_u16_le(file, pos));
                sum += static_cast<float>(sample) / 32768.0f;
            } else if (bits_per_sample == 24) {
                int32_t sample = static_cast<int32_t>(file[pos]) | (static_cast<int32_t>(file[pos + 1]) << 8) | (static_cast<int32_t>(file[pos + 2]) << 16);
                if (sample & 0x800000) sample |= ~0xffffff;
                sum += static_cast<float>(sample) / 8388608.0f;
            } else {
                const int32_t sample = static_cast<int32_t>(read_u32_le(file, pos));
                sum += static_cast<float>(sample) / 2147483648.0f;
            }
        }
        mono.push_back(sum / static_cast<float>(channels));
    }
    return WavAudio{static_cast<int>(sample_rate), mono};
}

static std::vector<float> resample_linear(const std::vector<float>& input, int source_rate) {
    if (source_rate == TARGET_SAMPLE_RATE) return input;
    if (input.empty()) return {};
    const size_t output_size = static_cast<size_t>((static_cast<uint64_t>(input.size()) * TARGET_SAMPLE_RATE) / source_rate);
    std::vector<float> output(output_size);
    const double step = static_cast<double>(source_rate) / TARGET_SAMPLE_RATE;
    for (size_t i = 0; i < output.size(); ++i) {
        const double source = static_cast<double>(i) * step;
        const size_t left = std::min(static_cast<size_t>(source), input.size() - 1);
        const size_t right = std::min(left + 1, input.size() - 1);
        const float t = static_cast<float>(source - static_cast<double>(left));
        output[i] = input[left] * (1.0f - t) + input[right] * t;
    }
    return output;
}

static size_t env_size(const char* name, size_t fallback) {
    const char* value = std::getenv(name);
    return value ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : fallback;
}

static std::string json_escape(const std::string& value) {
    std::string out;
    for (char c : value) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

static bool has_usable_file(const std::string& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error) && std::filesystem::file_size(path, error) > 0;
}

static std::string default_model_path() {
    const char* home = std::getenv("HOME");
    if (!home || std::string(home).empty()) throw std::runtime_error("HOME is not set; pass a model path explicitly");
    const char* model_file = std::getenv("PARAKEET_CPP_MODEL_FILE");
    return (std::filesystem::path(home) / "models/parakeet-cpp-gguf" / (model_file ? model_file : DEFAULT_MODEL_FILE)).string();
}

static void ensure_model_file(const std::string& path) {
    if (has_usable_file(path)) return;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    const std::string file = std::filesystem::path(path).filename().string();
    const std::string url = std::string(PARAKEET_CPP_REPO) + "/" + file;
    const std::string tmp_path = path + ".tmp-" + std::to_string(getpid());
    std::filesystem::remove(tmp_path);
    std::cerr << "downloading parakeet.cpp model file " << file << "\n";
    const std::string command = "curl -fL --progress-bar -o \"" + tmp_path + "\" \"" + url + "\"";
    if (std::system(command.c_str()) != 0) {
        std::filesystem::remove(tmp_path);
        throw std::runtime_error("failed to download model file: " + url);
    }
    std::filesystem::rename(tmp_path, path);
}

static std::string timestamp(size_t seconds) {
    const size_t minutes = seconds / 60;
    const size_t remaining_seconds = seconds % 60;
    std::string out;
    if (minutes < 10) out += "0";
    out += std::to_string(minutes) + ":";
    if (remaining_seconds < 10) out += "0";
    out += std::to_string(remaining_seconds);
    return out;
}

static void print_help(const char* program) {
    std::cout << "usage: " << program << " [--text] <audio.wav> [model.gguf]\n\n"
              << "Transcribe a WAV file with parakeet.cpp. Defaults to JSON with word timestamps.\n\n"
              << "Options:\n"
              << "  --text, -t     Output plain text grouped in 15 second timestamped chunks.\n"
              << "  --help, -h     Show this help.\n\n"
              << "Arguments:\n"
              << "  audio.wav      PCM 16/24/32-bit or float32 WAV. Multi-channel audio is mixed to mono.\n"
              << "  model.gguf     Optional GGUF model path. Overrides PARAKEET_CPP_MODEL_PATH.\n\n"
              << "Model resolution:\n"
              << "  1. model.gguf argument\n"
              << "  2. PARAKEET_CPP_MODEL_PATH\n"
              << "  3. ~/models/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf\n\n"
              << "If the resolved model file is missing, it is downloaded from:\n"
              << "  " << PARAKEET_CPP_REPO << "\n\n"
              << "Environment:\n"
              << "  PARAKEET_CPP_MODEL_PATH       Full model path override.\n"
              << "  PARAKEET_CPP_MODEL_FILE       File name under ~/models/parakeet-cpp-gguf.\n"
              << "  PARAKEET_CLI_CHUNK_SECONDS    Chunk length for long WAVs. Default: 30.\n\n"
              << "Output JSON format:\n"
              << "  {\n"
              << "    \"text\": \"full transcript\",\n"
              << "    \"words\": [\n"
              << "      {\"w\": \"hello\", \"start\": 0.48, \"end\": 0.72, \"conf\": 0.91}\n"
              << "    ]\n"
              << "  }\n\n"
              << "Times are seconds. conf is word confidence aggregated from token confidences.\n\n"
              << "Text output format:\n"
              << "  [00:00-00:15] transcript text for that chunk\n";
}

int main(int argc, char** argv) {
    try {
        bool text_output = false;
        std::vector<std::string> args;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_help(argv[0]);
                return 0;
            }
            if (arg == "--text" || arg == "-t") {
                text_output = true;
                continue;
            }
            args.push_back(arg);
        }
        if (args.empty() || args.size() > 2) {
            print_help(argv[0]);
            return 2;
        }
        const char* env_model = std::getenv("PARAKEET_CPP_MODEL_PATH");
        const std::string model_path = args.size() == 2 ? args[1] : (env_model ? env_model : default_model_path());
        ensure_model_file(model_path);

        WavAudio wav = load_wav(args[0]);
        std::vector<float> audio = resample_linear(wav.mono, wav.sample_rate);
        const size_t default_chunk_seconds = text_output ? TEXT_CHUNK_SECONDS : DEFAULT_CHUNK_SECONDS;
        const size_t chunk_seconds = env_size("PARAKEET_CLI_CHUNK_SECONDS", default_chunk_seconds);
        const size_t chunk_frames = std::max<size_t>(1, chunk_seconds * TARGET_SAMPLE_RATE);

        std::cerr << "loading parakeet.cpp model: " << model_path << "\n";
        std::unique_ptr<pk::Model> model = pk::Model::load(model_path);
        if (!model) throw std::runtime_error("failed to load parakeet.cpp model");

        std::string full_text;
        std::vector<pk::Word> words;
        for (size_t offset = 0; offset < audio.size(); offset += chunk_frames) {
            const size_t count = std::min(chunk_frames, audio.size() - offset);
            std::vector<float> chunk(audio.begin() + static_cast<std::ptrdiff_t>(offset), audio.begin() + static_cast<std::ptrdiff_t>(offset + count));
            pk::Transcription transcription = model->transcribe_with_timestamps(chunk, TARGET_SAMPLE_RATE, pk::Decoder::kDefault);
            if (!transcription.text.empty()) {
                if (!full_text.empty()) full_text += " ";
                full_text += transcription.text;
            }
            const float chunk_start = static_cast<float>(offset) / static_cast<float>(TARGET_SAMPLE_RATE);
            if (text_output) {
                const size_t start_seconds = offset / TARGET_SAMPLE_RATE;
                const size_t end_seconds = (offset + count + TARGET_SAMPLE_RATE - 1) / TARGET_SAMPLE_RATE;
                if (!transcription.text.empty()) {
                    std::cout << "[" << timestamp(start_seconds) << "-" << timestamp(end_seconds) << "] " << transcription.text << "\n";
                }
                continue;
            }
            for (pk::Word word : transcription.words) {
                word.start += chunk_start;
                word.end += chunk_start;
                words.push_back(std::move(word));
            }
        }

        if (text_output) {
            pk::shutdown_backend();
            return 0;
        }

        std::cout << "{\"text\":\"" << json_escape(full_text) << "\",\"words\":[";
        for (size_t i = 0; i < words.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << "{\"w\":\"" << json_escape(words[i].text) << "\",\"start\":" << words[i].start
                      << ",\"end\":" << words[i].end << ",\"conf\":" << words[i].conf << "}";
        }
        std::cout << "]}\n";

        pk::shutdown_backend();
        return 0;
    } catch (const std::exception& e) {
        pk::shutdown_backend();
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
