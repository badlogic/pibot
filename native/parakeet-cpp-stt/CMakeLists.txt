cmake_minimum_required(VERSION 3.18)
project(parakeet_cpp_stt_worker LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

include(FetchContent)

option(PARAKEET_CPP_STT_METAL "Enable GGML Metal backend on Apple platforms" ${APPLE})
if(UNIX AND NOT APPLE)
  set(PARAKEET_CPP_STT_VULKAN_DEFAULT ON)
else()
  set(PARAKEET_CPP_STT_VULKAN_DEFAULT OFF)
endif()
option(PARAKEET_CPP_STT_VULKAN "Enable GGML Vulkan backend on Linux/other non-Apple platforms" ${PARAKEET_CPP_STT_VULKAN_DEFAULT})
option(PARAKEET_CPP_STT_HIP "Enable GGML HIP/ROCm backend" OFF)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(PARAKEET_GGML_METAL ${PARAKEET_CPP_STT_METAL} CACHE BOOL "" FORCE)
set(PARAKEET_GGML_VULKAN ${PARAKEET_CPP_STT_VULKAN} CACHE BOOL "" FORCE)
set(PARAKEET_GGML_HIP ${PARAKEET_CPP_STT_HIP} CACHE BOOL "" FORCE)
set(GGML_METAL ${PARAKEET_CPP_STT_METAL} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${PARAKEET_CPP_STT_VULKAN} CACHE BOOL "" FORCE)
set(GGML_HIP ${PARAKEET_CPP_STT_HIP} CACHE BOOL "" FORCE)
set(GGML_HIP_NO_VMM ON CACHE BOOL "" FORCE)
set(GGML_HIP_ROCWMMA_FATTN OFF CACHE BOOL "" FORCE)
set(GGML_HIP_MMQ_MFMA ON CACHE BOOL "" FORCE)

set(PARAKEET_CPP_GIT_REPOSITORY "https://github.com/mudler/parakeet.cpp.git" CACHE STRING "parakeet.cpp repository")
set(PARAKEET_CPP_GIT_TAG "cb45f68068081af01e7092e91b038ee353eb56be" CACHE STRING "parakeet.cpp git tag" FORCE)
set(PARAKEET_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(PARAKEET_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PARAKEET_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  parakeet_cpp
  GIT_REPOSITORY ${PARAKEET_CPP_GIT_REPOSITORY}
  GIT_TAG ${PARAKEET_CPP_GIT_TAG}
  GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(parakeet_cpp)
target_include_directories(parakeet PRIVATE ${parakeet_cpp_SOURCE_DIR}/third_party)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/patches/backend.cpp ${parakeet_cpp_SOURCE_DIR}/src/backend.cpp COPYONLY)

# ggml Metal currently does not implement GGML_OP_CONV_2D_DW. parakeet.cpp's
# subsampling stack used ggml_conv_2d_dw_direct for two depthwise conv stages,
# which aborts at runtime on Metal. Use ggml_conv_2d_dw instead; it lowers to
# im2col + mul_mat, both supported by the Metal backend, and is mathematically
# equivalent to the direct depthwise op.
set(PARAKEET_CPP_SUBSAMPLING "${parakeet_cpp_SOURCE_DIR}/src/subsampling.cpp")
file(READ "${PARAKEET_CPP_SUBSAMPLING}" PARAKEET_CPP_SUBSAMPLING_SOURCE)
string(REPLACE "ggml_conv_2d_dw_direct" "ggml_conv_2d_dw" PARAKEET_CPP_SUBSAMPLING_SOURCE "${PARAKEET_CPP_SUBSAMPLING_SOURCE}")
file(WRITE "${PARAKEET_CPP_SUBSAMPLING}" "${PARAKEET_CPP_SUBSAMPLING_SOURCE}")

# ggml_conv_2d_dw lowers to im2col + mul_mat with an F16 im2col tensor by
# default. With F32 depthwise weights this creates a F32 x F16 mul_mv. The Metal
# dispatch code asks for kernel_mul_mv_f32_f16_short, but the embedded Metal
# library only defines f32_f32_short and f16_f32_short. Keep the lowered path on
# Metal-compatible kernels by making depthwise im2col F32.
set(PARAKEET_CPP_GGML_C "${parakeet_cpp_SOURCE_DIR}/third_party/ggml/src/ggml.c")
file(READ "${PARAKEET_CPP_GGML_C}" PARAKEET_CPP_GGML_C_SOURCE)
string(REPLACE "true, GGML_TYPE_F16); // [N * IC, OH, OW, KH * KW]" "true, GGML_TYPE_F32); // [N * IC, OH, OW, KH * KW]" PARAKEET_CPP_GGML_C_SOURCE "${PARAKEET_CPP_GGML_C_SOURCE}")
file(WRITE "${PARAKEET_CPP_GGML_C}" "${PARAKEET_CPP_GGML_C_SOURCE}")

if(WIN32)
  foreach(PARAKEET_CPP_PI_FILE IN ITEMS
    "${parakeet_cpp_SOURCE_DIR}/src/fft.cpp"
    "${parakeet_cpp_SOURCE_DIR}/src/mel_gpu.cpp"
  )
    file(READ "${PARAKEET_CPP_PI_FILE}" PARAKEET_CPP_PI_SOURCE)
    string(REPLACE "M_PI" "3.14159265358979323846" PARAKEET_CPP_PI_SOURCE "${PARAKEET_CPP_PI_SOURCE}")
    file(WRITE "${PARAKEET_CPP_PI_FILE}" "${PARAKEET_CPP_PI_SOURCE}")
  endforeach()
endif()

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2 OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  whisper_cpp
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG master
)
FetchContent_MakeAvailable(whisper_cpp)

add_executable(parakeet-cpp-stt-worker src/main.cpp)
target_include_directories(parakeet-cpp-stt-worker PRIVATE ${parakeet_cpp_SOURCE_DIR}/src)
target_link_libraries(parakeet-cpp-stt-worker PRIVATE parakeet whisper)

add_executable(parakeet-cpp-transcribe src/cli.cpp)
target_include_directories(parakeet-cpp-transcribe PRIVATE ${parakeet_cpp_SOURCE_DIR}/src)
target_link_libraries(parakeet-cpp-transcribe PRIVATE parakeet)
