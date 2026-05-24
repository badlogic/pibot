cmake_minimum_required(VERSION 3.16)
project(pibot_stt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../vendor/whisper.cpp ${CMAKE_BINARY_DIR}/whisper.cpp-build)

add_executable(pibot-stt pibot_stt.cpp)
target_link_libraries(pibot-stt PRIVATE whisper)
target_include_directories(pibot-stt PRIVATE
	${CMAKE_CURRENT_LIST_DIR}/../../vendor/whisper.cpp/include
	${CMAKE_CURRENT_LIST_DIR}/../../vendor/whisper.cpp/ggml/include
)
