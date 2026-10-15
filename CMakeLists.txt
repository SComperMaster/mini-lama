cmake_minimum_required(VERSION 3.20)

project(mini_llama_tutorial LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

add_subdirectory(stage01_tensor)
add_subdirectory(stage02_llama_block)
add_subdirectory(stage03_toy_model)
add_subdirectory(stage04_kv_cache)
add_subdirectory(stage05_kv_cells)
add_subdirectory(stage06_prompt_reuse)
add_subdirectory(stage07_ggml_graph)
