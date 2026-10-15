add_library(mini_llama_ggml_graph STATIC
    src/mini_ggml.cpp
)

target_include_directories(mini_llama_ggml_graph PUBLIC include)

if(MSVC)
    target_compile_options(mini_llama_ggml_graph PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_ggml_graph PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage07_ggml_graph src/main.cpp)
target_link_libraries(stage07_ggml_graph PRIVATE mini_llama_ggml_graph)

if(MSVC)
    target_compile_options(stage07_ggml_graph PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage07_ggml_graph PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage07_ggml_graph_tests COMMAND stage07_ggml_graph)
