add_library(mini_llama_graph_attention STATIC
    src/graph_attention.cpp
)

target_include_directories(mini_llama_graph_attention PUBLIC include)
target_link_libraries(mini_llama_graph_attention PUBLIC mini_llama_graph_kv_store)

if(MSVC)
    target_compile_options(mini_llama_graph_attention PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_graph_attention PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage09_graph_attention src/main.cpp)
target_link_libraries(stage09_graph_attention PRIVATE mini_llama_graph_attention)

if(MSVC)
    target_compile_options(stage09_graph_attention PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage09_graph_attention PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage09_graph_attention_tests COMMAND stage09_graph_attention)
