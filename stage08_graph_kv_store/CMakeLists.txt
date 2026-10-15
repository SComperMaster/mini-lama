add_library(mini_llama_graph_kv_store STATIC
    src/graph_kv_store.cpp
)

target_include_directories(mini_llama_graph_kv_store PUBLIC include)
target_link_libraries(mini_llama_graph_kv_store PUBLIC mini_llama_ggml_graph)

if(MSVC)
    target_compile_options(mini_llama_graph_kv_store PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_graph_kv_store PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage08_graph_kv_store src/main.cpp)
target_link_libraries(stage08_graph_kv_store PRIVATE mini_llama_graph_kv_store)

if(MSVC)
    target_compile_options(stage08_graph_kv_store PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage08_graph_kv_store PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage08_graph_kv_store_tests COMMAND stage08_graph_kv_store)
