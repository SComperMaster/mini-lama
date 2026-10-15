add_library(mini_llama_kv_cells STATIC
    src/kv_cells.cpp
)

target_include_directories(mini_llama_kv_cells PUBLIC include)
target_link_libraries(mini_llama_kv_cells PUBLIC mini_llama_model)

if(MSVC)
    target_compile_options(mini_llama_kv_cells PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_kv_cells PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage05_kv_cells src/main.cpp)
target_link_libraries(stage05_kv_cells PRIVATE mini_llama_kv_cells)

if(MSVC)
    target_compile_options(stage05_kv_cells PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage05_kv_cells PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage05_kv_cells_tests COMMAND stage05_kv_cells)
