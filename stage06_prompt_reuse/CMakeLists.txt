add_library(mini_llama_prompt_reuse STATIC
    src/prompt_reuse.cpp
)

target_include_directories(mini_llama_prompt_reuse PUBLIC include)
target_link_libraries(mini_llama_prompt_reuse PUBLIC mini_llama_kv_cells)

if(MSVC)
    target_compile_options(mini_llama_prompt_reuse PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_prompt_reuse PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage06_prompt_reuse src/main.cpp)
target_link_libraries(stage06_prompt_reuse PRIVATE mini_llama_prompt_reuse)

if(MSVC)
    target_compile_options(stage06_prompt_reuse PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage06_prompt_reuse PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage06_prompt_reuse_tests COMMAND stage06_prompt_reuse)
