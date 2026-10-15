add_library(mini_llama_block STATIC
    src/llama_block.cpp
)

target_include_directories(mini_llama_block PUBLIC include)
target_link_libraries(mini_llama_block PUBLIC mini_llama_tensor)

if(MSVC)
    target_compile_options(mini_llama_block PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_block PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage02_llama_block src/main.cpp)
target_link_libraries(stage02_llama_block PRIVATE mini_llama_block)

if(MSVC)
    target_compile_options(stage02_llama_block PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage02_llama_block PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage02_llama_block_tests COMMAND stage02_llama_block)
