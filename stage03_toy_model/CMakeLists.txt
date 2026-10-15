add_library(mini_llama_model STATIC
    src/model.cpp
)

target_include_directories(mini_llama_model PUBLIC include)
target_link_libraries(mini_llama_model PUBLIC mini_llama_block)

if(MSVC)
    target_compile_options(mini_llama_model PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_model PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage03_toy_model src/main.cpp)
target_link_libraries(stage03_toy_model PRIVATE mini_llama_model)

if(MSVC)
    target_compile_options(stage03_toy_model PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage03_toy_model PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage03_toy_model_tests COMMAND stage03_toy_model)
