add_library(mini_llama_tensor STATIC
    src/tensor.cpp
)

target_include_directories(mini_llama_tensor PUBLIC include)

if(MSVC)
    target_compile_options(mini_llama_tensor PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_tensor PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage01_tensor src/main.cpp)
target_link_libraries(stage01_tensor PRIVATE mini_llama_tensor)

if(MSVC)
    target_compile_options(stage01_tensor PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage01_tensor PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage01_tensor_tests COMMAND stage01_tensor)
