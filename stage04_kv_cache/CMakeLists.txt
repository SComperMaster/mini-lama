add_library(mini_llama_kv_cache STATIC
    src/kv_cache.cpp
)

target_include_directories(mini_llama_kv_cache PUBLIC include)
target_link_libraries(mini_llama_kv_cache PUBLIC mini_llama_model)

if(MSVC)
    target_compile_options(mini_llama_kv_cache PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(mini_llama_kv_cache PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(stage04_kv_cache src/main.cpp)
target_link_libraries(stage04_kv_cache PRIVATE mini_llama_kv_cache)

if(MSVC)
    target_compile_options(stage04_kv_cache PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(stage04_kv_cache PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME stage04_kv_cache_tests COMMAND stage04_kv_cache)
