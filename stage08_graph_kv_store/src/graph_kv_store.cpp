#include "mini_llama/graph_kv_store.h"

#include <limits>
#include <stdexcept>

namespace mini_llama {

GraphKVCache::GraphKVCache(
    const std::size_t kv_width,
    const std::size_t capacity)
    : kv_width_(kv_width), capacity_(capacity) {
    if (kv_width == 0 || capacity == 0) {
        throw std::invalid_argument("KV cache dimensions must be positive");
    }
    if (kv_width > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        capacity > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("KV cache dimensions exceed int64_t");
    }

    key_ = storage_context_.new_tensor_2d(
        static_cast<std::int64_t>(kv_width),
        static_cast<std::int64_t>(capacity),
        "key_cache");
    value_ = storage_context_.new_tensor_2d(
        static_cast<std::int64_t>(kv_width),
        static_cast<std::int64_t>(capacity),
        "value_cache");
}

std::size_t GraphKVCache::kv_width() const noexcept {
    return kv_width_;
}

std::size_t GraphKVCache::capacity() const noexcept {
    return capacity_;
}

ggml::Tensor* GraphKVCache::key() noexcept {
    return key_;
}

ggml::Tensor* GraphKVCache::value() noexcept {
    return value_;
}

const ggml::Tensor* GraphKVCache::key() const noexcept {
    return key_;
}

const ggml::Tensor* GraphKVCache::value() const noexcept {
    return value_;
}

KVStoreNodes GraphKVCache::build_store_nodes(
    ggml::Context& graph_context,
    ggml::Tensor* k_cur,
    ggml::Tensor* v_cur,
    ggml::Tensor* cell_indices) {
    if (k_cur == nullptr || v_cur == nullptr || cell_indices == nullptr) {
        throw std::invalid_argument("KV store inputs cannot be null");
    }
    if (k_cur->ne != v_cur->ne ||
        k_cur->ne[0] != static_cast<std::int64_t>(kv_width_) ||
        k_cur->ne[1] != cell_indices->ne[0]) {
        throw std::invalid_argument("KV store shapes do not match cache or indices");
    }

    KVStoreNodes result;
    result.key_write = graph_context.set_rows(
        key_, k_cur, cell_indices, "key_cache_write");
    result.value_write = graph_context.set_rows(
        value_, v_cur, cell_indices, "value_cache_write");
    return result;
}

} // namespace mini_llama
