#pragma once

#include "mini_llama/mini_ggml.h"

#include <cstddef>

namespace mini_llama {

struct KVStoreNodes {
    ggml::Tensor* key_write = nullptr;
    ggml::Tensor* value_write = nullptr;
};

// 这个对象的生命周期跨越多次 graph compute，对应 llama_context 中长期存在的
// KV 数据区。每次 decode 使用的 ggml::Context 可以销毁，但这里的 K/V 不销毁。
class GraphKVCache {
public:
    GraphKVCache(std::size_t kv_width, std::size_t capacity);

    [[nodiscard]] std::size_t kv_width() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;

    [[nodiscard]] ggml::Tensor* key() noexcept;
    [[nodiscard]] ggml::Tensor* value() noexcept;
    [[nodiscard]] const ggml::Tensor* key() const noexcept;
    [[nodiscard]] const ggml::Tensor* value() const noexcept;

    // k_cur/v_cur 是当前 batch 的临时计算结果；cell_indices 指定它们应该写入的
    // 物理 cache 行。这里只创建 SET_ROWS 节点，graph_compute 时才真正修改缓存。
    [[nodiscard]] KVStoreNodes build_store_nodes(
        ggml::Context& graph_context,
        ggml::Tensor* k_cur,
        ggml::Tensor* v_cur,
        ggml::Tensor* cell_indices);

private:
    std::size_t kv_width_ = 0;
    std::size_t capacity_ = 0;
    ggml::Context storage_context_;
    ggml::Tensor* key_ = nullptr;
    ggml::Tensor* value_ = nullptr;
};

} // namespace mini_llama
