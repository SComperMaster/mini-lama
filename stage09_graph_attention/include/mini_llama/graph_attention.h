#pragma once

#include "mini_llama/graph_kv_store.h"

#include <cstddef>
#include <string>
#include <vector>

namespace mini_llama {

// 构造教学版 Attention mask。每个 query token 都给出它能够看到的物理
// KV cell：可见位置写 0，不可见位置写 -infinity。
[[nodiscard]] ggml::Tensor* build_attention_mask(
    ggml::Context& graph_context,
    std::size_t cache_capacity,
    const std::vector<std::vector<std::size_t>>& visible_cells,
    std::string name = "attention_mask");

// SET_ROWS 返回的是“写操作节点”，而 Attention 读取的是持久 cache Tensor。
// 两者之间没有普通的数据依赖边，因此必须像 llama.cpp 一样把写节点先作为
// graph root 展开，然后再展开 Attention 输出。
void build_kv_attention_graph(
    ggml::Graph& graph,
    const KVStoreNodes& stores,
    ggml::Tensor* attention_output);

} // namespace mini_llama
