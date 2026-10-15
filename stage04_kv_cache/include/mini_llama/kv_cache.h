#pragma once

#include "mini_llama/model.h"

#include <cstddef>
#include <vector>

namespace mini_llama {

// 最容易理解的连续 KV Cache。它提前分配 [capacity, kv_width]，并用 size_ 记录
// 已写入多少行。stage05 才会把它升级为 llama.cpp 风格的 cell 管理。
class LayerKVCache {
public:
    LayerKVCache(std::size_t capacity, std::size_t kv_width);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t kv_width() const noexcept;

    void append(const Tensor& new_keys, const Tensor& new_values);
    void clear() noexcept;

    [[nodiscard]] float key_at(std::size_t token, std::size_t dimension) const;
    [[nodiscard]] float value_at(std::size_t token, std::size_t dimension) const;

private:
    Tensor keys_;
    Tensor values_;
    std::size_t size_ = 0;
};

class ModelKVCache {
public:
    ModelKVCache(const LlamaModelConfig& config, std::size_t max_context);

    [[nodiscard]] std::size_t layer_count() const noexcept;
    [[nodiscard]] std::size_t tokens_cached() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;

    LayerKVCache& layer(std::size_t index);
    const LayerKVCache& layer(std::size_t index) const;
    void clear() noexcept;

private:
    std::vector<LayerKVCache> layers_;
    std::size_t capacity_ = 0;
};

// query 只包含当前新输入，cache 已包含旧 token 和当前新 token 的 K/V。
[[nodiscard]] Tensor causal_gqa_attention_with_cache(
    const Tensor& query,
    const LayerKVCache& cache,
    std::size_t past_tokens,
    std::size_t query_heads,
    std::size_t kv_heads,
    std::size_t head_dim);

[[nodiscard]] Tensor llama_block_forward_with_kv_cache(
    const Tensor& input,
    const LlamaBlockWeights& weights,
    const LlamaBlockConfig& config,
    LayerKVCache& cache);

// new_tokens 可以是整个 prompt（prefill），也可以是单个 token（decode）。
// 返回值只包含本次 new_tokens 对应位置的 logits。
[[nodiscard]] Tensor model_forward_with_kv_cache(
    const std::vector<TokenId>& new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    ModelKVCache& cache);

[[nodiscard]] GenerationResult generate_greedy_with_kv_cache(
    const std::vector<TokenId>& prompt,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    ModelKVCache& cache);

} // namespace mini_llama
