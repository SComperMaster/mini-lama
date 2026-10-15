#pragma once

#include "mini_llama/llama_block.h"

#include <cstddef>
#include <vector>

namespace mini_llama {

using TokenId = std::size_t;

struct LlamaModelConfig {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 0;
    std::size_t layer_count = 0;
    std::size_t query_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t head_dim = 0;
    std::size_t intermediate_size = 0;
    float rms_epsilon = 1.0e-5F;
    float rope_theta = 10000.0F;
};

struct LlamaModelWeights {
    Tensor token_embeddings;               // [vocab_size, hidden_size]
    std::vector<LlamaBlockWeights> blocks; // layer_count 个 Transformer Block
    Tensor output_norm;                    // [hidden_size]
    Tensor lm_head;                        // [hidden_size, vocab_size]
};

// 根据 token id 从 embedding 表中复制对应行。
[[nodiscard]] Tensor embedding_lookup(
    const std::vector<TokenId>& tokens,
    const Tensor& embedding_table);

// 本阶段的完整前向过程。它一次处理所有 token，并返回每个位置的 logits：
// [token_count, vocab_size]。目前还没有 KV Cache。
[[nodiscard]] Tensor model_forward(
    const std::vector<TokenId>& tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config);

// 只在最后一个 token 的 logits 上执行 argmax。
[[nodiscard]] TokenId greedy_next_token(const Tensor& logits);

struct GenerationResult {
    std::vector<TokenId> tokens;
    std::size_t forward_passes = 0;

    // 所有 forward 实际处理的 token 数之和。无 KV Cache 时会形成
    // prompt + (prompt+1) + ... 的重复计算。
    std::size_t tokens_processed = 0;
};

[[nodiscard]] GenerationResult generate_greedy_without_kv_cache(
    const std::vector<TokenId>& prompt,
    std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config);

} // namespace mini_llama
