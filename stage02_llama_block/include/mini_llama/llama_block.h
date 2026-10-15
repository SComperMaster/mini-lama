#pragma once

#include "mini_llama/tensor.h"

#include <cstddef>

namespace mini_llama {

[[nodiscard]] float silu(float value);
[[nodiscard]] Tensor silu(const Tensor& input);

// 对二维 Tensor 的每一行分别执行数值稳定的 Softmax。
[[nodiscard]] Tensor softmax_rows(const Tensor& input);

// Llama FFN 的门控激活：SiLU(gate) * up。
[[nodiscard]] Tensor swiglu(const Tensor& gate, const Tensor& up);

// tensor 的布局是 [tokens, heads * head_dim]。当前教学实现采用相邻维度配对：
// (0, 1), (2, 3), ...。start_position 用于未来的增量 decode。
void apply_rope_in_place(
    Tensor& tensor,
    std::size_t heads,
    std::size_t head_dim,
    std::size_t start_position,
    float theta = 10000.0F);

// 不使用 KV Cache 的 causal Grouped-Query Attention。
// query: [tokens, query_heads * head_dim]
// key/value: [tokens, kv_heads * head_dim]
// output: [tokens, query_heads * head_dim]
[[nodiscard]] Tensor causal_gqa_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    std::size_t query_heads,
    std::size_t kv_heads,
    std::size_t head_dim);

struct LlamaBlockConfig {
    std::size_t hidden_size = 0;
    std::size_t query_heads = 0;
    std::size_t kv_heads = 0;
    std::size_t head_dim = 0;
    std::size_t intermediate_size = 0;
    float rms_epsilon = 1.0e-5F;
    float rope_theta = 10000.0F;
};

// 为了便于阅读，本阶段统一让线性层权重采用 [input_dim, output_dim]。
struct LlamaBlockWeights {
    Tensor attention_norm; // [hidden_size]
    Tensor wq;             // [hidden_size, query_heads * head_dim]
    Tensor wk;             // [hidden_size, kv_heads * head_dim]
    Tensor wv;             // [hidden_size, kv_heads * head_dim]
    Tensor wo;             // [query_heads * head_dim, hidden_size]

    Tensor ffn_norm; // [hidden_size]
    Tensor w_gate;   // [hidden_size, intermediate_size]
    Tensor w_up;     // [hidden_size, intermediate_size]
    Tensor w_down;   // [intermediate_size, hidden_size]
};

// 完整的 pre-norm Llama Transformer Block：
// x + Attention(RMSNorm(x))，然后再加 SwiGLU(RMSNorm(x))。
[[nodiscard]] Tensor llama_block_forward(
    const Tensor& input,
    const LlamaBlockWeights& weights,
    const LlamaBlockConfig& config,
    std::size_t start_position = 0);

} // namespace mini_llama
