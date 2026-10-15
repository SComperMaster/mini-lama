#include "mini_llama/llama_block.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_llama {
namespace {

void require_matrix(const Tensor& tensor, const char* name) {
    if (tensor.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D tensor");
    }
}

void require_same_shape(const Tensor& left, const Tensor& right, const char* operation) {
    if (left.shape() != right.shape()) {
        throw std::invalid_argument(std::string(operation) + " requires equal tensor shapes");
    }
}

Tensor add(const Tensor& left, const Tensor& right) {
    require_same_shape(left, right, "add");
    Tensor result(left.shape());
    for (std::size_t i = 0; i < left.size(); ++i) {
        result[i] = left[i] + right[i];
    }
    return result;
}

void validate_block_shapes(
    const Tensor& input,
    const LlamaBlockWeights& weights,
    const LlamaBlockConfig& config) {
    require_matrix(input, "input");

    if (config.hidden_size == 0 || config.query_heads == 0 || config.kv_heads == 0 ||
        config.head_dim == 0 || config.intermediate_size == 0) {
        throw std::invalid_argument("Llama block dimensions must be greater than zero");
    }
    if (input.shape()[1] != config.hidden_size) {
        throw std::invalid_argument("input hidden dimension does not match block config");
    }
    if (config.query_heads % config.kv_heads != 0) {
        throw std::invalid_argument("query_heads must be divisible by kv_heads");
    }

    const std::size_t query_size = config.query_heads * config.head_dim;
    const std::size_t kv_size = config.kv_heads * config.head_dim;

    const auto require_shape = [](const Tensor& tensor, const std::vector<std::size_t>& shape,
                                  const char* name) {
        if (tensor.shape() != shape) {
            throw std::invalid_argument(std::string(name) + " has an invalid shape");
        }
    };

    require_shape(weights.attention_norm, {config.hidden_size}, "attention_norm");
    require_shape(weights.wq, {config.hidden_size, query_size}, "wq");
    require_shape(weights.wk, {config.hidden_size, kv_size}, "wk");
    require_shape(weights.wv, {config.hidden_size, kv_size}, "wv");
    require_shape(weights.wo, {query_size, config.hidden_size}, "wo");
    require_shape(weights.ffn_norm, {config.hidden_size}, "ffn_norm");
    require_shape(weights.w_gate, {config.hidden_size, config.intermediate_size}, "w_gate");
    require_shape(weights.w_up, {config.hidden_size, config.intermediate_size}, "w_up");
    require_shape(weights.w_down, {config.intermediate_size, config.hidden_size}, "w_down");
}

} // namespace

float silu(const float value) {
    return value / (1.0F + std::exp(-value));
}

Tensor silu(const Tensor& input) {
    Tensor result(input.shape());
    for (std::size_t i = 0; i < input.size(); ++i) {
        result[i] = silu(input[i]);
    }
    return result;
}

Tensor softmax_rows(const Tensor& input) {
    require_matrix(input, "input");

    const std::size_t rows = input.shape()[0];
    const std::size_t cols = input.shape()[1];
    Tensor result(input.shape());

    for (std::size_t row = 0; row < rows; ++row) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t col = 0; col < cols; ++col) {
            maximum = std::max(maximum, input.at(row, col));
        }

        float exponential_sum = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) {
            const float exponential = std::exp(input.at(row, col) - maximum);
            result.at(row, col) = exponential;
            exponential_sum += exponential;
        }

        for (std::size_t col = 0; col < cols; ++col) {
            result.at(row, col) /= exponential_sum;
        }
    }

    return result;
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
    require_same_shape(gate, up, "SwiGLU");
    Tensor result(gate.shape());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        result[i] = silu(gate[i]) * up[i];
    }
    return result;
}

void apply_rope_in_place(
    Tensor& tensor,
    const std::size_t heads,
    const std::size_t head_dim,
    const std::size_t start_position,
    const float theta) {
    require_matrix(tensor, "RoPE tensor");
    if (heads == 0 || head_dim == 0 || head_dim % 2 != 0) {
        throw std::invalid_argument("RoPE requires non-zero heads and an even head_dim");
    }
    if (tensor.shape()[1] != heads * head_dim) {
        throw std::invalid_argument("RoPE tensor width does not match heads * head_dim");
    }
    if (theta <= 0.0F) {
        throw std::invalid_argument("RoPE theta must be positive");
    }

    const std::size_t tokens = tensor.shape()[0];
    for (std::size_t token = 0; token < tokens; ++token) {
        const float position = static_cast<float>(start_position + token);
        for (std::size_t head = 0; head < heads; ++head) {
            const std::size_t head_offset = head * head_dim;
            for (std::size_t dimension = 0; dimension < head_dim; dimension += 2) {
                const float exponent = -static_cast<float>(dimension) /
                                       static_cast<float>(head_dim);
                const float frequency = std::pow(theta, exponent);
                const float angle = position * frequency;
                const float cosine = std::cos(angle);
                const float sine = std::sin(angle);

                const float even = tensor.at(token, head_offset + dimension);
                const float odd = tensor.at(token, head_offset + dimension + 1);
                tensor.at(token, head_offset + dimension) = even * cosine - odd * sine;
                tensor.at(token, head_offset + dimension + 1) = even * sine + odd * cosine;
            }
        }
    }
}

Tensor causal_gqa_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::size_t query_heads,
    const std::size_t kv_heads,
    const std::size_t head_dim) {
    require_matrix(query, "query");
    require_matrix(key, "key");
    require_matrix(value, "value");

    if (query_heads == 0 || kv_heads == 0 || head_dim == 0 || query_heads % kv_heads != 0) {
        throw std::invalid_argument("GQA requires query_heads divisible by kv_heads");
    }

    const std::size_t tokens = query.shape()[0];
    if (key.shape()[0] != tokens || value.shape()[0] != tokens ||
        query.shape()[1] != query_heads * head_dim ||
        key.shape()[1] != kv_heads * head_dim || value.shape() != key.shape()) {
        throw std::invalid_argument("GQA tensor shapes do not match the head configuration");
    }

    Tensor output({tokens, query_heads * head_dim});
    const std::size_t query_heads_per_kv_head = query_heads / kv_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    for (std::size_t token = 0; token < tokens; ++token) {
        for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
            const std::size_t kv_head = query_head / query_heads_per_kv_head;
            std::vector<float> scores(token + 1, 0.0F);

            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t source = 0; source <= token; ++source) {
                float dot = 0.0F;
                for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                    dot += query.at(token, query_head * head_dim + dimension) *
                           key.at(source, kv_head * head_dim + dimension);
                }
                scores[source] = dot * scale;
                maximum = std::max(maximum, scores[source]);
            }

            float sum = 0.0F;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (float& score : scores) {
                score /= sum;
            }

            for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                float weighted_value = 0.0F;
                for (std::size_t source = 0; source <= token; ++source) {
                    weighted_value += scores[source] *
                                      value.at(source, kv_head * head_dim + dimension);
                }
                output.at(token, query_head * head_dim + dimension) = weighted_value;
            }
        }
    }

    return output;
}

Tensor llama_block_forward(
    const Tensor& input,
    const LlamaBlockWeights& weights,
    const LlamaBlockConfig& config,
    const std::size_t start_position) {
    validate_block_shapes(input, weights, config);

    const Tensor attention_input = rms_norm(input, weights.attention_norm, config.rms_epsilon);
    Tensor query = matmul(attention_input, weights.wq);
    Tensor key = matmul(attention_input, weights.wk);
    const Tensor value = matmul(attention_input, weights.wv);

    apply_rope_in_place(
        query, config.query_heads, config.head_dim, start_position, config.rope_theta);
    apply_rope_in_place(
        key, config.kv_heads, config.head_dim, start_position, config.rope_theta);

    const Tensor attention = causal_gqa_attention(
        query, key, value, config.query_heads, config.kv_heads, config.head_dim);
    const Tensor attention_output = matmul(attention, weights.wo);
    const Tensor attention_residual = add(input, attention_output);

    const Tensor ffn_input = rms_norm(
        attention_residual, weights.ffn_norm, config.rms_epsilon);
    const Tensor gate = matmul(ffn_input, weights.w_gate);
    const Tensor up = matmul(ffn_input, weights.w_up);
    const Tensor ffn_hidden = swiglu(gate, up);
    const Tensor ffn_output = matmul(ffn_hidden, weights.w_down);

    return add(attention_residual, ffn_output);
}

} // namespace mini_llama
