#include "mini_llama/kv_cache.h"

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

Tensor add(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape()) {
        throw std::invalid_argument("residual add requires equal tensor shapes");
    }
    Tensor result(left.shape());
    for (std::size_t i = 0; i < left.size(); ++i) {
        result[i] = left[i] + right[i];
    }
    return result;
}

LlamaBlockConfig block_config_from(const LlamaModelConfig& config) {
    return LlamaBlockConfig{
        .hidden_size = config.hidden_size,
        .query_heads = config.query_heads,
        .kv_heads = config.kv_heads,
        .head_dim = config.head_dim,
        .intermediate_size = config.intermediate_size,
        .rms_epsilon = config.rms_epsilon,
        .rope_theta = config.rope_theta,
    };
}

void validate_block_dimensions(
    const Tensor& input,
    const LlamaBlockWeights& weights,
    const LlamaBlockConfig& config,
    const LayerKVCache& cache) {
    require_matrix(input, "block input");
    if (input.shape()[1] != config.hidden_size) {
        throw std::invalid_argument("block input width does not match hidden_size");
    }
    if (config.query_heads == 0 || config.kv_heads == 0 || config.head_dim == 0 ||
        config.query_heads % config.kv_heads != 0) {
        throw std::invalid_argument("invalid GQA head configuration");
    }
    if (cache.kv_width() != config.kv_heads * config.head_dim) {
        throw std::invalid_argument("KV Cache width does not match block config");
    }

    const std::size_t query_width = config.query_heads * config.head_dim;
    const std::size_t kv_width = config.kv_heads * config.head_dim;
    const auto require_shape = [](const Tensor& tensor, const std::vector<std::size_t>& shape,
                                  const char* name) {
        if (tensor.shape() != shape) {
            throw std::invalid_argument(std::string(name) + " has an invalid shape");
        }
    };

    require_shape(weights.attention_norm, {config.hidden_size}, "attention_norm");
    require_shape(weights.wq, {config.hidden_size, query_width}, "wq");
    require_shape(weights.wk, {config.hidden_size, kv_width}, "wk");
    require_shape(weights.wv, {config.hidden_size, kv_width}, "wv");
    require_shape(weights.wo, {query_width, config.hidden_size}, "wo");
    require_shape(weights.ffn_norm, {config.hidden_size}, "ffn_norm");
    require_shape(weights.w_gate, {config.hidden_size, config.intermediate_size}, "w_gate");
    require_shape(weights.w_up, {config.hidden_size, config.intermediate_size}, "w_up");
    require_shape(weights.w_down, {config.intermediate_size, config.hidden_size}, "w_down");
}

} // namespace

LayerKVCache::LayerKVCache(const std::size_t capacity, const std::size_t kv_width)
    : keys_({capacity, kv_width}), values_({capacity, kv_width}) {
    if (capacity == 0 || kv_width == 0) {
        throw std::invalid_argument("KV Cache capacity and width must be greater than zero");
    }
}

std::size_t LayerKVCache::size() const noexcept {
    return size_;
}

std::size_t LayerKVCache::capacity() const noexcept {
    return keys_.shape()[0];
}

std::size_t LayerKVCache::kv_width() const noexcept {
    return keys_.shape()[1];
}

void LayerKVCache::append(const Tensor& new_keys, const Tensor& new_values) {
    require_matrix(new_keys, "new keys");
    require_matrix(new_values, "new values");
    if (new_keys.shape() != new_values.shape() || new_keys.shape()[1] != kv_width()) {
        throw std::invalid_argument("new K/V shapes do not match the cache");
    }
    if (new_keys.shape()[0] > capacity() - size_) {
        throw std::overflow_error("KV Cache capacity exceeded");
    }

    for (std::size_t row = 0; row < new_keys.shape()[0]; ++row) {
        for (std::size_t col = 0; col < kv_width(); ++col) {
            keys_.at(size_ + row, col) = new_keys.at(row, col);
            values_.at(size_ + row, col) = new_values.at(row, col);
        }
    }
    size_ += new_keys.shape()[0];
}

void LayerKVCache::clear() noexcept {
    size_ = 0;
}

float LayerKVCache::key_at(const std::size_t token, const std::size_t dimension) const {
    if (token >= size_) {
        throw std::out_of_range("K token index is outside the used cache");
    }
    return keys_.at(token, dimension);
}

float LayerKVCache::value_at(const std::size_t token, const std::size_t dimension) const {
    if (token >= size_) {
        throw std::out_of_range("V token index is outside the used cache");
    }
    return values_.at(token, dimension);
}

ModelKVCache::ModelKVCache(
    const LlamaModelConfig& config,
    const std::size_t max_context)
    : capacity_(max_context) {
    if (config.layer_count == 0 || config.kv_heads == 0 || config.head_dim == 0 ||
        max_context == 0) {
        throw std::invalid_argument("invalid model or KV Cache dimensions");
    }

    const std::size_t kv_width = config.kv_heads * config.head_dim;
    layers_.reserve(config.layer_count);
    for (std::size_t layer_index = 0; layer_index < config.layer_count; ++layer_index) {
        layers_.emplace_back(max_context, kv_width);
    }
}

std::size_t ModelKVCache::layer_count() const noexcept {
    return layers_.size();
}

std::size_t ModelKVCache::tokens_cached() const noexcept {
    return layers_.empty() ? 0 : layers_.front().size();
}

std::size_t ModelKVCache::capacity() const noexcept {
    return capacity_;
}

LayerKVCache& ModelKVCache::layer(const std::size_t index) {
    return layers_.at(index);
}

const LayerKVCache& ModelKVCache::layer(const std::size_t index) const {
    return layers_.at(index);
}

void ModelKVCache::clear() noexcept {
    for (LayerKVCache& cache : layers_) {
        cache.clear();
    }
}

Tensor causal_gqa_attention_with_cache(
    const Tensor& query,
    const LayerKVCache& cache,
    const std::size_t past_tokens,
    const std::size_t query_heads,
    const std::size_t kv_heads,
    const std::size_t head_dim) {
    require_matrix(query, "query");
    if (query_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        query_heads % kv_heads != 0) {
        throw std::invalid_argument("invalid cached GQA head configuration");
    }

    const std::size_t new_tokens = query.shape()[0];
    if (query.shape()[1] != query_heads * head_dim ||
        cache.kv_width() != kv_heads * head_dim ||
        cache.size() != past_tokens + new_tokens) {
        throw std::invalid_argument("cached GQA shapes or token positions are inconsistent");
    }

    Tensor output({new_tokens, query_heads * head_dim});
    const std::size_t query_heads_per_kv_head = query_heads / kv_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    for (std::size_t token = 0; token < new_tokens; ++token) {
        // 当前 chunk 的第 token 个 query 可以看到全部 past，以及 chunk 中到自己为止的 K/V。
        const std::size_t visible_tokens = past_tokens + token + 1;
        for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
            const std::size_t kv_head = query_head / query_heads_per_kv_head;
            std::vector<float> scores(visible_tokens, 0.0F);

            float maximum = -std::numeric_limits<float>::infinity();
            for (std::size_t source = 0; source < visible_tokens; ++source) {
                float dot = 0.0F;
                for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                    dot += query.at(token, query_head * head_dim + dimension) *
                           cache.key_at(source, kv_head * head_dim + dimension);
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
                for (std::size_t source = 0; source < visible_tokens; ++source) {
                    weighted_value += scores[source] *
                                      cache.value_at(source, kv_head * head_dim + dimension);
                }
                output.at(token, query_head * head_dim + dimension) = weighted_value;
            }
        }
    }

    return output;
}

Tensor llama_block_forward_with_kv_cache(
    const Tensor& input,
    const LlamaBlockWeights& weights,
    const LlamaBlockConfig& config,
    LayerKVCache& cache) {
    validate_block_dimensions(input, weights, config, cache);
    const std::size_t past_tokens = cache.size();

    const Tensor attention_input = rms_norm(input, weights.attention_norm, config.rms_epsilon);
    Tensor query = matmul(attention_input, weights.wq);
    Tensor key = matmul(attention_input, weights.wk);
    const Tensor value = matmul(attention_input, weights.wv);

    // 位置不能从零重新开始，而必须接在缓存已有 token 的后面。
    apply_rope_in_place(
        query, config.query_heads, config.head_dim, past_tokens, config.rope_theta);
    apply_rope_in_place(
        key, config.kv_heads, config.head_dim, past_tokens, config.rope_theta);

    cache.append(key, value);
    const Tensor attention = causal_gqa_attention_with_cache(
        query, cache, past_tokens, config.query_heads, config.kv_heads, config.head_dim);
    const Tensor attention_output = matmul(attention, weights.wo);
    const Tensor attention_residual = add(input, attention_output);

    const Tensor ffn_input = rms_norm(
        attention_residual, weights.ffn_norm, config.rms_epsilon);
    const Tensor gate = matmul(ffn_input, weights.w_gate);
    const Tensor up = matmul(ffn_input, weights.w_up);
    const Tensor ffn_output = matmul(swiglu(gate, up), weights.w_down);

    return add(attention_residual, ffn_output);
}

Tensor model_forward_with_kv_cache(
    const std::vector<TokenId>& new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    ModelKVCache& cache) {
    if (new_tokens.empty()) {
        throw std::invalid_argument("cached forward requires at least one new token");
    }
    if (weights.blocks.size() != config.layer_count ||
        cache.layer_count() != config.layer_count) {
        throw std::invalid_argument("model weights, config and cache layer counts differ");
    }
    if (new_tokens.size() > cache.capacity() - cache.tokens_cached()) {
        throw std::overflow_error("new tokens exceed the model KV Cache capacity");
    }

    const std::size_t original_position = cache.tokens_cached();
    Tensor hidden = embedding_lookup(new_tokens, weights.token_embeddings);
    const LlamaBlockConfig block_config = block_config_from(config);

    for (std::size_t layer_index = 0; layer_index < config.layer_count; ++layer_index) {
        if (cache.layer(layer_index).size() != original_position) {
            throw std::logic_error("model KV Cache layers are out of sync");
        }
        hidden = llama_block_forward_with_kv_cache(
            hidden, weights.blocks[layer_index], block_config, cache.layer(layer_index));
    }

    const Tensor normalized = rms_norm(hidden, weights.output_norm, config.rms_epsilon);
    return matmul(normalized, weights.lm_head);
}

GenerationResult generate_greedy_with_kv_cache(
    const std::vector<TokenId>& prompt,
    const std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    ModelKVCache& cache) {
    if (prompt.empty()) {
        throw std::invalid_argument("generation prompt cannot be empty");
    }
    if (cache.tokens_cached() != 0) {
        throw std::invalid_argument("generation requires an empty KV Cache");
    }

    GenerationResult result{
        .tokens = prompt,
        .forward_passes = 0,
        .tokens_processed = 0,
    };
    if (max_new_tokens == 0) {
        return result;
    }

    // Prefill：整个 prompt 一次进入模型，并填充每一层的 K/V。
    result.tokens_processed += prompt.size();
    Tensor logits = model_forward_with_kv_cache(prompt, weights, config, cache);
    result.tokens.push_back(greedy_next_token(logits));
    ++result.forward_passes;

    // Decode：为了预测下一个 token，只提交上一步刚生成的一个 token。
    for (std::size_t step = 1; step < max_new_tokens; ++step) {
        const TokenId newest_token = result.tokens.back();
        logits = model_forward_with_kv_cache({newest_token}, weights, config, cache);
        result.tokens.push_back(greedy_next_token(logits));
        ++result.forward_passes;
        ++result.tokens_processed;
    }

    return result;
}

} // namespace mini_llama
