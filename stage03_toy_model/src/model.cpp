#include "mini_llama/model.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace mini_llama {
namespace {

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

void validate_model(
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config) {
    if (config.vocab_size == 0 || config.hidden_size == 0 || config.layer_count == 0) {
        throw std::invalid_argument("model dimensions must be greater than zero");
    }
    if (weights.token_embeddings.shape() !=
        std::vector<std::size_t>{config.vocab_size, config.hidden_size}) {
        throw std::invalid_argument("token embedding table has an invalid shape");
    }
    if (weights.blocks.size() != config.layer_count) {
        throw std::invalid_argument("block count does not match model config");
    }
    if (weights.output_norm.shape() != std::vector<std::size_t>{config.hidden_size}) {
        throw std::invalid_argument("output norm has an invalid shape");
    }
    if (weights.lm_head.shape() !=
        std::vector<std::size_t>{config.hidden_size, config.vocab_size}) {
        throw std::invalid_argument("LM head has an invalid shape");
    }
}

} // namespace

Tensor embedding_lookup(
    const std::vector<TokenId>& tokens,
    const Tensor& embedding_table) {
    if (embedding_table.ndim() != 2) {
        throw std::invalid_argument("embedding table must be a 2D tensor");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("token sequence cannot be empty");
    }

    const std::size_t vocab_size = embedding_table.shape()[0];
    const std::size_t hidden_size = embedding_table.shape()[1];
    Tensor hidden({tokens.size(), hidden_size});

    for (std::size_t token_index = 0; token_index < tokens.size(); ++token_index) {
        const TokenId token = tokens[token_index];
        if (token >= vocab_size) {
            throw std::out_of_range("token id is outside the embedding table");
        }
        for (std::size_t hidden_index = 0; hidden_index < hidden_size; ++hidden_index) {
            hidden.at(token_index, hidden_index) = embedding_table.at(token, hidden_index);
        }
    }

    return hidden;
}

Tensor model_forward(
    const std::vector<TokenId>& tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config) {
    validate_model(weights, config);

    Tensor hidden = embedding_lookup(tokens, weights.token_embeddings);
    const LlamaBlockConfig block_config = block_config_from(config);

    for (const LlamaBlockWeights& block : weights.blocks) {
        hidden = llama_block_forward(hidden, block, block_config, 0);
    }

    const Tensor normalized = rms_norm(hidden, weights.output_norm, config.rms_epsilon);
    return matmul(normalized, weights.lm_head);
}

TokenId greedy_next_token(const Tensor& logits) {
    if (logits.ndim() != 2 || logits.shape()[0] == 0 || logits.shape()[1] == 0) {
        throw std::invalid_argument("logits must be a non-empty 2D tensor");
    }

    const std::size_t last_row = logits.shape()[0] - 1;
    TokenId best_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();

    for (TokenId token = 0; token < logits.shape()[1]; ++token) {
        const float candidate = logits.at(last_row, token);
        if (candidate > best_logit) {
            best_logit = candidate;
            best_token = token;
        }
    }

    return best_token;
}

GenerationResult generate_greedy_without_kv_cache(
    const std::vector<TokenId>& prompt,
    const std::size_t max_new_tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config) {
    if (prompt.empty()) {
        throw std::invalid_argument("generation prompt cannot be empty");
    }

    GenerationResult result{
        .tokens = prompt,
        .forward_passes = 0,
        .tokens_processed = 0,
    };

    for (std::size_t step = 0; step < max_new_tokens; ++step) {
        result.tokens_processed += result.tokens.size();
        const Tensor logits = model_forward(result.tokens, weights, config);
        result.tokens.push_back(greedy_next_token(logits));
        ++result.forward_passes;
    }

    return result;
}

} // namespace mini_llama
