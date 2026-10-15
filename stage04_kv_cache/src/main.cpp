#include "mini_llama/kv_cache.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

mini_llama::Tensor identity_matrix(const std::size_t size) {
    mini_llama::Tensor result({size, size});
    for (std::size_t i = 0; i < size; ++i) {
        result.at(i, i) = 1.0F;
    }
    return result;
}

mini_llama::LlamaModelConfig attention_config() {
    return mini_llama::LlamaModelConfig{
        .vocab_size = 3,
        .hidden_size = 2,
        .layer_count = 1,
        .query_heads = 1,
        .kv_heads = 1,
        .head_dim = 2,
        .intermediate_size = 2,
        .rms_epsilon = 1.0e-5F,
        .rope_theta = 10000.0F,
    };
}

mini_llama::LlamaModelWeights make_nonzero_attention_weights() {
    mini_llama::LlamaBlockWeights block{
        .attention_norm = mini_llama::Tensor({2}, {1.0F, 1.0F}),
        .wq = identity_matrix(2),
        .wk = identity_matrix(2),
        .wv = identity_matrix(2),
        .wo = identity_matrix(2),
        .ffn_norm = mini_llama::Tensor({2}, {1.0F, 1.0F}),
        .w_gate = mini_llama::Tensor({2, 2}),
        .w_up = mini_llama::Tensor({2, 2}),
        .w_down = mini_llama::Tensor({2, 2}),
    };

    std::vector<mini_llama::LlamaBlockWeights> blocks;
    blocks.push_back(std::move(block));
    return mini_llama::LlamaModelWeights{
        .token_embeddings = mini_llama::Tensor({3, 2}, {
            1.0F, 0.0F,
            0.0F, 1.0F,
            1.0F, 1.0F,
        }),
        .blocks = std::move(blocks),
        .output_norm = mini_llama::Tensor({2}, {1.0F, 1.0F}),
        .lm_head = mini_llama::Tensor({2, 3}, {
            1.0F, 0.0F, 0.5F,
            0.0F, 1.0F, 0.5F,
        }),
    };
}

mini_llama::LlamaModelConfig cyclic_config() {
    return mini_llama::LlamaModelConfig{
        .vocab_size = 4,
        .hidden_size = 4,
        .layer_count = 1,
        .query_heads = 2,
        .kv_heads = 1,
        .head_dim = 2,
        .intermediate_size = 6,
        .rms_epsilon = 1.0e-5F,
        .rope_theta = 10000.0F,
    };
}

mini_llama::LlamaModelWeights make_cyclic_weights() {
    std::vector<mini_llama::LlamaBlockWeights> blocks;
    blocks.push_back(mini_llama::LlamaBlockWeights{
        .attention_norm = mini_llama::Tensor({4}, {1.0F, 1.0F, 1.0F, 1.0F}),
        .wq = mini_llama::Tensor({4, 4}),
        .wk = mini_llama::Tensor({4, 2}),
        .wv = mini_llama::Tensor({4, 2}),
        .wo = mini_llama::Tensor({4, 4}),
        .ffn_norm = mini_llama::Tensor({4}, {1.0F, 1.0F, 1.0F, 1.0F}),
        .w_gate = mini_llama::Tensor({4, 6}),
        .w_up = mini_llama::Tensor({4, 6}),
        .w_down = mini_llama::Tensor({6, 4}),
    });

    return mini_llama::LlamaModelWeights{
        .token_embeddings = mini_llama::Tensor({4, 4}, {
            1.0F, 0.0F, 0.0F, 0.0F,
            0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F,
        }),
        .blocks = std::move(blocks),
        .output_norm = mini_llama::Tensor({4}, {1.0F, 1.0F, 1.0F, 1.0F}),
        .lm_head = mini_llama::Tensor({4, 4}, {
            0.0F, 1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 1.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F,
            1.0F, 0.0F, 0.0F, 0.0F,
        }),
    };
}

void test_layer_cache_append() {
    mini_llama::LayerKVCache cache(4, 2);
    cache.append(
        mini_llama::Tensor({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F}),
        mini_llama::Tensor({2, 2}, {5.0F, 6.0F, 7.0F, 8.0F}));

    require(cache.size() == 2, "cache append did not update size");
    require(cache.key_at(1, 0) == 3.0F, "cached K value is wrong");
    require(cache.value_at(0, 1) == 6.0F, "cached V value is wrong");

    cache.clear();
    require(cache.size() == 0, "cache clear did not reset the used size");
}

void test_incremental_logits_match_full_recomputation() {
    const mini_llama::LlamaModelConfig config = attention_config();
    const mini_llama::LlamaModelWeights weights = make_nonzero_attention_weights();

    // 参考结果：一次完整计算三个 token，Attention 权重为非零。
    const mini_llama::Tensor full_logits =
        mini_llama::model_forward({0, 1, 2}, weights, config);

    // 增量结果：先 prefill 两个 token，再 decode 第三个 token。
    mini_llama::ModelKVCache cache(config, 8);
    const mini_llama::Tensor prefill_logits =
        mini_llama::model_forward_with_kv_cache({0, 1}, weights, config, cache);
    const mini_llama::Tensor decode_logits =
        mini_llama::model_forward_with_kv_cache({2}, weights, config, cache);

    require(cache.tokens_cached() == 3, "incremental forward should cache three tokens");
    for (std::size_t token = 0; token < 2; ++token) {
        for (std::size_t vocab = 0; vocab < config.vocab_size; ++vocab) {
            require(std::abs(prefill_logits.at(token, vocab) - full_logits.at(token, vocab)) <
                        1.0e-5F,
                    "prefill logits differ from full recomputation");
        }
    }
    for (std::size_t vocab = 0; vocab < config.vocab_size; ++vocab) {
        require(std::abs(decode_logits.at(0, vocab) - full_logits.at(2, vocab)) < 1.0e-5F,
                "decode logits differ from full recomputation");
    }
}

void test_cached_generation_matches_and_avoids_recomputation() {
    const mini_llama::LlamaModelConfig config = cyclic_config();
    const mini_llama::LlamaModelWeights weights = make_cyclic_weights();

    const mini_llama::GenerationResult without_cache =
        mini_llama::generate_greedy_without_kv_cache({0}, 8, weights, config);

    mini_llama::ModelKVCache cache(config, 16);
    const mini_llama::GenerationResult with_cache =
        mini_llama::generate_greedy_with_kv_cache({0}, 8, weights, config, cache);

    require(with_cache.tokens == without_cache.tokens,
            "cached and uncached generation produced different tokens");
    require(without_cache.tokens_processed == 36,
            "uncached reference should process 1+...+8=36 tokens");
    require(with_cache.tokens_processed == 8,
            "cached generation should process the prompt once and then one token per step");
    require(cache.tokens_cached() == 8,
            "the last emitted token is not cached until another decode step needs it");
}

void print_tokens(const std::vector<mini_llama::TokenId>& tokens) {
    std::cout << '[';
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << tokens[i];
    }
    std::cout << ']';
}

} // namespace

int main() {
    try {
        test_layer_cache_append();
        test_incremental_logits_match_full_recomputation();
        test_cached_generation_matches_and_avoids_recomputation();

        const mini_llama::LlamaModelConfig config = cyclic_config();
        const mini_llama::LlamaModelWeights weights = make_cyclic_weights();
        const mini_llama::GenerationResult without_cache =
            mini_llama::generate_greedy_without_kv_cache({0}, 8, weights, config);
        mini_llama::ModelKVCache cache(config, 16);
        const mini_llama::GenerationResult with_cache =
            mini_llama::generate_greedy_with_kv_cache({0}, 8, weights, config, cache);

        std::cout << "stage04_kv_cache: all tests passed\nGenerated token ids: ";
        print_tokens(with_cache.tokens);
        std::cout << "\nTokens processed without cache: " << without_cache.tokens_processed
                  << "\nTokens processed with cache:    " << with_cache.tokens_processed
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage04_kv_cache: FAILED: " << error.what() << '\n';
        return 1;
    }
}
