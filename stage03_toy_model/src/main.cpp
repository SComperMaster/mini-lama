#include "mini_llama/model.h"

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

mini_llama::LlamaBlockWeights make_zero_block() {
    return mini_llama::LlamaBlockWeights{
        .attention_norm = mini_llama::Tensor({4}, {1.0F, 1.0F, 1.0F, 1.0F}),
        .wq = mini_llama::Tensor({4, 4}),
        .wk = mini_llama::Tensor({4, 2}),
        .wv = mini_llama::Tensor({4, 2}),
        .wo = mini_llama::Tensor({4, 4}),
        .ffn_norm = mini_llama::Tensor({4}, {1.0F, 1.0F, 1.0F, 1.0F}),
        .w_gate = mini_llama::Tensor({4, 6}),
        .w_up = mini_llama::Tensor({4, 6}),
        .w_down = mini_llama::Tensor({6, 4}),
    };
}

mini_llama::LlamaModelWeights make_cyclic_toy_weights() {
    // embedding 是单位矩阵：token 0/1/2/3 分别对应四个 one-hot 隐状态。
    mini_llama::Tensor embeddings({4, 4}, {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
    });

    // 当前 token=i 时，让 token=(i+1)%4 的 logit 最大。
    mini_llama::Tensor lm_head({4, 4}, {
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F,
        1.0F, 0.0F, 0.0F, 0.0F,
    });

    std::vector<mini_llama::LlamaBlockWeights> blocks;
    blocks.push_back(make_zero_block());

    return mini_llama::LlamaModelWeights{
        .token_embeddings = std::move(embeddings),
        .blocks = std::move(blocks),
        .output_norm = mini_llama::Tensor({4}, {1.0F, 1.0F, 1.0F, 1.0F}),
        .lm_head = std::move(lm_head),
    };
}

mini_llama::LlamaModelConfig toy_config() {
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

void test_embedding_lookup() {
    const mini_llama::Tensor table({3, 2}, {
        1.0F, 2.0F,
        3.0F, 4.0F,
        5.0F, 6.0F,
    });
    const mini_llama::Tensor expected({3, 2}, {
        5.0F, 6.0F,
        1.0F, 2.0F,
        5.0F, 6.0F,
    });

    const mini_llama::Tensor actual = mini_llama::embedding_lookup({2, 0, 2}, table);
    require(mini_llama::all_close(actual, expected),
            "embedding lookup test failed:\n" + actual.debug_string());
}

void test_full_model_logits() {
    const mini_llama::LlamaModelConfig config = toy_config();
    const mini_llama::LlamaModelWeights weights = make_cyclic_toy_weights();
    const mini_llama::Tensor logits = mini_llama::model_forward({0, 1, 2, 3}, weights, config);

    require(logits.shape() == std::vector<std::size_t>{4, 4},
            "full model returned an invalid logits shape");
    require(mini_llama::greedy_next_token(logits) == 0,
            "token 3 should predict token 0");

    // 验证每一个位置，而不只验证最后一行。
    for (std::size_t row = 0; row < 4; ++row) {
        std::size_t best = 0;
        for (std::size_t token = 1; token < 4; ++token) {
            if (logits.at(row, token) > logits.at(row, best)) {
                best = token;
            }
        }
        require(best == (row + 1) % 4, "cyclic LM head prediction is wrong");
    }
}

void test_greedy_generation_and_recomputation_count() {
    const mini_llama::LlamaModelConfig config = toy_config();
    const mini_llama::LlamaModelWeights weights = make_cyclic_toy_weights();

    const mini_llama::GenerationResult result =
        mini_llama::generate_greedy_without_kv_cache({0}, 5, weights, config);
    const std::vector<mini_llama::TokenId> expected{0, 1, 2, 3, 0, 1};

    require(result.tokens == expected, "greedy generation produced the wrong token sequence");
    require(result.forward_passes == 5, "generation should execute five forward passes");

    // 五次 forward 分别处理 1、2、3、4、5 个 token。
    require(result.tokens_processed == 15,
            "without KV Cache, processed token count should be 1+2+3+4+5=15");
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
        test_embedding_lookup();
        test_full_model_logits();
        test_greedy_generation_and_recomputation_count();

        const mini_llama::LlamaModelConfig config = toy_config();
        const mini_llama::LlamaModelWeights weights = make_cyclic_toy_weights();
        const mini_llama::GenerationResult demo =
            mini_llama::generate_greedy_without_kv_cache({0}, 8, weights, config);

        std::cout << "stage03_toy_model: all tests passed\n";
        std::cout << "Generated token ids: ";
        print_tokens(demo.tokens);
        std::cout << "\nForward passes: " << demo.forward_passes
                  << ", tokens processed without KV Cache: " << demo.tokens_processed << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage03_toy_model: FAILED: " << error.what() << '\n';
        return 1;
    }
}
