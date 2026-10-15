#include "mini_llama/prompt_reuse.h"

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

mini_llama::LlamaModelConfig model_config() {
    return mini_llama::LlamaModelConfig{
        .vocab_size = 32,
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

mini_llama::LlamaModelWeights make_weights() {
    const mini_llama::LlamaModelConfig config = model_config();
    mini_llama::Tensor embeddings({config.vocab_size, config.hidden_size});
    for (std::size_t token = 0; token < config.vocab_size; ++token) {
        embeddings.at(token, 0) = 0.25F + 0.03125F * static_cast<float>(token);
        embeddings.at(token, 1) = 1.25F - 0.015625F * static_cast<float>(token);
    }

    mini_llama::Tensor lm_head({config.hidden_size, config.vocab_size});
    for (std::size_t token = 0; token < config.vocab_size; ++token) {
        lm_head.at(0, token) = static_cast<float>((token % 7) + 1) / 8.0F;
        lm_head.at(1, token) = static_cast<float>((token % 5) + 1) / 6.0F;
    }

    std::vector<mini_llama::LlamaBlockWeights> blocks;
    blocks.push_back(mini_llama::LlamaBlockWeights{
        .attention_norm = mini_llama::Tensor({2}, {1.0F, 1.0F}),
        .wq = identity_matrix(2),
        .wk = identity_matrix(2),
        .wv = identity_matrix(2),
        .wo = identity_matrix(2),
        .ffn_norm = mini_llama::Tensor({2}, {1.0F, 1.0F}),
        .w_gate = mini_llama::Tensor({2, 2}),
        .w_up = mini_llama::Tensor({2, 2}),
        .w_down = mini_llama::Tensor({2, 2}),
    });

    return mini_llama::LlamaModelWeights{
        .token_embeddings = std::move(embeddings),
        .blocks = std::move(blocks),
        .output_norm = mini_llama::Tensor({2}, {1.0F, 1.0F}),
        .lm_head = std::move(lm_head),
    };
}

void require_suffix_logits_match(
    const mini_llama::Tensor& suffix_logits,
    const mini_llama::Tensor& full_logits,
    const std::size_t first_full_row,
    const std::string& message) {
    require(suffix_logits.shape()[1] == full_logits.shape()[1], message + ": width differs");
    require(first_full_row + suffix_logits.shape()[0] == full_logits.shape()[0],
            message + ": row count differs");

    for (std::size_t row = 0; row < suffix_logits.shape()[0]; ++row) {
        for (std::size_t col = 0; col < suffix_logits.shape()[1]; ++col) {
            if (std::abs(suffix_logits.at(row, col) -
                         full_logits.at(first_full_row + row, col)) > 1.0e-5F) {
                throw std::runtime_error(message);
            }
        }
    }
}

void test_longest_common_prefix() {
    require(mini_llama::longest_common_prefix({1, 10, 20}, {1, 2, 7}) == 1,
            "user example should have one shared prefix token");
    require(mini_llama::longest_common_prefix({1, 10}, {1, 10, 30}) == 2,
            "two equal initial tokens should produce prefix length two");
    require(mini_llama::longest_common_prefix({5, 1}, {9, 1}) == 0,
            "equal later tokens are not a common prefix");
}

void test_user_example_reuses_one_cell_automatically() {
    const mini_llama::LlamaModelConfig config = model_config();
    const mini_llama::LlamaModelWeights weights = make_weights();
    mini_llama::CellKVCache cache(config, 16);
    mini_llama::PromptReuseScheduler scheduler;

    const mini_llama::PromptRunResult req1 =
        scheduler.run(0, {1, 10, 20}, weights, config, cache);
    require(!req1.plan.source_sequence.has_value(), "first request cannot reuse a prefix");
    require(req1.plan.tokens_to_compute == 3, "first request should compute all three tokens");

    const mini_llama::PromptRunResult req2 =
        scheduler.run(1, {1, 2, 7}, weights, config, cache);
    require(req2.plan.source_sequence == std::optional<mini_llama::SequenceId>(0),
            "second request should reuse sequence 0");
    require(req2.plan.matched_tokens == 1 && req2.plan.shared_tokens == 1,
            "second request should share exactly token 1 at position zero");
    require(req2.plan.tokens_to_compute == 2,
            "second request should compute only tokens 2 and 7");

    const mini_llama::Tensor reference =
        mini_llama::model_forward({1, 2, 7}, weights, config);
    require_suffix_logits_match(req2.logits, reference, 1,
                                "reused request logits differ from full recomputation");

    require(cache.used_cells() == 5,
            "two three-token requests sharing one prefix should use five cells");
    require(cache.cells_for_sequence(0).size() + cache.cells_for_sequence(1).size() == 6,
            "both sequences should logically contain three cells");
    require(cache.kv_rows_written() == 5,
            "one shared prefix cell should avoid one physical KV row write");

    const std::vector<std::size_t> prefix_cells = cache.cells_for_sequence(0, 0);
    require(prefix_cells.size() == 1 && cache.sequence_count(prefix_cells.front()) == 2,
            "position zero cell should belong to both requests");
}

void test_same_token_without_same_prefix_is_not_reused() {
    const mini_llama::LlamaModelConfig config = model_config();
    const mini_llama::LlamaModelWeights weights = make_weights();
    mini_llama::CellKVCache cache(config, 16);
    mini_llama::PromptReuseScheduler scheduler;

    (void) scheduler.run(0, {5, 1}, weights, config, cache);
    const mini_llama::PromptRunResult second =
        scheduler.run(1, {9, 1}, weights, config, cache);

    require(second.plan.matched_tokens == 0 && second.plan.shared_tokens == 0,
            "matching token 1 at position one must not trigger prefix reuse");
    require(second.plan.tokens_to_compute == 2,
            "request with a different prefix must compute both tokens");
    require(cache.used_cells() == 4, "unrelated two-token requests should use four cells");
}

void test_full_match_recomputes_last_token_for_logits_and_release_is_safe() {
    const mini_llama::LlamaModelConfig config = model_config();
    const mini_llama::LlamaModelWeights weights = make_weights();
    mini_llama::CellKVCache cache(config, 16);
    mini_llama::PromptReuseScheduler scheduler;

    (void) scheduler.run(0, {1, 10, 20}, weights, config, cache);
    const mini_llama::PromptRunResult identical =
        scheduler.run(1, {1, 10, 20}, weights, config, cache);

    require(identical.plan.matched_tokens == 3, "all three prompt tokens should match");
    require(identical.plan.shared_tokens == 2,
            "the final token should be recomputed to produce prompt logits");
    require(identical.plan.tokens_to_compute == 1,
            "an identical prompt should compute only its last token");

    const mini_llama::Tensor reference =
        mini_llama::model_forward({1, 10, 20}, weights, config);
    require_suffix_logits_match(identical.logits, reference, 2,
                                "identical prompt's final logits differ from full recomputation");
    require(cache.used_cells() == 4,
            "three original cells plus one recomputed final cell should be used");

    scheduler.release(0, cache);
    require(scheduler.active_requests().size() == 1,
            "release should remove sequence 0 from scheduler state");
    require(cache.cells_for_sequence(0).empty(), "released sequence should own no cells");
    require(cache.cells_for_sequence(1).size() == 3,
            "sequence 1 should keep two shared cells and its private final cell");
    require(cache.used_cells() == 3,
            "releasing sequence 0 should preserve sequence 1 and free one private cell");
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
        test_longest_common_prefix();
        test_user_example_reuses_one_cell_automatically();
        test_same_token_without_same_prefix_is_not_reused();
        test_full_match_recomputes_last_token_for_logits_and_release_is_safe();

        const mini_llama::LlamaModelConfig config = model_config();
        const mini_llama::LlamaModelWeights weights = make_weights();
        mini_llama::CellKVCache cache(config, 16);
        mini_llama::PromptReuseScheduler scheduler;
        const std::vector<mini_llama::TokenId> tokens1{1, 10, 20};
        const std::vector<mini_llama::TokenId> tokens2{1, 2, 7};
        const mini_llama::PromptRunResult req1 =
            scheduler.run(0, tokens1, weights, config, cache);
        const mini_llama::PromptRunResult req2 =
            scheduler.run(1, tokens2, weights, config, cache);

        std::cout << "stage06_prompt_reuse: all tests passed\nreq1 ";
        print_tokens(tokens1);
        std::cout << ": computed " << req1.plan.tokens_to_compute << " tokens\nreq2 ";
        print_tokens(tokens2);
        std::cout << ": matched " << req2.plan.matched_tokens
                  << ", shared " << req2.plan.shared_tokens
                  << ", computed " << req2.plan.tokens_to_compute << " tokens\n";
        std::cout << "Logical sequence cells: "
                  << cache.cells_for_sequence(0).size() + cache.cells_for_sequence(1).size()
                  << ", physical cells: " << cache.used_cells()
                  << ", KV rows written: " << cache.kv_rows_written() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage06_prompt_reuse: FAILED: " << error.what() << '\n';
        return 1;
    }
}
