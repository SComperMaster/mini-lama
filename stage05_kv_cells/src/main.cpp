#include "mini_llama/kv_cells.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
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

mini_llama::LlamaModelWeights make_nonzero_weights() {
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

void require_row_close(
    const mini_llama::Tensor& actual,
    const std::size_t actual_row,
    const mini_llama::Tensor& expected,
    const std::size_t expected_row,
    const std::string& message) {
    require(actual.shape()[1] == expected.shape()[1], message + ": row widths differ");
    for (std::size_t col = 0; col < actual.shape()[1]; ++col) {
        if (std::abs(actual.at(actual_row, col) - expected.at(expected_row, col)) > 1.0e-5F) {
            throw std::runtime_error(message);
        }
    }
}

void test_ring_search_finds_non_contiguous_holes() {
    const mini_llama::LlamaModelConfig config = model_config();
    mini_llama::CellKVCache cache(config, 6);

    const mini_llama::CellSlotInfo initial = cache.allocate(0, {0, 1, 2, 3, 4});
    require(initial.cell_indices == std::vector<std::size_t>({0, 1, 2, 3, 4}),
            "initial allocation should use the first five cells");
    require(cache.search_head() == 5, "ring search head should follow the last allocation");

    cache.seq_remove(0, 1, 2);
    cache.seq_remove(0, 3, 4);
    require(cache.search_head() == 1, "freeing an earlier cell should move the search head");

    const mini_llama::CellSlotInfo reused = cache.allocate(0, {5, 6, 7});
    require(reused.cell_indices == std::vector<std::size_t>({1, 3, 5}),
            "allocator should gather non-contiguous holes while scanning the ring");
    require(cache.used_cells() == 6, "all physical cells should now be occupied");
    require(cache.used_max_plus_one() == 6, "used high-water mark should cover all cells");
}

void test_zero_copy_prefix_branching_and_sequence_mask() {
    const mini_llama::LlamaModelConfig config = model_config();
    const mini_llama::LlamaModelWeights weights = make_nonzero_weights();
    mini_llama::CellKVCache cache(config, 8);

    const mini_llama::Tensor prefix_logits =
        mini_llama::model_forward_with_kv_cells({0, 1}, 0, weights, config, cache);
    const mini_llama::Tensor prefix_reference =
        mini_llama::model_forward({0, 1}, weights, config);
    require(mini_llama::all_close(prefix_logits, prefix_reference, 1.0e-5F, 1.0e-5F),
            "cell prefill logits differ from full recomputation");

    const std::vector<std::size_t> shared_cells = cache.cells_for_sequence(0);
    require(shared_cells.size() == 2, "prefix should occupy two physical cells");
    const std::size_t writes_before_copy = cache.kv_rows_written();

    const std::size_t copied = cache.seq_copy(0, 1, 0, 2);
    require(copied == 2, "two prefix cells should be shared with sequence 1");
    require(cache.used_cells() == 2, "seq_copy must not allocate new physical cells");
    require(cache.kv_rows_written() == writes_before_copy,
            "seq_copy must not copy or rewrite K/V rows");
    for (const std::size_t cell : shared_cells) {
        require(cache.sequence_count(cell) == 2,
                "shared prefix cell should belong to both sequences");
    }

    // 两个 sequence 从同一前缀分叉，在相同逻辑位置 2 写入不同 token。
    const mini_llama::Tensor branch_zero =
        mini_llama::model_forward_with_kv_cells({2}, 0, weights, config, cache);
    const mini_llama::Tensor branch_one =
        mini_llama::model_forward_with_kv_cells({0}, 1, weights, config, cache);

    const mini_llama::Tensor reference_zero =
        mini_llama::model_forward({0, 1, 2}, weights, config);
    const mini_llama::Tensor reference_one =
        mini_llama::model_forward({0, 1, 0}, weights, config);
    require_row_close(branch_zero, 0, reference_zero, 2,
                      "sequence 0 branch logits differ from its full reference");
    require_row_close(branch_one, 0, reference_one, 2,
                      "sequence 1 branch saw the wrong sequence cells");

    require(cache.used_cells() == 4,
            "two shared prefix cells plus two private suffix cells should use four cells");
    require(cache.cells_for_sequence(0).size() == 3, "sequence 0 should see three cells");
    require(cache.cells_for_sequence(1).size() == 3, "sequence 1 should see three cells");

    // 删除 sequence 0：共享前缀继续由 sequence 1 持有，只有私有后缀真正释放。
    const std::size_t removed = cache.seq_remove(
        0, 0, std::numeric_limits<mini_llama::Position>::max());
    require(removed == 3, "sequence 0 should be removed from three cells");
    require(cache.cells_for_sequence(0).empty(), "sequence 0 should have no remaining cells");
    require(cache.cells_for_sequence(1).size() == 3,
            "removing sequence 0 must preserve sequence 1's shared prefix");
    require(cache.used_cells() == 3,
            "only sequence 0's private suffix cell should become empty");
}

void print_cell_map(const mini_llama::CellKVCache& cache) {
    for (std::size_t cell = 0; cell < cache.capacity(); ++cell) {
        std::cout << "cell " << cell << ": ";
        if (cache.is_empty(cell)) {
            std::cout << "empty\n";
            continue;
        }
        std::cout << "pos=" << cache.position(cell) << " seq={";
        bool first = true;
        for (std::size_t sequence = 0; sequence < 2; ++sequence) {
            if (cache.has_sequence(cell, sequence)) {
                if (!first) {
                    std::cout << ',';
                }
                std::cout << sequence;
                first = false;
            }
        }
        std::cout << "}\n";
    }
}

} // namespace

int main() {
    try {
        test_ring_search_finds_non_contiguous_holes();
        test_zero_copy_prefix_branching_and_sequence_mask();

        const mini_llama::LlamaModelConfig config = model_config();
        const mini_llama::LlamaModelWeights weights = make_nonzero_weights();
        mini_llama::CellKVCache cache(config, 8);
        (void) mini_llama::model_forward_with_kv_cells({0, 1}, 0, weights, config, cache);
        cache.seq_copy(0, 1, 0, 2);
        (void) mini_llama::model_forward_with_kv_cells({2}, 0, weights, config, cache);
        (void) mini_llama::model_forward_with_kv_cells({0}, 1, weights, config, cache);

        std::cout << "stage05_kv_cells: all tests passed\n";
        std::cout << "Cell map after two sequences branch from a shared prefix:\n";
        print_cell_map(cache);
        std::cout << "Used physical cells: " << cache.used_cells()
                  << ", logical sequence cells: "
                  << cache.cells_for_sequence(0).size() + cache.cells_for_sequence(1).size()
                  << ", KV rows written: " << cache.kv_rows_written() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage05_kv_cells: FAILED: " << error.what() << '\n';
        return 1;
    }
}
