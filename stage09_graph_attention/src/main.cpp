#include "mini_llama/graph_attention.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mini_llama::GraphKVCache;
using mini_llama::KVStoreNodes;
using mini_llama::ggml::Context;
using mini_llama::ggml::Graph;
using mini_llama::ggml::Op;
using mini_llama::ggml::Tensor;

constexpr float tolerance = 1.0e-5F;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(const float actual, const float expected, const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
}

std::array<float, 2> rotate_pair(
    const std::array<float, 2>& input,
    const std::int32_t position) {
    const float angle = static_cast<float>(position);
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    return {
        input[0] * cosine - input[1] * sine,
        input[0] * sine + input[1] * cosine,
    };
}

std::array<float, 2> reference_attention(
    const std::array<float, 2>& query,
    const GraphKVCache& cache,
    const std::vector<std::size_t>& visible_cells) {
    const float scale = 1.0F / std::sqrt(2.0F);
    std::vector<float> scores;
    scores.reserve(visible_cells.size());

    float maximum = -std::numeric_limits<float>::infinity();
    for (const std::size_t cell : visible_cells) {
        const float score = scale * (
            query[0] * cache.key()->at(0, cell) +
            query[1] * cache.key()->at(1, cell));
        scores.push_back(score);
        maximum = std::max(maximum, score);
    }

    float denominator = 0.0F;
    for (float& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
    }

    std::array<float, 2> result{0.0F, 0.0F};
    for (std::size_t i = 0; i < visible_cells.size(); ++i) {
        const float probability = scores[i] / denominator;
        const std::size_t cell = visible_cells[i];
        result[0] += probability * cache.value()->at(0, cell);
        result[1] += probability * cache.value()->at(1, cell);
    }
    return result;
}

struct IdentityWeights {
    Context context;
    Tensor* wq = nullptr;
    Tensor* wk = nullptr;
    Tensor* wv = nullptr;

    IdentityWeights() {
        wq = context.new_tensor_2d(2, 2, "wq");
        wk = context.new_tensor_2d(2, 2, "wk");
        wv = context.new_tensor_2d(2, 2, "wv");
        mini_llama::ggml::set_values(wq, {1.0F, 0.0F, 0.0F, 1.0F});
        mini_llama::ggml::set_values(wk, {1.0F, 0.0F, 0.0F, 1.0F});
        mini_llama::ggml::set_values(wv, {1.0F, 0.0F, 0.0F, 1.0F});
    }
};

void require_writes_before_attention(const Graph& graph) {
    std::vector<std::size_t> writes;
    std::size_t attention = graph.nodes.size();
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        if (graph.nodes[i]->op == Op::set_rows) {
            writes.push_back(i);
        } else if (graph.nodes[i]->op == Op::attention) {
            attention = i;
        }
    }
    require(writes.size() == 2, "graph must contain K and V SET_ROWS nodes");
    require(attention < graph.nodes.size(), "graph must contain an Attention node");
    require(writes[0] < attention && writes[1] < attention,
            "both cache writes must execute before Attention");
}

void run_prefill(
    const IdentityWeights& weights,
    GraphKVCache& cache) {
    Context graph_context;
    Tensor* input = graph_context.new_tensor_2d(2, 2, "prefill_input");
    mini_llama::ggml::set_values(input, {
        1.0F, 0.0F,
        0.0F, 1.0F,
    });

    Tensor* positions = graph_context.new_i32_tensor_1d(2, "prefill_positions");
    mini_llama::ggml::set_i32_values(positions, {0, 1});
    Tensor* cell_indices = graph_context.new_i32_tensor_1d(2, "prefill_cells");
    mini_llama::ggml::set_i32_values(cell_indices, {2, 5});

    Tensor* q_raw = graph_context.mul_mat(weights.wq, input, "prefill_q_raw");
    Tensor* k_raw = graph_context.mul_mat(weights.wk, input, "prefill_k_raw");
    Tensor* v_cur = graph_context.mul_mat(weights.wv, input, "prefill_v");
    Tensor* q_cur = graph_context.rope(q_raw, positions, 10000.0F, "prefill_q_rope");
    Tensor* k_cur = graph_context.rope(k_raw, positions, 10000.0F, "prefill_k_rope");

    const KVStoreNodes stores =
        cache.build_store_nodes(graph_context, k_cur, v_cur, cell_indices);
    Tensor* mask = mini_llama::build_attention_mask(
        graph_context, cache.capacity(), {{2}, {2, 5}}, "prefill_mask");
    Tensor* output = graph_context.attention(
        q_cur, cache.key(), cache.value(), mask, "prefill_attention");

    // 建图阶段只描述写操作，cache 还没有被修改。
    require_close(cache.key()->at(0, 2), 0.0F,
                  "SET_ROWS must stay lazy before graph_compute");

    Graph graph;
    mini_llama::build_kv_attention_graph(graph, stores, output);
    require_writes_before_attention(graph);
    mini_llama::ggml::graph_compute(graph);

    const auto key0 = rotate_pair({1.0F, 0.0F}, 0);
    const auto key1 = rotate_pair({0.0F, 1.0F}, 1);
    require_close(cache.key()->at(0, 2), key0[0], "prefill K cell 2 is wrong");
    require_close(cache.key()->at(1, 2), key0[1], "prefill K cell 2 is wrong");
    require_close(cache.key()->at(0, 5), key1[0], "prefill K cell 5 is wrong");
    require_close(cache.key()->at(1, 5), key1[1], "prefill K cell 5 is wrong");
    require_close(cache.value()->at(0, 2), 1.0F, "prefill V cell 2 is wrong");
    require_close(cache.value()->at(1, 5), 1.0F, "prefill V cell 5 is wrong");

    const auto expected0 = reference_attention(key0, cache, {2});
    const auto expected1 = reference_attention(key1, cache, {2, 5});
    require_close(output->at(0, 0), expected0[0], "prefill output token 0 is wrong");
    require_close(output->at(1, 0), expected0[1], "prefill output token 0 is wrong");
    require_close(output->at(0, 1), expected1[0], "prefill output token 1 is wrong");
    require_close(output->at(1, 1), expected1[1], "prefill output token 1 is wrong");

    std::cout << "prefill: logical positions [0, 1] -> physical cells [2, 5]\n";
    std::cout << "prefill output token 1 = [" << output->at(0, 1)
              << ", " << output->at(1, 1) << "]\n";
}

void run_decode(
    const IdentityWeights& weights,
    GraphKVCache& cache) {
    Context graph_context;
    Tensor* input = graph_context.new_tensor_2d(2, 1, "decode_input");
    mini_llama::ggml::set_values(input, {1.0F, 1.0F});

    Tensor* positions = graph_context.new_i32_tensor_1d(1, "decode_position");
    mini_llama::ggml::set_i32_values(positions, {2});
    Tensor* cell_indices = graph_context.new_i32_tensor_1d(1, "decode_cell");
    mini_llama::ggml::set_i32_values(cell_indices, {1});

    Tensor* q_raw = graph_context.mul_mat(weights.wq, input, "decode_q_raw");
    Tensor* k_raw = graph_context.mul_mat(weights.wk, input, "decode_k_raw");
    Tensor* v_cur = graph_context.mul_mat(weights.wv, input, "decode_v");
    Tensor* q_cur = graph_context.rope(q_raw, positions, 10000.0F, "decode_q_rope");
    Tensor* k_cur = graph_context.rope(k_raw, positions, 10000.0F, "decode_k_rope");

    const KVStoreNodes stores =
        cache.build_store_nodes(graph_context, k_cur, v_cur, cell_indices);
    Tensor* mask = mini_llama::build_attention_mask(
        graph_context, cache.capacity(), {{2, 5, 1}}, "decode_mask");
    Tensor* output = graph_context.attention(
        q_cur, cache.key(), cache.value(), mask, "decode_attention");

    Graph graph;
    mini_llama::build_kv_attention_graph(graph, stores, output);
    require_writes_before_attention(graph);
    mini_llama::ggml::graph_compute(graph);

    const auto query = rotate_pair({1.0F, 1.0F}, 2);
    const auto expected = reference_attention(query, cache, {2, 5, 1});
    require_close(output->at(0, 0), expected[0], "decode output is wrong");
    require_close(output->at(1, 0), expected[1], "decode output is wrong");

    const auto key2 = rotate_pair({1.0F, 1.0F}, 2);
    require_close(cache.key()->at(0, 1), key2[0], "decode K cell 1 is wrong");
    require_close(cache.key()->at(1, 1), key2[1], "decode K cell 1 is wrong");
    require_close(cache.value()->at(0, 1), 1.0F, "decode V cell 1 is wrong");
    require_close(cache.value()->at(1, 1), 1.0F, "decode V cell 1 is wrong");

    // 上一轮 graph_context 已销毁，但持久 cache 中的历史仍然存在。
    require_close(cache.value()->at(0, 2), 1.0F, "historical cell 2 was lost");
    require_close(cache.value()->at(1, 5), 1.0F, "historical cell 5 was lost");

    std::cout << "decode: logical positions [0, 1, 2] -> physical cells [2, 5, 1]\n";
    std::cout << "decode output = [" << output->at(0, 0)
              << ", " << output->at(1, 0) << "]\n";
}

void test_mask_validation() {
    Context context;
    bool empty_row_rejected = false;
    try {
        (void) mini_llama::build_attention_mask(context, 4, {{}});
    } catch (const std::invalid_argument&) {
        empty_row_rejected = true;
    }
    require(empty_row_rejected, "an empty visibility row must be rejected");

    bool out_of_range_rejected = false;
    try {
        (void) mini_llama::build_attention_mask(context, 4, {{4}});
    } catch (const std::out_of_range&) {
        out_of_range_rejected = true;
    }
    require(out_of_range_rejected, "an out-of-range cell must be rejected");
}

} // namespace

int main() {
    try {
        IdentityWeights weights;
        GraphKVCache cache(2, 6);
        run_prefill(weights, cache);
        run_decode(weights, cache);
        test_mask_validation();
        std::cout << "stage09 graph Attention tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage09 failure: " << error.what() << '\n';
        return 1;
    }
}
