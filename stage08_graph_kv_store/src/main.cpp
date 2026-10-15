#include "mini_llama/graph_kv_store.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using mini_llama::GraphKVCache;
using mini_llama::KVStoreNodes;
using mini_llama::ggml::Context;
using mini_llama::ggml::Graph;
using mini_llama::ggml::Tensor;

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(const float actual, const float expected, const std::string& message) {
    if (std::abs(actual - expected) > 1.0e-5F) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected));
    }
}

struct ProjectionWeights {
    Tensor* wq = nullptr;
    Tensor* wk = nullptr;
    Tensor* wv = nullptr;
};

ProjectionWeights make_projection_weights(Context& model_context) {
    ProjectionWeights weights;
    weights.wq = model_context.new_tensor_2d(3, 2, "wq");
    weights.wk = model_context.new_tensor_2d(3, 2, "wk");
    weights.wv = model_context.new_tensor_2d(3, 2, "wv");

    // q = [x0 + x2, x1 + x2]
    mini_llama::ggml::set_values(weights.wq, {
        1.0F, 0.0F, 1.0F,
        0.0F, 1.0F, 1.0F,
    });
    // k = [x0, x1]
    mini_llama::ggml::set_values(weights.wk, {
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
    });
    // v = [x2, x0 + x1 + x2]
    mini_llama::ggml::set_values(weights.wv, {
        0.0F, 0.0F, 1.0F,
        1.0F, 1.0F, 1.0F,
    });
    return weights;
}

void add_qkv_roots_and_compute(
    Graph& graph,
    Tensor* q_cur,
    const KVStoreNodes& store_nodes) {
    // llama.cpp 也会把 K/V 写入节点显式加入 forward graph。否则 Attention
    // 读取 cache 时，图中没有依赖边能保证写入一定发生。
    mini_llama::ggml::build_forward_expand(graph, store_nodes.key_write);
    mini_llama::ggml::build_forward_expand(graph, store_nodes.value_write);
    mini_llama::ggml::build_forward_expand(graph, q_cur);
    mini_llama::ggml::graph_compute(graph);
}

void test_temporary_qkv_write_persistent_cache() {
    // 模型权重和 KV Cache 都比一次 decode graph 活得更久。
    Context model_context;
    const ProjectionWeights weights = make_projection_weights(model_context);
    GraphKVCache cache(/*kv_width=*/2, /*capacity=*/6);

    float first_q0 = 0.0F;
    float first_q1 = 0.0F;

    {
        // 这个 Context 只服务于第一个 batch，离开作用域后 Q/K/V 临时节点都会销毁。
        Context graph_context;
        Tensor* input = graph_context.new_tensor_2d(3, 2, "prefill_input");
        mini_llama::ggml::set_values(input, {
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
        });

        Tensor* q_cur = graph_context.mul_mat(weights.wq, input, "q_cur");
        Tensor* k_cur = graph_context.mul_mat(weights.wk, input, "k_cur");
        Tensor* v_cur = graph_context.mul_mat(weights.wv, input, "v_cur");

        Tensor* cell_indices = graph_context.new_i32_tensor_1d(2, "cell_indices");
        mini_llama::ggml::set_i32_values(cell_indices, {2, 5});

        const KVStoreNodes stores =
            cache.build_store_nodes(graph_context, k_cur, v_cur, cell_indices);

        require_close(cache.key()->at(0, 2), 0.0F,
                      "building SET_ROWS must not write cache eagerly");

        Graph graph;
        add_qkv_roots_and_compute(graph, q_cur, stores);

        first_q0 = q_cur->at(0, 0);
        first_q1 = q_cur->at(1, 0);
        require_close(first_q0, 4.0F, "Q projection is wrong");
        require_close(first_q1, 5.0F, "Q projection is wrong");
    }

    // graph_context 已销毁，q_cur/k_cur/v_cur 不再存在；持久 KV 数据仍然存在。
    require_close(first_q0, 4.0F, "saved Q observation is wrong");
    require_close(first_q1, 5.0F, "saved Q observation is wrong");
    require_close(cache.key()->at(0, 2), 1.0F, "token 0 K was not stored in cell 2");
    require_close(cache.key()->at(1, 2), 2.0F, "token 0 K width is wrong");
    require_close(cache.key()->at(0, 5), 4.0F, "token 1 K was not stored in cell 5");
    require_close(cache.key()->at(1, 5), 5.0F, "token 1 K width is wrong");
    require_close(cache.value()->at(0, 2), 3.0F, "token 0 V was not stored in cell 2");
    require_close(cache.value()->at(1, 2), 6.0F, "token 0 V width is wrong");
    require_close(cache.value()->at(0, 5), 6.0F, "token 1 V was not stored in cell 5");
    require_close(cache.value()->at(1, 5), 15.0F, "token 1 V width is wrong");

    {
        // 第二次 decode 建一张全新的临时图，只写 cell 1。
        Context graph_context;
        Tensor* input = graph_context.new_tensor_2d(3, 1, "decode_input");
        mini_llama::ggml::set_values(input, {7.0F, 8.0F, 9.0F});

        Tensor* q_cur = graph_context.mul_mat(weights.wq, input, "q_cur");
        Tensor* k_cur = graph_context.mul_mat(weights.wk, input, "k_cur");
        Tensor* v_cur = graph_context.mul_mat(weights.wv, input, "v_cur");
        Tensor* cell_indices = graph_context.new_i32_tensor_1d(1, "cell_indices");
        mini_llama::ggml::set_i32_values(cell_indices, {1});

        const KVStoreNodes stores =
            cache.build_store_nodes(graph_context, k_cur, v_cur, cell_indices);
        Graph graph;
        add_qkv_roots_and_compute(graph, q_cur, stores);
    }

    require_close(cache.key()->at(0, 1), 7.0F, "decode K was not stored in cell 1");
    require_close(cache.key()->at(1, 1), 8.0F, "decode K width is wrong");
    require_close(cache.value()->at(0, 1), 9.0F, "decode V was not stored in cell 1");
    require_close(cache.value()->at(1, 1), 24.0F, "decode V width is wrong");

    // 第二张图没有破坏第一张图写入的非连续 cell。
    require_close(cache.key()->at(0, 2), 1.0F, "old cell 2 K should persist");
    require_close(cache.key()->at(0, 5), 4.0F, "old cell 5 K should persist");
}

void test_set_rows_rejects_invalid_cell() {
    GraphKVCache cache(/*kv_width=*/2, /*capacity=*/3);
    Context graph_context;
    Tensor* k_cur = graph_context.new_tensor_2d(2, 1, "k_cur");
    Tensor* v_cur = graph_context.new_tensor_2d(2, 1, "v_cur");
    Tensor* indices = graph_context.new_i32_tensor_1d(1, "bad_indices");
    mini_llama::ggml::set_i32_values(indices, {3});

    const KVStoreNodes stores =
        cache.build_store_nodes(graph_context, k_cur, v_cur, indices);
    Graph graph;
    mini_llama::ggml::build_forward_expand(graph, stores.key_write);

    bool threw = false;
    try {
        mini_llama::ggml::graph_compute(graph);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    require(threw, "SET_ROWS must reject a physical cell outside cache capacity");
}

} // namespace

int main() {
    try {
        test_temporary_qkv_write_persistent_cache();
        test_set_rows_rejects_invalid_cell();

        std::cout << "stage08_graph_kv_store: all tests passed\n";
        std::cout << "Implemented: temporary Q/K/V graph nodes, I32 cell indices, "
                     "SET_ROWS scatter writes and a persistent K/V storage context.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage08_graph_kv_store: FAILED: " << error.what() << '\n';
        return 1;
    }
}
