#include "mini_llama/mini_ggml.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using mini_llama::ggml::Context;
using mini_llama::ggml::Graph;
using mini_llama::ggml::Op;
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

void test_ne_nb_and_view() {
    Context context;
    Tensor* source = context.new_tensor_2d(4, 3, "source");
    mini_llama::ggml::set_values(source, {
         1.0F,  2.0F,  3.0F,  4.0F,
         5.0F,  6.0F,  7.0F,  8.0F,
         9.0F, 10.0F, 11.0F, 12.0F,
    });

    require(source->nb[0] == sizeof(float), "nb[0] must be one F32 value");
    require(source->nb[1] == 4 * sizeof(float), "nb[1] must skip one row");

    Tensor* view = context.view_2d(
        source,
        2,
        2,
        source->nb[1],
        sizeof(float),
        "middle_columns");

    require_close(view->at(0, 0), 2.0F, "view first value is wrong");
    require_close(view->at(1, 0), 3.0F, "view first row is wrong");
    require_close(view->at(0, 1), 6.0F, "view stride is wrong");
    view->at(1, 1) = 70.0F;
    require_close(source->at(2, 1), 70.0F, "view must share source storage");
}

void test_graph_is_lazy_and_topological() {
    Context context;

    // ggml 的二维布局是 [ne0, ne1]，ne0 连续。
    // weights: [input_features=3, output_features=2]
    Tensor* weights = context.new_tensor_2d(3, 2, "weights");
    mini_llama::ggml::set_values(weights, {
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    });

    // input: [features=3, tokens=2]
    Tensor* input = context.new_tensor_2d(3, 2, "input");
    mini_llama::ggml::set_values(input, {
         7.0F,  9.0F, 11.0F,
         8.0F, 10.0F, 12.0F,
    });

    Tensor* bias = context.new_tensor_1d(2, "bias");
    mini_llama::ggml::set_values(bias, {1.0F, -1.0F});

    Tensor* scale = context.new_tensor_1d(2, "norm_weight");
    mini_llama::ggml::set_values(scale, {2.0F, 0.5F});

    Tensor* projected = context.mul_mat(weights, input, "projected");
    Tensor* biased = context.add(projected, bias, "biased");
    Tensor* normalized = context.rms_norm(biased, 1.0e-6F, "normalized");
    Tensor* output = context.mul(normalized, scale, "output");

    // 创建算子只建图，不执行 kernel。
    require_close(output->at(0, 0), 0.0F, "operator construction must be lazy");

    Graph graph;
    mini_llama::ggml::build_forward_expand(graph, output);
    require(graph.leaves.size() == 4, "graph should contain four input/weight leaves");
    require(graph.nodes.size() == 4, "graph should contain four compute nodes");
    require(graph.nodes[0]->op == Op::mul_mat, "MUL_MAT must precede its users");
    require(graph.nodes[1]->op == Op::add, "ADD must follow MUL_MAT");
    require(graph.nodes[2]->op == Op::rms_norm, "RMS_NORM order is wrong");
    require(graph.nodes[3]->op == Op::mul, "MUL must be the output node");

    mini_llama::ggml::graph_compute(graph);

    const float row0_inverse_rms =
        1.0F / std::sqrt((59.0F * 59.0F + 138.0F * 138.0F) / 2.0F + 1.0e-6F);
    const float row1_inverse_rms =
        1.0F / std::sqrt((65.0F * 65.0F + 153.0F * 153.0F) / 2.0F + 1.0e-6F);

    require_close(output->at(0, 0), 59.0F * row0_inverse_rms * 2.0F,
                  "token 0, feature 0 is wrong");
    require_close(output->at(1, 0), 138.0F * row0_inverse_rms * 0.5F,
                  "token 0, feature 1 is wrong");
    require_close(output->at(0, 1), 65.0F * row1_inverse_rms * 2.0F,
                  "token 1, feature 0 is wrong");
    require_close(output->at(1, 1), 153.0F * row1_inverse_rms * 0.5F,
                  "token 1, feature 1 is wrong");
}

void test_shared_subgraph_is_visited_once() {
    Context context;
    Tensor* weights = context.new_tensor_2d(2, 1, "weights");
    Tensor* input = context.new_tensor_2d(2, 1, "input");
    mini_llama::ggml::set_values(weights, {2.0F, 3.0F});
    mini_llama::ggml::set_values(input, {5.0F, 7.0F});

    Tensor* product = context.mul_mat(weights, input, "product");
    Tensor* doubled = context.add(product, product, "doubled");

    Graph graph;
    mini_llama::ggml::build_forward_expand(graph, doubled);
    require(graph.nodes.size() == 2, "shared MUL_MAT node must only appear once");

    mini_llama::ggml::graph_compute(graph);
    require_close(doubled->at(0), 62.0F, "shared subgraph result is wrong");
}

} // namespace

int main() {
    try {
        test_ne_nb_and_view();
        test_graph_is_lazy_and_topological();
        test_shared_subgraph_is_visited_once();

        std::cout << "stage07_ggml_graph: all tests passed\n";
        std::cout << "Implemented: ggml-style ne/nb layout, storage-sharing views, "
                     "lazy op nodes, forward graph construction and scalar graph execution.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage07_ggml_graph: FAILED: " << error.what() << '\n';
        return 1;
    }
}
