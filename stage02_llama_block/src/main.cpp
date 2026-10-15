#include "mini_llama/llama_block.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    const float actual,
    const float expected,
    const float tolerance,
    const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
}

void test_softmax() {
    const mini_llama::Tensor input({2, 3}, {
        1.0F, 2.0F, 3.0F,
        1001.0F, 1002.0F, 1003.0F,
    });
    const mini_llama::Tensor expected({2, 3}, {
        0.09003057F, 0.24472848F, 0.66524094F,
        0.09003057F, 0.24472848F, 0.66524094F,
    });

    const mini_llama::Tensor actual = mini_llama::softmax_rows(input);
    require(mini_llama::all_close(actual, expected, 1.0e-6F, 1.0e-6F),
            "stable Softmax test failed:\n" + actual.debug_string());
}

void test_silu_and_swiglu() {
    require_close(mini_llama::silu(-1.0F), -0.26894143F, 1.0e-6F, "SiLU(-1)");
    require_close(mini_llama::silu(0.0F), 0.0F, 1.0e-6F, "SiLU(0)");
    require_close(mini_llama::silu(1.0F), 0.73105860F, 1.0e-6F, "SiLU(1)");

    const mini_llama::Tensor gate({1, 3}, {-1.0F, 0.0F, 1.0F});
    const mini_llama::Tensor up({1, 3}, {2.0F, 3.0F, 4.0F});
    const mini_llama::Tensor expected({1, 3}, {-0.53788286F, 0.0F, 2.92423439F});
    const mini_llama::Tensor actual = mini_llama::swiglu(gate, up);

    require(mini_llama::all_close(actual, expected, 1.0e-6F, 1.0e-6F),
            "SwiGLU test failed:\n" + actual.debug_string());
}

void test_rope() {
    mini_llama::Tensor tensor({2, 4}, {
        1.0F, 2.0F, 3.0F, 4.0F,
        1.0F, 0.0F, 0.0F, 1.0F,
    });

    mini_llama::apply_rope_in_place(tensor, 2, 2, 0);

    // position=0 不旋转。head_dim=2 时 position=1 的角度就是 1 radian。
    const float cosine = std::cos(1.0F);
    const float sine = std::sin(1.0F);
    const mini_llama::Tensor expected({2, 4}, {
        1.0F, 2.0F, 3.0F, 4.0F,
        cosine, sine, -sine, cosine,
    });

    require(mini_llama::all_close(tensor, expected, 1.0e-6F, 1.0e-6F),
            "RoPE test failed:\n" + tensor.debug_string());
}

void test_causal_gqa_attention() {
    // 两个 Q head 共享一个 KV head，验证 GQA 的 head 映射和 causal mask。
    const mini_llama::Tensor query({2, 4}, {
        1.0F, 0.0F, 0.0F, 1.0F,
        1.0F, 0.0F, 1.0F, 0.0F,
    });
    const mini_llama::Tensor key({2, 2}, {
        1.0F, 0.0F,
        0.0F, 1.0F,
    });
    const mini_llama::Tensor value({2, 2}, {
        10.0F, 0.0F,
        0.0F, 20.0F,
    });

    const mini_llama::Tensor actual =
        mini_llama::causal_gqa_attention(query, key, value, 2, 1, 2);

    const float first_weight = std::exp(1.0F / std::sqrt(2.0F)) /
                               (std::exp(1.0F / std::sqrt(2.0F)) + 1.0F);
    const float second_weight = 1.0F - first_weight;
    const mini_llama::Tensor expected({2, 4}, {
        10.0F, 0.0F, 10.0F, 0.0F,
        10.0F * first_weight, 20.0F * second_weight,
        10.0F * first_weight, 20.0F * second_weight,
    });

    require(mini_llama::all_close(actual, expected, 1.0e-5F, 1.0e-5F),
            "causal GQA test failed:\n" + actual.debug_string());
}

void test_zero_weight_block_is_residual_identity() {
    const mini_llama::LlamaBlockConfig config{
        .hidden_size = 4,
        .query_heads = 2,
        .kv_heads = 1,
        .head_dim = 2,
        .intermediate_size = 6,
        .rms_epsilon = 1.0e-5F,
        .rope_theta = 10000.0F,
    };

    mini_llama::LlamaBlockWeights weights{
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

    const mini_llama::Tensor input({2, 4}, {
        1.0F, 2.0F, 3.0F, 4.0F,
        5.0F, 6.0F, 7.0F, 8.0F,
    });
    const mini_llama::Tensor actual =
        mini_llama::llama_block_forward(input, weights, config);

    require(mini_llama::all_close(actual, input),
            "zero-weight block should preserve the residual input");
}

} // namespace

int main() {
    try {
        test_softmax();
        test_silu_and_swiglu();
        test_rope();
        test_causal_gqa_attention();
        test_zero_weight_block_is_residual_identity();

        std::cout << "stage02_llama_block: all tests passed\n";
        std::cout << "Implemented: stable Softmax, SiLU/SwiGLU, RoPE, causal GQA "
                     "and a pre-norm Llama block.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage02_llama_block: FAILED: " << error.what() << '\n';
        return 1;
    }
}
