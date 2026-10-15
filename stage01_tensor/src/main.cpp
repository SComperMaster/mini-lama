#include "mini_llama/tensor.h"

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

void test_row_major_layout() {
    const mini_llama::Tensor tensor({2, 3}, {
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    });

    require(tensor.at(0, 0) == 1.0F, "row-major test: element (0, 0) is wrong");
    require(tensor.at(1, 2) == 6.0F, "row-major test: element (1, 2) is wrong");
    require(tensor[4] == 5.0F, "row-major test: flat index 4 is wrong");
}

void test_matmul() {
    const mini_llama::Tensor left({2, 3}, {
        1.0F, 2.0F, 3.0F,
        4.0F, 5.0F, 6.0F,
    });
    const mini_llama::Tensor right({3, 2}, {
         7.0F,  8.0F,
         9.0F, 10.0F,
        11.0F, 12.0F,
    });
    const mini_llama::Tensor expected({2, 2}, {
         58.0F,  64.0F,
        139.0F, 154.0F,
    });

    const mini_llama::Tensor actual = mini_llama::matmul(left, right);
    require(mini_llama::all_close(actual, expected),
            "matmul test failed:\n" + actual.debug_string());
}

void test_rms_norm() {
    const mini_llama::Tensor input({2, 4}, {
         1.0F,  2.0F,  3.0F,  4.0F,
        -1.0F, -2.0F, -3.0F, -4.0F,
    });
    const mini_llama::Tensor weight({4}, {1.0F, 1.0F, 1.0F, 1.0F});

    // mean([1, 4, 9, 16]) = 7.5，epsilon 取极小值时 inverse_rms 约 0.36514837。
    const mini_llama::Tensor expected({2, 4}, {
         0.36514837F,  0.73029673F,  1.09544516F,  1.46059346F,
        -0.36514837F, -0.73029673F, -1.09544516F, -1.46059346F,
    });

    const mini_llama::Tensor actual = mini_llama::rms_norm(input, weight, 1.0e-6F);
    require(mini_llama::all_close(actual, expected, 1.0e-5F, 1.0e-5F),
            "RMSNorm test failed:\n" + actual.debug_string());
}

} // namespace

int main() {
    try {
        test_row_major_layout();
        test_matmul();
        test_rms_norm();

        std::cout << "stage01_tensor: all tests passed\n";
        std::cout << "Implemented: contiguous FP32 tensor, row-major indexing, "
                     "naive matmul and RMSNorm.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stage01_tensor: FAILED: " << error.what() << '\n';
        return 1;
    }
}
