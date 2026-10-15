#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace mini_llama {

// 第一阶段只实现连续、row-major、float32 Tensor。
// shape={rows, cols} 时，元素 (row, col) 位于 data[row * cols + col]。
class Tensor {
public:
    explicit Tensor(std::vector<std::size_t> shape);
    Tensor(std::vector<std::size_t> shape, std::initializer_list<float> values);

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;
    [[nodiscard]] std::size_t ndim() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;

    float& at(std::size_t row, std::size_t col);
    const float& at(std::size_t row, std::size_t col) const;

    [[nodiscard]] const std::vector<float>& data() const noexcept;
    [[nodiscard]] std::string debug_string() const;

private:
    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

// [M, K] x [K, N] -> [M, N]。
// 这是容易验证的三层循环参考实现，不追求性能。
[[nodiscard]] Tensor matmul(const Tensor& left, const Tensor& right);

// 对最后一个维度执行 Llama 使用的 RMSNorm：
// y_i = x_i * weight_i / sqrt(mean(x^2) + epsilon)
[[nodiscard]] Tensor rms_norm(
    const Tensor& input,
    const Tensor& weight,
    float epsilon);

[[nodiscard]] bool all_close(
    const Tensor& actual,
    const Tensor& expected,
    float absolute_tolerance = 1.0e-5F,
    float relative_tolerance = 1.0e-5F);

} // namespace mini_llama
