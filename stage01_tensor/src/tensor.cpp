#include "mini_llama/tensor.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mini_llama {
namespace {

std::size_t element_count(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }

    std::size_t count = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0) {
            throw std::invalid_argument("Tensor dimensions must be greater than zero");
        }
        count *= dimension;
    }
    return count;
}

void require_matrix(const Tensor& tensor, const char* name) {
    if (tensor.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D tensor");
    }
}

} // namespace

Tensor::Tensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), data_(element_count(shape_), 0.0F) {}

Tensor::Tensor(std::vector<std::size_t> shape, std::initializer_list<float> values)
    : shape_(std::move(shape)), data_(values) {
    if (data_.size() != element_count(shape_)) {
        throw std::invalid_argument("Value count does not match tensor shape");
    }
}

const std::vector<std::size_t>& Tensor::shape() const noexcept {
    return shape_;
}

std::size_t Tensor::ndim() const noexcept {
    return shape_.size();
}

std::size_t Tensor::size() const noexcept {
    return data_.size();
}

float& Tensor::operator[](const std::size_t index) {
    return data_.at(index);
}

const float& Tensor::operator[](const std::size_t index) const {
    return data_.at(index);
}

float& Tensor::at(const std::size_t row, const std::size_t col) {
    require_matrix(*this, "Tensor");
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("Tensor matrix index is out of range");
    }
    return data_[row * shape_[1] + col];
}

const float& Tensor::at(const std::size_t row, const std::size_t col) const {
    require_matrix(*this, "Tensor");
    if (row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("Tensor matrix index is out of range");
    }
    return data_[row * shape_[1] + col];
}

const std::vector<float>& Tensor::data() const noexcept {
    return data_;
}

std::string Tensor::debug_string() const {
    std::ostringstream output;
    output << std::fixed << std::setprecision(5);

    if (ndim() != 2) {
        output << '[';
        for (std::size_t i = 0; i < size(); ++i) {
            if (i != 0) {
                output << ", ";
            }
            output << data_[i];
        }
        output << ']';
        return output.str();
    }

    output << "[\n";
    for (std::size_t row = 0; row < shape_[0]; ++row) {
        output << "  [";
        for (std::size_t col = 0; col < shape_[1]; ++col) {
            if (col != 0) {
                output << ", ";
            }
            output << at(row, col);
        }
        output << "]";
        if (row + 1 != shape_[0]) {
            output << ',';
        }
        output << '\n';
    }
    output << ']';
    return output.str();
}

Tensor matmul(const Tensor& left, const Tensor& right) {
    require_matrix(left, "left");
    require_matrix(right, "right");

    const std::size_t rows = left.shape()[0];
    const std::size_t shared = left.shape()[1];
    const std::size_t right_rows = right.shape()[0];
    const std::size_t cols = right.shape()[1];

    if (shared != right_rows) {
        throw std::invalid_argument("matmul dimensions do not match");
    }

    Tensor result({rows, cols});
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            float sum = 0.0F;
            for (std::size_t inner = 0; inner < shared; ++inner) {
                sum += left.at(row, inner) * right.at(inner, col);
            }
            result.at(row, col) = sum;
        }
    }
    return result;
}

Tensor rms_norm(const Tensor& input, const Tensor& weight, const float epsilon) {
    require_matrix(input, "input");
    if (weight.ndim() != 1 || weight.shape()[0] != input.shape()[1]) {
        throw std::invalid_argument("RMSNorm weight must match input's last dimension");
    }
    if (epsilon <= 0.0F) {
        throw std::invalid_argument("RMSNorm epsilon must be positive");
    }

    const std::size_t rows = input.shape()[0];
    const std::size_t cols = input.shape()[1];
    Tensor result({rows, cols});

    for (std::size_t row = 0; row < rows; ++row) {
        float squared_sum = 0.0F;
        for (std::size_t col = 0; col < cols; ++col) {
            const float value = input.at(row, col);
            squared_sum += value * value;
        }

        const float mean_square = squared_sum / static_cast<float>(cols);
        const float inverse_rms = 1.0F / std::sqrt(mean_square + epsilon);

        for (std::size_t col = 0; col < cols; ++col) {
            result.at(row, col) = input.at(row, col) * inverse_rms * weight[col];
        }
    }

    return result;
}

bool all_close(
    const Tensor& actual,
    const Tensor& expected,
    const float absolute_tolerance,
    const float relative_tolerance) {
    if (actual.shape() != expected.shape()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::abs(actual[i] - expected[i]);
        const float allowed = absolute_tolerance + relative_tolerance * std::abs(expected[i]);
        if (difference > allowed) {
            return false;
        }
    }
    return true;
}

} // namespace mini_llama
