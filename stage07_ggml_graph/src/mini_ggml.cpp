#include "mini_llama/mini_ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_llama::ggml {
namespace {

constexpr std::size_t type_size = sizeof(float);

std::size_t extent(const std::int64_t value) {
    if (value <= 0) {
        throw std::invalid_argument("Tensor dimensions must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::size_t checked_element_count(const std::array<std::int64_t, max_dims>& ne) {
    std::size_t count = 1;
    for (const std::int64_t dimension : ne) {
        const std::size_t current = extent(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / current) {
            throw std::overflow_error("Tensor element count overflows size_t");
        }
        count *= current;
    }
    return count;
}

bool can_repeat(const Tensor& smaller, const Tensor& larger) {
    for (std::size_t dim = 0; dim < max_dims; ++dim) {
        if (smaller.ne[dim] != 1 && smaller.ne[dim] != larger.ne[dim]) {
            return false;
        }
    }
    return true;
}

void require_tensor(Tensor* tensor, const char* name) {
    if (tensor == nullptr) {
        throw std::invalid_argument(std::string(name) + " cannot be null");
    }
}

template <typename Function>
void for_each_index(const Tensor& tensor, Function function) {
    for (std::size_t i3 = 0; i3 < extent(tensor.ne[3]); ++i3) {
        for (std::size_t i2 = 0; i2 < extent(tensor.ne[2]); ++i2) {
            for (std::size_t i1 = 0; i1 < extent(tensor.ne[1]); ++i1) {
                for (std::size_t i0 = 0; i0 < extent(tensor.ne[0]); ++i0) {
                    function(i0, i1, i2, i3);
                }
            }
        }
    }
}

void compute_elementwise(Tensor& output, const bool multiply) {
    const Tensor& left = *output.src[0];
    const Tensor& right = *output.src[1];

    for_each_index(output, [&](const std::size_t i0,
                               const std::size_t i1,
                               const std::size_t i2,
                               const std::size_t i3) {
        const std::size_t r0 = right.ne[0] == 1 ? 0 : i0;
        const std::size_t r1 = right.ne[1] == 1 ? 0 : i1;
        const std::size_t r2 = right.ne[2] == 1 ? 0 : i2;
        const std::size_t r3 = right.ne[3] == 1 ? 0 : i3;
        const float lhs = left.at(i0, i1, i2, i3);
        const float rhs = right.at(r0, r1, r2, r3);
        output.at(i0, i1, i2, i3) = multiply ? lhs * rhs : lhs + rhs;
    });
}

void compute_mul_mat(Tensor& output) {
    const Tensor& weights = *output.src[0];
    const Tensor& input = *output.src[1];
    const std::size_t shared = extent(weights.ne[0]);

    for (std::size_t token = 0; token < extent(input.ne[1]); ++token) {
        for (std::size_t output_feature = 0;
             output_feature < extent(weights.ne[1]);
             ++output_feature) {
            float sum = 0.0F;
            for (std::size_t inner = 0; inner < shared; ++inner) {
                sum += weights.at(inner, output_feature) * input.at(inner, token);
            }
            output.at(output_feature, token) = sum;
        }
    }
}

float read_epsilon(const Tensor& tensor) {
    float epsilon = 0.0F;
    static_assert(sizeof(epsilon) <= sizeof(tensor.op_params));
    std::memcpy(&epsilon, tensor.op_params.data(), sizeof(epsilon));
    return epsilon;
}

void compute_rms_norm(Tensor& output) {
    const Tensor& input = *output.src[0];
    const std::size_t row_width = extent(input.ne[0]);
    const float epsilon = read_epsilon(output);

    for (std::size_t i3 = 0; i3 < extent(input.ne[3]); ++i3) {
        for (std::size_t i2 = 0; i2 < extent(input.ne[2]); ++i2) {
            for (std::size_t i1 = 0; i1 < extent(input.ne[1]); ++i1) {
                float squared_sum = 0.0F;
                for (std::size_t i0 = 0; i0 < row_width; ++i0) {
                    const float value = input.at(i0, i1, i2, i3);
                    squared_sum += value * value;
                }
                const float inverse_rms = 1.0F /
                    std::sqrt(squared_sum / static_cast<float>(row_width) + epsilon);
                for (std::size_t i0 = 0; i0 < row_width; ++i0) {
                    output.at(i0, i1, i2, i3) =
                        input.at(i0, i1, i2, i3) * inverse_rms;
                }
            }
        }
    }
}

void visit(Graph& graph, Tensor* tensor) {
    require_tensor(tensor, "graph output");
    if (!graph.visited.insert(tensor).second) {
        return;
    }

    for (Tensor* source : tensor->src) {
        if (source != nullptr) {
            visit(graph, source);
        }
    }

    if (tensor->op == Op::none) {
        graph.leaves.push_back(tensor);
    } else {
        graph.nodes.push_back(tensor);
    }
}

} // namespace

std::size_t Tensor::n_elements() const noexcept {
    std::size_t count = 1;
    for (const std::int64_t dimension : ne) {
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

std::size_t Tensor::n_dims() const noexcept {
    for (std::size_t dim = max_dims; dim > 1; --dim) {
        if (ne[dim - 1] != 1) {
            return dim;
        }
    }
    return 1;
}

float& Tensor::at(
    const std::size_t i0,
    const std::size_t i1,
    const std::size_t i2,
    const std::size_t i3) {
    return const_cast<float&>(std::as_const(*this).at(i0, i1, i2, i3));
}

const float& Tensor::at(
    const std::size_t i0,
    const std::size_t i1,
    const std::size_t i2,
    const std::size_t i3) const {
    const std::array<std::size_t, max_dims> index{i0, i1, i2, i3};
    std::size_t byte_offset = storage_offset_bytes;
    for (std::size_t dim = 0; dim < max_dims; ++dim) {
        if (index[dim] >= static_cast<std::size_t>(ne[dim])) {
            throw std::out_of_range("Tensor index is out of range");
        }
        byte_offset += index[dim] * nb[dim];
    }
    if (byte_offset % type_size != 0 || storage == nullptr) {
        throw std::logic_error("Tensor has invalid F32 storage");
    }
    return storage->at(byte_offset / type_size);
}

Tensor* Context::new_tensor_1d(const std::int64_t ne0, std::string name) {
    return new_tensor({ne0, 1, 1, 1}, std::move(name));
}

Tensor* Context::new_tensor_2d(
    const std::int64_t ne0,
    const std::int64_t ne1,
    std::string name) {
    return new_tensor({ne0, ne1, 1, 1}, std::move(name));
}

Tensor* Context::new_tensor(
    const std::array<std::int64_t, max_dims> ne,
    std::string name) {
    const std::size_t count = checked_element_count(ne);
    auto tensor = std::make_unique<Tensor>();
    tensor->ne = ne;
    tensor->nb[0] = type_size;
    for (std::size_t dim = 1; dim < max_dims; ++dim) {
        tensor->nb[dim] = tensor->nb[dim - 1] * extent(ne[dim - 1]);
    }
    tensor->name = std::move(name);
    tensor->storage = std::make_shared<std::vector<float>>(count, 0.0F);

    Tensor* result = tensor.get();
    tensors_.push_back(std::move(tensor));
    return result;
}

Tensor* Context::view_2d(
    Tensor* source,
    const std::int64_t ne0,
    const std::int64_t ne1,
    const std::size_t nb1,
    const std::size_t offset,
    std::string name) {
    require_tensor(source, "view source");
    const std::size_t columns = extent(ne0);
    const std::size_t rows = extent(ne1);
    if (nb1 < columns * type_size || nb1 % type_size != 0 || offset % type_size != 0) {
        throw std::invalid_argument("view_2d requires aligned, non-overlapping rows");
    }

    const std::size_t last_byte = source->storage_offset_bytes + offset +
        (rows - 1) * nb1 + columns * type_size;
    const std::size_t storage_bytes = source->storage->size() * type_size;
    if (last_byte > storage_bytes) {
        throw std::out_of_range("view exceeds source storage");
    }

    auto tensor = std::make_unique<Tensor>();
    tensor->ne = {ne0, ne1, 1, 1};
    tensor->nb = {type_size, nb1, nb1 * rows, nb1 * rows};
    tensor->op = Op::view;
    tensor->src[0] = source;
    tensor->view_src = source;
    tensor->view_offs = offset;
    tensor->name = std::move(name);
    tensor->storage = source->storage;
    tensor->storage_offset_bytes = source->storage_offset_bytes + offset;

    Tensor* result = tensor.get();
    tensors_.push_back(std::move(tensor));
    return result;
}

Tensor* Context::new_elementwise_node(
    const Op op,
    Tensor* left,
    Tensor* right,
    std::string name) {
    require_tensor(left, "left tensor");
    require_tensor(right, "right tensor");
    if (!can_repeat(*right, *left)) {
        throw std::invalid_argument("right tensor cannot be repeated to the left tensor shape");
    }
    Tensor* result = new_tensor(left->ne, std::move(name));
    result->op = op;
    result->src[0] = left;
    result->src[1] = right;
    return result;
}

Tensor* Context::add(Tensor* left, Tensor* right, std::string name) {
    return new_elementwise_node(Op::add, left, right, std::move(name));
}

Tensor* Context::mul(Tensor* left, Tensor* right, std::string name) {
    return new_elementwise_node(Op::mul, left, right, std::move(name));
}

Tensor* Context::mul_mat(Tensor* weights, Tensor* input, std::string name) {
    require_tensor(weights, "weights");
    require_tensor(input, "input");
    if (weights->n_dims() > 2 || input->n_dims() > 2) {
        throw std::invalid_argument("stage07 mul_mat only supports 2D tensors");
    }
    if (weights->ne[0] != input->ne[0]) {
        throw std::invalid_argument("mul_mat inner dimensions do not match");
    }

    Tensor* result = new_tensor_2d(weights->ne[1], input->ne[1], std::move(name));
    result->op = Op::mul_mat;
    result->src[0] = weights;
    result->src[1] = input;
    return result;
}

Tensor* Context::rms_norm(Tensor* input, const float epsilon, std::string name) {
    require_tensor(input, "rms_norm input");
    if (!(epsilon > 0.0F)) {
        throw std::invalid_argument("rms_norm epsilon must be positive");
    }
    Tensor* result = new_tensor(input->ne, std::move(name));
    result->op = Op::rms_norm;
    result->src[0] = input;
    std::memcpy(result->op_params.data(), &epsilon, sizeof(epsilon));
    return result;
}

void Graph::clear() {
    leaves.clear();
    nodes.clear();
    visited.clear();
}

void set_values(Tensor* tensor, const std::initializer_list<float> values) {
    require_tensor(tensor, "tensor");
    if (values.size() != tensor->n_elements()) {
        throw std::invalid_argument("Value count does not match tensor shape");
    }

    auto current = values.begin();
    for_each_index(*tensor, [&](const std::size_t i0,
                                const std::size_t i1,
                                const std::size_t i2,
                                const std::size_t i3) {
        tensor->at(i0, i1, i2, i3) = *current;
        ++current;
    });
}

void build_forward_expand(Graph& graph, Tensor* output) {
    visit(graph, output);
}

void graph_compute(Graph& graph) {
    for (Tensor* node : graph.nodes) {
        switch (node->op) {
        case Op::view:
            // View 与源 Tensor 共享 storage，只改变 ne/nb/offset，不需要 kernel。
            break;
        case Op::add:
            compute_elementwise(*node, false);
            break;
        case Op::mul:
            compute_elementwise(*node, true);
            break;
        case Op::mul_mat:
            compute_mul_mat(*node);
            break;
        case Op::rms_norm:
            compute_rms_norm(*node);
            break;
        case Op::none:
            throw std::logic_error("A leaf Tensor was incorrectly placed in graph.nodes");
        }
    }
}

const char* op_name(const Op op) noexcept {
    switch (op) {
    case Op::none: return "NONE";
    case Op::view: return "VIEW";
    case Op::add: return "ADD";
    case Op::mul: return "MUL";
    case Op::mul_mat: return "MUL_MAT";
    case Op::rms_norm: return "RMS_NORM";
    }
    return "UNKNOWN";
}

} // namespace mini_llama::ggml
