#include "mini_llama/mini_ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_llama::ggml {
namespace {

std::size_t type_size(const Type type) {
    switch (type) {
    case Type::f32: return sizeof(float);
    case Type::i32: return sizeof(std::int32_t);
    }
    throw std::logic_error("Unknown Tensor type");
}

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

float read_float_param(const Tensor& tensor) {
    float value = 0.0F;
    static_assert(sizeof(value) <= sizeof(tensor.op_params));
    std::memcpy(&value, tensor.op_params.data(), sizeof(value));
    return value;
}

void compute_rms_norm(Tensor& output) {
    const Tensor& input = *output.src[0];
    const std::size_t row_width = extent(input.ne[0]);
    const float epsilon = read_float_param(output);

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

void compute_rope(Tensor& output) {
    const Tensor& input = *output.src[0];
    const Tensor& positions = *output.src[1];
    const std::size_t head_dim = extent(input.ne[0]);
    const float theta = read_float_param(output);

    for (std::size_t token = 0; token < extent(input.ne[1]); ++token) {
        const std::int32_t position_i32 = positions.at_i32(token);
        if (position_i32 < 0) {
            throw std::invalid_argument("RoPE positions must be non-negative");
        }
        const float position = static_cast<float>(position_i32);
        for (std::size_t dimension = 0; dimension < head_dim; dimension += 2) {
            const float exponent =
                -static_cast<float>(dimension) / static_cast<float>(head_dim);
            const float frequency = std::pow(theta, exponent);
            const float angle = position * frequency;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);
            const float even = input.at(dimension, token);
            const float odd = input.at(dimension + 1, token);
            output.at(dimension, token) = even * cosine - odd * sine;
            output.at(dimension + 1, token) = even * sine + odd * cosine;
        }
    }
}

void compute_attention(Tensor& output) {
    const Tensor& query = *output.src[0];
    const Tensor& key = *output.src[1];
    const Tensor& value = *output.src[2];
    const Tensor& mask = *output.src[3];
    const std::size_t head_dim = extent(query.ne[0]);
    const std::size_t cache_capacity = extent(key.ne[1]);
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(cache_capacity, 0.0F);

    for (std::size_t token = 0; token < extent(query.ne[1]); ++token) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t cell = 0; cell < cache_capacity; ++cell) {
            float dot = 0.0F;
            for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                dot += query.at(dimension, token) * key.at(dimension, cell);
            }
            scores[cell] = dot * scale + mask.at(cell, token);
            maximum = std::max(maximum, scores[cell]);
        }
        if (!std::isfinite(maximum)) {
            throw std::invalid_argument("Attention mask hides every KV cell");
        }

        float sum = 0.0F;
        for (float& score : scores) {
            score = std::isfinite(score) ? std::exp(score - maximum) : 0.0F;
            sum += score;
        }
        if (!(sum > 0.0F)) {
            throw std::runtime_error("Attention softmax has an invalid sum");
        }

        for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
            float weighted_value = 0.0F;
            for (std::size_t cell = 0; cell < cache_capacity; ++cell) {
                weighted_value += (scores[cell] / sum) * value.at(dimension, cell);
            }
            output.at(dimension, token) = weighted_value;
        }
    }
}

void compute_set_rows(Tensor& output) {
    const Tensor& source = *output.src[0];
    const Tensor& row_indices = *output.src[1];

    for (std::size_t source_row = 0; source_row < extent(source.ne[1]); ++source_row) {
        const std::int32_t destination_row_i32 = row_indices.at_i32(source_row);
        if (destination_row_i32 < 0 || destination_row_i32 >= output.ne[1]) {
            throw std::out_of_range("SET_ROWS destination row is outside the cache");
        }
        const std::size_t destination_row =
            static_cast<std::size_t>(destination_row_i32);
        for (std::size_t i0 = 0; i0 < extent(source.ne[0]); ++i0) {
            output.at(i0, destination_row) = source.at(i0, source_row);
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
    if (type != Type::f32 || f32_storage == nullptr) {
        throw std::logic_error("Tensor is not backed by F32 storage");
    }
    const std::array<std::size_t, max_dims> index{i0, i1, i2, i3};
    std::size_t byte_offset = storage_offset_bytes;
    for (std::size_t dim = 0; dim < max_dims; ++dim) {
        if (index[dim] >= static_cast<std::size_t>(ne[dim])) {
            throw std::out_of_range("Tensor index is out of range");
        }
        byte_offset += index[dim] * nb[dim];
    }
    if (byte_offset % sizeof(float) != 0) {
        throw std::logic_error("Tensor has invalid F32 storage");
    }
    return f32_storage->at(byte_offset / sizeof(float));
}

std::int32_t& Tensor::at_i32(
    const std::size_t i0,
    const std::size_t i1,
    const std::size_t i2,
    const std::size_t i3) {
    return const_cast<std::int32_t&>(std::as_const(*this).at_i32(i0, i1, i2, i3));
}

const std::int32_t& Tensor::at_i32(
    const std::size_t i0,
    const std::size_t i1,
    const std::size_t i2,
    const std::size_t i3) const {
    if (type != Type::i32 || i32_storage == nullptr) {
        throw std::logic_error("Tensor is not backed by I32 storage");
    }
    const std::array<std::size_t, max_dims> index{i0, i1, i2, i3};
    std::size_t byte_offset = storage_offset_bytes;
    for (std::size_t dim = 0; dim < max_dims; ++dim) {
        if (index[dim] >= static_cast<std::size_t>(ne[dim])) {
            throw std::out_of_range("Tensor index is out of range");
        }
        byte_offset += index[dim] * nb[dim];
    }
    if (byte_offset % sizeof(std::int32_t) != 0) {
        throw std::logic_error("Tensor has invalid I32 storage");
    }
    return i32_storage->at(byte_offset / sizeof(std::int32_t));
}

Tensor* Context::new_tensor_1d(const std::int64_t ne0, std::string name) {
    return new_tensor(Type::f32, {ne0, 1, 1, 1}, std::move(name));
}

Tensor* Context::new_tensor_2d(
    const std::int64_t ne0,
    const std::int64_t ne1,
    std::string name) {
    return new_tensor(Type::f32, {ne0, ne1, 1, 1}, std::move(name));
}

Tensor* Context::new_i32_tensor_1d(const std::int64_t ne0, std::string name) {
    return new_tensor(Type::i32, {ne0, 1, 1, 1}, std::move(name));
}

Tensor* Context::new_tensor(
    const Type type,
    const std::array<std::int64_t, max_dims> ne,
    std::string name) {
    const std::size_t count = checked_element_count(ne);
    auto tensor = std::make_unique<Tensor>();
    tensor->type = type;
    tensor->ne = ne;
    tensor->nb[0] = type_size(type);
    for (std::size_t dim = 1; dim < max_dims; ++dim) {
        tensor->nb[dim] = tensor->nb[dim - 1] * extent(ne[dim - 1]);
    }
    tensor->name = std::move(name);
    if (type == Type::f32) {
        tensor->f32_storage = std::make_shared<std::vector<float>>(count, 0.0F);
    } else {
        tensor->i32_storage =
            std::make_shared<std::vector<std::int32_t>>(count, std::int32_t{0});
    }

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
    const std::size_t element_size = type_size(source->type);
    if (nb1 < columns * element_size ||
        nb1 % element_size != 0 ||
        offset % element_size != 0) {
        throw std::invalid_argument("view_2d requires aligned, non-overlapping rows");
    }

    const std::size_t last_byte = source->storage_offset_bytes + offset +
        (rows - 1) * nb1 + columns * element_size;
    const std::size_t storage_elements = source->type == Type::f32
        ? source->f32_storage->size()
        : source->i32_storage->size();
    const std::size_t storage_bytes = storage_elements * element_size;
    if (last_byte > storage_bytes) {
        throw std::out_of_range("view exceeds source storage");
    }

    auto tensor = std::make_unique<Tensor>();
    tensor->type = source->type;
    tensor->ne = {ne0, ne1, 1, 1};
    tensor->nb = {element_size, nb1, nb1 * rows, nb1 * rows};
    tensor->op = Op::view;
    tensor->src[0] = source;
    tensor->view_src = source;
    tensor->view_offs = offset;
    tensor->name = std::move(name);
    tensor->f32_storage = source->f32_storage;
    tensor->i32_storage = source->i32_storage;
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
    if (left->type != Type::f32 || right->type != Type::f32) {
        throw std::invalid_argument("stage07 elementwise ops require F32 tensors");
    }
    Tensor* result = new_tensor(Type::f32, left->ne, std::move(name));
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
    if (weights->type != Type::f32 || input->type != Type::f32) {
        throw std::invalid_argument("mul_mat requires F32 tensors");
    }
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
    if (input->type != Type::f32) {
        throw std::invalid_argument("rms_norm requires an F32 tensor");
    }
    Tensor* result = new_tensor(Type::f32, input->ne, std::move(name));
    result->op = Op::rms_norm;
    result->src[0] = input;
    std::memcpy(result->op_params.data(), &epsilon, sizeof(epsilon));
    return result;
}

Tensor* Context::rope(
    Tensor* input,
    Tensor* positions,
    const float theta,
    std::string name) {
    require_tensor(input, "RoPE input");
    require_tensor(positions, "RoPE positions");
    if (input->type != Type::f32 || positions->type != Type::i32) {
        throw std::invalid_argument("RoPE requires F32 input and I32 positions");
    }
    if (input->n_dims() > 2 || positions->n_dims() != 1 ||
        input->ne[1] != positions->ne[0] || input->ne[0] % 2 != 0) {
        throw std::invalid_argument("RoPE shapes require even head_dim and one position per token");
    }
    if (!(theta > 0.0F)) {
        throw std::invalid_argument("RoPE theta must be positive");
    }

    Tensor* result = new_tensor(Type::f32, input->ne, std::move(name));
    result->op = Op::rope;
    result->src[0] = input;
    result->src[1] = positions;
    std::memcpy(result->op_params.data(), &theta, sizeof(theta));
    return result;
}

Tensor* Context::set_rows(
    Tensor* destination,
    Tensor* source,
    Tensor* row_indices,
    std::string name) {
    require_tensor(destination, "SET_ROWS destination");
    require_tensor(source, "SET_ROWS source");
    require_tensor(row_indices, "SET_ROWS row indices");
    if (destination->type != Type::f32 || source->type != Type::f32 ||
        row_indices->type != Type::i32) {
        throw std::invalid_argument("SET_ROWS requires F32 data and I32 indices");
    }
    if (destination->n_dims() > 2 || source->n_dims() > 2 ||
        row_indices->n_dims() != 1 || destination->ne[0] != source->ne[0] ||
        source->ne[1] != row_indices->ne[0]) {
        throw std::invalid_argument("SET_ROWS shapes do not match");
    }

    auto tensor = std::make_unique<Tensor>();
    tensor->type = destination->type;
    tensor->ne = destination->ne;
    tensor->nb = destination->nb;
    tensor->op = Op::set_rows;
    // 与 ggml_set_rows 保持相同的 src 顺序：source、indices、destination。
    tensor->src[0] = source;
    tensor->src[1] = row_indices;
    tensor->src[2] = destination;
    tensor->view_src = destination;
    tensor->view_offs = 0;
    tensor->name = std::move(name);
    tensor->f32_storage = destination->f32_storage;
    tensor->storage_offset_bytes = destination->storage_offset_bytes;

    Tensor* result = tensor.get();
    tensors_.push_back(std::move(tensor));
    return result;
}

Tensor* Context::attention(
    Tensor* query,
    Tensor* key_cache,
    Tensor* value_cache,
    Tensor* mask,
    std::string name) {
    require_tensor(query, "Attention query");
    require_tensor(key_cache, "Attention key cache");
    require_tensor(value_cache, "Attention value cache");
    require_tensor(mask, "Attention mask");
    if (query->type != Type::f32 || key_cache->type != Type::f32 ||
        value_cache->type != Type::f32 || mask->type != Type::f32) {
        throw std::invalid_argument("Attention requires F32 tensors");
    }
    if (query->n_dims() > 2 || key_cache->n_dims() > 2 ||
        value_cache->n_dims() > 2 || mask->n_dims() > 2 ||
        key_cache->ne != value_cache->ne || query->ne[0] != key_cache->ne[0] ||
        mask->ne[0] != key_cache->ne[1] || mask->ne[1] != query->ne[1]) {
        throw std::invalid_argument("Attention tensor shapes do not match");
    }

    Tensor* result = new_tensor(Type::f32, query->ne, std::move(name));
    result->op = Op::attention;
    result->src[0] = query;
    result->src[1] = key_cache;
    result->src[2] = value_cache;
    result->src[3] = mask;
    return result;
}

void Graph::clear() {
    leaves.clear();
    nodes.clear();
    visited.clear();
}

void set_values(Tensor* tensor, const std::initializer_list<float> values) {
    require_tensor(tensor, "tensor");
    if (tensor->type != Type::f32) {
        throw std::invalid_argument("set_values requires an F32 tensor");
    }
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

void set_i32_values(
    Tensor* tensor,
    const std::initializer_list<std::int32_t> values) {
    require_tensor(tensor, "tensor");
    if (tensor->type != Type::i32) {
        throw std::invalid_argument("set_i32_values requires an I32 tensor");
    }
    if (values.size() != tensor->n_elements()) {
        throw std::invalid_argument("Value count does not match tensor shape");
    }

    auto current = values.begin();
    for (std::size_t i = 0; i < tensor->n_elements(); ++i) {
        tensor->at_i32(i) = *current;
        ++current;
    }
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
        case Op::rope:
            compute_rope(*node);
            break;
        case Op::set_rows:
            compute_set_rows(*node);
            break;
        case Op::attention:
            compute_attention(*node);
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
    case Op::rope: return "ROPE";
    case Op::set_rows: return "SET_ROWS";
    case Op::attention: return "ATTENTION";
    }
    return "UNKNOWN";
}

} // namespace mini_llama::ggml
