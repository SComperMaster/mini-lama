#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace mini_llama::ggml {

constexpr std::size_t max_dims = 4;
constexpr std::size_t max_sources = 4;
constexpr std::size_t max_op_params = 8;

// 这一阶段只实现 F32，但保留 type 字段，因为真实 ggml_tensor 的布局和
// 后端选择都依赖它。
enum class Type {
    f32,
};

// 对应 ggml_op 的最小子集。NONE 表示叶子 Tensor，不需要执行计算。
enum class Op {
    none,
    view,
    add,
    mul,
    mul_mat,
    rms_norm,
};

struct Tensor {
    Type type = Type::f32;

    // 与 ggml 一样，ne[0] 是最内层、连续的维度，而不是传统 C++ 矩阵的行数。
    std::array<std::int64_t, max_dims> ne{1, 1, 1, 1};
    std::array<std::size_t, max_dims> nb{0, 0, 0, 0};

    Op op = Op::none;
    std::array<std::int32_t, max_op_params> op_params{};
    std::array<Tensor*, max_sources> src{};

    Tensor* view_src = nullptr;
    std::size_t view_offs = 0;
    std::string name;

    [[nodiscard]] std::size_t n_elements() const noexcept;
    [[nodiscard]] std::size_t n_dims() const noexcept;

    float& at(
        std::size_t i0,
        std::size_t i1 = 0,
        std::size_t i2 = 0,
        std::size_t i3 = 0);
    const float& at(
        std::size_t i0,
        std::size_t i1 = 0,
        std::size_t i2 = 0,
        std::size_t i3 = 0) const;

private:
    friend class Context;

    std::shared_ptr<std::vector<float>> storage;
    std::size_t storage_offset_bytes = 0;
};

// Context 负责 Tensor 对象的生命周期，对应 ggml_context 的最小职责。
// 算子函数只创建结果 Tensor 和依赖关系，不立即计算。
class Context {
public:
    Tensor* new_tensor_1d(std::int64_t ne0, std::string name = {});
    Tensor* new_tensor_2d(std::int64_t ne0, std::int64_t ne1, std::string name = {});

    Tensor* view_2d(
        Tensor* source,
        std::int64_t ne0,
        std::int64_t ne1,
        std::size_t nb1,
        std::size_t offset,
        std::string name = {});

    Tensor* add(Tensor* left, Tensor* right, std::string name = {});
    Tensor* mul(Tensor* left, Tensor* right, std::string name = {});
    Tensor* mul_mat(Tensor* weights, Tensor* input, std::string name = {});
    Tensor* rms_norm(Tensor* input, float epsilon, std::string name = {});

private:
    Tensor* new_tensor(
        std::array<std::int64_t, max_dims> ne,
        std::string name);
    Tensor* new_elementwise_node(
        Op op,
        Tensor* left,
        Tensor* right,
        std::string name);

    std::vector<std::unique_ptr<Tensor>> tensors_;
};

struct Graph {
    // 叶子是权重或输入；nodes 按依赖在前、使用者在后的顺序排列。
    std::vector<Tensor*> leaves;
    std::vector<Tensor*> nodes;
    std::unordered_set<Tensor*> visited;

    void clear();
};

void set_values(Tensor* tensor, std::initializer_list<float> values);

// 从输出 Tensor 递归访问 src，构建 forward graph。共享子图只记录一次。
void build_forward_expand(Graph& graph, Tensor* output);

// 单线程标量执行器。性能很朴素，但执行边界与 ggml_graph_compute 一致。
void graph_compute(Graph& graph);

[[nodiscard]] const char* op_name(Op op) noexcept;

} // namespace mini_llama::ggml
