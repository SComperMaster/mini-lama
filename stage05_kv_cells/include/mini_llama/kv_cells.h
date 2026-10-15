#pragma once

#include "mini_llama/model.h"

#include <bitset>
#include <cstddef>
#include <limits>
#include <vector>

namespace mini_llama {

using SequenceId = std::size_t;
using Position = std::size_t;

inline constexpr std::size_t max_sequences = 32;

struct CellSlotInfo {
    // 第 i 个新 token 要写入的物理 cell。
    std::vector<std::size_t> cell_indices;
    std::vector<Position> positions;
};

// 多层共享一张 cell 元数据表；每层有自己的物理 K/V 张量，但使用相同 cell index。
class CellKVCache {
public:
    CellKVCache(const LlamaModelConfig& config, std::size_t capacity);

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t layer_count() const noexcept;
    [[nodiscard]] std::size_t kv_width() const noexcept;
    [[nodiscard]] std::size_t used_cells() const noexcept;
    [[nodiscard]] std::size_t used_max_plus_one() const noexcept;
    [[nodiscard]] std::size_t search_head() const noexcept;
    [[nodiscard]] std::size_t kv_rows_written() const noexcept;

    [[nodiscard]] bool is_empty(std::size_t cell_index) const;
    [[nodiscard]] Position position(std::size_t cell_index) const;
    [[nodiscard]] bool has_sequence(std::size_t cell_index, SequenceId sequence) const;
    [[nodiscard]] std::size_t sequence_count(std::size_t cell_index) const;

    // 先寻找全部空槽，只有全部找到后才修改元数据。
    [[nodiscard]] CellSlotInfo allocate(
        SequenceId sequence,
        const std::vector<Position>& positions);

    // 把某一层的新 K/V scatter 写入 allocate 返回的物理 cell。
    void write_layer(
        std::size_t layer_index,
        const CellSlotInfo& slots,
        const Tensor& keys,
        const Tensor& values);

    [[nodiscard]] float key_at(
        std::size_t layer_index,
        std::size_t cell_index,
        std::size_t dimension) const;
    [[nodiscard]] float value_at(
        std::size_t layer_index,
        std::size_t cell_index,
        std::size_t dimension) const;

    [[nodiscard]] std::vector<std::size_t> cells_for_sequence(
        SequenceId sequence,
        Position max_position = std::numeric_limits<Position>::max()) const;
    [[nodiscard]] Position next_position(SequenceId sequence) const;

    // 区间采用 [position_begin, position_end)。返回改变的序列关联数量。
    std::size_t seq_copy(
        SequenceId source,
        SequenceId destination,
        Position position_begin,
        Position position_end);
    std::size_t seq_remove(
        SequenceId sequence,
        Position position_begin,
        Position position_end);

private:
    struct Cell {
        Position position = 0;
        bool occupied = false;
        std::bitset<max_sequences> sequences;
    };

    std::vector<Cell> cells_;
    std::vector<Tensor> keys_by_layer_;
    std::vector<Tensor> values_by_layer_;
    std::size_t kv_width_ = 0;
    std::size_t head_ = 0;
    std::size_t kv_rows_written_ = 0;
};

[[nodiscard]] Tensor causal_gqa_attention_with_cells(
    const Tensor& query,
    const std::vector<Position>& query_positions,
    const CellKVCache& cache,
    std::size_t layer_index,
    SequenceId sequence,
    std::size_t query_heads,
    std::size_t kv_heads,
    std::size_t head_dim);

// new_tokens 会从该 sequence 的 next_position() 开始连续放置。
[[nodiscard]] Tensor model_forward_with_kv_cells(
    const std::vector<TokenId>& new_tokens,
    SequenceId sequence,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    CellKVCache& cache);

} // namespace mini_llama
