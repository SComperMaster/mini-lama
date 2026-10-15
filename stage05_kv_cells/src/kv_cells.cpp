#include "mini_llama/kv_cells.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mini_llama {
namespace {

void validate_sequence(const SequenceId sequence) {
    if (sequence >= max_sequences) {
        throw std::out_of_range("sequence id exceeds max_sequences");
    }
}

Tensor add(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape()) {
        throw std::invalid_argument("residual add requires equal tensor shapes");
    }
    Tensor result(left.shape());
    for (std::size_t i = 0; i < left.size(); ++i) {
        result[i] = left[i] + right[i];
    }
    return result;
}

LlamaBlockConfig block_config_from(const LlamaModelConfig& config) {
    return LlamaBlockConfig{
        .hidden_size = config.hidden_size,
        .query_heads = config.query_heads,
        .kv_heads = config.kv_heads,
        .head_dim = config.head_dim,
        .intermediate_size = config.intermediate_size,
        .rms_epsilon = config.rms_epsilon,
        .rope_theta = config.rope_theta,
    };
}

} // namespace

CellKVCache::CellKVCache(
    const LlamaModelConfig& config,
    const std::size_t capacity)
    : cells_(capacity), kv_width_(config.kv_heads * config.head_dim) {
    if (capacity == 0 || config.layer_count == 0 || config.kv_heads == 0 ||
        config.head_dim == 0) {
        throw std::invalid_argument("invalid cell KV Cache dimensions");
    }

    keys_by_layer_.reserve(config.layer_count);
    values_by_layer_.reserve(config.layer_count);
    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        keys_by_layer_.emplace_back(std::vector<std::size_t>{capacity, kv_width_});
        values_by_layer_.emplace_back(std::vector<std::size_t>{capacity, kv_width_});
    }
}

std::size_t CellKVCache::capacity() const noexcept {
    return cells_.size();
}

std::size_t CellKVCache::layer_count() const noexcept {
    return keys_by_layer_.size();
}

std::size_t CellKVCache::kv_width() const noexcept {
    return kv_width_;
}

std::size_t CellKVCache::used_cells() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        cells_.begin(), cells_.end(), [](const Cell& cell) { return cell.occupied; }));
}

std::size_t CellKVCache::used_max_plus_one() const noexcept {
    for (std::size_t i = cells_.size(); i > 0; --i) {
        if (cells_[i - 1].occupied) {
            return i;
        }
    }
    return 0;
}

std::size_t CellKVCache::search_head() const noexcept {
    return head_;
}

std::size_t CellKVCache::kv_rows_written() const noexcept {
    return kv_rows_written_;
}

bool CellKVCache::is_empty(const std::size_t cell_index) const {
    return !cells_.at(cell_index).occupied;
}

Position CellKVCache::position(const std::size_t cell_index) const {
    const Cell& cell = cells_.at(cell_index);
    if (!cell.occupied) {
        throw std::logic_error("empty cell has no logical position");
    }
    return cell.position;
}

bool CellKVCache::has_sequence(
    const std::size_t cell_index,
    const SequenceId sequence) const {
    validate_sequence(sequence);
    return cells_.at(cell_index).occupied && cells_.at(cell_index).sequences.test(sequence);
}

std::size_t CellKVCache::sequence_count(const std::size_t cell_index) const {
    return cells_.at(cell_index).sequences.count();
}

CellSlotInfo CellKVCache::allocate(
    const SequenceId sequence,
    const std::vector<Position>& positions) {
    validate_sequence(sequence);
    if (positions.empty()) {
        throw std::invalid_argument("cell allocation requires at least one position");
    }
    if (!std::is_sorted(positions.begin(), positions.end()) ||
        std::adjacent_find(positions.begin(), positions.end()) != positions.end()) {
        throw std::invalid_argument("new positions must be strictly increasing");
    }
    for (const Cell& cell : cells_) {
        if (!cell.occupied || !cell.sequences.test(sequence)) {
            continue;
        }
        if (std::binary_search(positions.begin(), positions.end(), cell.position)) {
            throw std::invalid_argument("sequence already has a cell at a requested position");
        }
    }

    CellSlotInfo result;
    result.positions = positions;
    result.cell_indices.reserve(positions.size());

    std::size_t tested = 0;
    std::size_t candidate = head_;
    while (tested < cells_.size() && result.cell_indices.size() < positions.size()) {
        if (!cells_[candidate].occupied) {
            result.cell_indices.push_back(candidate);
        }
        candidate = (candidate + 1) % cells_.size();
        ++tested;
    }

    if (result.cell_indices.size() != positions.size()) {
        throw std::overflow_error("not enough empty KV cells");
    }

    for (std::size_t i = 0; i < result.cell_indices.size(); ++i) {
        Cell& cell = cells_[result.cell_indices[i]];
        cell.position = positions[i];
        cell.occupied = true;
        cell.sequences.reset();
        cell.sequences.set(sequence);
    }
    head_ = (result.cell_indices.back() + 1) % cells_.size();
    return result;
}

void CellKVCache::write_layer(
    const std::size_t layer_index,
    const CellSlotInfo& slots,
    const Tensor& keys,
    const Tensor& values) {
    if (layer_index >= layer_count()) {
        throw std::out_of_range("layer index exceeds KV Cache layer count");
    }
    if (keys.ndim() != 2 || values.ndim() != 2 || keys.shape() != values.shape() ||
        keys.shape() != std::vector<std::size_t>{slots.cell_indices.size(), kv_width_}) {
        throw std::invalid_argument("K/V rows do not match allocated cell slots");
    }

    for (std::size_t row = 0; row < slots.cell_indices.size(); ++row) {
        const std::size_t cell_index = slots.cell_indices[row];
        if (is_empty(cell_index)) {
            throw std::logic_error("cannot write K/V into an unallocated cell");
        }
        for (std::size_t col = 0; col < kv_width_; ++col) {
            keys_by_layer_[layer_index].at(cell_index, col) = keys.at(row, col);
            values_by_layer_[layer_index].at(cell_index, col) = values.at(row, col);
        }
    }
    kv_rows_written_ += slots.cell_indices.size();
}

float CellKVCache::key_at(
    const std::size_t layer_index,
    const std::size_t cell_index,
    const std::size_t dimension) const {
    if (is_empty(cell_index)) {
        throw std::logic_error("cannot read K from an empty cell");
    }
    return keys_by_layer_.at(layer_index).at(cell_index, dimension);
}

float CellKVCache::value_at(
    const std::size_t layer_index,
    const std::size_t cell_index,
    const std::size_t dimension) const {
    if (is_empty(cell_index)) {
        throw std::logic_error("cannot read V from an empty cell");
    }
    return values_by_layer_.at(layer_index).at(cell_index, dimension);
}

std::vector<std::size_t> CellKVCache::cells_for_sequence(
    const SequenceId sequence,
    const Position max_position) const {
    validate_sequence(sequence);
    std::vector<std::size_t> result;
    for (std::size_t i = 0; i < cells_.size(); ++i) {
        const Cell& cell = cells_[i];
        if (cell.occupied && cell.sequences.test(sequence) && cell.position <= max_position) {
            result.push_back(i);
        }
    }
    std::sort(result.begin(), result.end(), [this](const std::size_t left, const std::size_t right) {
        return cells_[left].position < cells_[right].position;
    });
    return result;
}

Position CellKVCache::next_position(const SequenceId sequence) const {
    const std::vector<std::size_t> indices = cells_for_sequence(sequence);
    if (indices.empty()) {
        return 0;
    }
    return position(indices.back()) + 1;
}

std::size_t CellKVCache::seq_copy(
    const SequenceId source,
    const SequenceId destination,
    const Position position_begin,
    const Position position_end) {
    validate_sequence(source);
    validate_sequence(destination);
    if (position_end < position_begin) {
        throw std::invalid_argument("invalid sequence copy range");
    }

    std::size_t changed = 0;
    for (Cell& cell : cells_) {
        if (cell.occupied && cell.position >= position_begin && cell.position < position_end &&
            cell.sequences.test(source) && !cell.sequences.test(destination)) {
            cell.sequences.set(destination);
            ++changed;
        }
    }
    return changed;
}

std::size_t CellKVCache::seq_remove(
    const SequenceId sequence,
    const Position position_begin,
    const Position position_end) {
    validate_sequence(sequence);
    if (position_end < position_begin) {
        throw std::invalid_argument("invalid sequence remove range");
    }

    std::size_t changed = 0;
    std::size_t first_freed = cells_.size();
    for (std::size_t i = 0; i < cells_.size(); ++i) {
        Cell& cell = cells_[i];
        if (!cell.occupied || cell.position < position_begin || cell.position >= position_end ||
            !cell.sequences.test(sequence)) {
            continue;
        }

        cell.sequences.reset(sequence);
        ++changed;
        if (cell.sequences.none()) {
            cell.occupied = false;
            cell.position = 0;
            first_freed = std::min(first_freed, i);
        }
    }

    if (first_freed < head_) {
        head_ = first_freed;
    }
    return changed;
}

Tensor causal_gqa_attention_with_cells(
    const Tensor& query,
    const std::vector<Position>& query_positions,
    const CellKVCache& cache,
    const std::size_t layer_index,
    const SequenceId sequence,
    const std::size_t query_heads,
    const std::size_t kv_heads,
    const std::size_t head_dim) {
    if (query.ndim() != 2 || query.shape()[0] != query_positions.size() ||
        query.shape()[1] != query_heads * head_dim || query_heads == 0 || kv_heads == 0 ||
        head_dim == 0 || query_heads % kv_heads != 0 ||
        cache.kv_width() != kv_heads * head_dim) {
        throw std::invalid_argument("invalid cell attention dimensions");
    }

    Tensor output({query_positions.size(), query_heads * head_dim});
    const std::size_t query_heads_per_kv_head = query_heads / kv_heads;
    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim));

    for (std::size_t token = 0; token < query_positions.size(); ++token) {
        const std::vector<std::size_t> visible_cells =
            cache.cells_for_sequence(sequence, query_positions[token]);
        if (visible_cells.empty()) {
            throw std::logic_error("query has no visible KV cells");
        }

        for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
            const std::size_t kv_head = query_head / query_heads_per_kv_head;
            std::vector<float> scores(visible_cells.size(), 0.0F);
            float maximum = -std::numeric_limits<float>::infinity();

            for (std::size_t source = 0; source < visible_cells.size(); ++source) {
                float dot = 0.0F;
                for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                    dot += query.at(token, query_head * head_dim + dimension) *
                           cache.key_at(
                               layer_index,
                               visible_cells[source],
                               kv_head * head_dim + dimension);
                }
                scores[source] = dot * scale;
                maximum = std::max(maximum, scores[source]);
            }

            float sum = 0.0F;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                sum += score;
            }
            for (float& score : scores) {
                score /= sum;
            }

            for (std::size_t dimension = 0; dimension < head_dim; ++dimension) {
                float weighted_value = 0.0F;
                for (std::size_t source = 0; source < visible_cells.size(); ++source) {
                    weighted_value += scores[source] *
                                      cache.value_at(
                                          layer_index,
                                          visible_cells[source],
                                          kv_head * head_dim + dimension);
                }
                output.at(token, query_head * head_dim + dimension) = weighted_value;
            }
        }
    }
    return output;
}

Tensor model_forward_with_kv_cells(
    const std::vector<TokenId>& new_tokens,
    const SequenceId sequence,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    CellKVCache& cache) {
    validate_sequence(sequence);
    if (new_tokens.empty()) {
        throw std::invalid_argument("cell forward requires at least one token");
    }
    if (weights.blocks.size() != config.layer_count ||
        cache.layer_count() != config.layer_count ||
        cache.kv_width() != config.kv_heads * config.head_dim) {
        throw std::invalid_argument("model and cell cache dimensions differ");
    }

    Tensor hidden = embedding_lookup(new_tokens, weights.token_embeddings);
    const Position start_position = cache.next_position(sequence);
    std::vector<Position> positions(new_tokens.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        positions[i] = start_position + i;
    }
    const CellSlotInfo slots = cache.allocate(sequence, positions);
    const LlamaBlockConfig block_config = block_config_from(config);

    for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
        const LlamaBlockWeights& block = weights.blocks[layer];
        const Tensor attention_input = rms_norm(
            hidden, block.attention_norm, block_config.rms_epsilon);
        Tensor query = matmul(attention_input, block.wq);
        Tensor key = matmul(attention_input, block.wk);
        const Tensor value = matmul(attention_input, block.wv);

        apply_rope_in_place(
            query,
            block_config.query_heads,
            block_config.head_dim,
            start_position,
            block_config.rope_theta);
        apply_rope_in_place(
            key,
            block_config.kv_heads,
            block_config.head_dim,
            start_position,
            block_config.rope_theta);

        cache.write_layer(layer, slots, key, value);
        const Tensor attention = causal_gqa_attention_with_cells(
            query,
            positions,
            cache,
            layer,
            sequence,
            block_config.query_heads,
            block_config.kv_heads,
            block_config.head_dim);
        const Tensor attention_residual = add(hidden, matmul(attention, block.wo));

        const Tensor ffn_input = rms_norm(
            attention_residual, block.ffn_norm, block_config.rms_epsilon);
        const Tensor gate = matmul(ffn_input, block.w_gate);
        const Tensor up = matmul(ffn_input, block.w_up);
        hidden = add(
            attention_residual,
            matmul(swiglu(gate, up), block.w_down));
    }

    const Tensor normalized = rms_norm(hidden, weights.output_norm, config.rms_epsilon);
    return matmul(normalized, weights.lm_head);
}

} // namespace mini_llama
