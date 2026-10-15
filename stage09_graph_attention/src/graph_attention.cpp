#include "mini_llama/graph_attention.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_llama {

ggml::Tensor* build_attention_mask(
    ggml::Context& graph_context,
    const std::size_t cache_capacity,
    const std::vector<std::vector<std::size_t>>& visible_cells,
    std::string name) {
    if (cache_capacity == 0) {
        throw std::invalid_argument("cache capacity must be positive");
    }
    if (visible_cells.empty()) {
        throw std::invalid_argument("attention needs at least one query token");
    }

    ggml::Tensor* mask = graph_context.new_tensor_2d(
        static_cast<std::int64_t>(cache_capacity),
        static_cast<std::int64_t>(visible_cells.size()),
        std::move(name));

    const float hidden = -std::numeric_limits<float>::infinity();
    for (std::size_t query = 0; query < visible_cells.size(); ++query) {
        if (visible_cells[query].empty()) {
            throw std::invalid_argument("each query token must see at least one KV cell");
        }
        for (std::size_t cell = 0; cell < cache_capacity; ++cell) {
            mask->at(cell, query) = hidden;
        }
        for (const std::size_t cell : visible_cells[query]) {
            if (cell >= cache_capacity) {
                throw std::out_of_range("visible KV cell exceeds cache capacity");
            }
            mask->at(cell, query) = 0.0F;
        }
    }
    return mask;
}

void build_kv_attention_graph(
    ggml::Graph& graph,
    const KVStoreNodes& stores,
    ggml::Tensor* attention_output) {
    if (stores.key_write == nullptr || stores.value_write == nullptr ||
        attention_output == nullptr) {
        throw std::invalid_argument("KV writes and attention output cannot be null");
    }

    ggml::build_forward_expand(graph, stores.key_write);
    ggml::build_forward_expand(graph, stores.value_write);
    ggml::build_forward_expand(graph, attention_output);
}

} // namespace mini_llama
