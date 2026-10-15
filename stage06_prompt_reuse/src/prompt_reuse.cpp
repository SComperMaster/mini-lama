#include "mini_llama/prompt_reuse.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mini_llama {

std::size_t longest_common_prefix(
    const std::vector<TokenId>& left,
    const std::vector<TokenId>& right) {
    const std::size_t limit = std::min(left.size(), right.size());
    std::size_t length = 0;
    while (length < limit && left[length] == right[length]) {
        ++length;
    }
    return length;
}

const std::vector<ActiveRequest>& PromptReuseScheduler::active_requests() const noexcept {
    return requests_;
}

PromptReusePlan PromptReuseScheduler::plan(
    const SequenceId destination,
    const std::vector<TokenId>& tokens) const {
    if (destination >= max_sequences) {
        throw std::out_of_range("destination sequence exceeds max_sequences");
    }
    if (tokens.empty()) {
        throw std::invalid_argument("prompt cannot be empty");
    }
    if (std::any_of(requests_.begin(), requests_.end(), [destination](const ActiveRequest& request) {
            return request.sequence == destination;
        })) {
        throw std::invalid_argument("destination sequence is already active");
    }

    PromptReusePlan result;
    for (const ActiveRequest& candidate : requests_) {
        const std::size_t matched = longest_common_prefix(candidate.tokens, tokens);
        if (matched > result.matched_tokens) {
            result.source_sequence = candidate.sequence;
            result.matched_tokens = matched;
        }
    }

    // To produce logits for the prompt, the final token must pass through the model.
    // A fully matched prompt therefore reuses all but its last token.
    result.shared_tokens = std::min(result.matched_tokens, tokens.size() - 1);
    result.tokens_to_compute = tokens.size() - result.shared_tokens;
    if (result.shared_tokens == 0) {
        result.source_sequence.reset();
    }
    return result;
}

PromptRunResult PromptReuseScheduler::run(
    const SequenceId destination,
    const std::vector<TokenId>& tokens,
    const LlamaModelWeights& weights,
    const LlamaModelConfig& config,
    CellKVCache& cache) {
    const PromptReusePlan reuse = plan(destination, tokens);

    if (reuse.source_sequence.has_value()) {
        const std::size_t copied = cache.seq_copy(
            *reuse.source_sequence, destination, 0, reuse.shared_tokens);
        if (copied != reuse.shared_tokens) {
            cache.seq_remove(
                destination, 0, std::numeric_limits<Position>::max());
            throw std::logic_error("source request token history does not match its KV cells");
        }
    }

    const auto suffix_begin = tokens.begin() + static_cast<std::ptrdiff_t>(reuse.shared_tokens);
    const std::vector<TokenId> suffix(suffix_begin, tokens.end());

    try {
        Tensor logits = model_forward_with_kv_cells(
            suffix, destination, weights, config, cache);
        requests_.push_back(ActiveRequest{
            .sequence = destination,
            .tokens = tokens,
        });
        return PromptRunResult{
            .logits = std::move(logits),
            .plan = reuse,
        };
    } catch (...) {
        cache.seq_remove(
            destination, 0, std::numeric_limits<Position>::max());
        throw;
    }
}

void PromptReuseScheduler::release(
    const SequenceId sequence,
    CellKVCache& cache) {
    const auto request = std::find_if(
        requests_.begin(), requests_.end(), [sequence](const ActiveRequest& item) {
            return item.sequence == sequence;
        });
    if (request == requests_.end()) {
        throw std::invalid_argument("cannot release an inactive sequence");
    }

    cache.seq_remove(sequence, 0, std::numeric_limits<Position>::max());
    requests_.erase(request);
}

} // namespace mini_llama
