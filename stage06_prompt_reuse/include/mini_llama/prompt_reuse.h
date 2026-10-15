#pragma once

#include "mini_llama/kv_cells.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace mini_llama {

[[nodiscard]] std::size_t longest_common_prefix(
    const std::vector<TokenId>& left,
    const std::vector<TokenId>& right);

struct ActiveRequest {
    SequenceId sequence = 0;
    std::vector<TokenId> tokens;
};

struct PromptReusePlan {
    std::optional<SequenceId> source_sequence;
    std::size_t matched_tokens = 0;
    std::size_t shared_tokens = 0;
    std::size_t tokens_to_compute = 0;
};

struct PromptRunResult {
    Tensor logits;
    PromptReusePlan plan;
};

// 一个 scheduler 实例只能服务于同一组模型权重、adapter 和位置编码配置。
class PromptReuseScheduler {
public:
    [[nodiscard]] const std::vector<ActiveRequest>& active_requests() const noexcept;

    [[nodiscard]] PromptReusePlan plan(
        SequenceId destination,
        const std::vector<TokenId>& tokens) const;

    // 注册新请求，自动共享公共前缀，并只计算剩余 suffix。
    [[nodiscard]] PromptRunResult run(
        SequenceId destination,
        const std::vector<TokenId>& tokens,
        const LlamaModelWeights& weights,
        const LlamaModelConfig& config,
        CellKVCache& cache);

    // 删除请求的 sequence 关联。共享 cell 只有在没有其他 sequence 使用时才释放。
    void release(SequenceId sequence, CellKVCache& cache);

private:
    std::vector<ActiveRequest> requests_;
};

} // namespace mini_llama
