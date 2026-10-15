# Stage 06：请求层与自动前缀复用

KV cell 不按 token 内容查找缓存。本阶段在 `CellKVCache` 上增加请求调度层：

```text
ActiveRequest 保存原始 token
        -> 比较 longest common prefix
        -> 选择匹配最长的活跃 sequence
        -> 调用 seq_copy 共享 cell
        -> 只把未共享 suffix 送进模型
```

例如：

```text
req1 = [1, 10, 20]
req2 = [1,  2,  7]
```

公共前缀长度是 1。req2 共享 req1 的 position 0 cell，只计算 `[2, 7]`。

## 为什么不能按单个 token 复用

`[5, 1]` 和 `[9, 1]` 都在 position 1 出现 token 1，但该 token 的 K/V 已经受到
前面不同上下文影响，所以公共前缀长度是 0，不能共享任何 cell。

## 完整命中的边界情况

如果两个 prompt 完全相同，单纯复制全部 K/V 后没有最终 hidden state 和 logits。
为了继续生成，本阶段共享除最后一个 token 外的全部前缀，并重新计算最后一个 token。
真实服务也可以通过保存额外输出状态采用其他策略，但 KV 本身不包含 logits。

## 安全边界

一个 scheduler 实例只允许服务同一模型、同一 adapter、同一 RoPE 和推理配置。
如果权重或影响 hidden state 的配置不同，即使 token 前缀相同也不能共享 K/V。
