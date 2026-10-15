# 实现路线和边界

## 原则

- 第一目标是正确和可解释，第二目标才是性能。
- 每加入一个优化，都保留可比较的朴素实现。
- 前几个阶段只支持 CPU、float32、batch=1 和一种 Llama 架构。
- 暂时不追求兼容 llama.cpp 支持的所有模型。

## 与 llama.cpp 的对应关系

| 教学阶段 | llama.cpp 中的概念 |
| --- | --- |
| Tensor 与 stride | `ggml_tensor::ne`、`ggml_tensor::nb` |
| 朴素算子 | ggml CPU kernels |
| Transformer Block | llama.cpp model graph builder |
| 连续 KV Cache | K/V tensors 和 decode |
| KV cells | `llama_kv_cache`、`llama_kv_cells` |
| Prompt reuse | 请求 token 匹配和 `seq_cp` |
| 最小计算图 | `ggml_tensor::op/src`、`ggml_cgraph`、forward graph compute |
| 图写入 KV | `ggml_set_rows`、`cpy_k/cpy_v`、持久 cache Tensor |
| 图读取 KV | RoPE、Attention mask、`get_k/get_v` 和显式写后读顺序 |
| GGUF reader | model loader 和 GGUF metadata |
| Q8/Q4 | ggml quantization types |
| backend 替换 | ggml graph、allocator、backend |

## 为什么第一阶段不用 ggml

如果一开始直接调用 `ggml_mul_mat`，我们可以很快得到结果，但看不到张量布局、
索引以及矩阵乘法到底怎样发生。Stage 01 先写了标量参考实现；Stage 07 开始
自己实现 GGML 风格的 Tensor 节点和计算图。这样既能观察核心架构，也能保留
容易核对的标量 kernel。
