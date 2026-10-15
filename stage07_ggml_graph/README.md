# Stage 07：最小 GGML 风格计算图

这一阶段不追求 ggml 的性能，而是复现它最核心的执行模型：算子调用只创建
Tensor 节点，真正的计算由 forward graph executor 完成。

## 与 llama.cpp/ggml 源码的对应关系

| 教学实现 | ggml 源码概念 |
| --- | --- |
| `Tensor::ne`、`Tensor::nb` | `ggml_tensor::ne`、`ggml_tensor::nb` |
| `Tensor::op`、`Tensor::src` | `ggml_tensor::op`、`ggml_tensor::src` |
| `view_src`、`view_offs` | Tensor view 共享底层 storage |
| `Context` | `ggml_context` 对 Tensor 对象生命周期的管理 |
| `build_forward_expand` | 从输出递归访问父节点并生成拓扑序 |
| `graph_compute` | `ggml_graph_compute` 的单线程标量教学版 |

真实源码入口：

- `ggml/include/ggml.h` 中的 `struct ggml_tensor`
- `ggml/src/ggml.c` 中的 `ggml_add_impl`、`ggml_mul_mat`
- `ggml/src/ggml.c` 中的 `ggml_visit_parents_graph`
- `ggml/src/ggml.c` 中的 `ggml_build_forward_expand`

## 本阶段刻意省略

- F16、BF16 和量化类型；
- backend buffer 与 graph allocator；
- SIMD、多线程、CUDA 等 kernel；
- in-place 算子和完整广播规则；
- 自动求导；
- 超过二维的 `MUL_MAT` batch 广播。

这些不会改变本阶段要观察的核心关系：

```text
叶子 Tensor（输入、权重）
        ↓
算子创建新的 Tensor，并记录 op/src
        ↓
从最终输出反向遍历依赖
        ↓
得到依赖在前的 graph.nodes
        ↓
executor 按顺序运行每个节点的 kernel
```

下一阶段会用这里的图接口重新搭建一个 Llama Block。届时模型代码负责“描述
计算”，executor 才负责“执行计算”。
