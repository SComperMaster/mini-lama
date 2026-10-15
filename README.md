# mini-llama

这是一个从零开始理解 llama.cpp 的教学项目。项目不会一开始复制完整的
llama.cpp，而是先实现正确、容易阅读的版本，再逐步加入 KV Cache、GGUF、
量化、ggml 和性能优化。

## 学习路线

1. `stage01_tensor`：连续内存 Tensor、矩阵乘法、RMSNorm 和数值测试。
2. `stage02_llama_block`：RoPE、GQA Attention、SwiGLU 和一个 Transformer Block。
3. `stage03_toy_model`：词嵌入、多个 Block、LM Head 和逐 token 生成。
4. `stage04_kv_cache`：先实现连续 KV Cache，对比有缓存和无缓存的计算量。
5. `stage05_kv_cells`：实现接近 llama.cpp 的 cell、sequence 和槽位复用。
6. `stage06_prompt_reuse`：公共前缀匹配并复用已有 KV cells。
7. `stage07_ggml_graph`：实现 `ne/nb`、view、算子节点、建图和标量执行器。
8. `stage08_graph_kv_store`：用 `SET_ROWS` 连接临时 Q/K/V 图和持久 KV Cache。
9. `stage09_graph_attention`：加入 RoPE 和读取缓存的 Attention 图。
10. 后续阶段：完整 graph Llama Block、KV prepare/apply、GGUF、量化与 backend。

每个阶段都必须满足：能独立编译、能运行测试、能与参考结果比较。

## Visual Studio 2022 构建

在“x64 Native Tools Command Prompt for VS 2022”中执行：

```powershell
cd E:\preparefor_lama\mini-llama
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

也可以在 Visual Studio 2022 中选择“打开本地文件夹”，打开本目录。Visual
Studio 会识别顶层 `CMakeLists.txt`。

## 当前阶段

- `stage01_tensor`：连续 FP32 Tensor、朴素矩阵乘法和 RMSNorm。
- `stage02_llama_block`：Softmax、SiLU/SwiGLU、RoPE、GQA Attention 和完整的
  pre-norm Llama Transformer Block。
- `stage03_toy_model`：token embedding、多层 Block、LM Head、logits 和无 KV
  Cache 的 greedy generation。
- `stage04_kv_cache`：连续 K/V 缓存、prefill、单 token decode，以及缓存版与完整
  重算的逐元素 logits 对比。
- `stage05_kv_cells`：物理 K/V 池、逻辑 cell、ring slot search、sequence mask、
  `seq_copy` 零拷贝前缀共享和 `seq_remove` 引用释放。
- `stage06_prompt_reuse`：保存活跃请求 token、最长公共前缀匹配、自动 `seq_copy`、
  suffix-only 计算和安全释放。
- `stage07_ggml_graph`：GGML 风格 `ne/nb` 布局、共享 storage 的 view、惰性
  算子节点、forward graph 构建和单线程标量执行。
- `stage08_graph_kv_store`：I32 cell 索引、`SET_ROWS` 散写节点，以及跨多次
  graph compute 长期存在的 K/V Tensor。

第七阶段开始从“数学功能相同”收紧为“核心执行结构对齐”。第八阶段把临时图和
持久 KV 数据区接通；下一步让 Attention 从缓存读取历史 K/V。
