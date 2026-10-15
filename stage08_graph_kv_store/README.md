# Stage 08：临时 Q/K/V 图与持久 KV Cache 的桥接

这一阶段专门回答一个生命周期问题：Q、K、V 是一次 graph compute 中的临时
Tensor，而 KV Cache 必须跨越很多次 decode 长期存在，它们怎样连接？

答案是 `SET_ROWS`：

```text
临时图 Context                       持久 Cache Context

input ── MUL_MAT ── q_cur            key_cache   [width, capacity]
      ├─ MUL_MAT ── k_cur ──┐        value_cache [width, capacity]
      └─ MUL_MAT ── v_cur ──┼─ SET_ROWS ────────→ 指定物理 cell
cell_indices (I32) ──────────┘
```

- Q 只用于当前 Attention，不写入缓存；
- 当前 token 的 K/V 是临时图节点；
- `cell_indices` 给出 KV cell 分配器选出的物理槽位；
- `SET_ROWS` 节点执行时把 K/V 散写到长期缓存；
- 临时 graph Context 销毁后，K/V Cache 的底层数据仍然存在。

## 与 llama.cpp 的源码对应

当前 llama.cpp 中：

- `llama_kv_cache::cpy_k()` 使用 `ggml_set_rows(ctx, k, k_cur, k_idxs)`；
- `llama_kv_cache::cpy_v()` 使用同样方式写 V；
- `llm_graph_context::build_attn()` 把 K/V 写入节点显式加入 forward graph；
- 随后通过 `get_k()`、`get_v()` 创建缓存视图供 Attention 使用。

我们的 `Context::set_rows()` 特意采用与 `ggml_set_rows` 相同的输入顺序：

```text
src[0] = source K/V
src[1] = I32 row indices
src[2] = destination cache
```

## 当前边界

本阶段只实现二维 F32 数据的 `SET_ROWS`，还没有把 Attention 本身放入图中。
下一阶段会增加 RoPE 和缓存版 Attention，并把写缓存节点排在 Attention 节点之前。
