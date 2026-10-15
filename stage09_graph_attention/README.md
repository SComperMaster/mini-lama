# Stage 09：让 Attention 真正读取 KV Cache

这一阶段把 Stage 08 的“图中写入缓存”继续向前推进：当前 batch 的 Q/K/V
先由投影节点产生，Q/K 经过 RoPE，K/V 通过 `SET_ROWS` 写进持久缓存，随后
Attention 从这块持久缓存读取历史信息。

```text
                         +--> Q projection --> RoPE ------------------+
input ------------------+                                             |
                         +--> K projection --> RoPE --> SET_ROWS(K) ---+--> Attention
                         +--> V projection ---------> SET_ROWS(V) -----+       |
                                                                             output
```

## 为什么写节点必须显式加入图

`SET_ROWS(K)` 的结果与 K cache 共享存储，但 Attention 的输入是原来的 K cache
Tensor。对普通依赖遍历来说，Attention 看不到 `SET_ROWS(K) -> K cache` 这条“副
作用依赖”。因此 `build_kv_attention_graph()` 按下面的顺序把三个 root 展开：

1. K 的 `SET_ROWS`；
2. V 的 `SET_ROWS`；
3. Attention 输出。

执行器按节点顺序计算，于是 Attention 一定读到本轮刚写入的 K/V。这与
llama.cpp 在构图时先展开 `cpy_k/cpy_v`，再构造并展开 Attention 的核心处理
方式一致。

## 物理 cell 与逻辑位置

测试故意把逻辑位置 `[0, 1, 2]` 放进物理 cell `[2, 5, 1]`。Attention mask
的形状是 `[cache_capacity, query_tokens]`：

- 值为 `0`：这个 query 可以读取该物理 cell；
- 值为 `-infinity`：这个 cell 被 softmax 排除。

因此 Attention 不要求缓存物理地址按逻辑顺序连续。cell 管理器负责决定“数据
放在哪里”，mask 负责决定“本次计算能看见谁”。

## 本阶段有意保留的限制

- 仅实现单头 FP32 Attention；
- 用普通标量循环实现 softmax 和加权求和；
- mask 由测试直接给出物理 cell，还没有自动连接 Stage 05 的 sequence 元数据；
- 暂未实现 GQA、多层 cache 和 backend 调度。

下一阶段会把 Stage 05 的 cell/sequence 管理接入图构建，让 mask 与写入位置都
由一次 KV prepare 结果自动产生，然后逐步拼成完整的 graph Llama Block。
