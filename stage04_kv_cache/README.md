# Stage 04：连续 KV Cache

这一阶段实现最直接的 KV Cache。每一层提前分配：

```text
K cache: [max_context, kv_heads * head_dim]
V cache: [max_context, kv_heads * head_dim]
```

`size` 表示已经缓存的 token 数。新 token 的 K/V 总是追加到 `size` 后面。

## Prefill

整个 prompt 一次进入模型。每层计算 prompt 的 Q/K/V，把 K/V 写入缓存，然后让
prompt 中第 i 个 query 只看到 `0..i` 的 K/V。

## Decode

后续每次只提交一个刚生成的 token：

1. 只为这个 token 计算 Q/K/V；
2. 使用 `past_tokens` 作为 RoPE position；
3. 把新 K/V 追加到缓存；
4. 新 Q 读取缓存中的全部历史 K/V；
5. 输出下一个 token 的 logits。

## 正确性验证

测试使用非零的 Q/K/V/O 权重，比较：

```text
完整计算 [token0, token1, token2]

和

prefill [token0, token1] + decode [token2]
```

两者的 logits 必须逐元素接近。只比较最终生成 token 不够严格，因为不同 logits
也可能碰巧得到相同的 argmax。

## 当前局限

缓存必须从位置 0 连续写到 size，没有空洞、多个 sequence、删除、共享前缀或环形
复用。下一阶段将用逻辑 cell table 替换这个限制，逐步靠近 llama.cpp。
