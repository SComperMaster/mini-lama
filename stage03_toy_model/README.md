# Stage 03：从 token 到下一个 token

本阶段把前两阶段的算子组装为结构完整的最小语言模型：

```text
token ids
  -> embedding lookup
  -> N 个 Llama Transformer Block
  -> final RMSNorm
  -> LM Head
  -> logits
  -> greedy argmax
```

## 为什么暂时使用整数 token

Tokenizer 负责“文本和 token id 之间的转换”，模型实际接收的输入就是整数 id。
为了先看清模型推理主路径，本阶段跳过文本 tokenizer，直接使用 0、1、2、3。

## 玩具权重

测试模型使用四维 one-hot embedding，并让 LM Head 实现固定循环：

```text
0 -> 1 -> 2 -> 3 -> 0
```

Transformer Block 的投影权重暂时为零，因此残差路径保持 embedding 不变。Block
内部的 Attention、RoPE 和 SwiGLU 已经由第二阶段分别做过非零数值测试。本阶段
测试的是模型组装、logits、argmax 和生成循环。

## 故意保留的低效行为

生成第一个新 token 时处理整个 prompt；生成第二个新 token 时，又把 prompt 和
第一个新 token 全部重新计算。例如初始长度为 1，生成 5 个 token 时，处理量是：

```text
1 + 2 + 3 + 4 + 5 = 15 token
```

下一阶段加入 KV Cache 后，prefill 只执行一次，之后每一步只处理一个新 token。
