# Stage 05：llama.cpp 风格的 KV cells

第四阶段要求 K/V 从第 0 行连续写到 `size-1`。本阶段把“逻辑 token 位置”和
“物理 K/V 行号”分开。

```text
逻辑 token:  pos 5   pos 6   pos 7
                  \     |     /
物理 cell:       cell1 cell3 cell5
```

每个 cell 保存：

```text
position
occupied
sequence bitset
```

每层 K/V 张量仍然是预分配的连续物理池，但所有层共享相同的 cell index。新 K/V
使用 `CellSlotInfo::cell_indices` scatter 写入。

## Ring search

分配器从 `head` 开始循环扫描空 cell，不要求结果连续。如果找到 `[1, 3, 5]`，
三个新 token 的各层 K/V 就分别写到这些物理行。

## Sequence mask

Attention 为当前 sequence 收集 cell，并同时检查：

```text
cell belongs to sequence
cell.position <= query.position
```

因此统一物理池中的其他 sequence 后缀不会被当前 query 看见。

## 零拷贝前缀共享

`seq_copy(0, 1, 0, 2)` 不分配新 cell，也不复制 K/V，只在两个前缀 cell 的
sequence bitset 中增加 sequence 1。分叉以后，两个 sequence 各自写入私有后缀。

删除 sequence 0 时，只有 sequence 0 独占的 cell 变为空闲；仍被 sequence 1
引用的共享前缀继续保留。

## 与本地 llama.cpp 的对应关系

- `src/llama-kv-cells.h`：`pos`、`seq`、used cells 和 `seq_rm`。
- `src/llama-kv-cache.cpp::find_slot`：从 ring head 搜索连续或非连续槽位。
- `src/llama-kv-cache.cpp::apply_ubatch`：把逻辑 token 信息提交到选中的 cell。
- `src/llama-kv-cache.cpp::seq_cp`：同一 unified stream 中只更新元数据。

真实实现还处理多 stream、SWA、context shift、状态保存、2D position、不同 backend
和量化 K/V。本阶段只保留理解核心 cell 机制所需的部分。
