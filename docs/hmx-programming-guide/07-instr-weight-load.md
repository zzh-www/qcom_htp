# 07 · `weight.*` load

**TL;DR**：HMX 有 9 种 weight 类型：`.b`（int8）、`.n`（int4）、`.c`（int2 signed
crumb）、`.sc`（signed int2 alt）、`.ubit` / `.sbit`（1-bit）、`.hf`（fp16）、
`.f8`（fp8）、`.sm`（sparse mask）。大多数 sub-byte 类型在 QNN v75 skel 里只有
`.b / .n / .c / .hf` 被实际使用；其他是**硬件支持但 QNN 没用**。

## 基本语法 (F)

```c
asm volatile("{ activation.ub = mxmem(%0, %1)\n"
             "  weight.<type> = mxmem(%2, %3)<modifier> }"
             :: "r"(act), "r"(2047),
                "r"(wt),  "r"(2047)
             : "memory");
```

和 `activation.* = mxmem` 必须同一 VLIW packet。

## 整数 weight 类型：tile 字节数不同 (F)

| 类型 | bits | sub-byte/byte | tile 字节（32³）| signed? |
|------|:----:|:-------------:|:---------------:|:-------:|
| `.b` | 8    | 1             | 1024            | signed  |
| `.n` | 4    | 2             | **512**         | signed  |
| `.c` | 2    | 4             | **256**         | signed (crumb) |
| `.sc` | 2   | 4             | 256             | signed alt mapping |
| `.ubit` | 1 | 8             | **128**         | unsigned |
| `.sbit` | 1 | 8             | 128             | signed ({+1,-1}) |

### 关键事实（硅级验证）

一 byte 的 N 个 sub-byte 值**全部累加到同一个输出 cell**——不是分发到 N 个
不同 K-stride cell。所以：

- **每 cell 每 packet 仍然 1 个 int32 输出** + K=32 次累加
- **weight tile 字节数 = 1/N**（int4 用 int8 一半，int2 四分之一，int1 八分之一）
- **吞吐优势在 VTCM 带宽，不在 MAC 速率**

demo05 的实测结果印证这点：

```
weight.b byte=0x01 → out = 32   (32 × 1)
weight.n byte=0x07 → out = 112  (16 × (0+7))
weight.c byte=0x55 → out = 32   (8 × 4×1)
weight.ubit 0xFF  → out = 32   (4 × 8×1)
```

详见 `Agent/hmx_int4_combos_analysis.md` §3.3。

### 公式（用于写 C oracle）

对一个 byte `B`，固定 A=1 全 tile、其它 byte=0：
- `weight.b`：`out += (bytes/cell) × int8_reinterp(B)` = `32 × signed(B)`
- `weight.n`：`out += 16 × (hi_nib_i4(B) + lo_nib_i4(B))`
- `weight.c`：`out += 8 × Σ(4 signed crumbs of B)`
- `weight.ubit`：`out += 4 × popcount(B)`
- `weight.sbit`：`out += 4 × (popcount(B) + (8-popcount(B))·(-1))` = 4×(2·popcount − 8)（未实测，推算自 signed 1-bit 语义）

### `weight.n:2x` —— int4 双倍宽度变体 (F + P)

**ISA 汇编**：`weight.n = mxmem(Rs, Rt):2x`（也存在 `:2x:after / :2x:deep / :2x:single / :2x:drop / :2x:dilate`）

**已知**：
- `weight.n*` 在 v75 skel 里用了 232 次，其中 `:2x` 家族 81 次。
- cycle: `.n:2x` 比普通 `.n` 慢 **16%**（HAP_perf_get_pcycles 实测）
- 功能上异常：`byte=0x11` 给 `out = 544`（= 32 × 17）而非预期 32 或 64。sim 和硅
  输出一致，说明不是 sim artifact 但语义未解。

**推测** (P)：
- `:2x` 读 2 条 128-B 线（1 KiB 而非 512 B）
- 当你给的 tile 不足时，HMX 可能绕回读同一块 memory 多次，导致异常累加
- 没有给配套的精确语义文档

**生产建议**：**不要在 matmul 里用 `weight.n:2x`**。ch13 的 int4×int8 tutorial **完全
不用 `:2x`**。如果你写 conv kernel 迁 QNN 的模式，照抄 QNN 同样的 Rt 字段 + tile
大小，别自己发明。

### `weight.sm` —— sparse mask (P，QNN 未使用)

和稀疏权重编码配合。QNN v75 skel 里 0 次使用。本指南不展开。

## 浮点 weight 类型 (F)

### `.hf` — fp16 weight

和 `activation.hf` 配对的唯一路径。

```c
asm volatile("mxclracc.hf" ::: "memory");
asm volatile("{ activation.hf = mxmem(%0,%1)\n"
             "  weight.hf     = mxmem(%2,%3) }"
             :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
asm volatile("mxmem(%0,%1):after.hf = acc" :: "r"(out), "r"(0) : "memory");
```

- tile 2 KiB = 32×32×2 bytes = 1024 fp16
- 走独立的 xfp 浮点 MAC 路径（`hmx_mult_xfp`），和整数 MAC 阵列并联
- 详见 `Agent/hmx_int4_combos_analysis.md` §3.1

### `.f8` — fp8 weight (QNN 未使用)

同样 0 次使用。不展开。

## 修饰符速查表

每种 weight 类型都可搭配一套"位置修饰符"。和 `activation.*` 基本同名同义：

| 修饰符 | 含义（推测）| matmul 用？ | QNN 实测 |
|--------|-------------|:----------:|:--------:|
| (无) | 标准读 | ✓ | ✓ |
| `:dilate` | Conv2D dilate=1 | ✗ | ✓ |
| `:single` | Conv 单空间点 | ✗ | ✓ |
| `:deep` | 深通道 | ✗ | ✓ |
| `:drop` | 丢最后一线 | ✗ | ✗ |
| `:after` | convert 之后发 MAC | 特殊 | 部分 |
| `:2x` | （仅 `.n`）双倍宽度 | **不推荐** | ✓ |

## 完整 intrinsic 速查（`hmx_hexagon_protos.h` 提取）

```
weight.b     = mxmem(Rs, Rt)                  ; Q6_weight_b_mxmem_RR
weight.b     = mxmem(Rs, Rt):after
weight.b     = mxmem(Rs, Rt):dilate
weight.b     = mxmem(Rs, Rt):deep
weight.b     = mxmem(Rs, Rt):drop
weight.b     = mxmem(Rs, Rt):single
weight.n     = mxmem(Rs, Rt)                  ; Q6_weight_n_mxmem_RR
weight.n     = mxmem(Rs, Rt):after
weight.n     = mxmem(Rs, Rt):dilate
weight.n     = mxmem(Rs, Rt):deep
weight.n     = mxmem(Rs, Rt):drop
weight.n     = mxmem(Rs, Rt):single
weight.n     = mxmem(Rs, Rt):2x               ; Q6_weight_n_mxmem_RR_2x (独有)
weight.n     = mxmem(Rs, Rt):2x:after
weight.n     = mxmem(Rs, Rt):2x:dilate
weight.n     = mxmem(Rs, Rt):2x:deep
weight.n     = mxmem(Rs, Rt):2x:drop
weight.n     = mxmem(Rs, Rt):2x:single
weight.c     = mxmem(Rs, Rt)                  ; Q6_weight_c_mxmem_RR
weight.c     = mxmem(Rs, Rt):after/dilate/deep/drop/single
weight.sc    = mxmem(Rs, Rt)                  ; Q6_weight_sc_mxmem_RR
weight.ubit  = mxmem(Rs, Rt)                  ; Q6_weight_ubit_mxmem_RR
weight.sbit  = mxmem(Rs, Rt)                  ; Q6_weight_sbit_mxmem_RR
weight.sm    = mxmem(Rs, Rt)                  ; Q6_weight_sm_mxmem_RR
weight.hf    = mxmem(Rs, Rt)                  ; Q6_weight_hf_mxmem_RR
weight.hf    = mxmem(Rs, Rt):after/dilate/deep/drop/single
weight.f8    = mxmem(Rs, Rt)                  ; Q6_weight_f8_mxmem_RR
```

## 选型决策树

```
需要什么精度的 weight？
├── int8           → weight.b
├── int4           → weight.n (推荐) 或 Cast to int8 + weight.b
├── int2           → weight.c
├── 1-bit (unsigned) → weight.ubit
├── 1-bit (signed)   → weight.sbit
├── fp16           → weight.hf (配 activation.hf)
└── 其他            → 软件侧分解成上面之一

要不要尝试 weight.n:2x？
└── 除非你在复现 QNN conv pattern 且有实测参考 → 否则不要。
```

## 踩坑提醒

1. **`.b` 当 unsigned**：`.b` 是 **signed** int8。`0xFF` 被当成 -1 不是 255。
2. **`.c` 的 signed 语义**：demo05 用 `0xFF`（4 个 `11` crumb）给 out = -32，证实 **signed 2-bit**，不是 unsigned。
3. **忘了 tile 字节数缩放**：写 `.n` 时仍然 alloc 1 KiB 没问题（多分配不报错），但如果你 alloc 512 B 又 `memset` 1024 B 就会越界踩到下一块 VTCM。
4. **`:2x` 语义不明**：除非你很清楚，否则不要用。
5. **sub-byte tile 内 (K,col) 精确字节映射**：本指南目前**仅保证全 tile 同 byte 值情况下公式成立**。要打 tile 里单个 (K,col) 的精确 nibble/bit 位置，需要 single-hot-nibble 探针验证（ch13 tutorial step 3 会处理）。

## 参考

- 设备上 8 种 sub-byte unpacker 的证据：`Agent/hmx_int4_combos_analysis.md` §3.2
- `weight.n:2x` 异常值 544 的记录：`Agent/hmx_int4_combos_analysis.md` §6

## 下一章

[08 VLIW packet](08-instr-mac-packet.md) —— `activation + weight` 为什么必须同 packet，
合法/非法组合。
