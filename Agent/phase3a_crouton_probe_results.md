# Phase 3A — Crouton_8 consume probe results (2026-04-23)

> ⚠️ Historical note from Phase 3A/B research. Files referenced below
> (`kernel/pack_act_hvx.c`, `kernel/hmx_core{,_v2}.c`, `src/HmxMatMulV{2-7}Op.cpp`,
> `src/run_phase3_probe.cpp`, etc.) were moved to
> `example/hmx_matmul_phase3/_archive/` in the 2026-04-25 V8-only cleanup.
> Content of this note is still accurate as RE history. See
> `docs/qnn_custom_op_sop.md` for the current V8 path.


Device: SM8650 v75 (ssh oneplus). Probe binary:
`example/hmx_matmul_phase3/build/aarch64/run_phase3_probe`.

## 核心结论

**QNN 确实把 Crouton_8 + Indirect 签名 auto-link 到 ForceFormat_Crouton_b 上游 pack op**，把 block_table 指针数组传给我们的 kernel。机制完全可用。

但 **Crouton_8 block 格式不是 HMX mxmem 直接消费格式** ——每块 2048 字节里只有 256 字节真实数据 (8 rows × 32 cols × 1 byte)，其余 1792 字节是 padding。HMX 一次 mxmem load 读 2KB，但需要 32 logical rows，Crouton_8 只能给 8 rows/块 —— **需要 4 块拼接成一个 HMX tile**。

## 具体测量（32×32 int8 tensor）

输入形状：`[1, 1, 32, 32]`，testpattern `aBuf[i] = (i*37) & 0xFF`。

```
act_nblk = 4
wt_nblk  = 4
block_shape = [1, 8, 8, 32]
```

### Block 0 实际字节（activation）

```
offset  00  01  02  03  04  05  06  07  08  09  0a  0b  0c  0d  0e  0f
0x00    00  25  4a  6f  94  b9  de  03  28  4d  72  97  bc  e1  06  2b
0x10    50  75  9a  bf  e4  09  2e  53  78  9d  c2  e7  0c  31  56  7b
0x20    a0  c5  ea  0f  34  59  7e  a3  c8  ed  12  37  5c  81  a6  cb
0x30    f0  15  3a  5f  84  a9  ce  f3  18  3d  62  87  ac  d1  f6  1b
```

对照 `aBuf`:
- `aBuf[0..15]  = (0*37, 1*37, ..., 15*37) & 0xFF = 00 25 4a 6f 94 b9 de 03 28 4d 72 97 bc e1 06 2b` ✓
- `aBuf[16..31] = 50 75 9a bf e4 09 2e 53 78 9d c2 e7 0c 31 56 7b` ✓
- `aBuf[32..47] = a0 c5 ea 0f ...` ✓

**结论**: Block 0 bytes 0..31 = row 0 (32 K-bytes), bytes 32..63 = row 1, bytes 64..255 = rows 2..7。之后 256..2047 字节是 padding。

### Block layout 规则

`block_shape = [1, 8, 8, 32]`:
- Dim 0 = 1 (tensor dim 0 batch, chunk size 1)
- Dim 1 = 8 (tensor dim 1 "H", chunk size 8, padded from 1 to 8)
- Dim 2 = 8 (tensor dim 2 "W" = M, chunk size 8)
- Dim 3 = 32 (tensor dim 3 "C" = K, chunk size 32)

对 `[1, 1, 32, 32]` 张量:
- 块数 = ⌈1/1⌉ × ⌈1/8⌉ × ⌈32/8⌉ × ⌈32/32⌉ = 1 × 1 × 4 × 1 = 4 ✓

每块 2048 字节 = 1×8×8×32。真实数据密度 = 1×1×8×32 / 2048 = 256/2048 = **12.5%**。

## 对 Phase 3B 的含义

### 问题

我们 Phase 2 HMX tile 格式：
- 2-stream interleaved 4-byte cell (bytes [pad, s0, pad, s1])
- 16 phys_rows × 128 bytes = 2048 B
- 32 logical rows per tile (via 2-stream)

Crouton_8 block 格式：
- Row-major 连续 (8 rows × 32 cols × 1 byte)
- 256 B 数据 + 1792 B padding per 2048 B block
- **只有 8 rows per block**

不一一对应。

### 两条路（Phase 3B 需要决定）

**路 X**: 接受 Crouton 密度低，用 `:cm` modifier
- QNN ConvLayer 用 `activation.ub = mxmem(p, Rt):cm`。Phase 1 P4 probe 在 2-stream tile 上测得输出是 plain 的一半 (output=16 vs 32) —— 但那时没有在 **row-major tile** 上测过 `:cm`。
- 假设: `:cm` 是 "convolution mode" = 从 row-major 8-row block 按 column 自动广播 / interpretive 地读成某种内部格式
- 需要：新一轮硅探针，测 `:cm` + Crouton row-major input 的 MAC 输出

**路 Y**: 寻找更 HMX-friendly 的 QNN layout
- `QHPI_Layout_Weights8x4` (enum=8) —— 未文档化，可能针对 weight matrix
- `QHPI_Layout_Crouton4x1` / `Crouton2x2` —— 不同 chunk 比例
- 各自探测 block_shape，找密度高的

**路 Z**: block-level gather
- 接受 Crouton_8 低密度
- Kernel 里 1 次 mxmem 读 1 个 Crouton block = 8 rows 数据 + padding
- 需要多次 mxmem 完成 32 rows HMX tile
- HMX MAC 可能天然接受 "只读 8 rows" 的场景（某种独立 modifier）

**路 W**: 在上游放独立 HVX op 把 Crouton 转成我们期望的 HMX tile 格式
- 回到 Phase 2 的 tile 格式，但把 pack 从 matmul op 里挪到单独 HVX op
- QNN scheduler 可以并行这个 HVX pack + 其它 HVX 工作 → 4 HVX threads
- 保留 Phase 2 的 HMX consumption 方式

## 推荐 Phase 3B 策略

**先做路 W（最低风险）**:
- Phase 2 的 HMX binding 已经验证且 6.5×-8.7× over baseline
- 把现有 pack_weight_32x32 / prepack_activation HVX code 抽成独立 QHPI op "PackToHmxTile"
- 声明 MT=true, QHPI_RESOURCE_HVX → QNN scheduler 给它 4 HVX threads
- 我们 matmul op 不变，但 input 签名改成 "consume packed HMX tiles directly"（从 PackToHmxTile 的输出）
- 只要 QNN 支持 custom op → custom op 的 dataflow, 就可以实现

**再看路 X/Y（理想 but 需额外 RE）**:
- 如果路 W 验证了 "分离 pack 到独立 HVX op 真的并行"，且性能和 QNN 接近
- 那可以进一步 RE `:cm` + row-major 让 HMX 直接消费 Crouton

## 下一步命令

```bash
# Phase 3B 路 W：把 pack 变成独立 op
# 1. 新 op: PackActivationToHmxTile (HVX, MT=true)
#    - input: uint16 activation [1,1,M,K] Flat4 Direct
#    - output: uint8 HMX tile bytes [M/32, K/32, 2048] Flat4 Direct
# 2. 同 PackWeightToHmxTile (HVX, MT=true)
# 3. Matmul op: 只 HMX MAC, 从上游 op 的输出读已 packed 的 HMX tiles
# 4. QNN scheduler 会把 3 个 op 挂成 graph, Pack ops 并行 4 HVX threads
```

## 验证代码位置

- 探针: `example/hmx_matmul_phase3/src/HmxMatMulPhase3Op.cpp` 的 `phase3_hmx_matmul_kernel`
- Host: `example/hmx_matmul_phase3/src/run_phase3_probe.cpp`
- Build: `example/hmx_matmul_phase3/build.sh`
- Run: `bash example/hmx_matmul_phase3/run_on_device.sh --shape 32,32,32`
