# HMX 如何做 int4×int16 / int4×int8 / int4×int4 矩阵乘法

**范围**：v75 HMX 整数 MatMul 的架构解读 + 三种 int4×* 组合的分解方案。

**证据链**（三端一致，硅上验证过）：
- **ISA 头**：`tools/hexagon-sdk/.../hmx_hexagon_protos.h` — 给出指令清单
- **x86 ISS 模拟器**：`.../Tools/lib/iss/libhexagonissv75.so` — 带调试符号的 v75 功能模型
- **设备二进制**：`tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so` — 实际跑在 v75 cDSP
- **真机探针**：`example/hmx_matmul_device/probe_subbyte_device.c` — SM8650 v75 `HAP_perf_get_pcycles` 结果（2026-04-20，20/20 PASS）

**证据级别**：
- **(F)** 由上述四种证据之一或多个直接支撑
- **(P)** 从 (F) 推断的最合理架构假设，可被 RTL 或更深探针证伪

---

## 1. HMX 文档性：为什么只能靠反汇编

Qualcomm 公开 SDK **没有 HMX 架构手册**：

| 文档 | 有 HMX 章节 | 有 int4/nibble/weight.n |
|------|:---:|:---:|
| V75 Programmer's Reference Manual (661 页) | ✗ | ✗ |
| V75 HVX Programmer's Reference Manual (304 页) | ✗ | ✗ |
| QAIRT/QNN docs | 仅出现为 profiling 资源名 | ✗ |
| `hmx_hexagon_protos.h` | ✓ 列 intrinsic | ✓ 列 `weight.n`，无语义 |

所以"想搞懂 HMX"的唯一路径就是 ISA 头 + 反汇编 + 真机探针。本文就是这么做的。

---

## 2. HMX ISA 清单 (F)

从 `hmx_hexagon_protos.h` 枚举。

### 2.1 MAC operand 类型

| 侧 | 汇编后缀 | 含义 | 本文简称 |
|----|---------|------|---------|
| activation | `.ub` | uint8 | int 侧唯一 |
| activation | `.hf` | fp16 | fp 路径 |
| activation | `.f8` | fp8 | — |
| weight | `.b`     | int8 (signed) | |
| weight | `.n`     | int4 (signed nibble) | 独有 `:2x` 变体 |
| weight | `.c`     | int2 (signed "crumb") | **不是 compressed** |
| weight | `.sc`    | signed 2-bit alt mapping | |
| weight | `.ubit`  | unsigned 1-bit | |
| weight | `.sbit`  | signed 1-bit | |
| weight | `.hf`    | fp16 | |
| weight | `.f8`    | fp8 | |
| weight | `.sm`    | sparse mask | |

**硬约束**：
- **没有** `activation.n` / `activation.b` / `activation.h/.uh`。所有"非 uint8"
  激活（int4, int8-signed, int16）都必须在 CPU 侧偏移/拆分后喂 `.ub`。
- 累加器只有两种：**int32 定点**（`mxclracc` 清零）和 **xfp 浮点**
  （`mxclracc.hf` 清零）。两者不能并存，切换前必须重新清零。

### 2.2 读回（convert）

- `mxmem(Rs, Rt):after.uh = acc:2x1` — int32 acc → uint16（可加 `:sat`）
- `mxmem(Rs, Rt):after:retain.uh = acc:2x1` — 同上但保留 acc 供再次 convert
- `mxmem(Rs, Rt):after.hf = acc` — xfp → fp16
- 没有原生 int32 读回。**要完整读 int32 必须 dual-scale**（retain 一次 scale=1.0，再一次 scale=2⁻⁸），本仓库 `example/hmx_matmul_int16/` 已验证 bit-exact。

---

## 3. HMX 微架构（硅级验证过）

### 3.1 三个物理 MAC 基元 (F)

模拟器里有一张 **3 项函数指针表** `hmx_mult_body_ptr_table @ 0xc809c0`：

```
[0] 0x41b630  hmx_mult_fxp           定点 byte MAC
[1] 0x41b6c0  hmx_mult_fxp_subbyte   定点 sub-byte (int4/int2/int1)
[2] 0x41b430  hmx_mult_xfp           扩展浮点 MAC (fp16)
```

`hmx_mult_fxp` 的核心 4 行 x86（一次 MAC 的功能语义）：

```asm
movsbl  %r9b, %edx       ; edx = sign_extend(wt_byte as int8)
movzbl  %r8b, %esi       ; esi = zero_extend(act_byte as uint8)
imul    %edx, %esi       ; 32-bit signed product
add     %esi, ACC[cell]  ; int32 accumulate
```

`hmx_mult_fxp_subbyte` 同结构，只是在调用前由上层把一个 byte 拆成 N 个
sub-byte；每个 sub-byte 通过同一个 `imul` 路径乘以同一 activation byte。

`hmx_mult_xfp` 是独立的浮点流水（函数体 frame 0x68 B，含 `divb` 归一化），
走单独的 xfp accumulator。

### 3.2 sub-byte unpack：`hmx_unpack_ptr_table` (F)

另一张 **8 项函数指针表** `@ 0xc809e0`，索引对应 weight type：

| idx | 函数 | 每字节产生 | 对应 ISA |
|:---:|------|:---:|---|
| 0 | `hmx_unpack_byte_from_byte` | 1 个 int8 | `weight.b` |
| 1 | `hmx_unpack_sm_from_byte` | (稀疏 mask) | `weight.sm` |
| 2 | `hmx_unpack_nibble_from_byte` | **2 个 int4** | `weight.n` |
| 3 | `hmx_unpack_crumb_from_byte` | **4 个 int2** | `weight.c` |
| 4 | `hmx_unpack_scrumb_from_byte` | 4 个 signed 2-bit | `weight.sc` |
| 5 | `hmx_unpack_1bit_from_byte` | **8 个 unsigned 1-bit** | `weight.ubit` |
| 6 | `hmx_unpack_1sbit_from_byte` | 8 个 signed 1-bit | `weight.sbit` |
| 7 | `hmx_unpack_none` | 1 (passthrough) | `weight.hf` |

每条 unpack 函数都是几行 shift+mask，例如 nibble：
```asm
sar %cl, %eax       ; cl = 0 (低 nibble) 或 4 (高 nibble)
shl $4, %al
sar $4, %al         ; sign-extend 4-bit → 8-bit
```

### 3.3 sub-byte MAC 的本质：一 byte N 值 → 累加到**同一**输出 cell (F，硅级验证)

**`probe_subbyte_device.c` 在 SM8650 真机上的核心实测**（固定 A=1 everywhere）：

| weight type | byte_val | 硅实测 out[0][0] | 公式（硅级验证） |
|-------------|---------:|:---------------:|------|
| `.b`  int8  | 0x01 → 32, 0x02 → 64, 0x7F → 4064, 0xFF → 65504(=-32) | ✅ | `32 × byte_i8` |
| `.n`  int4  | 0x01 → 16, 0x10 → 16, 0x11 → 32, 0x07 → 112, 0x77 → 224, 0x88 → 65280(=-256) | ✅ | `16 × (hi_i4 + lo_i4)` |
| `.c`  int2  | 0x01 → 8, 0x55 → 32, 0xFF → 65504(=-32) | ✅ | `8 × Σ(4 signed crumbs_i2)` |
| `.ubit` int1 | 0x01 → 4, 0x80 → 4, 0xFF → 32 | ✅ | `4 × popcount(byte)` |

**`bytes/cell × N = 32 K-step` 不变量**永远成立：

| weight type | N sub-byte/byte | bytes 读/cell/packet |
|-------------|:---:|:---:|
| `.b`  | 1 | 32 |
| `.n`  | 2 | 16 |
| `.c`  | 4 |  8 |
| `.ubit` | 8 |  4 |

即：**sub-byte 类型不是"同 cycle 多 MAC"**，而是**"同 MAC 数但 tile 字节更少"**。
一 byte 里的 N 个 sub-byte 值在 cell 内部沿 K 维**顺序累加到同一输出**。

### 3.4 cycle 吞吐硅实测（2026-04-20）

200 MAC packets per run，`HAP_perf_get_pcycles()`：

| weight type | pcycles | cyc/MAC | vs int8 |
|-------------|--------:|--------:|:-------:|
| `weight.b`     |  9 155 | 45.77 | 1.000× |
| `weight.n`     |  9 507 | 47.53 | 0.963× |
| `weight.c`     |  9 507 | 47.53 | 0.963× |
| `weight.ubit`  |  9 526 | 47.63 | 0.961× |
| `weight.n:2x`  | 10 609 | 53.05 | **0.863×** |

sub-byte 与 int8 的 cyc/packet 差别 < 4%（测量噪声）。**`.n:2x` 慢 16%**
是因为一次读 2×128 B 权重线 → VTCM 带宽压力翻倍，MAC cell 等数据。

### 3.5 sub-byte 的吞吐优势到底体现在哪里

**不在每 packet 的 cycle 数或 MAC 数**，而在 **VTCM 权重带宽**：

- int8: 1024 B/tile/K=32
- int4:  512 B/tile/K=32
- int2:  256 B/tile/K=32
- int1:  128 B/tile/K=32

`Agent/int4_matmul_optimization_log.md` iter 4 已独立发现 v75 HMX MatMul
在这个硬件上是 VTCM-bound。所以权重 tile 减半/四分/八分字节直接翻译
成等比例的实际 speedup——marketing 说的 "int4 = 2× int8 throughput"
就是这个意思。

### 3.6 物理乘法器到底多宽？(P)

sim 与硅上的 cycle 数全部相等（int8/int4/int2/int1 ~4% 以内）说明：**每个
MAC cell 每 cycle 仍只产出 1 个 int32 累加结果**。一字节内 N 个 sub-byte
值是在 cell 内部**时序累加**到同一输出 cell，不是并联输出到 N 个 cell。

最简单且与所有观测一致的硅结构猜想：

> **每个 cell 一个 int32 累加器 + 一个可配置精度的乘加路径**。当 weight 类型
> 为 `.n/.c/.ubit` 时，**load 阶段**把一字节拆成 N 个 sub-byte，**MAC 阶段**
> 按 K 维循环 N 次，把 N 个 sub-byte × activation 的乘积依次累加。

这和"cell 内是 8 个 u8×1bit 并联输出"的假设不同——后者会让每 packet 产出
更多结果，但**硅实测否认了这一点**。

物理上的乘法器位宽无法从功能/cycle 探针区分，要 RTL 或 die-shot。

---

## 4. Tile 布局

### 4.1 已实测（F，`hexagon_hmx_matmul_native_int.md`）

**activation tile (2 KiB)**：
```
A_byte(phys_row, K, stream) = 128·phys_row + 4·K + (stream ? 3 : 1)
    phys_row ∈ 0..15, K ∈ 0..31, stream ∈ {0,1}
    logical_row ir → (phys_row = ir & 15, stream = ir >> 4)
```
`4·K + 0` 和 `4·K + 2` 字节必须为 0（16-bit slot 的低字节被忽略，留给 fp16）。

**weight tile (int8, 1 KiB)**：
```
W_byte(K, col) = 128·(K >> 2) + 4·col + (K & 3)
```
每 128 B 一条 HVX 线装 4 K-rows × 32 cols × int8。

### 4.2 其他 weight 类型 tile 字节数（F，硅验证）

| 类型 | tile 字节 / 32³ | 128-B 线数 | 每线 K-rows |
|------|:---------------:|:----------:|:-----------:|
| int8 `.b` | 1024 | 8 | 4 |
| int4 `.n` |  512 | 4 | 8 |
| int2 `.c` |  256 | 2 | 16 |
| int1 `.ubit` | 128 | 1 | 32 |
| fp16 `.hf` | 2048 | 16 | 2 |

### 4.3 sub-byte tile 内 nibble/bit 的打包次序 (P)

硅探针只验证了"全字节同值时每 cell 累加得多少"，**没验证精确的位-到-(K,col) 映射**。
要完整决定 int4 tile layout，需要 single-hot-nibble 探针（在 512 B 里只有一个
位置写 nibble=0x7、其余 0，dump 输出并反推）。已列为 §6 的遗留实验。

### 4.4 为什么 act 2 KiB 而 wt int8 只有 1 KiB

activation tile byte layout **要兼容 fp16**（fp16 每元素 2 B，32 列 × 32 K
= 2 KiB），所以 uint8 activation 只用每 16-bit slot 的高字节。weight 侧不共
享：weight tile 字节数就是 `N_K × N_col × element_bytes`。

---

## 5. 三种组合的 int4×* 分解方案

所有三种共享同一骨架：
```
for each (m_tile, n_tile):
    mxclracc
    for each k_tile:
        pack activation into .ub tile(s)
        pack weight into .n tile
        { activation.ub = mxmem(...); weight.n = mxmem(...) }
    readback
    scalar combine + zero-point correction
```

差别只在 (a) activation 分解成几个 `.ub` 流、(b) 需要几个 partial、(c) 修正项。

### 5.1 int4 × int4

输入 `a_q ∈ [-8,7] (i4), w_q ∈ [-8,7] (i4)`。

```
a_u = a_q + 8           ∈ [0, 15]          存入 .ub 低 4 位
a_q · w_q = (a_u - 8) · w_q = a_u · w_q - 8 · w_q
```

**1 条 MAC packet**：`{ activation.ub; weight.n }`。
读回范围：K=32 下 ≤ 15·8·32 = 3840，**单次 `.uh:sat` 即可**，无需 dual-scale。
合并：`out[m,n] = acc[m,n] − 8·ColSumW[n]`。

**短 K 成本：1 MAC + 1 convert = 2 HMX packets / 32³-tile**（硬件下限）。

### 5.2 int4 × int8

输入 `a_q ∈ [-128,127] (i8), w_q ∈ [-8,7] (i4)`。

```
a_u = a_q + 128         ∈ [0, 255]
a_q · w_q = a_u · w_q - 128 · w_q
```

**1 条 MAC + 修正**。读回范围：
- K ≤ 32：≤ 255·8·32 = 65 280，**单 convert** 即可
- K > 32：需 dual-scale

合并：`out[m,n] = acc[m,n] − 128·ColSumW[n]`。

### 5.3 int4 × int16

输入 `a_q ∈ [-32768,32767] (i16), w_q ∈ [-8,7] (i4)`。

int16 超 uint8，必须分两路：
```
a_u = a_q + 32768  ∈ [0, 65535] = 256·A_h + A_l
a_q · w_q = 256·(A_h · w_q) + A_l · w_q − 32768·w_q
```

**2 条 partial MAC**（A_h · W, A_l · W），每 partial 每 K 步 ≤ 255·8 = 2040。
K=32 可单 convert，K > 32 仍需 dual-scale。

合并：`out[m,n] = (P_hi[m,n] << 8) + P_lo[m,n] − 32768·ColSumW[n]`。

### 5.4 三表对比

| 组合 | act 分解 | wt | partials | 修正 | 短 K packets/tile |
|------|---------|----|:---:|------|:---:|
| int4×int4  | a+8     | `.n` | 1 | −8·ColSumW    | 2 |
| int4×int8  | a+128   | `.n` | 1 | −128·ColSumW  | 2 |
| int4×int16 | A_h, A_l | `.n` | 2 | (P_hi<<8)+P_lo−32768·ColSumW | 4–6 |

`.n:2x` 在 K ≥ 64 时可再砍一半 MAC issue 数，但 cyc/MAC 实测 +16%
（见 §3.4），所以仅在 VTCM 吞吐盈余且 K 足够大时才值得切换。

---

## 6. 遗留问题

1. **int4 tile 精确 nibble-to-(K,col) 映射**：§4.3，需 single-hot-nibble 探针。
   最小验证：一个 byte 里写 `0x70` 其余全 0，dump 输出反推哪个 K 坐标被激活。

2. **`weight.n:2x` 语义**：sim 与硅都给出 `byte=0x11 → 544`（=32×17），pattern
   不明。`:2x` 比 `.n` 慢 ~16%，但多做了多少 MAC / 覆盖多大 K 待定。

3. **`hmx_matmul_qnn` HMX 路径 graphExecute err 1003**：本仓库现有 kernel
   走 `weight.b`（int8 签名重解释），迁 `weight.n` 同时就能验证 err 1003 是
   tile-shape 问题还是 OpPackage signature 问题。

---

## 附录 A：关键反汇编锚点（v75）

**`libQnnHtpV75Skel.so`**（设备二进制，跑在 cDSP）：

| 符号 | 地址 | 说明 |
|------|-----|------|
| `hmx_convhbh_5x5_stride1` | 0x213bc0 | int8-weight conv |
| `hmx_convhnh_5x5_stride1` | 0x213e60 | **int4-weight conv**，VLIW `{ activation.ub:above; weight.n:dilate }` |
| `hmx_convhnh_NxN_stride1` | 0x214ac0 | |
| `matmul_qu8xqi8_32` | 0x21be60 | HVX MatMul fallback (非 HMX) |
| `expand_bq_pkweights_s8` | 0x2ea040 | LPBQ int4 → int8 展开器 |

**`libhexagonissv75.so`**（x86 模拟器，带调试符号）：

| 符号 | 地址 | 说明 |
|------|-----|------|
| `hmx_mult_body_ptr_table` | 0xc809c0 | 3 项：fxp / fxp_subbyte / xfp |
| `hmx_unpack_ptr_table` | 0xc809e0 | 8 项 sub-byte unpacker |
| `hmx_mult_fxp` | 0x41b630 | 核心 u8×i8 MAC（见 §3.1）|
| `hmx_mult_fxp_subbyte` | 0x41b6c0 | sub-byte 变体 |
| `hmx_mult_xfp` | 0x41b430 | fp16 路径 |

**ISA 指令编码**（从设备二进制抠出）：所有 HMX MAC-load 共享 `ICLASS=0x9, op24=0x2`，
低 4 位 type code：`0x0=b, 0x1=n, 0x2=c, 0xf=hf`。`.f` 刻意跨到最远值，和定点家族隔开。

---

## 附录 B：探针代码与可复现步骤

**x86 ISS（模拟器）**：
```sh
cd /home/zzh/work/qcom_htp
source scripts/env.sh
bash example/hmx_matmul_int16/build.sh

H2=tools/h2-install
hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    -- "$H2/bin/booter" --ext_power 1 --use_ext 1 --fence_hi 0xfe000000 \
       example/hmx_matmul_int16/probe_subbyte_full

hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    -- "$H2/bin/booter" --ext_power 1 --use_ext 1 --fence_hi 0xfe000000 \
       example/hmx_matmul_int16/probe_weight_types
```

**真机**（SM8650 v75，USB adb）：
```sh
cd example/hmx_matmul_device
bash build.sh                     # 编 libprobe_subbyte_device.so
bash run_subbyte_probe.sh         # 自动 adb push + run
./verify_subbyte_results.py build/probe_subbyte_result.txt
# → Summary: 20 PASS / 0 FAIL
```

---

## 参考

- `Agent/hmx_u8xi8_matmul_layers.md` — u8×i8 32³ MatMul 的 L1→L8 分层实现
- `Agent/int4_matmul_optimization_log.md` — 本仓库 int4×int16 QNN kernel 的优化迭代
- `example/hmx_matmul_int16/` — bit-exact int16 实现 + 各类 sim 探针
- `example/hmx_matmul_device/` — 真机探针 + 自动校验器
- `example/hexagon_hmx_matmul_native_int.md` — tile 布局实测日志
