# 03 · Tile Layout Cheatsheet

**TL;DR**：一张表记住 act / wt / out / bias 的字节布局公式。所有布局公式都是
硬件写死的，来自 `example/hexagon_hmx_matmul_native_int.md` 的 single-hot-byte
探针实测 (F)，并已在本仓库 `example/hmx_matmul_int16/` 端到端 bit-exact 验证过。

## 速查表

### Activation tile（2 KiB，`.ub` 和 `.hf` 共享 geometry）

| 参数 | 范围 | 含义 |
|------|------|------|
| `phys_row` | 0..15 | 物理行 |
| `K` | 0..31 | K 维度索引 |
| `stream` | 0..1 | 每 phys_row 装两个 logical row |

```
A_byte(phys_row, K, stream) = 128·phys_row + 4·K + (stream ? 3 : 1)
```

**逻辑行到物理的映射**：`logical ir ∈ 0..31 → (phys_row = ir & 15, stream = ir >> 4)`
- `ir=0..15`  → `stream=0`
- `ir=16..31` → `stream=1`

**重要约束**：
- 字节 `4·K + 0` 和 `4·K + 2` **必须为 0**（硬件会读，int 模式下应忽略但不清零可能污染）
- 总字节 = 2 KiB = 16 lines × 128 B/line
- 每 128 B 一条 HVX 线装 2 phys_row 的数据？不对——**每条线对应 1 个 phys_row** 的全部 32 K（因为 phys_row 是外层 index）

**fp16 activation** 用同一个 geometry 但把整个 4-byte 槽当 2 个 fp16（每 stream 一个）。

### Weight tile（int8 时 1 KiB）

```
W_byte(K, col) = 128·(K >> 2) + 4·col + (K & 3)
    K ∈ 0..31, col ∈ 0..31
```

**每 128 B 一条 HVX 线** = 4 个连续 K × 32 cols × int8。8 条线覆盖 K=0..31。

### Weight tile（其它类型，字节数随 element bit-width 线性缩放）

| 类型 | 元素字节 | 单 32³ tile 字节 | 每 128B 线 K-rows |
|------|:-------:|:---------------:|:-----------------:|
| `.b`  int8  | 1     | **1024** | 4 |
| `.n`  int4  | 0.5   | **512**  | 8 |
| `.c`  int2  | 0.25  | **256**  | 16 |
| `.ubit` int1 | 0.125 | **128** | 32 |
| `.hf` fp16  | 2     | **2048** | 2 |
| `.f8` fp8   | 1     | 1024 (P) | 4 (P) |

> **sub-byte weight 的 (K, col) → nibble/bit 精确映射** 目前 (P)：推测 sub-byte
> 沿用 int8 的外层结构 `128·(K>>?) + 4·col + ...`，只是每 byte 装 N 个值。
> 已在本仓库硅级探针 (`probe_subbyte_device.c`) 验证了"一字节里 N 个 sub-byte 值
> 全部累加到**同一**输出 cell"的语义 (F)，但具体位到 K 的映射没 single-hot 探。
> 详见 `Agent/hmx_int4_combos_analysis.md` §4.3。

### Output tile（convert `.uh:2x1` 时 2 KiB = 1024 u16）

```
out_u16[phys_row * 64 + 2 * col + stream]
   logical (ir, jc) → (phys_row = ir & 15, col = jc, stream = ir >> 4)
```

- 每 phys_row 占 64 u16 = 128 B
- 16 phys_row × 128 B = 2 KiB

**convert `.uh:2x2`（未实测，QNN 在 v75 上不用）**输出 geometry 可能不同，
目前本指南不涉及。

### Bias tile（256 B）

- 128 个 `u16`，每个是 f16 值
- 一列（逻辑 `jc`）一个 bias：`bias[jc]` 对应输出列 `jc`
- `mxmem(Rs)` 从 `Rs` 读 256 B

常用 bias 值：

| u16 值 | f16 = | 有效 scale | 用场景 |
|-------|------|:---------:|--------|
| `0x4000` | 2.0 | **1.0** | identity，读 acc 低 16 bit |
| `0x2000` | 2⁻⁷ | **2⁻⁸** | dual-scale 高 bit readback |
| `0x3C00` | 1.0 | 0.5 | 读 acc/2 |
| `0x0000` | 0.0 | 0 | 输出全 0 |

dual-scale 详细 readback 手法见 v2 章节 09。

## 逆向查找函数

给定**逻辑坐标**，速写 pack 代码：

```c
/* logical A[ir][K] → act byte offset */
static inline int act_off(int ir, int K) {
    int phys_row = ir & 15;
    int stream   = ir >> 4;
    return 128 * phys_row + 4 * K + (stream ? 3 : 1);
}

/* logical W[K][jc] → wt byte offset (int8) */
static inline int wt_off_b(int K, int jc) {
    return 128 * (K >> 2) + 4 * jc + (K & 3);
}

/* logical C[ir][jc] → out u16 index */
static inline int out_idx(int ir, int jc) {
    int phys_row = ir & 15;
    int stream   = ir >> 4;
    return phys_row * 64 + 2 * jc + stream;
}
```

demo01 里的 pack 循环其实就是这三个函数的"枚举全部坐标"展开。后续 demo
会把这几个 helper 做成公共 header。

## pack 常见错法与症状

| 错法 | 症状 |
|------|------|
| 把 stream 弄反（`4·K + 3` 当成 stream 0） | `out[0..15]` 全 0，`out[16..31]` 有值 |
| 忘了清理 `4·K + 0/2` 的 "ignored" 字节 | 随机小偏差（取决于上次 VTCM 里的数据）|
| wt `(K>>2)` 写成 `(K>>3)` | out 值变成预期的 2 倍（K 覆盖范围被错摆） |
| 读 out 按 `[ir*32 + jc]` 平铺 | 只有 col=0/stream=0 的 cell 对，别的乱 |

**诊断法**：把 A、W 全填同一个已知值跑，拿 out[0]/out[1]/out[0,16]/out[31,31]
这 4 个"角"比对。`demo01_hello_hmx.c` 的 verify 部分就是这么做的。

## 为什么 act 2 KiB 但 wt 1 KiB？

| | act tile | wt tile（int8） |
|---|---|---|
| 总字节 | 2 KiB | 1 KiB |
| 存多少元素 | 32×32 = 1024 u8 | 32×32 = 1024 i8 |
| 字节密度 | 50%（一半位置必须 0） | 100% |

原因：**act 路径要兼容 fp16 activation**，而 fp16 每元素 2 B，32×32 fp16 正好 2 KiB。
硬件不想给 int/fp 两套 act-FIFO geometry，所以 uint8 activation 也得占 2 KiB，
只用每 16-bit slot 的高字节（`4·K + 1` 和 `4·K + 3`）。weight 侧不共享几何，
按元素 bit-width 线性装。

详细一点的讨论见 `Agent/hmx_u8xi8_matmul_layers.md` 附录 B。

## 最小可运行示例

想实际打开一份 hex dump 看 tile 字节？`example/hmx_matmul_int16/probe_hmx_acc.c`
里有 single-hot-byte 的实测代码。跑 `bash build.sh` 再用 hexagon-sim 跑它，
你能看到"在 act tile 第 N 个字节写 1，out 的第 M 个 cell 就变"的对应关系。

## 下一章

v1 到这里完结。v2 会上：
- **04**：`mxclracc / mxclracc.hf / mxswapacc` 的使用场景
- **05**：bias load 的 mxmem / mxmem2 差别
- **09**：convert 家族（dual-scale readback 原理）
- **11**：修饰符大全

以及新增 demo02–demo05 覆盖溢出、bias scale、tile probe、weight 类型对比。
