# 05 · `bias = mxmem` / `mxmem2`

**TL;DR**：convert 的 per-column scale（128 个 u16 f16）来自 bias slot。
`bias = mxmem(Rs)` 读 256 B（128 × u16），`mxmem2(Rs)` 读 512 B 支持扩展
bias 格式。大多数情况用 `mxmem`，带 relu6 等 clamp 时用 `mxmem2`。

## 作用 (F)

HMX 的 **convert 阶段**需要两个参数：
1. 要 convert 的 acc（内部寄存器里）
2. **一列 128 个 f16 scale**（存在 bias slot 里）

convert 大致做：
```
out_u16[i][j] = saturate_or_wrap(acc[i][j] × f16_to_float(bias[j]) / 2)
```

所以 bias 是**每输出列独立的 scale**，不是整个 kernel 的 global scale。这就是
为什么同 output tile 里不同 col 可以有不同的量化 step。

## `bias = mxmem(Rs)` —— 标准 bias load (F)

**汇编**：`bias = mxmem(Rs32)`
**Intrinsic**：`Q6_bias_mxmem_R`（看 `hmx_hexagon_protos.h`）

读 `Rs` 指针处的 **256 字节 = 128 × u16** 作为 bias。

```c
uint16_t *bias_vtcm = ...;  // 必须 VTCM 地址
asm volatile("bias = mxmem(%0)" :: "r"(bias_vtcm) : "memory");
```

**对齐要求**：`Rs` 128 B 对齐（HVX 向量线）。推荐把 bias 放在 VTCM 某 2 KiB
边界处，一定安全。

**失败模式**：
- `Rs` 不在 VTCM → sim 报权限错；真机直接 hang
- 没预先填数据 → 读进来是垃圾，convert 出一片噪声

## `bias = mxmem2(Rs)` —— 双倍宽 bias load (F)

**汇编**：`bias = mxmem2(Rs32)`
**Intrinsic**：`Q6_bias_mxmem2_R`

读 **512 字节 = 256 × u16**，扩展的 bias 格式，有两个分量：

- 前 128 u16：per-col scale（和 `mxmem` 同）
- 后 128 u16：per-col clamp / offset（具体格式随 convert 类型）

`example/hmx_matmul_int16/probe_f16_profile.c` 用 `mxmem2` 因为 fp16 convert
的 identity bias 就是 `mxmem2` 格式（256 个 u16：`val + 128 个零`）：

```c
/* fp16 path 典型: mxmem2 bias 前 128 是 f16 scale，后 128 是 zero */
((HVX_Vector *)scale)[0] = Q6_V_vsplat_R((F16_ONE << 16) | F16_ONE);
((HVX_Vector *)scale)[1] = Q6_V_vzero();
asm volatile("bias = mxmem2(%0)" :: "r"(scale) : "memory");
```

## 选 `mxmem` 还是 `mxmem2`？

| 场景 | 用 |
|------|-----|
| int 路径，想读 acc mod 2^16 或 × 某 scale | `mxmem` |
| int 路径，需要 per-col ReLU6 clamp / offset | `mxmem2` |
| fp16 路径，identity scale | `mxmem2`（QNN 成规）|
| fp16 路径，带 clamp / output offset | `mxmem2` |

本仓库 `int16_matmul_hmx.c` 一直用 `mxmem`（int 路径 + 纯 scale 需求）。

## bias 值的选择（demo03 里展开过）

| u16 | f16 | 有效 scale | 用途 |
|-----|-----|:---------:|------|
| `0x4000` | 2.0 | 1.0 | identity（读 acc mod 2^16）|
| `0x3C00` | 1.0 | 0.5 | 读 acc/2 |
| `0x2000` | 2⁻⁷ | 2⁻⁸ = 1/256 | **dual-scale 高 bit** |
| `0x0800` | 2⁻¹³ | 2⁻¹⁴ | 读 acc 高 bit（测试用）|
| `0x0400` | 2⁻¹⁴ | 2⁻¹⁵ | smallest normal 范围 |
| `0x0200` 及更小 | f16 denormal | ≠ nominal | **有 ~1.5× artifact，避免使用**|
| `0x0000` | 0.0 | 0 / 非 0 | **行为有 quirks**（见下）|

### 关于 `0x0000` / denormal 的警告 (F)

硅级探针（`probe_subbyte_device.c` 和本指南 demo03 的早期版本）发现：

- **`bias = 0x0000`**：数学上是 0 × acc = 0，但硅上输出非 0（1-2 之类小值）。
  推测 HMX 内部先做浮点归一化再乘，denormal 0 被当作最小 denormal，产生残余。
- **`bias ≤ 0x0200` (denormal)**：有效 scale ≈ `3 × 2^(-17)` 而不是 f16 标称值——多
  出 1.5×。

**生产代码规则**：**bias 值限制在 f16 normal range（u16 ≥ 0x0400）**。想"不输出"就
不 convert（skip 那条 `mxmem:after.uh = acc` 指令）。

## 典型代码段

```c
/* dual-scale readback 的两次 bias 配对 */
static uint16_t bias_lo[128] __attribute__((aligned(128)));
static uint16_t bias_hi[128] __attribute__((aligned(128)));

void setup_biases(void) {
    for (int i = 0; i < 128; i++) {
        bias_lo[i] = 0x4000;  /* scale 1.0 : 读低 16 bit */
        bias_hi[i] = 0x2000;  /* scale 2^-8: 读高 16 bit */
    }
}

void dual_scale_readback(void *out_lo, void *out_hi) {
    asm volatile("mxclracc" ::: "memory");
    /* ... MAC packet ... */

    /* 第一次读: 低 bit */
    asm volatile("bias = mxmem(%0)" :: "r"(bias_lo) : "memory");
    asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"  /* :retain 保 acc */
                 :: "r"(out_lo), "r"(0) : "memory");

    /* 第二次读: 高 bit (换 bias) */
    asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out_hi), "r"(0) : "memory");
}
```

详见 ch09 convert 章节 + `example/hmx_matmul_int16/int16_matmul_hmx.c:116`。

## 踩坑提醒

1. **忘了 `bias = mxmem`** → convert 读回全 0。
2. **bias 填 f16 denormal** → 数值飘 1.5×，算完发现和 C ref 不 bit-exact。
3. **bias 未对齐 128 B** → sim 可能报 misaligned，硅上可能直接挂起。

## 参考

- Intrinsic 定义：`hmx_hexagon_protos.h`，grep `bias_mxmem`
- 探针数据：`example/hmx_matmul_int16/probe_dual_scale.c`
- 正式 kernel 用法：`example/hmx_matmul_int16/int16_matmul_hmx.c`

## 下一章

[06 activation load](06-instr-activation-load.md) —— `activation.ub/.hf/.f8` + 修饰符。
