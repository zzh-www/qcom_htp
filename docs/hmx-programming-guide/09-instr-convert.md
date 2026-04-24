# 09 · Convert = 读回 acc

**TL;DR**：convert 是把 HMX 内部 acc 写回 VTCM 的唯一方式。整数 acc 一次只读
16 bit，要完整 int32 得用 `:retain` + 两次 convert（dual-scale）。浮点 acc 一次
直接读 fp16。

## 基本语法 (F)

整数路径：
```c
asm volatile("mxmem(%0, %1):after.uh = acc:2x1"
             :: "r"(out), "r"(0) : "memory");
```

浮点路径：
```c
asm volatile("mxmem(%0, %1):after.hf = acc"
             :: "r"(out), "r"(0) : "memory");
```

- `Rs`（`%0`）= output tile 的 VTCM 起始地址
- `Rt`（`%1`）= 0（不分段写）
- `:after` = "MAC 完成之后做 convert"
- `.uh` / `.hf` = 输出类型
- `:2x1` = 输出 geometry（只 int 路径需要）

## 位置修饰符：`:after` vs `:before` (F)

- **`:after`**（常用）：等 MAC packet 队列里所有 MAC 完成后再发 convert
- **`:before`**（少见）：下一次 MAC 之前就 convert，不等当前 MAC 结束

matmul 里**永远用 `:after`**。`:before` 是给 pipeline conv 用的。

## 输出类型：`.uh / .ub / .hf` (F)

### `.uh` — uint16 (int 路径标准)

输出 16-bit unsigned half-word。acc 超 [0, 2^16) 时 **wrap mod 2^16**（默认）
或 **saturate 到 0xFFFF**（加 `:sat`）。

```c
/* wrap: */  asm volatile("mxmem(%0,%1):after.uh     = acc:2x1" ...);
/* sat:  */  asm volatile("mxmem(%0,%1):after:sat.uh = acc:2x1" ...);
```

demo02 验证了两种行为的区别：
- wrap: `A=255·W=127·K=32 → acc=1036320, out = acc mod 2^16 = 53280 (0xD020)`
- sat: 同样 acc → out = 0xFFFF

### `.ub` — uint8 (int 路径, 8-bit 输出)

输出 8-bit，配合 `:cm` / `:sat` 等修饰。

```c
asm volatile("mxmem(%0,%1):after:cm:sat.ub = acc"
             :: "r"(out), "r"(0) : "memory");
```

**matmul 里罕用**（信息太窄）。QNN 在 8-bit 量化 activation 输出场景用。

### `.hf` — half-float (fp 路径)

浮点 acc 转 fp16 写回。没有 saturate 问题（浮点自带 inf/NaN）。

```c
asm volatile("mxmem(%0,%1):after.hf = acc"
             :: "r"(out), "r"(0) : "memory");
```

### `:pos.hf` — ReLU 融合 (F，QNN 用)

`:pos` 是"正数通过，负数截断为 0"的变体。相当于把 ReLU 融合进 convert。

```c
asm volatile("mxmem(%0,%1):after:pos.hf = acc" ...);
```

## Output geometry：`:2x1` vs `:2x2` (F + P)

- **`:2x1`**（标准）：输出 geometry 和 ch03 cheatsheet 给的公式一致
  `out_u16[phys_row*64 + 2*col + stream]`，总 2 KiB = 1024 u16。
- **`:2x2`**（QNN 未用）：可能是 2×2 空间布局变体，未实测。

matmul **只用 `:2x1`**。

## `:retain` —— 保留 acc 给下次读 (F)

默认 convert 会**消耗** acc（后续 MAC 要先 `mxclracc`）。`:retain` 让 acc 保留，
允许同一 acc 用**不同 bias** 多次 convert —— 这是 dual-scale readback 的关键。

```c
/* 错误：没 :retain，第二次 convert 读到的是已消耗的 acc */
asm volatile("mxmem(%0,%1):after.uh = acc:2x1" :: "r"(out_lo), "r"(0) : "memory");
/* 换 bias ... */
asm volatile("mxmem(%0,%1):after.uh = acc:2x1" :: "r"(out_hi), "r"(0) : "memory");  // 错！

/* 正确：前一次 :retain 保留 acc */
asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1" :: "r"(out_lo), "r"(0) : "memory");
asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");
asm volatile("mxmem(%0,%1):after.uh = acc:2x1" :: "r"(out_hi), "r"(0) : "memory");  // 正常
```

## Dual-scale readback：从 int32 acc 拿完整 32-bit (F)

HMX int acc 是 int32，但 convert 只能读 16 bit。dual-scale 用两次 convert 在
CPU 侧拼回：

### 原理

一次 convert 做：`out_u16[j] = (acc[j] × f16_to_float(bias[j]) / 2) mod 2^16`

选两组 bias：
- **bias_lo = 0x4000** (f16 2.0) → 有效 scale 1.0 → `OUT_LO[j] = acc[j] mod 2^16`（低 16 bit）
- **bias_hi = 0x2000** (f16 2⁻⁷) → 有效 scale 2⁻⁸ → `OUT_HI[j] = floor(acc[j] / 256) mod 2^16`

`OUT_HI` 当 int16 解释（自动带符号位），正好是 acc 的 bit[8..23]；`OUT_LO`
取低 8 bit 给 acc 的 bit[0..7]。

### CPU 侧重建公式

```c
int32_t acc_j = ((int32_t)(int16_t)OUT_HI[j] << 8) | ((int32_t)OUT_LO[j] & 0xFF);
```

完整推导 + bit-exact 验证见 `example/hmx_matmul_int16/int16_matmul_hmx.c`，
本仓库 int16 matmul 已 end-to-end 验证。

### 完整代码段

```c
uint16_t bias_lo[128] __attribute__((aligned(128)));
uint16_t bias_hi[128] __attribute__((aligned(128)));
for (int i = 0; i < 128; i++) { bias_lo[i] = 0x4000; bias_hi[i] = 0x2000; }

/* MAC */
asm volatile("mxclracc" ::: "memory");
asm volatile("bias = mxmem(%0)" :: "r"(bias_lo) : "memory");
asm volatile("{ activation.ub = mxmem(%0,%1); weight.b = mxmem(%2,%3) }"
             :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");

/* 第一次 convert: 低 16 bit (scale 1.0, :retain 保 acc) */
asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"
             :: "r"(out_lo), "r"(0) : "memory");

/* 第二次 convert: 高 bit (scale 2^-8) */
asm volatile("bias = mxmem(%0)" :: "r"(bias_hi) : "memory");
asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
             :: "r"(out_hi), "r"(0) : "memory");

/* CPU 侧拼 */
for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 32; j++) {
        int phys_row = i & 15, stream = i >> 4;
        int idx = phys_row * 64 + 2 * j + stream;
        int32_t acc = ((int32_t)(int16_t)out_hi[idx] << 8) |
                      ((int32_t)out_lo[idx] & 0xFF);
        C[i][j] = acc;
    }
}
```

## 全 convert intrinsic 速查

```
/* int 路径: .uh 输出 (2 KiB) */
mxmem(Rs, Rt):after.uh           = acc:2x1   ; Q6_mxmem_AR_after_uh_2x1
mxmem(Rs, Rt):after:sat.uh       = acc:2x1   ; Q6_mxmem_AR_after_sat_uh_2x1
mxmem(Rs, Rt):after:retain.uh    = acc:2x1   ; Q6_mxmem_AR_after_retain_uh_2x1
mxmem(Rs, Rt):after:retain:sat.uh = acc:2x1
mxmem(Rs, Rt):before.uh          = acc:2x1
mxmem(Rs, Rt):before:sat.uh      = acc:2x1
mxmem(Rs, Rt):before:retain.uh   = acc:2x1
mxmem(Rs, Rt):before:retain:sat.uh = acc:2x1
/* .uh :2x2 变体 (QNN 未用) */
mxmem(Rs, Rt):after.uh           = acc:2x2
...
/* int 路径: .ub 输出 (1 KiB, u8) */
mxmem(Rs, Rt):after:cm:sat.ub    = acc      ; Q6_mxmem_AR_after_cm_sat_ub
mxmem(Rs, Rt):after:retain:cm:sat.ub = acc
mxmem(Rs, Rt):before:cm:sat.ub   = acc
/* fp 路径 */
mxmem(Rs, Rt):after.hf           = acc      ; Q6_mxmem_AR_after_hf
mxmem(Rs, Rt):after:retain.hf    = acc
mxmem(Rs, Rt):after:pos.hf       = acc      ; ReLU fused
mxmem(Rs, Rt):after:retain:pos.hf = acc
mxmem(Rs, Rt):before.hf          = acc
mxmem(Rs, Rt):before:pos.hf      = acc
```

> 命名规律：`Q6_mxmem_AR_<when>_<extras>_<type>` — `when` ∈ `{after, before}`，
> `extras` ∈ `{retain, sat, pos, cm, retain_sat, ...}` 组合，`type` ∈ `{uh, uh_2x1, ub, hf}`。

## 踩坑提醒

1. **没 `:retain` 就做第二次 convert** → 第二次读到未定义的 acc 数据。
2. **dual-scale bias 换错顺序**（先 2⁻⁸ 后 1.0）→ CPU 拼公式也得反，否则结果错位。
3. **int32 重建忘了 `(int16_t)` cast**：
   ```c
   acc = ((int32_t)out_hi[idx] << 8) | ...;   /* 错！负数也被当 unsigned 左移 */
   acc = ((int32_t)(int16_t)out_hi[idx] << 8) | ...;   /* 对 */
   ```
4. **K 很大时单次 convert 溢出**：K=32 u8·i8 acc 上界 2^20，超过 16-bit。不切 dual-scale
   就 wrap 掉高位。`demo09`（v3）会演示。

## 参考

- 探针数据：`example/hmx_matmul_int16/probe_dual_scale.c`
- 生产实现：`example/hmx_matmul_int16/int16_matmul_hmx.c`
- 硬件读回路径解释：`Agent/hmx_u8xi8_matmul_layers.md` §L8

## 下一章

11 modifiers — 所有修饰符横向对比（暂推迟到 v2.5）。
[13 tutorial int4×i8](13-tutorial-int4xi8-matmul/) 是 v3 的主内容。
