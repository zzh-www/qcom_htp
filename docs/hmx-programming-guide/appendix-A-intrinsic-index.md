# 附录 A · Intrinsic 索引（指针）

本指南按"功能族"讲 HMX 指令。想查**具体某个 `Q6_*` intrinsic 在讲哪章**，用这里的索引。

## 说明

本附录只给 **intrinsic 名称 → 对应章节**的指针。每条 intrinsic 的具体语义
参考对应章节；完整参数/宏展开查 `hmx_hexagon_protos.h`。

## 清零 / acc 切换 — [ch04](04-instr-clracc-swap.md)

| Intrinsic | 汇编 |
|-----------|------|
| `Q6_mxclracc` | `mxclracc` |
| `Q6_mxclracc_hf` | `mxclracc.hf` |
| `Q6_mxswapacc` | `mxswapacc` |
| `Q6_mxswapacc_hf` | `mxswapacc.hf` |
| `Q6_acc_mxshl_acc` | `acc = mxshl(acc, #16)` |

## Bias load — [ch05](05-instr-bias.md)

| Intrinsic | 汇编 |
|-----------|------|
| `Q6_bias_mxmem_R` | `bias = mxmem(Rs)` |
| `Q6_bias_mxmem2_R` | `bias = mxmem2(Rs)` |
| `Q6_mxmem_st_bias_R` | `mxmem(Rs) = bias` (导出 bias 到 VTCM) |
| `Q6_mxmem2_st_bias_R` | `mxmem2(Rs) = bias` |

## Activation load — [ch06](06-instr-activation-load.md)

| Intrinsic 族 | 汇编模板 |
|--------------|----------|
| `Q6_activation_ub_mxmem_RR[_<mod>]` | `activation.ub = mxmem(Rs, Rt)[<:mod>]` |
| `Q6_activation_hf_mxmem_RR[_<mod>]` | `activation.hf = mxmem(Rs, Rt)[<:mod>]` |
| `Q6_activation_f8_mxmem_RR[_<mod>]` | `activation.f8 = mxmem(Rs, Rt)[<:mod>]` |

`<mod>` 可能值：`(none), above, cm, deep, deep_cm, dilate, dilate_cm, single, single_cm, above_cm`。

## Weight load — [ch07](07-instr-weight-load.md)

| 类型 | Intrinsic 前缀 | 汇编 |
|------|----------------|------|
| int8 | `Q6_weight_b_mxmem_RR[_<mod>]` | `weight.b = mxmem(Rs, Rt)` |
| int4 | `Q6_weight_n_mxmem_RR[_<mod>]` | `weight.n = mxmem(Rs, Rt)` |
| int4 :2x | `Q6_weight_n_mxmem_RR_2x[_<mod>]` | `weight.n = mxmem(Rs, Rt):2x` |
| int2 crumb | `Q6_weight_c_mxmem_RR[_<mod>]` | `weight.c = mxmem(Rs, Rt)` |
| int2 signed alt | `Q6_weight_sc_mxmem_RR[_<mod>]` | `weight.sc = mxmem(Rs, Rt)` |
| 1-bit unsigned | `Q6_weight_ubit_mxmem_RR[_<mod>]` | `weight.ubit = mxmem(Rs, Rt)` |
| 1-bit signed | `Q6_weight_sbit_mxmem_RR[_<mod>]` | `weight.sbit = mxmem(Rs, Rt)` |
| sparse mask | `Q6_weight_sm_mxmem_RR[_<mod>]` | `weight.sm = mxmem(Rs, Rt)` |
| fp16 | `Q6_weight_hf_mxmem_RR[_<mod>]` | `weight.hf = mxmem(Rs, Rt)` |
| fp8 | `Q6_weight_f8_mxmem_RR[_<mod>]` | `weight.f8 = mxmem(Rs, Rt)` |

`<mod>` 可能值：`(none), after, dilate, deep, drop, single`。`.n` 还有 `:2x` 前缀可加。

## MAC packet — [ch08](08-instr-mac-packet.md)

MAC 本身不是单独 intrinsic——由 activation + weight 的两条 load **同 VLIW packet**
发射触发。所以没有 "Q6_mac_*" 一类；靠汇编里的花括号语法组合。

## Convert / readback — [ch09](09-instr-convert.md)

| Intrinsic 族 | 汇编模板 |
|--------------|----------|
| `Q6_mxmem_AR_<when>_<extras>_uh_<geom>` | `mxmem(Rs, Rt):<when><:extras>.uh = acc<:geom>` |
| `Q6_mxmem_AR_<when>_<extras>_ub` | `mxmem(Rs, Rt):<when><:extras>.ub = acc` |
| `Q6_mxmem_AR_<when>_<extras>_hf` | `mxmem(Rs, Rt):<when><:extras>.hf = acc` |
| `Q6_mxmem_cvt_RR_<geom>=cvt` | `mxmem(Rs, Rt)<:geom> = cvt` |

- `when` ∈ `{after, before}`
- `extras` ∈ `{(none), retain, sat, pos, cm, retain_sat, retain_pos, retain_cm, sat_cm, ...}`
- `geom` ∈ `{(none), 2x1, 2x2}`（仅 .uh 需要；.hf/.ub 无 geom）

## 修饰符 — [ch11](11-modifiers.md)

每个修饰符的语义见 ch11，不逐条列 intrinsic。

## 旧 cvt 指令 (v1 起遗留)

| Intrinsic | 汇编 | 用途 |
|-----------|------|------|
| `Q6_mxcvt_ub_acc_R` | `cvt.ub = acc(Rs)` | legacy，现代 kernel 用 `mxmem = acc` 代替 |
| `Q6_mxcvt_ub_acc_R_sc0` | `cvt.ub = acc(Rs):sc0` | |
| `Q6_mxcvt_ub_acc_R_sc1` | `cvt.ub = acc(Rs):sc1` | |
| `Q6_mxcvt_hf_acc_R` | `cvt.hf = acc(Rs)` | |

本指南不详细讲这些 legacy 变体；通常现代 HMX kernel 只用 `mxmem(Rs, Rt):after.* = acc`
家族。

## QNN v75 skel 实际使用频次参考

来自 `libQnnHtpV75Skel.so` 反汇编（2026-04）：

| 指令族 | 次数 | 本指南关键章节 |
|--------|-----:|---------------|
| `activation.ub` | 724 | ch06 |
| `weight.b` | 448 | ch07 |
| `mxmem2` (bias) | 480 | ch05 |
| `mxclracc` | 158 | ch04 |
| `cvt.uh` (convert) | 350 | ch09 |
| `weight.n*`（含 :2x） | 232 | ch07 |
| `cvt.ub` | 144 | ch09 |
| `weight.hf` / `activation.hf` | 110 | ch06/07 |
| `acc:2x1` | 76 | ch09 |
| `weight.c` | 44 | ch07 |
| `mxclracc.hf` | 31 | ch04 |
| `cvt.hf` | 24 | ch09 |
| `mxswapacc` | 22 | ch04 |

**未出现**：`weight.ubit`、`weight.sbit`、`weight.sc`、`weight.sm`、`weight.f8`、
`activation.f8`、`acc:2x2`。这些 ISA 支持但 QNN 没用——本指南只做最小说明。

## 完整列表

`hmx_hexagon_protos.h` 头文件是权威列表。约 200 条 `#define Q6_*_mxmem_*`。
想看**全部**枚举：

```sh
grep -E '^#define Q6_' \
  tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hmx_hexagon_protos.h \
  | head -200
```

## 参考

- 头文件：`tools/hexagon-sdk/.../hmx_hexagon_protos.h`
- 反汇编工具：`tools/hexagon-sdk/.../Tools/bin/hexagon-llvm-objdump`
- 所有指令族的使用场景讲解：见本指南对应章节
