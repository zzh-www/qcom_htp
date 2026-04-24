# 附录 B · 汇编 / Intrinsic 速查对照

两种写法都 hexagon-clang 支持：**inline asm**（本指南全程用）和 **Q6 intrinsic**。
一表对照，帮你在两种风格之间翻译。

## 清零

| Inline asm | Intrinsic |
|------------|-----------|
| `asm volatile("mxclracc" ::: "memory");` | `Q6_mxclracc();` |
| `asm volatile("mxclracc.hf" ::: "memory");` | `Q6_mxclracc_hf();` |
| `asm volatile("mxswapacc" ::: "memory");` | `Q6_mxswapacc();` |

## Bias

| Inline asm | Intrinsic |
|------------|-----------|
| `asm volatile("bias = mxmem(%0)" :: "r"(p) : "memory");` | `Q6_bias_mxmem_R(p);` |
| `asm volatile("bias = mxmem2(%0)" :: "r"(p) : "memory");` | `Q6_bias_mxmem2_R(p);` |

## Activation / Weight load（单条，不常用）

| Inline asm | Intrinsic |
|------------|-----------|
| `asm volatile("activation.ub = mxmem(%0,%1)" :: "r"(a),"r"(2047) : "memory");` | `Q6_activation_ub_mxmem_RR(a, 2047);` |
| `asm volatile("weight.n = mxmem(%0,%1)" :: "r"(w),"r"(2047) : "memory");` | `Q6_weight_n_mxmem_RR(w, 2047);` |

## MAC packet（关键）

**不能用单条 Intrinsic**，因为两条 load 必须同 VLIW。**只能 inline asm**：

```c
asm volatile(
    "{ activation.ub = mxmem(%0, %1)\n"
    "  weight.n      = mxmem(%2, %3) }"
    :: "r"(act), "r"(2047), "r"(wt), "r"(2047)
    : "memory");
```

如果你**先调 `Q6_activation_ub_mxmem_RR(act, 2047); Q6_weight_n_mxmem_RR(wt, 2047);`**
—— 编译器**可能**把它们放到同 packet，但不保证。**生产代码用 inline asm 明确绑定**。

## Convert

| Inline asm | Intrinsic |
|------------|-----------|
| `asm volatile("mxmem(%0,%1):after.uh = acc:2x1" :: "r"(o),"r"(0) : "memory");` | `Q6_mxmem_AR_after_uh_2x1(o, 0);` |
| `asm volatile("mxmem(%0,%1):after:sat.uh = acc:2x1" :: "r"(o),"r"(0) : "memory");` | `Q6_mxmem_AR_after_sat_uh_2x1(o, 0);` |
| `asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1" :: "r"(o),"r"(0) : "memory");` | `Q6_mxmem_AR_after_retain_uh_2x1(o, 0);` |
| `asm volatile("mxmem(%0,%1):after.hf = acc" :: "r"(o),"r"(0) : "memory");` | `Q6_mxmem_AR_after_hf(o, 0);` |

## 风格对比

### 何时用 inline asm

- MAC packet（必须同 VLIW）
- 想精确控制 packet 组合（比如把 scalar 地址计算塞进 packet 的其他 slot）
- 教学 / debug（看到什么汇编编译出什么指令）

### 何时用 Intrinsic

- 单条非 packet 指令（`mxclracc`、`bias = ...`）
- 简洁性优先的 app 代码
- 需要编译器做寄存器分配 / 重排优化

### 混合用

**常见做法**：
```c
Q6_mxclracc();                                       /* intrinsic */
Q6_bias_mxmem_R(bias);                               /* intrinsic */

asm volatile(                                        /* inline asm 保证 packet */
    "{ activation.ub = mxmem(%0, %1)\n"
    "  weight.n      = mxmem(%2, %3) }"
    :: "r"(act), "r"(2047), "r"(wt), "r"(2047)
    : "memory");

Q6_mxmem_AR_after_uh_2x1(out, 0);                    /* intrinsic */
```

## 编译时差异验证

两种写法编出来的汇编应该基本一致。用 `-S` flag 输出汇编：

```sh
hexagon-clang -mv75 -mhmx -mhvx -mhvx-length=128B -O2 -S your_file.c -o /tmp/a.s
grep -E "mxmem|mxclracc|weight\.|activation\." /tmp/a.s
```

**警告**：O0 下 intrinsic 可能被拆包；O2 以上编译器会推 packet optimization。
本指南所有 demo 都 `-O2`。

## 参考

- ISA 头：`hmx_hexagon_protos.h`
- LLVM 内部 builtin 映射：`__builtin_HEXAGON_M8_*`
