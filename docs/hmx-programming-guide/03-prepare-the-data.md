# 03 先把数据准备成 HMX 想要的样子

进入 HMX body 前，数据已经不再是普通矩阵。

当前 custom MatMul 最重要的四类数据是：

```text
bias
  native folded bias/scale record

weight
  K-major packed HMX tile

activation
  Crouton_8 TCM block，通过 pointer table 间接访问

output
  Crouton_8 TCM block，通过 pointer table 间接写回
```

HMX body 不负责重新解释这些数据。它只按 native 约定读。

## Weight: 从普通矩阵变成 K-major tile

逻辑上的 weight 是：

```text
W[K, N]
```

HMX body 想要的是 32x32 tile，而且 tile 顺序是 K-major：

```text
K tile 0, N tile 0
K tile 0, N tile 1
...
K tile 1, N tile 0
K tile 1, N tile 1
...
```

一个 32x32 tile 内部也不是普通 row-major。当前 generator 使用的核心地址公式是：

```text
dst = (k_row / 4) * 128 + n_col * 4 + (k_row % 4)
```

可以用图理解：

```text
普通 W[K,N]

        N0 N1 N2 ... N31
K0      w  w  w      w
K1      w  w  w      w
K2      w  w  w      w
K3      w  w  w      w
...
K31     w  w  w      w

打包后，一个 N column 的连续 4 个 K row 放在一起：

N0: K0 K1 K2 K3
N1: K0 K1 K2 K3
N2: K0 K1 K2 K3
...
然后进入下一组 K4 K5 K6 K7
```

真正重要的不是背公式，而是记住这件事发生在哪里：

```text
gen_u8i8_chain.py 提前 pack weight
HMX body 只顺着 packed bytes 读
```

如果 hot callback 里还在 pack weight，就已经偏离 native 设计了。

## Activation 和 Output: 不是 dense matrix

activation 和 output 走 QNN 的 Crouton_8 TCM 布局：

```text
QNN tensor
  |
  v
block pointer table
  |
  v
TCM payload block
```

HMX body 通过 pointer table 找到 block，然后直接读写 TCM payload。它不会先把 Crouton block 展平成普通矩阵。

对当前 256 case：

```text
M_t = 8
K_t = 8
N_t = 8
mt_groups = 4

activation table entries = 4 * 8 = 32
output table entries     = 4 * 8 = 32
```

QHPI precompute 会把这些 pointer-table 值记录到 custom op 的 precomputed data 里。payload block 本身仍然是 QNN 管理的 TCM block。

## Bias / Scale / Zero Point: 提前折叠

量化这里最容易误解。HMX hot path 里不做这些事：

```text
不在 C++ 里乘 runtime scale
不在 MAC loop 里动态减 activation zero point
不把 bias 当成一个简单 int32 vector 读
```

我们提前构造 native bias record。每个 N tile 256 bytes：

```text
bytes 0..127
  32 x (fp16 scale, fp16 baseline)

bytes 128..255
  32 x int32 effective_bias
```

HMX body 后面会执行：

```text
bias = mxmem2(...)
cvt.ub = acc(...)
```

可以把 convert 理解成：

```text
out_u8 = clamp(acc * scale_fp16 / 512 + baseline)
```

当前 replica 的测试流使用 identity scale：

```text
runtime scale = 1.0
scale_fp16    = 512.0
baseline      = 0
ACT_ZP        = 128
```

因为 `512.0 / 512 = 1.0`，所以 convert 的 scale 等价于不缩放。

activation zero point 则折进 int32 bias：

```text
(act_u8 - 128) @ W + bias_q
  = act_u8 @ W + (-128 * sum_k(W[k,c]) + bias_q[c])

effective_bias[c] = -128 * sum_k(W[k,c]) + bias_q[c]
```

这样 HMX MAC loop 就可以直接做：

```text
unsigned activation byte x signed int8 weight
```

它不需要知道 activation 的 zero point 是 128。

## 如果以后要支持真实 per-channel scale

HMX body 仍然不应该改。应该改的是提前准备的 bias record：

```text
runtime_scale[c]  = input_scale * weight_scale[c] / output_scale
scale_fp16[c]     = 512.0 * runtime_scale[c]
baseline[c]       = output_zp << 7
effective_bias[c] = bias_q[c] - input_zp * sum_k(W[k,c])
```

规则仍然是：

```text
改准备好的数据
不要改 hot HMX body
```

## 进入 HMX 前的检查表

调 HMX 前，先确认：

```text
weight 已经是 K-major packed bytes
activation/output 已经是 Crouton_8 TCM block
pointer table 已经在 precompute 中准备好
bias record 已经包含 scale/baseline/effective_bias
zero point 已经被折进 effective_bias
tile count 和 stride 已经换成 native 单位
```

如果这里有一项没准备好，问题不在 HMX 指令，而在输入组织。
