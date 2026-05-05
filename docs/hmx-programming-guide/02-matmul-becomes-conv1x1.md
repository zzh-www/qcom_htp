# 02 MatMul 为什么变成 Conv1x1

当前 custom op 的名字是：

```text
HmxU8I8ToU8MatMul
```

但它最终调用的 HMX body，本质上是 QNN native 的 Conv1x1 body。

这件事听起来绕，其实数学上很简单。

MatMul 是：

```text
Y[m, n] = sum_k A[m, k] * W[k, n]
```

Conv1x1 是：

```text
Y[position, out_channel] =
  sum_input_channel X[position, input_channel] * Filter[input_channel, out_channel]
```

把名字换一下：

```text
position       -> m
input_channel  -> k
out_channel    -> n
```

公式就完全一样。

## QNN 为什么喜欢这个形式

QNN native 已经有成熟的 HMX Conv1x1 kernel。这个 kernel 已经知道怎么做这些事：

```text
读 Crouton activation block
读 K-major packed weight
用 HMX deep packet 做 MAC
把 accumulator 转成 u8
写 Crouton output block
```

所以 native 不需要重新发明一个普通矩阵乘 kernel。它把 MatMul 组织成 Conv1x1 的输入状态，然后复用这套 HMX 计算路径。

这也是 custom op 要模仿的核心架构。

## 数据怎么改名

用户眼里的数据：

```text
A[M,K]
W[K,N]
Y[M,N]
```

HMX Conv1x1 眼里的数据：

```text
activation:
  M 个 position
  每个 position 有 K 个 input channel

weight:
  K 个 input channel
  N 个 output channel

output:
  M 个 position
  每个 position 有 N 个 output channel
```

对当前 256 case：

```text
M = 256
K = 256
N = 256

每 32 个元素切成一个 tile:
M_t = 8
K_t = 8
N_t = 8
```

## HMX body 真正假设了什么

HMX body 不会收到 `M=256,K=256,N=256` 这种高层信息。它只假设：

```text
activation 已经是 Crouton_8 TCM block
output 已经是 Crouton_8 TCM block
weight 已经按 K-major HMX tile 打包
bias record 已经包含 scale / baseline / zero-point correction
descriptor 里的 stride 和 count 已经换成 native 单位
mask state 已经选好 V73DEEP 路径
```

只要这些都成立，HMX body 就可以短而快。任何一个不成立，都不应该让 HMX body 自己补救，而应该回到准备层修数据。

## 设计 kernel 时先问什么

不要先问：

```text
我要怎么写一个矩阵乘循环？
```

应该先问：

```text
native Conv1x1 HMX body 期待什么输入状态？
我的代码怎样提前构造这个状态？
```

这就是从普通 MatMul 走到 QNN/HMX kernel 的第一步。
