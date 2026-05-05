# 05 读懂 HMX body

HMX body 是唯一真正做计算的部分。

当前代码在：

```text
example/qnn_hmx_matmul_u8i8/src/v73deep_conv1x1_kernel.inc
```

它是我们持有的 inline-asm replica，对齐的是 QNN native V73DEEP Conv1x1 body：

```text
hmx_v73_convbbb1x1deep_stride1
1132 bytes
```

读这段 asm 时，不需要一开始就背每个 packet。先按四段看。

## 第一段: prologue

body 一进来先读 ABI：

```text
out_desc
act_desc
mask_desc
extra_param
```

然后得到内部循环要用的状态：

```text
loop count
pointer stride
runtime mask
weight step
activation/output base pointer
```

这段的作用是把 descriptor 字段翻译成 HMX body 自己的循环变量。

## 第二段: MAC feed

真正喂 HMX 计算管线的核心 packet 是：

```text
activation.ub = mxmem(...):deep:cm
weight.b      = mxmem(...):deep
```

可以直白理解为：

```text
读一个 activation tile
读一个 weight tile
把两者送进 HMX deep Conv1x1 MAC pipeline
```

activation 地址来自 activation pointer table。weight 地址沿着 K-major packed stream 前进。

accumulator 在 HMX 内部。C++ 不会读它，也不应该读它。

## 第三段: convert 和 store

K 方向累加够了以后，body 会把 accumulator 转成 u8：

```text
bias = mxmem2(...)
cvt.ub = acc(...)
mxmem(...):cm = cvt
```

这三句分别代表：

```text
读取提前准备好的 bias/scale/baseline record
把 accumulator 转成 u8
把 u8 output 写回 Crouton output block
```

所以 scale、baseline、zero-point correction 都不是在 C++ 里临时算，而是在准备好的 record 里被 HMX convert 路径使用。

## 第四段: drain 和 epilogue

最后 body 会处理剩余的 pipeline 状态，然后退出：

```text
完成最后的 K work
convert/store 最后一批 accumulator
mxclracc
恢复 register
return
```

`mxclracc` 很重要。HMX accumulator 不是普通 C 局部变量，body 必须把它清干净，避免影响下一次调用。

## 简化后的整体形状

可以先把整个 body 看成这样：

```text
read descriptors

for each K tile group:
  for each M/N tile group:
    for each K step:
      activation.ub = mxmem(...):deep:cm
      weight.b      = mxmem(...):deep

    bias = mxmem2(...)
    cvt.ub = acc(...)
    store output tile

final drain
mxclracc
return
```

真实 asm 更 packetized，也有一些分支和调度细节。但理解 kernel 设计时，上面这张图足够抓住主线。

## 为什么我们保留 inline asm bytes

这个项目最初通过 reverse engineering 找到了 native body。现在保留的实现不是运行时 `dlsym` native symbol，而是直接嵌入 owned inline asm。

这样做的好处是：

```text
不依赖 native symbol swap
compute body byte-verified
custom op package 行为稳定
wrapper/precompute 可以自己控制
```

当前 body 已经对齐 native 的 1132-byte compute slice。后续优化重点应该放在 body 周围的数据准备和 wrapper 边界，而不是随意改 MAC loop。
