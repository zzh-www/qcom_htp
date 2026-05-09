# HMX 编程指南

这份指南按当前 `HmxU8I8ToU8MatMul` 的实践重写。它不是一份指令字典，也不是把所有历史实验重新列一遍，而是回答一个更实际的问题：

```text
如果我要写一个接近 QNN native 风格的 HMX kernel，
我应该先理解什么，再写什么，最后怎么判断自己写对了？
```

先记住一句话：

```text
HMX 只负责计算。
计算之外的事情，都要在 HMX 运行前准备好。
```

这句话是整份指南的主线。不要把 HMX 当成一个会理解普通 Tensor 的通用函数。更准确的模型是：

```text
普通 MatMul 语义
  |
  |  Python generator / QNN converter / QHPI precompute / sidecar op
  v
HMX 已经能直接吃的数据和描述符
  |
  |  很薄的 C++ wrapper
  v
inline asm HMX body
  |
  v
MAC + convert + store
```

## 阅读顺序

| 顺序 | 章节 | 先解决的问题 |
|---|---|---|
| 1 | [先看全图](01-start-with-the-picture.md) | HMX 到底负责什么，不负责什么。 |
| 2 | [MatMul 为什么变成 Conv1x1](02-matmul-becomes-conv1x1.md) | 为什么 native QNN 会用 Conv1x1 的 kernel 来跑 MatMul。 |
| 3 | [先把数据准备成 HMX 想要的样子](03-prepare-the-data.md) | weight、activation、output、bias、scale、zero point 分别怎么提前处理。 |
| 4 | [C++ wrapper 只搭一个很小的 ABI](04-build-the-small-abi.md) | hot callback 为什么不能做重活，它到底给 asm 传什么。 |
| 5 | [读懂 HMX body](05-read-the-hmx-body.md) | inline asm 里哪些包是真计算，哪些只是前后处理。 |
| 6 | [调试和性能怎么看](06-debug-and-measure.md) | 出错先查哪一层，cycles/pkts/cpp 分别说明什么。 |

## 当前主线代码

本指南只围绕当前保留的代表性实现讲：

```text
example/qnn_hmx_matmul_u8i8/
  standard_flow/custom_u8i8/gen_u8i8_chain.py
  src/HmxU8I8ToU8MatMulOp.cpp
  src/v73deep_conv1x1_kernel.h
  src/v73deep_conv1x1_kernel.inc
```

端到端 custom op 文档在：

```text
docs/qnn_custom_op_matmul_e2e.md
```

混合精度 HMX MatMul 的数据流和 accumulator/drain 说明在：

```text
docs/hmx_mixed_precision_matmul_dataflow.md
```

这份指南只保留当前 QNN native 对齐实践里的有效主线：真正快的设计不是让 HMX 在运行时理解数据，而是让 HMX 一进来就面对已经准备好的 TCM/VTCM 状态。

## 当前结果

256x256x256 chain8 真机测试中，当前 custom op 的核心结果是：

```text
custom HmxU8I8ToU8MatMul: 1165 cycles / 337 packets
native ConvLayer_s1.opt:  1140 cycles / 346 packets
```

这说明 custom 路径没有多执行 MatMul 计算包。剩下的小 cycle 差距主要来自 custom QHPI/profiling 边界和很薄的 descriptor glue，不是 HMX MAC loop 多算了。
