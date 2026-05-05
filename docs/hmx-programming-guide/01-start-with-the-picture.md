# 01 先看全图

先不要急着看汇编。HMX 最容易理解的方式，是把它看成 HTP 里的一个专用计算机器。

它很擅长：

```text
读已经排好的 activation tile
读已经排好的 weight tile
做很多 MAC
把 accumulator 转成 u8
把 output tile 写回去
```

它不擅长：

```text
理解 QNN Tensor
判断 layout
运行时打包 weight
运行时处理各种 shape
运行时动态减 zero point
运行时复制大量 pointer table
在 C++ 里做复杂控制流
```

所以 HMX kernel 的正确写法，不是：

```text
先写一个普通 C++ MatMul
再把里面几行换成 HMX 指令
```

而是：

```text
先把输入整理成 HMX 原生格式
再让 HMX 做一段固定、短、直接的计算
```

## 一张总图

当前 custom MatMul 的数据流可以这样看：

```text
用户看到的 MatMul
  A[M,K] x W[K,N] -> Y[M,N]
        |
        |  graph 生成和加载阶段
        |  - weight 提前打包成 K-major HMX tile
        |  - bias / scale / zero point 折进 native bias record
        |  - activation/output 变成 Crouton_8 TCM block
        |  - pointer table、tile count、mask state 准备好
        v
HMX 能直接吃的状态
        |
        |  hot callback
        |  - 组 out_desc
        |  - 组 act_desc
        |  - 放好 extra_param
        v
很小的 native ABI
        |
        |  owned V73DEEP inline asm body
        v
HMX 做 MAC + convert + store
```

这里最重要的边界是：

```text
准备层
  负责把普通 Tensor 变成 HMX 原生状态

计算层
  只负责用 HMX body 消耗这些状态
```

## 为什么这个边界重要

如果你把准备工作放进 hot callback，profile 就会把这部分时间算到你的 custom op 上。之前 custom 和 native 的 gap，核心就是这个问题。

旧路径像这样：

```text
hot callback
  查 tensor
  恢复 shape
  复制 pointer table
  修 mask
  组 descriptor
  调 HMX body
```

native QNN 更像这样：

```text
graph load / sidecar op / precompute
  准备 weight、bias、pointer table、mask

hot kernel event
  组很薄的 descriptor
  调 HMX body
```

所以我们后来对齐 native 的关键，不是把 MAC loop 魔改得更复杂，而是把非计算工作从 hot path 搬出去。

## 一个简单类比

可以把 HMX 想成高速机床。

低效方式：

```text
把原材料、图纸、测量任务都塞给机床
让机床边理解边加工
```

高效方式：

```text
人先把夹具、材料、坐标都准备好
机床只执行一段固定加工程序
```

在我们的 kernel 里：

```text
夹具和材料 = packed weight + Crouton block + folded bias + descriptor
固定程序   = V73DEEP HMX body
```

后面的章节都会围绕这件事展开。
