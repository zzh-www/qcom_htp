# HMX 混合精度 MatMul 数据流说明

本文不是官方 HMX ISA 文档，而是基于本仓库已复刻和 byte-verified 的
QNN native HMX body、packer、descriptor 行为整理出的工程说明。它的目标是
解释 `u8i8`、`w4a8`、`w4a16` 在 HMX 上的数据如何流动，特别是
accumulator banks 到 drain/output 这条链路。

## 0. 先给结论

`u8i8`、`w4a8` 和 `w4a16` 不是同一个通用 matmul kernel 通过参数切换精度。
它们更像是把同一个逻辑 MatMul 问题，分别喂给 HMX 硬件里几条不同的
固定数据通路：

```text
u8i8:
  U8 activation stream
  I8 byte weight stream
  internal accumulator
  U8 drain/output

w4a8:
  U8 activation stream
  I4 nibble weight stream
  internal accumulator
  U8 drain/output

w4a16:
  U16 activation stream
  I4 nibble weight stream
  internal accumulator
  U16 drain/output
```

HMX 的核心局限是：

```text
它不是通用矩阵乘单元。
它是固定 tile 形状、固定输入端口、固定累加器、固定 drain 路径的
typed tile-MAC 机器。
```

对 W4 要特别区分两件事：

```text
有证据支持:
  HMX 有原生 nibble weight 输入/解码/路由通路。

当前性能证据不支持:
  HMX 有相对 W8 吞吐翻倍的独立 2x INT4 MAC 阵列。
```

所以 W4 不是纯软件模拟；但也不能把 W4 理解成“物理 INT4 计算阵列翻倍”。

所以 QNN/HMX kernel 的主要工作不是在热路径里写普通三重循环，而是：

```text
把逻辑 tensor 预先整理成 HMX 某条数据通路能直接消费的物理 stream。
```

## 1. 术语表

下面的词后文会反复出现，先统一含义。

`HMX`
: Hexagon/HTP 里的矩阵/卷积类专用计算单元。本文把它抽象成一个有
  activation port、weight port、accumulator、convert/drain、output port
  的 tile-MAC 机器。

`HTP`
: Qualcomm Hexagon Tensor Processor。QNN 的 HTP backend 会把部分图节点
  lowering 到 HTP/HMX 执行。

`QNN`
: Qualcomm Neural Network SDK。这里主要关注 QNN native HTP kernel 的数据
  契约和 custom op 如何复刻它。

`dtype`
: data type，数据类型。这里常见的是 `U8`、`I8`、`I4`、`U16`。

`U8` / `I8` / `I4` / `U16`
: `U` 是 unsigned，无符号；`I` 是 signed integer，有符号整数。数字是 bit
  宽度。`U8` 表示 8-bit 无符号整数，`I8` 表示 8-bit 有符号整数，`I4`
  表示 4-bit 有符号整数，`U16` 表示 16-bit 无符号整数。

`native body`
: QNN SDK 自带或我们复刻出来的底层 HMX 计算体。它是真正执行 MAC/drain 的
  Hexagon/HMX 指令序列，不是外层 C++ wrapper。

`body family`
: 一类 native body 的数据通路家族。例如本文里的 `convbbb`、`convbnb`
  和 `convhnh`。

`convbbb`
: 本文用来描述 U8/I8/U8 路径的 body family 名字。三个 `b` 可以按数据
  通路直觉理解成 byte activation、byte weight、byte output。

`convbnb`
: 本文用来描述 W4A8 路径的 body family 名字。它的关键差异是中间的 `n`：
  activation/output 仍是 byte 语义，但 weight port 走 nibble weight 路径。

`convhnh`
: 本文用来描述 W4A16 路径的 body family 名字。`h` 表示 halfword 侧的
  activation/output 语义，`n` 表示 nibble weight 路径。

`byte-verified`
: 指我们编译复刻出来的 inline asm 后，把 `.text` 字节和 QNN native `.so`
  中截取的原始 slice 逐字节比较一致。它证明“这段 body 字节相同”，不自动
  证明我们已经理解了所有硬件语义。

`lowering`
: 编译器或 runtime 把高层图节点变成更底层执行形式的过程。比如把 MatMul
  组织成 Conv1x1-style HMX kernel 能消费的状态。

`hot path` / `hot callback`
: 性能计时中真正反复执行的路径。对 custom op 来说，hot callback 是每次
  inference 执行该 op 时进入的函数。这里应该尽量薄，避免做大量打包和复制。

`pack` / `packer`
: `pack` 是把逻辑 tensor 重排、压缩成 HMX 物理 stream 的动作；`packer` 是
  做这件事的脚本或代码。W4 的 packer 会把两个 int4 塞进一个 byte。

`row-major`
: 普通常见的二维数组行优先内存布局。比如一行的数据连续存放。HMX 不直接
  消费任意 row-major MatMul，而是消费 tiled/packed 后的格式。

`tile`
: 小块数据。HMX 不直接按任意 row-major 矩阵逐元素访问，而是按固定小块
  消费 activation、weight、output。本文用 32x32 weight tile 举例，因为
  当前 packer 和 native body 的证据都围绕这个粒度展开。

`stream`
: 连续喂给 HMX 的物理数据流。它不一定是逻辑矩阵的自然顺序，而是已经
  重排、打包、分块后的内存序列或地址序列。

`activation`
: 神经网络里当前层的输入张量。映射到 MatMul 时就是左矩阵 `A[M,K]`。

`weight`
: 权重张量。映射到 MatMul 时就是右矩阵 `W[K,N]`。

`output`
: 输出张量。映射到 MatMul 时就是结果矩阵 `C[M,N]`。

`MAC`
: Multiply-Accumulate，乘加。形式是 `acc += a * w`。

`INT4 MAC array`
: 直觉上可以理解为“专门以 4-bit 权重/输入做乘加的物理计算阵列”。本文的
  结论是：当前证据能确认 W4/nibble weight 通路，但不能确认存在相对 W8
  吞吐翻倍的独立 `2x INT4 MAC array`。

`accumulator`
: 累加器。保存 K 方向 reduce 的中间结果。它不是最终输出 dtype，通常应
  理解成比输入和输出更宽的内部整数累加槽。

`accumulator banks`
: 一组并行 accumulator。HMX 同时计算一个 output tile 里的多个输出位置，
  所以内部不是一个 scalar `acc`，而是一批并行累加槽。

`drain`
: 把 accumulator 里的中间结果“排出”成最终 output tile 的阶段。这个阶段
  会结合 bias/control 做 convert、round、saturate/clamp 和 pack/store。

`convert`
: 把内部 accumulator 变成目标输出 dtype 的过程。比如 `u8i8` 的 drain 输出
  `U8`，`w4a16` 的 drain 输出 `U16`。

`clamp` / `saturate`
: 把数值限制在目标 dtype 能表示的范围内。例如 U8 是 `[0,255]`，U16 是
  `[0,65535]`。

`bias/control`
: drain 阶段使用的 side data。它不只是传统神经网络里的 bias，也包含
  convert 所需的常量、scale、zero-point 补偿、lane/parity 选择等信息。

`descriptor`
: 传给 native body 的小结构，描述 activation/output 地址表、stride、tile
  count、K 长度、mask/routing 等。它不是原始 tensor metadata，而是 HMX body
  能直接使用的低层控制信息。

`ABI`
: Application Binary Interface，二进制调用约定。这里指 custom wrapper 和
  HMX body 之间“参数按什么顺序、什么结构传入”的低层契约。

`pointer table`
: 地址表。activation/output 经常不是一个线性 row-major 指针直接喂给 HMX，
  而是先整理成每个 tile 或 block 对应的地址表。

`mask` / `routing`
: 控制 HMX 端口、tile 路径或 lane 选择的低层参数。本文不把它展开成官方
  bitfield，只把它当作“告诉 HMX 数据从哪条路线进出”的控制状态。

`Crouton`
: QNN/HTP 使用的一类 tiled tensor layout。本文只把它当作“已经适合 HMX
  访问的 tiled activation/output 布局”，不展开官方内部细节。

`byte`
: 8-bit 数据。`u8i8` 中 activation 是 U8，weight 是 I8，output 是 U8。

`half`
: 16-bit 数据。`w4a16` 中 activation/output 是 U16。

`nibble`
: 4-bit 半字节。`w4a8` 和 `w4a16` 的 weight 是 I4，一个 byte 里通常
  放两个 int4。

`lane`
: 并行数据通道中的一个位置。可以粗略理解为 SIMD/tile 内部的一个元素槽。

`parity`
: 奇偶半区或成对输出中的一半。W4/A16 drain 里会出现分半、分批 drain 的
  现象，本文用 parity 泛指这种 lane 分组，不强行声明完整硬件编码。

`scale`
: 量化缩放因子。accumulator 是内部整数和，最终要经过 scale/等价控制常量
  转成目标 dtype 的数值域。

`zero-point`
: 量化零点。无符号量化值里的某个整数代表数学上的 0。为了减少 hot path
  工作，zero-point 相关补偿常被提前 fold 到 bias/control record。

`round`
: 舍入。把 scale 后的中间值变成整数输出时需要的取整步骤。

`fold`
: 提前合并。比如把 activation zero-point 对结果的影响提前合进 effective
  bias，运行时就不用逐元素做这部分计算。

`mnemonic`
: 汇编指令的文本名字。例如 `weight.n`、`cvt.uh`。本文用这些名字作为已
  恢复 HMX 数据路线的证据，但不把它们扩展成完整官方 ISA 解释。

`typed pipeline`
: 固定数据类型组合的数据通路。例如 `byte activation + byte weight + byte
  output`、`byte activation + nibble weight + byte output`，或者
  `half activation + nibble weight + half output`。

`convbbb` / `convbnb` / `convhnh`
: 本仓库用来区分 native HMX body family 的名字。可以粗略记成：

```text
convbbb: byte / byte   / byte  路径
convbnb: byte / nibble / byte  路径
convhnh: half / nibble / half  路径
```

这是便于理解的命名解释，不等价于完整官方 ISA 定义。

## 2. HMX 硬件抽象

可以先把 HMX 想成下面这台机器：

```text
                         activation port
                               |
                               v
weight port -----> MAC fabric -----> accumulator banks -----> convert/drain -----> output port
                               ^               ^
                               |               |
                         tile routing     bias/control port
```

各部分含义如下。

`activation port`
: 输入 activation tile。对 MatMul 来说，就是 `A[M,K]` 中某些 M 行和 K
  通道组成的小块。

`weight port`
: 输入 weight tile。这里的数据必须已经按 HMX 预期格式打包。`u8i8` 是 byte
  权重流，`w4a8` 和 `w4a16` 是 nibble-packed int4 权重流。

`MAC fabric`
: 真正做乘加的硬件阵列。它接收 activation lane 和 weight lane，执行大量
  并行 `acc += activation * weight`。

`accumulator banks`
: MAC fabric 的输出不会直接写到最终 tensor。它先进入内部累加器组。每个
  bank 保存一部分 output tile 的中间和。

`bias/control port`
: drain 时读取的辅助数据入口。它告诉 HMX 如何把 accumulator 转成最终 dtype。

`convert/drain`
: 读取 accumulator banks，结合 bias/control，做量化转换、round、saturate，
  最后形成 output tile。

`output port`
: 把 convert 后的 tile 写回 output layout。`u8i8` 和 `w4a8` 写 U8，
  `w4a16` 写 U16。

这一点很关键：

```text
MAC fabric 的输出是内部 accumulator。
kernel 的最终输出是 drain/convert 之后的 tensor dtype。
```

所以 `accumulator banks` 不是 U8，也不是 U16。对本文而言，可以把它理解成
“HMX 内部宽整数累加槽”。当前逆向证据能稳定说明的是 drain 结果：

```text
u8i8:   accumulator -> U8 output
w4a8:  accumulator -> U8 output
w4a16: accumulator -> U16 output
```

不要把 “内部宽整数” 解读成我们已经确认了官方精确 accumulator 位宽。

## 3. HMX 的设计局限

HMX 快，是因为它不做很多通用处理。它要求软件提前准备好一切。

它不适合做：

```text
解析任意 QNN tensor metadata
理解任意 row-major / column-major layout
运行时动态打包 weight
运行时决定 weight 是 int8 还是 int4
暴露 accumulator 给 C++ 随便读写
在热路径里做复杂 shape 推导
```

它适合做：

```text
按固定路线读取 activation tile
按固定路线读取 packed weight tile
在固定 typed pipeline 里做大量 MAC
把 accumulator 用固定 drain recipe 转成目标 dtype
写回固定 output tile layout
```

因此一个 HMX kernel 的正确设计边界是：

```text
准备层:
  整理 tensor、打包 weight、生成 pointer table、生成 bias/control。

计算层:
  用很薄的 descriptor 调 HMX body，让硬件消耗这些 stream。
```

如果把准备层的大量工作放到 hot callback 里，性能就会偏离 QNN native。
这也是我们复刻 native kernel 时反复遇到的核心问题。

## 4. MatMul 在 HMX 眼里是什么

逻辑 MatMul 是：

```text
C[M,N] = A[M,K] x W[K,N]
```

单个输出元素是：

```text
C[m,n] = sum_k A[m,k] * W[k,n]
```

用小矩阵画出来：

```text
A[M,K]                    W[K,N]                    C[M,N]

      k0 k1 k2 k3              n0 n1 n2 n3              n0 n1 n2 n3
m0   a00 a01 a02 a03      k0   w00 w01 w02 w03      m0   c00 c01 c02 c03
m1   a10 a11 a12 a13  x   k1   w10 w11 w12 w13  =   m1   c10 c11 c12 c13
m2   a20 a21 a22 a23      k2   w20 w21 w22 w23      m2   c20 c21 c22 c23
m3   a30 a31 a32 a33      k3   w30 w31 w32 w33      m3   c30 c31 c32 c33
```

HMX 不会按这个二维表逐元素访问。它看到的是 tile stream：

```text
A tile stream:
  选中一批 M 位置和一批 K 通道，对应 activation tile。

W tile stream:
  选中同一批 K 通道和 32 个 N 输出通道，对应 packed weight tile。

C tile stream:
  选中一批 M 位置和 32 个 N 输出通道，对应 output tile 地址。

bias/control stream:
  对应这 32 个 N 输出通道的 drain recipe。
```

HMX 的工作更接近：

```text
for each C tile:
    accumulator banks = 0

    for each K tile:
        activation port receives A tile
        weight port receives W tile
        MAC fabric accumulates into banks

    convert/drain banks with bias/control
    store output tile
```

这就是所谓 `tile reduce`：

```text
tile:   一次处理一块 M/N/K 数据
reduce: 沿 K 方向不断累加
```

## 5. 为什么 MatMul 会伪装成 Conv1x1

Conv1x1 的数学形式是：

```text
Y[position, out_channel] =
    sum_input_channel X[position, input_channel]
                      * Filter[input_channel, out_channel]
```

MatMul 是：

```text
C[m,n] = sum_k A[m,k] * W[k,n]
```

把名字对应起来：

```text
position       -> m
input_channel  -> k
out_channel    -> n
```

两者数学上就是同一个 reduce。QNN native 已经有成熟 Conv1x1 HMX body，所以
custom MatMul 最合理的复刻方式不是手写一个全新 GEMM，而是把 MatMul 的数据
整理成 Conv1x1 body 期待的状态：

```text
A[M,K]  -> activation positions/channels
W[K,N]  -> conv filter input/output channels
C[M,N]  -> output positions/channels
```

这就是“MatMul 变成 Conv1x1”的本质。

## 6. 一次 HMX 调用的完整过程

从图构建到真正执行，可以分成五层。

### 6.1 逻辑层

用户或模型看到的是普通矩阵：

```text
A[M,K] x W[K,N] -> C[M,N]
```

此时 dtype 只是逻辑契约：

```text
u8i8:   A 是 U8，W 是 I8，C 是 U8
w4a8:  A 是 U8，W 是 I4，C 是 U8
w4a16: A 是 U16，W 是 I4，C 是 U16
```

### 6.2 预处理层

QNN/native 或我们的 custom op 准备层会做：

```text
切 tile
重排 activation/output layout
打包 weight
生成 bias/control record
生成 pointer table
生成 descriptor/mask/routing 信息
```

这一步越像 native，hot path 就越薄，性能也越像 native。

### 6.3 调用层

hot callback 只应该做很少的事：

```text
拿到已经准备好的 raw pointers
填少量 descriptor 字段
选择正确 HMX body
跳进去执行
```

### 6.4 MAC 层

HMX body 在循环里喂数据：

```text
activation tile -> activation port
weight tile     -> weight port
```

硬件内部做：

```text
accumulator banks += activation tile * weight tile
```

这里输出的是 accumulator，不是最终 tensor。

### 6.5 Drain 层

K 方向累加结束后：

```text
accumulator banks
    + bias/control
    -> convert
    -> clamp/saturate
    -> output tile
```

这一步才决定最终输出 dtype：

```text
u8i8:   drain 后是 U8
w4a8:  drain 后是 U8
w4a16: drain 后是 U16
```

## 7. U8I8 的数据流：byte / byte / byte

`u8i8` 对应 byte-oriented pipeline：

```text
A:   U8 activation tile
W:   I8 weight tile
ACC: internal wider integer accumulator
C:   U8 output tile
```

图示：

```text
U8 activation tile
        |
        v
 byte activation lanes ----\
                            +--> accumulator banks --> U8 convert --> U8 output tile
 byte weight lanes     ----/
        ^
        |
I8 packed weight tile
```

单个输出元素可以理解成：

```text
A row:
[a0 a1 a2 a3]       each a is uint8

W column:
[w0 w1 w2 w3]       each w is int8

acc = a0*w0 + a1*w1 + a2*w2 + a3*w3
out = quantize_to_u8(acc + folded_bias/control)
```

这里 `quantize_to_u8` 代表 drain 里的组合操作：

```text
加 bias 或 zero-point 补偿
乘 scale 或应用等价控制常量
round
clamp 到 U8 范围
打包并 store
```

`u8i8` 的 weight stream 是 byte 粒度。逻辑 32x32 weight tile 的体积是：

```text
32 * 32 * 8 bits = 8192 bits = 1024 bytes
```

所以它的物理契约可以压缩成：

```text
U8 activation stream
+ I8 byte-packed weight stream
+ U8 drain recipe
```

在当前复刻 body 里，对应的关键路线是：

```text
activation.ub = ...
weight.b      = ...
cvt.ub        = acc(...)
```

含义是：

```text
activation.ub: activation 输入侧按 unsigned byte 相关路径喂入
weight.b:      weight 输入侧按 byte 权重路径喂入
cvt.ub:        accumulator drain 成 unsigned byte output
```

注意这里不是在逐条解释 asm，而是把这些 mnemonic 当成数据通路证据。

## 8. W4A8 的数据流：byte / nibble / byte

`w4a8` 是理解 W4 的最好中间态，因为它只把 weight 侧换成 W4/nibble，
activation/output 仍然保持 A8/U8 路径：

```text
A:   U8 activation tile
W:   I4 weight tile, packed as nibbles
ACC: internal wider integer accumulator
C:   U8 output tile
```

图示：

```text
U8 activation tile
        |
        v
 byte activation lanes ----\
                            +--> accumulator banks --> U8 convert --> U8 output tile
 nibble weight lanes   ----/
        ^
        |
packed I4 weight tile
```

所以它和 `u8i8` 的差异不是 output，也不是 activation，而是 weight port：

```text
u8i8:
  activation byte stream
  weight byte stream
  output byte drain

w4a8:
  activation byte stream
  weight nibble stream
  output byte drain
```

这正好说明“W4 支持”应该怎么理解：

```text
不是 hot path 里把 int4 先软件 unpack 成 int8。
而是 graph/packer 预先生成 HMX 需要的 nibble-packed weight stream，
然后 convbnb body 让 HMX weight port 按 nibble 路径消费它。
```

### 8.1 W4A8 weight stream 是怎样进来的

当前 W4A8 packer 的关键约束是：

```text
logical W shape:
  W[K,N], signed int4

HMX W4A8 physical tile:
  K32 x N64 logical tile
  one byte = two int4 weights
  paired output channels are n and n+32
```

注意最后一句。W4A8 不是把相邻的 `n0/n1` 塞进一个 byte，而是在每个
`K32 x N64` tile 里，把同一个 K row 上相隔 32 的 output channel 配对：

```text
same k row, within one N64 tile

logical:

        n0  n1  n2  ... n31 | n32 n33 n34 ... n63
k0      a   b   c       ... | A   B   C       ...

packed bytes:

  byte0 = pack(W[k0,n0],  W[k0,n32])
  byte1 = pack(W[k0,n1],  W[k0,n33])
  byte2 = pack(W[k0,n2],  W[k0,n34])
  ...
  byte31= pack(W[k0,n31], W[k0,n63])
```

更小一点画成 `K=4, N=8` 的示意，可以把真实的 `n+32` 缩小成 `n+4`：

```text
logical W[K,N]

        n0  n1  n2  n3 | n4  n5  n6  n7
k0      a   b   c   d  | A   B   C   D
k1      e   f   g   h  | E   F   G   H
k2      i   j   k   l  | I   J   K   L
k3      m   n   o   p  | M   N   O   P

packed stream, conceptually:

k0:  pack(a,A) pack(b,B) pack(c,C) pack(d,D)
k1:  pack(e,E) pack(f,F) pack(g,G) pack(h,H)
k2:  pack(i,I) pack(j,J) pack(k,K) pack(l,L)
k3:  pack(m,M) pack(n,N) pack(o,O) pack(p,P)
```

真实 packer 还会按 `K32` 分块、把 K 进一步拆成 `8` 个 group，每个 group
里连续放 `4` 个 K row：

```text
for each K32 tile:
  for each N64 tile:
    for kg in 0..7:
      for nc in 0..31:
        for kr in 0..3:
          k_idx = K_tile_base + kg*4 + kr
          low  = W[k_idx, N_tile_base + nc]
          high = W[k_idx, N_tile_base + nc + 32]
          byte = low | (high << 4)
```

一个真实 `K32 x N64` W4 tile 的体积是：

```text
32 * 64 * 4 bits = 8192 bits = 1024 bytes
```

这有两个含义：

```text
1. 相同 N64 范围内，W4 的字节数是 W8 的一半。
2. HMX tile 仍然喜欢固定大小的物理 stream；
   W4A8 用 N64 逻辑宽度把一个 tile 填回 1024B。
```

因此 W4A8 的 weight tensor 在 QNN/custom op 边界看起来会是：

```text
[1, 1, K, N/2] byte carrier
```

这里的 `N/2` 不是逻辑 output channel 变少，而是两个 int4 output-channel
权重共用一个 byte carrier。

### 8.2 carrier 和 nibble encoding 是什么

`carrier` 指“承载 packed int4 的 byte tensor”。它不是逻辑 I8 权重，而是
为了让 QNN/ctxgen/runtime 能把常量送到 HTP 的外层容器。

当前 W4A8 路径有一个容易忽略的细节：

```text
custom-op constant -> QNN generic weights_to_vtcm sidecar -> HMX raw weight stream
```

这个 sidecar 会对 INT8 carrier 做 sign-bit toggle。因此 packer 先把已经
排好的 native W4 byte stream 再异或一次 `0x80`：

```text
DLC/constant carrier byte = native_packed_byte ^ 0x80
runtime sidecar toggle    = ^ 0x80
HMX sees                  = native_packed_byte
```

这一步不是 W4 计算本身，只是为了穿过 QNN 的常量搬运路径后，HMX 最终看到
正确的 nibble stream。

### 8.3 activation/output 不重排 payload，只改 pointer table 视图

W4A8 的 activation 和 output 是 `Crouton_8 Indirect`：

```text
activation:
  QNN gives Crouton_8 block table
  HMX body consumes act_desc -> table entries -> U8 payload blocks

output:
  QNN gives Crouton_8 block table
  HMX body consumes out_desc -> table entries -> U8 output blocks
```

当前 custom op 的 hot path 不应该复制 activation/output payload：

```text
payload blocks:
  stay in QNN-owned TCM blocks

metadata:
  copied/reshaped into native descriptor ABI
```

因此 W4A8 wrapper 的主要工作是把 QHPI 的 block table 变成 native BNB
entry 想看的 descriptor：

```text
M_t = M / 16
K_t = K / 32
N_t = N / 32

out_desc:
  [out_table, N_t, desc_m_t*4, desc_m_t*4, 8, N_t*32]

act_desc:
  [act_table, K_t, desc_m_t*4]
```

对当前 `M=K=N=256` 的 native-compact surface：

```text
M_t = 16
K_t = 8
N_t = 8
desc_m_t = 8

out_desc = [table, 8, 32, 32, 8, 256]
act_desc = [table, 8, 32]
```

这里的 `desc_m_t` 不是随便选的循环次数。它决定 HMX body 怎样在 M 方向
推进 pointer table。如果这个值或 table 物理视图错了，可能出现“kernel
cycles 变好但只算对一半 rows”的假阳性；这个坑在之前的 W4A8 native-layout
实验里已经踩过。

### 8.4 convbnb body 负责什么

W4A8 使用的是 native `hmx_v73_convbnb1x1_stride1` 路线。当前仓库里的
embedded `.inc` 是从 `libQnnHtpV75Skel.so` 的 `0x2f0780` slice 复刻来的
`2624` 字节 wrapper/deep contiguous slice。

这个 `.inc` 的状态要如实看待：

```text
byte-proven:
  wrapper ABI、branch/loop 边界、部分普通 Hexagon 指令、部分 drain/store 路线

仍保留 raw words:
  branch-sensitive packets
  native padding
  尚未 byte-proven 的 unknown HMX packets
```

所以本文不把 W4A8 写成一张完整 mnemonic 清单。更稳妥的抽象是：

```text
输入:
  r0 -> out_desc
  r1 -> act_desc
  r2 -> packed W4 weight stream
  r3 -> bias/control bytes
  r4 -> W4 mask/routing desc
  r5 -> extra_param

核心语义:
  Crouton_8 activation blocks
  + K-major 32x64 W4 nibble-packed weight stream
  + convw4b1x1 mask/routing
  -> accumulator banks
  -> U8 drain/store into Crouton_8 output blocks
```

`convbnb` 这个名字本身就是一个很好的数据流标签：

```text
b = byte activation side
n = nibble weight side
b = byte output/drain side
```

### 8.5 为什么 W4A8 没有 2x

W4A8 只减少了 weight stream 的体积和 weight port 侧的数据宽度。它没有把
整条 pipeline 都变成“两个 INT4 MAC 阵列并行跑”：

```text
仍然存在:
  U8 activation fetch
  Crouton_8 pointer table traversal
  accumulator bank lifetime
  U8 drain/pack/store
  wrapper/descriptor/profiling envelope
```

所以当前 V75/QAIRT 2.45.0 的 256^3 chain8 结果里，W4A8 比 U8I8 只是
kernel-level 小幅变快：

```text
custom main:
  u8i8  = 10891 cycles
  w4a8  = 10213 cycles
  gain  = about 1.07x

native kernel:
  u8i8  = 12435 cycles
  w4a8  = 11546 cycles
  gain  = about 1.08x
```

这个结果和硬件心智模型一致：

```text
W4A8 有原生 nibble weight 输入/路由。
但当前证据不支持 W4A8 背后有相对 W8A8 翻倍的 2x INT4 MAC 阵列。
```

## 9. W4A16 的数据流：half / nibble / half

`w4a16` 对应 HNH-style pipeline：

```text
A:   U16 activation tile
W:   I4 weight tile, packed as nibbles
ACC: internal wider integer accumulator
C:   U16 output tile
```

图示：

```text
U16 activation tile
        |
        v
 half-side activation lanes ----\
                                 +--> accumulator banks --> U16 convert --> U16 output tile
 nibble weight lanes       ----/
        ^
        |
packed I4 weight tile
```

单个输出元素可以理解成：

```text
A row:
[a0 a1 a2 a3]       each a is uint16

W column:
[w0 w1 w2 w3]       each w is signed int4, packed in nibbles

acc = a0*w0 + a1*w1 + a2*w2 + a3*w3
out = quantize_to_u16(acc + A16/W4 drain recipe)
```

关键点：

```text
int4 unpack 不是 C++ wrapper 在热路径里做的。
weight 已经提前 pack 成 nibble stream。
HMX 的 nibble weight path 负责按硬件路线消费这些 nibble。
```

这说明 W4 不是纯软件方式支持。如果是纯软件模拟，热路径应该先把 int4
unpack/expand 成 int8 或 int16，再喂给普通 weight path。当前复刻 body 的
证据不是这样；它直接走 W4/nibble weight route。

但这也不等价于存在“吞吐翻倍的 INT4 计算单元”。更准确的硬件抽象是：

```text
W4 支持点:
  nibble-packed weight ingestion
  W4-specific weight route
  W4-specific body family and drain contract

没有被当前性能证据支持的点:
  W4 MAC lanes = 2 * W8 MAC lanes
  W4A8/W4A16 kernel cycles = W8A8/W8A16 的一半
  通用 INT4 tensor core 式的任意 INT4 matmul
```

因此本文后续提到“原生 W4”时，指的是原生 W4 权重输入/解码/路由契约，
不是指有一个表现出 2x 吞吐的独立 INT4 MAC 阵列。

逻辑 32x32 W4 weight tile 的体积是：

```text
32 * 32 * 4 bits = 4096 bits = 512 bytes
```

对比 `u8i8`：

```text
weight stream:       1024B -> 512B per 32x32 tile
activation stream:   U8    -> U16
output stream:       U8    -> U16
```

所以 W4A16 的优势和成本同时存在：

```text
weight 带宽下降
activation/output 带宽上升
drain/control 语义更复杂
```

在当前复刻 body 里，对应关键路线是：

```text
activation.ub = ...
weight.n      = ...
cvt.uh        = acc(...):2x2
```

这里最重要的是 `weight.n` 和 `cvt.uh`：

```text
weight.n: weight 输入侧走 nibble 权重路径。
cvt.uh:   accumulator drain 成 unsigned halfword，也就是 U16 output。
```

`activation.ub` 这个 mnemonic 容易让人误会。本文不把它直接等同于逻辑
activation dtype。W4A16 的 A16 语义来自整条契约：

```text
QNN tensor surface 是 U16/Crouton_16
HNH body family
W4 nibble weight path
U16 cvt/drain path
A16/W4 bias/control record
```

## 10. W4A16 packing 的小矩阵例子

用一个小的 `K=8, N=4` 权重矩阵说明 W4A16 这条 HNH 路径里的
nibble packing：

```text
logical W[K,N]

        n0  n1  n2  n3
k0      a   b   c   d
k1      e   f   g   h
k2      i   j   k   l
k3      m   n   o   p
k4      q   r   s   t
k5      u   v   w   x
k6      y   z   A   B
k7      C   D   E   F
```

每个元素是 signed int4。一个 byte 可以装两个 int4：

```text
byte = low_nibble | (high_nibble << 4)
```

当前 W4A16 native-style pack 可以抽象成按列取 `k` 和 `k+4` 配对：

```text
for n0:
  byte0 = pack(k0:n0, k4:n0)
  byte1 = pack(k1:n0, k5:n0)
  byte2 = pack(k2:n0, k6:n0)
  byte3 = pack(k3:n0, k7:n0)

for n1:
  byte4 = pack(k0:n1, k4:n1)
  byte5 = pack(k1:n1, k5:n1)
  byte6 = pack(k2:n1, k6:n1)
  byte7 = pack(k3:n1, k7:n1)
```

真实 32x32 路径还会按 K32 block、N32 tile、K8 group 组织。小例子的重点是：

```text
逻辑 int4 矩阵在进入 HMX 前已经变成 byte stream。
每个 byte 里有两个权重。
HMX weight.n 路径按 nibble 权重解释这条 stream。
```

## 11. Accumulator banks 到底输出什么

这部分容易混淆，单独展开。

对一个小 output tile：

```text
        n0      n1      n2      n3
m0    acc00   acc01   acc02   acc03
m1    acc10   acc11   acc12   acc13
m2    acc20   acc21   acc22   acc23
m3    acc30   acc31   acc32   acc33
```

每个 `accXY` 都是沿 K 方向累加：

```text
acc00 = A[m0,k0]*W[k0,n0]
      + A[m0,k1]*W[k1,n0]
      + A[m0,k2]*W[k2,n0]
      + ...

acc01 = A[m0,k0]*W[k0,n1]
      + A[m0,k1]*W[k1,n1]
      + A[m0,k2]*W[k2,n1]
      + ...
```

`banks` 这个词强调两点：

```text
不是只有一个 accumulator，而是一组并行 accumulator。
这些 accumulator 对应 output tile 里的多个 m/n lane。
```

它们的输出不是最终 tensor。更准确地说：

```text
MAC fabric 输出 internal accumulator。
convert/drain 输出 tensor dtype。
```

所以：

```text
u8i8:
  U8 x I8 -> internal accumulator -> U8

w4a8:
  U8 x I4 -> internal accumulator -> U8

w4a16:
  U16 x I4 -> internal accumulator -> U16
```

内部 accumulator 的精确位宽本文不声明；我们只依赖当前 body 能确认的 drain
路径和最终输出 dtype。

### 11.1 U8I8 accumulator -> drain 的逆向结果

以当前 `u8i8` 为例，body 里能静态确认的 drain 片段是：

```text
bias = mxmem2(r3)
cvt.ub = acc(r25)
mxmem(r10,r11):cm = cvt
```

含义不是“把 accumulator 直接当 U8 写出”，而是：

```text
accumulator banks
  + 当前 bias/control record
  + r25 选择的 drain route
  -> cvt.ub
  -> U8 output tile store
```

当前 retained artifact 可以把这条链路的数值中间量重建出来。对
`example/qnn_matmul_profile/output_u8i8_aligned_e2e_256`，U8I8 的 hot
kernel 实际累加的是 raw unsigned activation：

```text
raw_acc[m,n] = sum_k act_u8[m,k] * weight_i8[k,n]
```

activation zero-point 没有在 MAC 前逐元素减掉，而是提前 fold 进 bias record：

```text
effective[n] = -128 * sum_k(weight_i8[k,n]) + bias_q[n]
```

所以 drain 入口的可重建整数是：

```text
drain_in[m,n] = raw_acc[m,n] + effective[n]
              = sum_k (act_u8[m,k] - 128) * weight_i8[k,n] + bias_q[n]
```

结合受控输入和当前 retained artifact，`cvt.ub` 的外部可见效果可以细化为：

```text
out_u8[m,n] =
  clamp(
    trunc(drain_in[m,n] * scale_f16 / 512)
    + (baseline_u16 >> 7),
    0,
    255
  )
```

这个结论不是只看源码推出来的。通用 A8 drain 重建脚本：

```text
scripts/reconstruct_hmx_u8_drain.py
```

对 U8I8，它会从当前 artifact 读取：

```text
runtime_inputs_u8/act_u8i8.raw
u8i8.onnx.wRaw_KN.npy
u8i8.onnx.bias_q_int32.npy
u8i8.onnx.effective_int32.npy
device_out/out.raw
```

然后逐 chain 重建：

```text
raw_acc -> drain_in -> scale/baseline -> out_u8
```

典型命令：

```bash
python3 scripts/reconstruct_hmx_u8_drain.py \
  example/qnn_matmul_profile/output_u8i8_aligned_e2e_256
```

当前 `256^3, CHAIN=8` 的重建结果：

```text
effective matches saved file: True
reconstructed == ref:        True
reconstructed == device:     True
reconstructed == native:     True
final sha256:
  09ea6b13f2f1c80438c473f450590ae6440cecd13598c912b9daad3fd0bf093b
```

stage 0 的一个 sample：

```text
m0,n0:
  raw_acc   = 1060
  sum_w     = -7
  bias_q    = 1
  effective = -128 * (-7) + 1 = 897
  drain_in  = 1060 + 897 = 1957
  scale     = 512
  baseline  = 0
  out_u8    = 255
```

因此，U8I8 这条链路目前可以这样理解：

```text
HMX accumulator bank 内部值:
  对应 raw act_u8 * weight_i8 的 reduce 结果。

drain 前可重建整数:
  raw accumulator + folded effective bias。

drain 后可观测结果:
  经过 scale_f16 / 512、baseline_u16 >> 7、U8 clamp/saturate 后写到
  Crouton_8 output。
```

### 11.2 U8 drain record 的精简结论

把历史受控输入收敛成结论，U8 drain record 的前 128B 不是填充，而是
per-channel scale/baseline 控制区：

```text
per-channel 4 bytes:
  +0: scale_f16 bits
  +2: baseline_u16 raw bits

observed:
  scale contributes as scale_f16 / 512
  baseline contributes as baseline_u16 >> 7
```

后 128B 是每个 output channel 的 `effective_i32`：

```text
effective_i32[n] = -act_zp * sum_k(weight_i8[k,n]) + bias_q[n]
```

当 weight 全 0 时，raw accumulator 为 0，输出只由
`effective/scale/baseline` 决定；当只保留一个 `K` lane 时，输出变成
`act_u8[m,k0] - 128 + bias_q[n]` 后再进入同一套 scale/baseline/clamp。
这两类受控输入共同排除了“zero-weight 特殊路径”：raw accumulator 和
effective 确实是先相加，然后一起进入 `cvt.ub`。

因此 U8 drain record 可以画成：

```text
one N32 tile bias/control record = 256B

0x000..0x07f:
  32 lanes * 4B control
  lane c:
    u16 scale_f16_bits
    u16 baseline_raw_bits

0x080..0x0ff:
  32 lanes * 4B effective_i32
  lane c:
    i32 effective_i32[n_base + c]
```

数值上就是：

```text
raw accumulator banks
  + effective_i32
  -> scale_f16 / 512
  -> baseline_u16 >> 7
  -> U8 saturate
```

仍然保留的边界是：我们没有直接读取 HMX accumulator bank 的物理寄存器，也不
声明它的官方精确位宽。这里确认的是当前 U8I8 契约下，bank 经
`bias/control + cvt.ub` 后的外部等价模型。

### 11.3 r25 bit12 的结论

静态 body 里 main path 的两次 drain 是：

```text
M12: r25 = setbit(r25,#0xc); cvt.ub = acc(r25)   -> first cvt/store
M14: r25 = clrbit(r25,#0xc); cvt.ub = acc(r25)   -> second cvt/store
```

早期把 `r25 bit12` 叫作 “M half selector”，这个说法现在要收窄。更准确的
结论是：

```text
r25 bit12 是 cvt.ub = acc(r25) 的 drain route selector 之一。
它影响 accumulator lanes、当前 bias/control record、output tile lanes 之间的配对。
它不是一个可以简单叫作 "raw accumulator M half selector" 的位。
```

关键依据可以压成两个事实。

第一，只有 raw accumulator 带 M 信息、bias/control 不带 N 差异时：

```text
weight pattern:      single_k0
activation pattern:  row_ramp
bias pattern:        constant, bias_q = 128
scale_f16:           512
baseline_u16:        0
```

这时：

```text
effective[n] = -128 * 1 + 128 = 0
raw_acc[m,n] = act_u8[m,k0] = m
out[m,n]     = m
```

强制同向 route 仍能保持 bit-exact。如果 bit12 只是 raw accumulator 的
M-half 选择位，这里应该立刻出现行块重复或错位；实际没有。

第二，去掉 raw MAC、只让 bias/control 带 N 信息时：

```text
weight pattern:      zero
activation pattern:  constant
bias pattern:        ramp0
scale_f16:           512
baseline_u16:        0
```

这时：

```text
raw_acc      = 0
effective[n] = n
out[m,n]     = n
```

正确输出每一行的 N tile 起点应当是：

```text
N tile starts:
  [0, 32, 64, 96, 128, 160, 192, 224]
```

一旦把两次 drain 的 route 强制成同向，错误主要表现为 32 列 N tile pair
复用或交换：

```text
both clear: [0,0,64,64,128,128,192,192]
both set:   [0,32,96,96,160,160,224,224]
swapped:    [0,0,96,64,160,128,224,192]
```

这说明 bit12 更直接地控制 `cvt.ub` drain 时对当前
`bias = mxmem2(r3)` record / per-N control lane 的配对。更细的现象是：
第一个 32x32 output tile 内不是整 tile 统一错，而是有 8-row sublane 差异：

```text
rows 0..7,   col0 -> 0
rows 8..31,  col0 -> 32
```

因此 drain route 的粒度不是一个抽象的完整矩阵 half，而是 HMX `cvt` 内部的
lane/bank 配对。它同时涉及：

```text
accumulator lanes
bias/control lanes
N tile pair selection
output Crouton lane layout
```

当前更稳妥的模型是：

```text
             bias/control record for N tile pair
                         |
                         v
raw accumulator banks -> cvt.ub = acc(r25) -> U8 output tile
                         ^
                         |
                 r25 route bits, including bit12
```

也就是说，`r25 bit12` 不改变数值公式：

```text
out_u8 = clamp(trunc((raw_acc + effective) * scale_f16 / 512)
               + (baseline_u16 >> 7))
```

它决定的是 `raw_acc`、`effective/scale/baseline` 和 output lane 如何配对。
配对正确时公式 bit-exact；配对错误时，公式本身仍成立，但使用了错误 N tile
或错误 sublane 的 control record。

清理后的结论：

```text
我们仍没有直接读取 HMX accumulator bank 的物理寄存器/位宽。
我们已经确认的是：
  - accumulator 输出不是最终 U8；
  - drain/cvt 才输出 U8；
  - r25 bit12 是 drain route selector；
  - 它不能简单解释为 raw accumulator M half selector。
```

### 11.4 W4A16 accumulator -> drain 的当前理解

W4A16 的 drain 路线和 U8I8 明显不同。U8I8 是：

```text
cvt.ub = acc(r25)  -> U8 output
```

W4A16 HNH body 用的是：

```text
bias = mxmem2(r3)
cvt.uh = acc(r25):2x2
mxmem(r10,r11) = cvt
```

这里的 `cvt.uh` 表示 convert to unsigned halfword，也就是 drain 之后写出的
tensor dtype 是 U16。`:2x2` 是这条 HNH drain 的 lane/tile 组织后缀；
目前最稳妥的理解是它让一次 `cvt` 覆盖一个 2x2 风格的内部 lane group，
不要把它解读成普通 C 代码里的二维循环。

#### HNH drain 的关键形态

W4A16 main path 不是“一个 bias load + 一个 cvt + 一个 store”。它的一个
输出 store 前会连续做两次 bias/control load 和两次 `cvt.uh`：

```text
first half-record:
  r3 += 0x100
  r25 = next control word
  bias = mxmem2(r3)
  cvt.uh = acc(r25):2x2

second half-record:
  r3 += 0x100
  r25 = next control word
  bias = mxmem2(r3)
  cvt.uh = acc(r25):2x2

store:
  mxmem(r10,r11) = cvt
```

这说明 W4A16 的 `cvt` 更像一个 staged drain register：先用一个
bias/control half-record 填一部分 U16 lanes，再用下一个 half-record 填另一
部分 lanes，最后一次性把 `cvt` 写入 Crouton16 output tile。

对比 U8I8：

```text
U8I8:
  one bias/control record
  one cvt.ub
  one U8 store

W4A16:
  two 256B bias/control half-records
  two cvt.uh :2x2
  one U16 store
```

这就是 W4A16 drain/control 面更重的直接原因之一。

#### 512B native A16 bias/control record

当前 W4A16 `native_a16` bias/control record 是每个 N32 tile 一条 512B
记录：

```text
one N32 tile bias/control record = 512B

0x000..0x0ff: half-record 0
0x100..0x1ff: half-record 1
```

这两个 half-record 分别覆盖 N32 tile 内的偶数列和奇数列：

```text
logical N lanes in one N32 tile:

  n:       0  1  2  3  4  5  6  7  ... 30 31
           |     |     |     |        |
half 0:    0     2     4     6  ... 30

              |     |     |     |        |
half 1:       1     3     5     7  ... 31
```

每个 half-record 是 16 个 lane，每个 lane 占 8B。前 128B 是每个 lane 的
4 个 U16 control words，后 128B 是每个 lane 的 effective slot：

```text
half-record layout, 256B:

  +0x00 .. +0x7f:
    16 lanes * 8B constants

    lane i constants:
      u16[0] = 0x5524
      u16[1] = 0x8040
      u16[2] = 0x0092
      u16[3] = 0x4000

  +0x80 .. +0xff:
    16 lanes * 8B effective slots

    lane i effective slot:
      i32 effective_i32[col]
      i32 zero/pad
```

对第一个 N32 tile，小矩阵式地看就是：

```text
half-record 0:
  lane0 -> n0  -> effective[0]
  lane1 -> n2  -> effective[2]
  lane2 -> n4  -> effective[4]
  lane3 -> n6  -> effective[6]
  ...

half-record 1:
  lane0 -> n1  -> effective[1]
  lane1 -> n3  -> effective[3]
  lane2 -> n5  -> effective[5]
  lane3 -> n7  -> effective[7]
  ...
```

当前生成器里的 `effective_i32` 是：

```text
effective_i32[n] = -128 * sum_k(weight_i4[k,n])
```

这个 `-128` 不应按 U8I8 那样直接解释成 A16 activation zero-point。W4A16
公开 tensor contract 是 native A16：

```text
U16 code 32768 represents numeric zero
scale = 1 / 32767
```

因此 W4A16 的 bias/control record 是 HMX drain recipe 的一部分：它把
W4 nibble scale、A16 zero/scale、lane parity、rounding/saturation 控制和
少量 folded correction 放在一起。`effective_i32` 只是这个 recipe 的一个
字段，不是完整的神经网络 bias。

#### 外部可见数值模型

从 retained artifact 可见，W4A16 的近似数值模型是：

```text
centered_acc[m,n] =
  sum_k (act_u16[m,k] - 32768) * weight_i4[k,n]

out_u16[m,n] ~= saturate_u16(
  round(centered_acc[m,n] / 7 + 32768)
)
```

这里的 `/7` 来自 W4 signed int4 的 `qmax = 7`。这能解释 W4A16 为什么输出
是 U16，并且为什么很多输出会 saturate 到 0 或 65535。

但这个公式目前只能叫“外部可见近似模型”，不能叫 bit-exact drain 模型。
当前 canonical `256^3, CHAIN=8` artifact 的证据是：

```text
custom W4A16 output vs native QNN oracle:
  native-exact = 65536/65536
  maxdiff      = 0

custom W4A16 output vs Python analytic model:
  analytic-exact = 63422/65536
  maxdiff        = 65535
```

因此真正 bit-exact 的 drain oracle 仍然是 native `cvt.uh = acc(...):2x2`
路径，而不是 Python 的 `round(acc/7 + 32768)`。差异很可能来自 HMX drain
内部的定点 rounding、lane pairing、saturate 顺序或 control word 细节。

现在用工具固定这个差异分布：

```bash
python3 scripts/analyze_a16_drain_delta.py \
  example/qnn_matmul_profile/output_w4a16_aligned_e2e_256

python3 scripts/analyze_a16_drain_delta.py \
  example/qnn_matmul_profile/output_w8a16_aligned_e2e_256
```

当前结果说明两件事：

```text
W4A16:
  custom_vs_native:   65536/65536
  custom_vs_analytic: 63422/65536
  abs<=3:             63643/65536
  major signed deltas: -36, -65535, +65535

W8A16:
  custom_vs_native:   65536/65536
  custom_vs_analytic: 52003/65536
  abs<=3:             52014/65536
```

所以 A16 当前最可靠的 oracle 是 native output，不是外部 analytic formula。
`analyze_a16_drain_delta.py` 的价值在于把差异按 signed delta、N32 tile、
row32 tile 和 saturation crossing 拆开，后续继续破解 `cvt.uh` rounding /
control word 时可以直接看差异是否被某个假设消掉。

#### 当前边界

W4A16 这条链路目前可以可靠说明到这里：

```text
U16 activation tile
  + W4/nibble weight.n tile
  -> internal accumulator banks
  -> two-stage HNH cvt.uh drain with 512B native A16 bias/control
  -> U16 Crouton16 output tile
```

仍然不能声明的是：

```text
HMX accumulator bank 的物理位宽
每个 r25 bit 的完整语义
cvt.uh :2x2 的逐 lane bit-exact rounding 公式
```

但和 U8I8 相比，已经可以确定：

```text
U8I8 drain:  U8 output, one record path, cvt.ub
W4A16 drain: U16 output, two half-record staged path, cvt.uh:2x2
```

### 11.5 W4A8 accumulator -> drain 的当前理解

W4A8 位于 U8I8 和 W4A16 中间：

```text
U8 activation tile
  + W4/nibble weight tile
  -> internal accumulator banks
  -> U8-equivalent cvt/drain with 256B bias/control
  -> U8 Crouton_8 output tile
```

这里要先划清证据边界。当前 W4A8 `.inc` 是
`hmx_v73_convbnb1x1_stride1` 在 V75 skel `0x2f0780` 的 `2624` 字节
replica，已经和 native slice byte-identical。静态文本里能稳定看到：

```text
bias = mxmem2(r3)
...
mxmem(r10,r11):cm = cvt
```

中间的 HMX convert packet 仍有 raw `.word` 保留，所以本文不把它写成已经完整
decode 的 `cvt.ub` mnemonic。更稳妥的说法是：这条 path 的外部语义和输出
contract 是 U8 drain，`cvt` 最终被写到 Crouton_8 output。

现在有一个专门的 packet inventory 工具来固定这条边界：

```bash
python3 scripts/analyze_w4a8_cvt_packets.py
```

它从 W4A8 `.inc` 里抽取 accumulator conversion/writeback 周围的 raw packet。
当前结果是：

```text
pre_store_cvt_tail:  10
post_bias_cvt_tail:   2
post_store_tail:     12

repeated cvt-like words:
  0x75594000
  0x10bf40f6
  0x10bf40f8
  0x5cdf68f6
  0x5cdf68f8

plain cvt.ub = acc(rX) candidate words:
  0xa6f7d710 ... 0xa6ffd710
  matches in W4A8 raw inventory: {}
```

这说明 W4A8 的 raw groups 不是随机未知字节，而是稳定出现在
`bias = mxmem2(r3)` 和 `mxmem(...):cm = cvt` 附近的 mixed HMX/control
模板。并且这些 raw `a6..dc/dd` word 不等于 plain `cvt.ub = acc(rX)` 的
已汇编候选编码。因此结论也更精确：W4A8 是 U8 drain 语义，但 exact
BNB-specific `cvt` mnemonic 仍保持未声明，直到能 byte-prove 对应 asm。

数值链路和 U8I8 很接近，只是 weight 侧从 I8 byte 变成 signed I4 nibble：

```text
raw_acc[m,n] = sum_k act_u8[m,k] * weight_i4[k,n]
effective[n] = -128 * sum_k(weight_i4[k,n]) + bias_q[n]

drain_in[m,n] = raw_acc[m,n] + effective[n]
              = sum_k (act_u8[m,k] - 128) * weight_i4[k,n] + bias_q[n]
```

W4A8 的 bias/control record 仍是 U8-style 的 `256B / N32 tile`：

```text
one N32 tile bias/control record = 256B

0x000..0x07f:
  32 lanes * 4B control
  lane c:
    u16 scale_f16_bits
    u16 baseline_raw_bits

0x080..0x0ff:
  32 lanes * 4B effective_i32
  lane c:
    i32 effective_i32[n_base + c]
```

所以外部可见 drain 公式也是 U8 公式：

```text
out_u8[m,n] =
  clamp(
    trunc(drain_in[m,n] * scale_f16 / 512)
    + (baseline_u16 >> 7),
    0,
    255
  )
```

当前 canonical artifact 使用 `scale_f16 = 512`、`baseline_u16 = 0`，因此公式
退化为：

```text
out_u8[m,n] = clamp(drain_in[m,n], 0, 255)
```

用一个小矩阵看 W4A8 的 drain 入口：

```text
A_u8 row m0:       [a0, a1, a2, a3]
W_i4 col n0:       [w0, w1, w2, w3]

raw_acc[m0,n0]:
  a0*w0 + a1*w1 + a2*w2 + a3*w3

effective[n0]:
  -128 * (w0 + w1 + w2 + w3) + bias_q[n0]

drain_in[m0,n0]:
  (a0-128)*w0 + (a1-128)*w1 + (a2-128)*w2 + (a3-128)*w3
  + bias_q[n0]
```

当前 `256^3, CHAIN=8` 的实测重建结果：

```bash
python3 scripts/reconstruct_hmx_u8_drain.py \
  example/qnn_matmul_profile/output_w4a8_aligned_e2e_256
```

```text
effective matches formula:      True
reconstructed == ref:           True
reconstructed == custom device: True
reconstructed == native:        True
custom device == native device: True
native compare:                 65536/65536, maxdiff 0
final sha256:
  09ea6b13f2f1c80438c473f450590ae6440cecd13598c912b9daad3fd0bf093b
```

stage 0 的一个 sample 和 U8I8 很像：

```text
m0,n0:
  raw_acc   = 1060
  sum_w     = -7
  bias_q    = 1
  effective = -128 * (-7) + 1 = 897
  drain_in  = 1060 + 897 = 1957
  scale     = 512
  baseline  = 0
  out_u8    = 255
```

最后一个 chain 的同一位置会因为前一层输出已经被 U8 saturate 过而进入另一种
accumulator 分布：

```text
chain7 m0,n0:
  raw_acc   = -45135
  effective = 897
  drain_in  = -44238
  out_u8    = 0
```

这说明 W4A8 的 accumulator -> drain 可以稳定理解为：

```text
HMX accumulator bank 内部值:
  raw act_u8 * weight_i4 的 reduce 结果。

drain 前可重建整数:
  raw accumulator + folded effective bias。

drain 后可观测结果:
  U8-style scale/baseline/clamp 后写到 Crouton_8 output。
```

和 U8I8 的核心差异不在 drain dtype，而在 weight port：

```text
U8I8:
  weight stream = I8 byte
  body family   = convbbb
  drain output  = U8

W4A8:
  weight stream = signed I4 nibble, K32xN64 physical route
  body family   = convbnb
  drain output  = U8
```

和 W4A16 的差异则主要在 drain 面：

```text
W4A8:
  256B U8-style record
  one U8 output path
  external model is bit-exact for current artifact

W4A16:
  512B native A16 record
  two half-record staged U16 path
  external analytic model is only approximate
```

## 12. Bias/control 为什么是硬件契约的一部分

普通神经网络里 bias 可能只是：

```text
acc += bias[n]
```

但 HMX body 里的 bias/control 更像 drain recipe：

```text
accumulator banks
    + constants
    + effective bias
    + scale/baseline
    + route/lane selection
    -> output tile
```

对于 `u8i8`，可以理解成：

```text
(act_u8 - act_zp) * weight_i8 + bias
```

其中 activation zero-point 相关项可以提前 fold 到 effective bias：

```text
act_u8 * weight_i8 + (-act_zp * sum(weight_i8)) + bias
```

这样 hot kernel 里不需要逐元素减 zero-point。drain record 负责把 accumulator
转换到 U8。

对于 `w4a16`，情况更复杂：

```text
activation 是 U16
weight 是 signed int4 nibble
output 是 U16
```

因此 drain record 不只是 bias，还要携带 A16/W4 路径需要的常量和控制信息。
它决定 accumulator 如何被分批 drain、如何转成 U16、如何写入 output tile。

这也是为什么 W4A16 的 side data 比 U8I8 更重。

W4A8 介于两者之间：drain 仍是 U8，但 weight 侧需要 W4 mask/routing 和
nibble-packed carrier，所以它的控制面比纯 U8I8 更重，却没有 W4A16 的
U16 output/drain 负担。

## 13. 这些 kernel 的共同设计原则

这些路径共享同一个总体框架：

```text
logical QNN MatMul
    |
    v
reinterpret as Conv1x1 tile problem
    |
    v
prepare activation/output tile layout
    |
    v
pack weight into selected HMX physical format
    |
    v
build bias/control drain records
    |
    v
construct descriptors and routing/mask state
    |
    v
call matching HMX body
```

不同的是 selected HMX body family：

```text
u8i8:
  body family: convbbb
  activation:  U8 tile stream
  weight:      I8 byte-packed stream
  accumulator: internal wide integer banks
  drain:       U8 output path

w4a8:
  body family: convbnb
  activation:  U8 tile stream
  weight:      I4 nibble-packed K32xN64 stream
  accumulator: internal wide integer banks
  drain:       U8 output path

w4a16:
  body family: convhnh
  activation:  U16 tile stream
  weight:      I4 nibble-packed stream
  accumulator: internal wide integer banks
  drain:       U16 output path
```

所以最关键的抽象是：

```text
u8i8  不是 "普通 matmul + int8 参数"。
它是 "把 MatMul 组织成 HMX convbbb 能吃的 byte/byte/byte tile streams"。

w4a8 也不是 "u8i8 的 weight tensor 压缩一下"。
它是 "把 MatMul 组织成 HMX convbnb 能吃的 byte/nibble/byte tile streams"。

w4a16 不是 "u8i8 kernel 里把 weight 改成 4-bit"。
它是 "把 MatMul 组织成 HMX convhnh 能吃的 half/nibble/half tile streams"。
```

## 14. 为什么 wrapper 看起来像，物理契约却不同

这些 custom op wrapper 都会做类似事情：

```text
准备 out_desc
准备 act_desc
准备 mask/routing
传入 weight pointer
传入 bias/control pointer
调用 inline asm body
```

这会造成一个错觉：

```text
它们是不是同一个 kernel 模板换了 dtype?
```

不是。相似的是“如何把数据喂给 HMX body”的外壳；真正不同的是：

```text
weight port 怎么解释数据
activation/output tile 是 8-bit 还是 16-bit 语义
accumulator drain 成 U8 还是 U16
bias/control record 的内容和步进方式
native body family 的 HMX 指令路线
```

换句话说：

```text
descriptor ABI 相似，不代表物理数据契约相同。
```

## 15. 当前 repo 证据边界

这个工作模型来自当前仓库中的可验证材料。

W4A16 body：

```text
example/qnn_hmx_matmul_w4a16/src/v73deep_conv1x1_kernel.inc

native slice:
  hmx_v73_convhnh1x1deep_stride1

关键路线:
  weight.n
  cvt.uh = acc(...):2x2
```

W4A8 body：

```text
example/qnn_hmx_matmul_w4a8/src/v73deep_conv1x1_kernel.inc

native slice:
  hmx_v73_convbnb1x1_stride1 at 0x2f0780

当前状态:
  2624-byte hybrid readable asm/word replica
  byte-proven packets promoted to asm
  unknown HMX / branch-sensitive packets remain raw words

关键路线:
  byte activation/output surface
  nibble-packed W4 weight stream
  U8 drain/store semantics
```

U8I8 body：

```text
example/qnn_hmx_matmul_u8i8/src/v73deep_conv1x1_kernel.inc

native slice:
  hmx_v73_convbbb1x1deep_stride1

关键路线:
  weight.b
  cvt.ub = acc(...)
```

W4 packer：

```text
example/qnn_hmx_matmul_common/gen_quant_chain.py

W4A8:
  K-major 32x64 W4 tile
  one byte packs output channels n and n+32 for the same K row
  carrier shape [1,1,K,N/2]

W4A16:
  native_kblock32_nmajor_k4_lohi style packing for the HNH path
```

U8I8 packer：

```text
example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/gen_u8i8_chain.py
```

必须保留的边界：

```text
我们可以确认当前复刻 body 的 byte identity 和关键 HMX mnemonic。
我们可以确认 packer 生成的物理 stream 形状。
我们不能把本文写成完整官方 HMX ISA 规范。
```

## 16. 实验边界和性能数据

本文关于 W4 是否有 2x INT4 MAC 阵列的判断，是当前实验边界下的结论：

```text
HTP/HMX 架构:
  Hexagon V75 / HTP V75

native skel:
  tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so

SDK:
  tools/qnn-sdk/sdk.yaml
  product: QAIRT
  version: 2.45.0
  build_id: 260326154327
  qnn_backend_api_version: 2.18.0

HMX asm 验证目标:
  clang-19 -target hexagon -mcpu=hexagonv75 -mhmx

shape / graph:
  M=256, K=256, N=256
  CHAIN=8
  perf_profile=burst
  当前仓库保留的 *_aligned_e2e_256 和 *_native_ref_e2e_256 artifacts
```

所以本文的性能判断不应外推成：

```text
所有 Hexagon 架构都如此
所有 QNN/QAIRT 版本都如此
所有 shape 都如此
官方 HMX ISA 对 INT4 没有任何硬件支持
```

更准确的边界是：

```text
在 V75 + QAIRT 2.45.0 + 当前 256^3 chain8 artifacts 下，
W4 有原生 nibble weight 通路，但没有表现出相对 W8 的 2x kernel-cycle 吞吐。
```

### 16.1 Kernel-level cycles

下面的 `custom main` 和 `native kernel` 是最接近“纯 matmul kernel”的
optrace 口径。它们不包含 `InputSlice`、`OutputSlice`、`ForceFormat`、
`weights_to_vtcm` 等整图开销，但仍包含 kernel wrapper、descriptor glue、
HMX body entry/exit、accumulator drain 和 profiling envelope。

数据来自当前磁盘上的 `optrace/summary.json`，现在可用脚本直接复现：

```bash
python3 scripts/summarize_hmx_perf.py
```

| Pair | W8 custom main | W4 custom main | W4/W8 custom cycles | Custom packets/event | W8 native kernel | W4 native kernel | W4/W8 native cycles |
|---|---:|---:|---:|---:|---:|---:|---:|
| A8 output (`u8i8` vs `w4a8`) | `10891` | `10213` | `0.938` (`1.07x`) | `337` -> `325` | `12435` | `11546` | `0.929` (`1.08x`) |
| A16 output (`w8a16` vs `w4a16`) | `29203` | `31419` | `1.076` (`0.93x`) | `682` -> `800` | `30182` | `29815` | `0.988` (`1.01x`) |

解读：

```text
A8:
  W4A8 kernel-level cycles 比 W8A8 略低，大约好 7-8%。

A16:
  W4A16 和 W8A16 基本同档。
  当前 live custom artifact 里 W4A16 反而慢一些；
  native kernel 口径下 W4A16 只快约 1%。

两组都没有接近 2x。
```

这就是本文采用下面表述的性能依据：

```text
W4 有原生 nibble weight 输入通路。
但在 V75/QAIRT 2.45.0 的当前实验里，W4 没有表现出 2x INT4 MAC 阵列吞吐。
```

### 16.2 Full timeline cycles

Full timeline 包含 graph-boundary 和 sidecar 工作，所以不是纯 kernel 比较。
它仍然有用，因为它能显示 layout/format 工作很容易吞掉 W4 weight-size
收益。

| Pair | W8 custom timeline | W4 custom timeline | W4/W8 custom timeline | W8 native timeline | W4 native timeline | W4/W8 native timeline |
|---|---:|---:|---:|---:|---:|---:|
| A8 output (`u8i8` vs `w4a8`) | `36342` | `47136` | `1.297` | `53946` | `48831` | `0.905` |
| A16 output (`w8a16` vs `w4a16`) | `76615` | `77854` | `1.016` | `79095` | `253245` | `3.202` |

说明：

```text
A8:
  native timeline 下 W4A8 有小幅收益；
  但当前 custom W4A8 timeline 更差，因为 graph-boundary/layout overhead
  吞掉了小幅 kernel gain。

A16:
  custom W4A16/W8A16 timeline 基本同档。
  native W4A16 timeline 不能和 native W8A16 timeline 直接 apples-to-apples:
  runnable W4A16 Conv reference 包含大量 SlicePad/Transpose/Concat layout work。
  因此 kernel-only 比较应使用 `q::ConvLayer_s1.opt`。
```

部分 status 文档仍记录旧的 accepted summary 数字，例如 W8A16 custom
`30871` / timeline `80217` 和 W4A8 custom `10025` / timeline `38644`。
上表使用当前磁盘上的 live `optrace/summary.json`。两种口径的结论一致：
W4 没有表现出 2x kernel throughput。

## 17. 一句话心智模型

```text
HMX = 固定 typed tile-MAC 机器。

QNN mixed precision kernel =
  选择一条 HMX typed pipeline
  + 把逻辑 tensor 预处理成这条 pipeline 的物理 stream
  + 用对应 body 把 stream 喂进 MAC fabric
  + 用对应 drain recipe 得到目标 dtype 输出。

u8i8 选择 byte/byte/byte。
w4a8 选择 byte/nibble/byte。
w4a16 选择 half/nibble/half。

W4 的含义:
  原生 nibble weight 通路，不是热路径软件 unpack 模拟。

W4 不应被理解成:
  相对 W8 有 2x 吞吐的独立 INT4 MAC 阵列。
```
