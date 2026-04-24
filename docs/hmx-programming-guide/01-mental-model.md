# 01 · HMX 的心智模型：accumulator + tile + convert

**TL;DR**：把 HMX 当成一个"**带 32×32 cell 的矩阵 MAC 协处理器**"。你给它三样东西——
(1) **一片累加器**（HMX 内部的 32×32 int32 寄存器阵列）、
(2) **两张 tile**（act + wt，住在 VTCM 里的按固定字节序摆好的 2D 矩阵）、
(3) **一个 convert 配置**（f16 scale + 读回格式）——它自己就能把 `C += A·W` 算出来。

本章目标：读完你能画出一次 HMX MAC 的数据流，并知道下一章要在代码里拼哪些零件。

## 三件套

### 1. Accumulator（acc）——HMX 内部的计算寄存器

- HMX 有**两套**独立累加器：**int32 定点**和 **xfp 扩展浮点**。前者对应整数路径
  （`.b/.n/.c/.ubit` 各类 weight），后者对应 `.hf` fp16 路径。
- 两套 acc **不能并存**——切模式前要 `mxclracc`（int）或 `mxclracc.hf`（fp）清零。
- 逻辑上是 32×32 cell，**每 cell 1 个 int32（或 1 个 xfp）**。1024 个 MAC 单元并行算。
- 一次 MAC packet 把 "act · wt 的 outer-product" 累加到整个 32×32 acc 上。

> 详细硬件结构见 `Agent/hmx_u8xi8_matmul_layers.md` §L7 (F)。

### 2. Tile——住在 VTCM 里的 2D 矩阵

**act tile**（2 KiB，固定 32 逻辑行 × 32 K）：
- 每个字节放什么由**写死的字节布局公式**决定——不是平铺（row-major）也不是简单转置。
- 硬件要 128 B aligned（实际 VTCM 在 4 KiB 边界，更安全）。
- HMX 只接受 **uint8 / fp16 / fp8** 三种 activation——其他（int16、signed int8、int4）
  全都要**软件侧偏移或拆分**成 uint8 再喂。

**wt tile**（int8 时 1 KiB，随 weight 字节宽度成比例缩小：int4 = 512 B、int2 = 256 B、int1 = 128 B、fp16 = 2 KiB）：
- 同样有特定字节布局。
- 支持的 weight 类型：`.b`(int8) / `.n`(int4) / `.c`(int2) / `.ubit`(1-bit) / `.sbit`(signed 1-bit) / `.sc`(signed 2-bit) / `.hf`(fp16) / `.f8`(fp8) / `.sm`(sparse mask)。

**bias tile**（256 B = 128 个 u16 fp16）：
- 是 convert 阶段用的，**不参与乘加本身**。
- 每个输出列一个 u16（格式：f16），convert 时 HMX 把 `acc[i][j] × bias[j] / 2` 做饱和/折叠后写到 output 格子里。

> 具体字节公式留到 `03-tile-layout-cheatsheet.md` 再看；现在记住"有公式就行"。

### 3. Convert——把 acc 写到 VTCM

- HMX 的 acc 是 int32（定点路径），但 **读回只能是 16-bit**（`.uh` / `.uh:sat`）或 fp16（`.hf`）。
- 想拿完整 int32 → 得做 **dual-scale**：两次 convert，各用不同 bias scale，CPU 侧拼回。这在后续章节讲。
- **convert 消耗 acc**：读完默认被清。需要保留（做 dual-scale 等）→ 加 `:retain`。

## 一次 HMX MAC 的标准数据流

```
       CPU prep                          HMX 执行
┌───────────────────────┐          ┌────────────────────────────┐
│ 1. pack A[i][k] -> act │          │                            │
│    按 act tile 字节公式 │          │                            │
├───────────────────────┤          │                            │
│ 2. pack W[k][j] -> wt  │          │                            │
│    按 wt  tile 字节公式 │          │                            │
├───────────────────────┤          │                            │
│ 3. fill bias[j]        │          │                            │
│    128 个 u16 f16     │          │                            │
└───────────┬───────────┘          └────────────▲───────────────┘
            │                                   │
            ▼                                   │
        mxclracc            ─── 清零 int acc ───┤
        bias = mxmem(bias)  ─── 把 bias 送入 HMX ┤
        { activation.ub = mxmem(act);            │
          weight.b      = mxmem(wt)  } ─── MAC  ─┤ acc += A·W (1024 个 cell 并行)
        mxmem(out):after.uh = acc:2x1  ─── convert + write ── out[i][j] in VTCM
```

"**5 条 HMX 指令搞定一次 MAC**"——这就是 v1 的 demo01 的骨架。

## 为什么需要 pack？

`matmul` 的数学是对称的，但硬件的物理线路不是。HMX 的乘法阵列希望每 cycle
读一条**128 B HVX 向量线**就能并行喂满 32×32 cell——这就要求 tile 的字节
顺序和 cell 的空间几何对齐。pack 函数就是把逻辑 `(i, k)` / `(k, j)` **映射
到物理字节偏移**。不按公式摆 → HMX 读到的"A[i][k]"根本不是你以为的元素，
结果完全乱。

> **pack 公式是硬件写死的不是软件约定**。v1 里用既有经验公式
> （`Agent/hmx_u8xi8_matmul_layers.md` §L3 已 silicon-verified）直接搬，
> 不重复推导。

## 为什么需要 bias？

因为 convert 写到 VTCM 时必须经过 **乘 bias 再 / 2** 的硬件路径——HMX 不能
直接 dump `acc` 的原始字节。bias 决定 "`acc` 到 `out` 的缩放比"：

- `bias[j] = f16(2.0) = 0x4000` → 有效 scale = 1.0（identity，拿 acc mod 2^16）
- `bias[j] = f16(2^-7) = 0x2000` → 有效 scale = 2⁻⁸（高位 16 bit，用于 dual-scale）
- `bias[j] = 0` → 全零输出

**首次使用 HMX 的常见踩坑**：忘了 bias，convert 读回来全 0，以为是 MAC 没执行——其实是 scale 被 bias=0 吃掉了。

## 为什么 `activation.ub` 和 `weight.b` 必须**同一 VLIW packet**

- HMX 的 MAC 阵列**同时需要两个 operand**才能发射。
- 两条 `mxmem` load 都是 slot-0 指令，但 HMX 特批它们作为"伴生对"共占一个 packet。
- 分开两个 packet 发 → HMX 前一拍的 act 缓存被覆盖，后一拍只看到 wt 没有 act，结果非预期。

> demo01 用花括号 `{ ... ; ... }` 正是把它们绑一起的 asm 写法。

## 两套不能并存的状态机

HMX 有两个"**模式**"：

| 模式 | 清零指令 | MAC 伴生 act | MAC 伴生 wt | readback |
|------|---------|------------|-------------|----------|
| 定点 | `mxclracc` | `activation.ub/.f8` | `.b/.n/.c/.ubit/.sbit/.sc/.sm/.f8` | `.uh(:sat):2x1/2x2` 或 `.ub` |
| 浮点 | `mxclracc.hf` | `activation.hf` | `.hf` | `.hf` |

切换模式 = 重新清零。一次 kernel 内通常固定一种模式。

## 本章要记住的 5 件事

1. **acc 是 HMX 内部的 32×32 寄存器阵列**，不是 VTCM 也不是普通 reg。
2. **tile 的字节摆法是硬件写死的公式**，pack 写错 = 全错。
3. **bias 必须 load**，否则 convert 输出全 0。
4. **activation.* 和 weight.* 必须同 VLIW packet**。
5. **int 和 fp 是两套 state**，用对应的 `mxclracc(.hf)` 清零。

下一章看 demo01 的完整代码——这 5 条的实际汇编写法。

## 参考

- HMX 三件套更底层的细节：`Agent/hmx_u8xi8_matmul_layers.md`
- 字节布局实测：`example/hexagon_hmx_matmul_native_int.md`
- 所有 HMX intrinsic 清单：`tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hmx_hexagon_protos.h`
