# 02 · First Demo：5 条指令跑通一次 MAC

**目标**：让你**从零** → **看到 `[PASS]`**。

完整 demo 源码：`example/hmx_programming_guide/demo01_hello_hmx.c`（150 行左右，
已在本仓库）。本章把这 150 行拆成 7 段讲。

## 数学输入

```
A ∈ u8^{32×32},  每个元素 = 1
W ∈ i8^{32×32},  每个元素 = 1
C ∈ i32^{32×32}, C[i][j] = Σ_{k=0..31} A[i][k] · W[k][j]
```

**每个 cell 都是 1×1 × 32 = 32**。我们期望 `out[i][j] == 32` for all (i,j)。

我们把 acc 用 `bias = f16(2.0)`（硬件再 /2 → 有效 scale = 1.0）读回
`.uh:2x1`，K=32 下 acc = 32，单次 convert 就能原值读出。

## 跑一下

```sh
source scripts/env.sh
bash tests/test_hmx_programming_guide.sh
```

期望最后看到：

```
--- demo01_hello_hmx ---
      out[0,0] = 32  (expect 32)
      out[0,1] = 32   out[16,0] = 32   out[31,31] = 32   (all expect 32)
      [PASS] demo01
    PASS

=== Summary ===
  1 / 1 PASS
```

下面拆解。

## 段 1：环境 bootstrap（h2 hypervisor）

```c
unsigned int vtcm_base = h2_info(INFO_VTCM_BASE);

h2_vecaccess_state_t vacc;
h2_vecaccess_unit_init(&vacc, H2_VECACCESS_HVX_128, CFG_TYPE_VXU0,
                       CFG_SUBTYPE_VXU0, CFG_HVX_CONTEXTS, 0x1);
h2_vecaccess_acquire(&vacc);
h2_mxaccess_state_t mxacc;
h2_mxaccess_unit_init(&mxacc, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                      CFG_HMX_CONTEXTS, 0x1);
h2_mxaccess_acquire(&mxacc);
```

这段**只在 hexagon-sim 下用**——H2 是 Qualcomm 为 sim 提供的轻量 hypervisor，
负责 VTCM 映射 + HVX/HMX 上下文的申请。真机上要用 `HAP_power_set` + 
`HAP_compute_res_acquire`（参考 `example/hmx_matmul_device/`），与本指南 v1 范围无关。

记忆点：**五个函数调用要按这个顺序**，少一步 HMX 就发不了指令。

## 段 2：VTCM 分区

```c
uint8_t  *vt   = (uint8_t *)(unsigned long)vtcm_base;
uint8_t  *act  = vt + 0 * 2048;  /* 2 KiB */
int8_t   *wt   = (int8_t *)(vt + 2 * 2048);
uint16_t *bias = (uint16_t *)(vt + 4 * 2048);
uint16_t *out  = (uint16_t *)(vt + 6 * 2048);
```

VTCM 是一整块连续的 SRAM，你自己切分。每 2 KiB 分一段便于对齐——HMX 的
tile load 硬性要求 128 B 对齐，2 KiB 冗余对齐让你不用再精算。

记忆点：**act 2 KiB、wt 1 KiB（实际使用，这里分 2 KiB 以备别的 wt 类型）、
bias 256 B、output 2 KiB**——都住在 VTCM，不是 DDR。

## 段 3：填 activation tile

```c
memset(act, 0, 2048);
for (int phys_row = 0; phys_row < 16; phys_row++) {
    for (int K = 0; K < 32; K++) {
        act[128 * phys_row + 4 * K + 1] = 1;   /* stream 0 */
        act[128 * phys_row + 4 * K + 3] = 1;   /* stream 1 */
    }
}
```

### 布局公式（硬件写死）

```
A_byte(phys_row, K, stream) = 128·phys_row + 4·K + (stream ? 3 : 1)
    phys_row ∈ 0..15, K ∈ 0..31, stream ∈ {0, 1}
    logical row ir → (phys_row = ir & 15, stream = ir >> 4)
```

- 每 128 B 一行 = 32 K × 4 B = 1 条 HVX 线
- 每 K 槽 4 B 里只用 byte 1 和 byte 3，byte 0/2 必须为 0（留给 fp16 activation）
- **1 个 byte 存 1 个 logical 元素**：`A_byte(phys_row, K, stream=0)` = logical `A[phys_row][K]`；
  `A_byte(phys_row, K, stream=1)` = logical `A[phys_row + 16][K]`

所以 `for phys_row in 0..15, for K in 0..31, 写两个 stream` 正好填满 32 行 × 32 K 的逻辑矩阵。

> **为什么 2 KiB 要用 1024 B 去装 32×32 u8 的数据** —— 因为硬件的 act FIFO
> 路径给 fp16 也用同一个，fp16 每元素占 2 B，32×32 fp16 就是 2 KiB。int8 版本
> 被迫沿用这个几何，一半字节空着。详见 `Agent/hmx_u8xi8_matmul_layers.md` 附录 B。

## 段 4：填 weight tile

```c
memset(wt, 0, 1024);
for (int K = 0; K < 32; K++) {
    for (int col = 0; col < 32; col++) {
        wt[128 * (K >> 2) + 4 * col + (K & 3)] = 1;
    }
}
```

### 布局公式（硬件写死）

```
W_byte(K, col) = 128·(K >> 2) + 4·col + (K & 3)
    K ∈ 0..31, col ∈ 0..31
```

每 128 B 一条 HVX 线 = 4 K × 32 col × int8，8 条线覆盖完 K=0..31。

这里数据密度 100%（不像 act 浪费一半），因为 weight 侧的 tile 格式**不与
fp16 共享**——int8 就用 int8 的紧凑布局。

> 换 weight 类型时这个布局会变：int4 tile 512 B，int2 tile 256 B。
> 见 `03-tile-layout-cheatsheet.md`。

## 段 5：填 bias

```c
for (int i = 0; i < 128; i++) bias[i] = 0x4000;  /* f16(2.0) */
```

128 个 u16，每个是 f16 值 2.0。这个选择让 convert 的有效 scale = 1.0
（HMX 内部硬性 / 2）→ `out = acc mod 2^16`。

`mxmem(Rs)` load bias 时会从 `Rs` 开始读 **256 字节**（128 × u16）。
填 128 个是刚好；多填的字节会被忽略。

## 段 6：5 条 HMX 指令

```c
asm volatile("mxclracc" ::: "memory");                      /* ① */
asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");   /* ② */
asm volatile("{ activation.ub = mxmem(%0, %1)\n"            /* ③ VLIW packet */
             "  weight.b      = mxmem(%2, %3) }"
             :: "r"(act), "r"(2047),
                "r"(wt),  "r"(2047)
             : "memory");
asm volatile("mxmem(%0, %1):after.uh = acc:2x1"             /* ④ */
             :: "r"(out), "r"(0)
             : "memory");
```

**① `mxclracc`** —— 清零 32×32 int32 acc。**必须**在 MAC 之前。

**② `bias = mxmem(Rs)`** —— 从 `Rs` 指针读 256 B 放进 HMX 的 bias slot。
这条不是 MAC，是给 convert 用的配置。

**③ MAC packet**：花括号里的两条指令**同 VLIW**。这个 packet 会触发 HMX 
启动一个"消费完整 tile 的 multi-cycle MAC 任务"——它不会阻塞 Hexagon 核，核
可以立刻继续发后面的指令；HMX 自己在后台吃完 tile。

- `activation.ub = mxmem(Rs, Rt)` —— `Rs` 指 act tile，`Rt=2047` 是控制字
  （编码"读完整个 2 KiB"）。其它 Rt 值可以读部分 tile 或加 stride，见 v2 的
  06 章。
- `weight.b = mxmem(Rs, Rt)` —— 同上但读 wt tile。`.b` = 纯 int8（不偏移不修饰）。

**④ convert**：`mxmem(Rs, 0):after.uh = acc:2x1`
- `after` 表示 MAC 完成后才读
- `.uh` = uint16 输出（不 saturate，wrap mod 2^16）
- `:2x1` = 输出布局是"每 phys_row 装 2×32 u16"（64 u16/行 × 16 phys_row = 1024 u16 = 2 KiB）
- 第二个参数 `0` 是 offset，读整个 output tile

## 段 7：验证

```c
uint16_t got = out[0];
if (got != 32) { printf("  [FAIL] ...\n"); fail++; }
```

### 输出布局公式

```
out_u16[phys_row * 64 + 2 * col + stream]
   logical (ir, jc) → (phys_row = ir & 15, col = jc, stream = ir >> 4)
```

所以 `out[0]` = logical `C[0][0]`；`out[0*64 + 2*1 + 0]` = `C[0][1]`；
`out[0*64 + 2*0 + 1]` = `C[16][0]`。

demo01 检查了 4 个 cell（0,0 / 0,1 / 16,0 / 31,31）都等于 32，证明不是某
一个 cell 偶然对了。如果布局公式错了（比如 stream 搞反），这 4 个里至少
有一个会 != 32。

## 对应 C reference（为什么预期是 32）

```c
for (int i = 0; i < 32; i++)
  for (int j = 0; j < 32; j++)
    for (int k = 0; k < 32; k++)
      C[i][j] += A[i][k] * W[k][j];   // 1 * 1 accumulated 32 times = 32
```

## 常见报错/排查

| 症状 | 原因 | 修复 |
|------|------|------|
| `hexagon-sim` 直接 Segfault | 没 source `scripts/env.sh` 或 H2 没编 | `source scripts/env.sh` + 参考仓库 README 编 hypervisor |
| `out[0] == 0` | 忘了 `bias = mxmem` | 加 bias load |
| `out[0]` 奇怪的大数 (如 33 或 65505) | act/wt 布局公式写错 | 用 demo04（v2）的 single-hot-byte probe 验证 |
| `out[0] == 32` 但 `out[0,1] == 0` | 只填了第一个 cell 的 act/wt | 检查 pack 循环覆盖完 32 K × 32 col |
| sim 报 `FAIL` / non-zero exit | kernel 逻辑错 | 看 sim log 里 `[FAIL]` 打印的具体值 |

## 本章你学到了

1. HMX kernel 的最小骨架：h2 bootstrap → pack → clracc → bias → MAC packet → convert → 读 out。
2. act / wt / out 的字节布局公式（但不用背——03 章是速查表）。
3. 5 条 HMX 指令的基本语义。
4. 验证 HMX 结果对不对的"多 cell 抽查"做法。

## 下一章

[03 tile-layout-cheatsheet](03-tile-layout-cheatsheet.md) —— 把本章出现的 3 条
布局公式（act / wt / out）集中成一张速查表，加上 pack 常见错法的对照示意图。
