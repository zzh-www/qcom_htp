# 12 · 踩坑 + Debug 技巧

**TL;DR**：HMX kernel 第一次写不对大概率是 pack / bias / packet / acc 清零
四类坑之一。本章给**症状 → 原因 → 诊断法**的对照。

## 常见症状速查

### 症状 1：`out[0] == 0`（整块全 0）

**最常见原因**：忘了 `bias = mxmem`。convert 时 `bias_f16[j]` 如果是未初始化
或全 0，`out = acc · 0 = 0`。

**诊断**：加 `printf("%04x\n", bias[0])` 在 `mxmem` 之前看 bias 是不是 0x4000。

### 症状 2：`out[i][j]` 看起来接近预期但值偏大

**原因**：忘了 `mxclracc`，acc 里残留上次 kernel 的累加值。

**诊断**：在 kernel 开始跑一个 "A=0, W=0 应得全 0" 的测试。如果不是 0 → 没清 acc。

### 症状 3：`out[0]` 对但 `out[i≠0][j≠0]` 乱

**原因**：pack 写错（只填了第一个 cell）。

**诊断**：跑 demo04 的 single-hot-byte probe，看 pack 公式是不是你写的那样。

### 症状 4：一半 cells 对，一半全 0

**可能原因**：activation 的 `stream` 搞反——`phys_row = ir & 15; stream = ir >> 4`。
如果把 `stream` 条件写成 `(stream ? 1 : 3)` 而不是 `(stream ? 3 : 1)`，`ir < 16` 的那一
半填对了，`ir ≥ 16` 的那一半全 0。

**诊断**：检查 `out[0][0]` 和 `out[16][0]` 都对吗？如果只有一个对就是 stream 错。

### 症状 5：`out[phys_row][jc]` 全对但 `C[ir=phys_row + 16][jc]` 位置错乱

**原因**：unpack 忘了 stream 维度。正确公式：
```c
out_u16[phys_row * 64 + 2 * col + stream]
```
错写成 `out_u16[phys_row * 32 + col]` 会让 stream 数据永远读不到。

### 症状 6：值是"正确但 wrap 过 2^16"

典型值：`got = 45536, expect = 1036320`（45536 = 1036320 mod 65536）。

**原因**：acc 超 16 bit，单次 convert 又没 `:sat`。

**修法**：
- 降 K 或输入范围（教学场景）
- 加 `:sat.uh` 让溢出饱和（量化后输出）
- 换 dual-scale readback（要 bit-exact 读 int32）

### 症状 7：int8 signed activation 的结果偏一常数

**原因**：忘了减 `128 · ColSumW[j]` 修正。

**诊断**：`got - ref` 对所有 (i, j) 来说等于 `128 · ColSumW[j]`（和 i 无关）→ 就是这问题。

### 症状 8：sim 直接 segfault / hang

**可能原因**：
1. `Rs` 不是 VTCM 地址（普通 DDR/stack 指针 → HMX 拒读）
2. `Rs` 未 128 B 对齐
3. h2 hypervisor 没 init（忘了 `h2_mxaccess_acquire`）

**诊断**：在每条 mxmem 前加 `printf("mxmem Rs=%p\n", ptr)`，确认 `ptr` 在 VTCM 范围内
（通常 `0xFC000000+` 或 sim 报的 base）且 `((uintptr_t)ptr & 127) == 0`。

### 症状 9：fp16 kernel 全 nan

**原因**：`mxclracc` 错用在 fp 路径（应 `mxclracc.hf`），acc 里是 int32 的垃圾当 xfp 读。

**修法**：fp 路径一致用 `.hf` 系列 + `mxclracc.hf`。

## Debug 工具箱

### 1. `hexagon-llvm-objdump` 反汇编 ELF

看你的 HMX packet 是否真的是一个 VLIW packet：

```sh
source scripts/env.sh
hexagon-llvm-objdump -d --mcpu=hexagonv75 --mattr=+hmxv75,+hvxv75 \
    example/hmx_programming_guide/demo08_i4xi8_tile \
    | grep -B 1 -A 3 "activation.ub\|weight.n"
```

期望看到的 packet 形式（最后一行带 `}`）：
```
1234:  xx xx xx xx   activation.ub = mxmem(r17, r4):above
1238:  yy yy yy yy   weight.n      = mxmem(r19, r4) } 
```

如果 `activation.ub = ...` 那行就单独有 `}`，说明编译器把 packet 拆了。

### 2. hexagon-sim 的 stats 输出

```sh
hexagon-sim --mv75 --mhmx 1 --statsfile /tmp/sim.stats \
    -- booter --ext_power 1 --use_ext 1 demo_xxx
grep -E ":[0-9]+$" /tmp/sim.stats | awk -F: '$3+0 > 0'
```

看 non-zero counter，能看到 HMX packet 数、cycle 数等。

### 3. printf 打印中间值

HMX kernel 里不能在 MAC 和 convert 之间 printf（会打断 pipeline），但可以：
- kernel 开始前 print 输入
- kernel 结束后 print out[0..4]
- CPU 修正前 print `Chmx[0][0]`
- 修正后 print `Cfinal[0][0]`

### 4. Single-hot probe（参见 demo04）

定位 pack 问题的万能招式：只在一个字节写 1，其余全 0，看 out 的哪个 cell 亮。
反推你的 pack 公式对不对。

```c
memset(act, 0, 2048);
memset(wt, 0, 1024);
act[ACT_OFFSET] = 1;
wt[WT_OFFSET] = 1;
/* ... MAC + convert ... */
for (int i = 0; i < 1024; i++) if (out[i]) printf("out[%d] = %u\n", i, out[i]);
```

### 5. 逐步 PASS 阶梯

如果 demo08 失败：先确认 demo01 → demo04 → demo06 → demo07 都 PASS。找到**第一个**挂
的 demo，从那里 debug。大多数情况是 demo 之间某步细微改动引入的 bug，找到 delta
点极大缩小搜索空间。

## 高频 coding 错误对照

```c
/* ❌ 把 packet 拆了 */
asm volatile("activation.ub = mxmem(%0, %1)" ...);
asm volatile("weight.b      = mxmem(%2, %3)" ...);

/* ✅ 同一 packet */
asm volatile("{ activation.ub = mxmem(%0, %1)\n"
             "  weight.b      = mxmem(%2, %3) }"
             ...);
```

```c
/* ❌ clobber 忘写 "memory" */
asm volatile("mxclracc" :::);

/* ✅ 必须有 memory clobber，否则编译器重排 VTCM 写 */
asm volatile("mxclracc" ::: "memory");
```

```c
/* ❌ out tile 当 row-major 读 */
int16_t C[32][32];
for (int i = 0; i < 32; i++)
    for (int j = 0; j < 32; j++)
        C[i][j] = ((int16_t *)out)[i * 32 + j];   /* 错! */

/* ✅ 按硬件公式 */
for (int i = 0; i < 32; i++) {
    int pr = i & 15, st = i >> 4;
    for (int j = 0; j < 32; j++)
        C[i][j] = ((int16_t *)out)[pr * 64 + 2 * j + st];
}
```

```c
/* ❌ signed int4 填 weight.n 忘保留符号 bit */
for (int k = 0; k < K; k++)
    for (int j = 0; j < N; j++)
        wt[byte_off(k, j)] |= (W[k][j] << (hi ? 4 : 0));   /* 负数高位污染 */

/* ✅ 先 & 0x0F 把值限制在 4 bit */
uint8_t nib = (uint8_t)(W[k][j] & 0x0F);
if (hi) wt[off] = (wt[off] & 0x0F) | (nib << 4);
else    wt[off] = (wt[off] & 0xF0) | nib;
```

## 参考

- hexagon-sim 使用：`docs/hexagon_sim_handbook.md`
- 本指南每个 demo 都带 oracle，照抄进你的 debug
- `tests/test_hmx_programming_guide.sh` 是 sanity runner

## 到这里你应该能

- 读懂 sim 报错信号
- 用 single-hot probe 定位 pack bug
- 用反汇编确认 packet 真的同发
- 在自己的 kernel 里快速二分到第一个挂的 demo
