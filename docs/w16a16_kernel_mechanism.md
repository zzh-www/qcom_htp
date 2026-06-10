# w16a16 HMX kernel 机理（QNN `hmx_v73_convhhh1x1_stride1` 反编译）

QNN 原生 w16a16 matmul（uint16 激活 × int16 权重 → uint16）在 v75 HMX 上的内核，
逆向自 `tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so` 的
`hmx_v73_convhhh1x1_stride1`（VMA `0x2fa740`，0x708 字节）。仓库内的可读副本 =
`example/handwritten_hmx_matmul/kernels/w16a16/v73deep_conv1x1_kernel.inc`
（== `example/gdn_native/baremetal/inc/v73deep_conv1x1_kernel_i16.inc`，逐字节相同）。
完整反汇编落盘在 `Agent/qnn_re/hmx_v73_convhhh1x1_w16a16.disasm`。

> 命名：`convhhh` = **h**alf-act × **h**alf-weight → **h**alf-out（half = int16）。
> 对照：`convhbh`(w8a16)= int16-act × **int8**-weight，`convbbb`(u8i8)= 全 int8。

---

## 1. 它算什么 + ABI

一次 `conv1x1 stride1` = 一个 M×K×N 矩阵乘（GDN solve 里按 64³ tile 调用）。
寄存器契约（6 个指针参数，与 u8i8/w8a16 内核同一 `hmx_conv_*_desc_t` ABI）：

| reg | 参数 | 关键字段（偏移） |
|---|---|---|
| r0 | `out_desc` | tile_ptr_table(+0)、table_stride(+4)、y_stride(+8)、**n_tiles_pow2(+0xc)**、**m_total_minus_step(+0x10)**、**k_total_bytes(+0x14)** |
| r1 | `act_desc` | act_ptr_pairs(+0)、**n_act_pairs(+4)**、act_y_stride(+8) |
| r2 | `packed_weight` | k-major、**dilated** int16 权重流 |
| r3 | `folded_bias` | bias + **drain 控制字**（增益指数 + baseline，`mxmem2` 载入） |
| r4 | `mask_desc` | out_check(+0)、out_rt_mask(+4)、act_check(+8)、act_rt(+0xc)、…、alt_rt(+0x18) |
| r5 | `extra_param` | 控制元组 |

---

## 2. 核心机理：int16×int16 怎么落到 byte-MAC

HMX 阵列只做 **8bit×8bit→int32 累加**。int16×int16 必须按字节展开成 **4 个 byte-product**：

```
act = act_hi·256 + act_lo        wt = wt_hi·256 + wt_lo
act·wt = act_hi·wt_hi·65536      ┐
       + act_hi·wt_lo·256        │ 4 个 byte-product，各自带 256 的幂位权
       + act_lo·wt_hi·256        │
       + act_lo·wt_lo·1          ┘
```

这颗 HMX **不是**跑 4 次独立的 u8i8 matmul。w16a16 = 在 w8a16（int16-激活）的基础上**再加 int16-权重**，
两件正交的硬件原语叠起来，在**一个累加器里就地**做完 4 个 byte-product。三档内核对照：

| | byte-product | 激活喂数 | 权重喂数 | 排空 |
|---|---|---|---|---|
| u8i8 `convbbb` | 1 (a·w) | `:deep:cm` | `:deep` | `cvt.ub = acc`（无 2x2） |
| w8a16 `convhbh` | 2 (a_hi·w, a_lo·w) | **pairs → `2x2`** | `:deep` | `cvt.uh = acc:2x2` |
| **w16a16 `convhhh`** | **4** | **pairs → `2x2`** | **`:dilate`** | `cvt.uh = acc:2x2` |

### (a) `2x2` 累加 —— int16 **激活**（w8a16 已有，非 w16a16 独有）
激活以**字节平面 pairs** 喂入（`act_desc` 的 `n_act_pairs` / `act_ptr_pairs` = act_hi/act_lo 两平面）。
累加器按 **`2x2`** 瓦片组织：把 2 个 act 字节平面 × 权重的 MAC 结果连同 256 幂位权累进同一瓦片。
**这是 int16-激活的机制，w8a16 和 w16a16 都有**（所以两者都 `cvt.uh = acc:2x2`）。

### (b) `weight.b = mxmem(r8,r9):dilate` —— int16 **权重**（w16a16 独有）
w16a16 在上面基础上，权重再以 **`:dilate`** 模式载入：把 int16 权重的 hi/lo 两个字节平面"膨胀"展开，
使 MAC 同时覆盖 wt_hi、wt_lo 两个通道（w8a16 权重是 int8 走 `:deep`、单平面、不膨胀）。
**`:dilate` 正是 w16a16 比 w8a16 多出来的那一份**（2 act 平面 × 2 wt 平面 = 4 byte-product），
即"w16a16 ≈ 2× w8a16"在指令层的体现。未对齐路径退化为 `:single`（一次一平面，更慢兜底）。

### (c) `cvt.uh = acc(rXX):2x2` —— 唯一一次排空
K 维由 `loop0`/`loop1` 硬件循环遍历、跨 K 累进同一累加器。K 累完后载入 `bias = mxmem2(r3)`，然后：
```asm
cvt.uh = acc(r31):2x2          ; 累加器 → unsigned halfword(uint16)，按 2x2 折叠
mxmem(r10,r11):2x2 = cvt       ; 写出到 crouton16 输出面
```
- `cvt.uh` = 转 **uint16**（对照 u8i8 `cvt.ub` 转 uint8 —— 排空宽度 2× 字节）。
- `:2x2` 把累加器的 2×2 结构连同幂位权**折叠成最终 uint16**，并施加 drain 增益
  （**2 的幂移位**，指数取自 r3 控制字 —— 见 `docs/w16a16_is_two_w8a16.md`）。
- **每个输出 tile 只此一次 drain**（4 个 byte-product 已在累加器合并，无需多次排空再相加）。

---

## 3. 控制流：3 个函数 / 4 条路径，drain 模式完全一致

prologue（`0x2fa740`–`0x2fa800`）读全部 descriptor、算循环次数
（`r28 = n_act_pairs>>1` 驱动 loop0，`r20 = (n_tiles_pow2+3)>>2` 驱动 loop1），
再按对齐位（`bitsclr(.,0x783)`）和 `n_act_pairs` 分派：

| 路径 | 入口 | MAC 模式 | drain |
|---|---|---|---|
| 对齐 · n_act_pairs>1（主路） | `0x2fa800` | `:dilate` | `cvt.uh=acc:2x2` ×1/tile |
| 对齐 · n_act_pairs≤1 | `0x2fa900` | `:dilate` | 同上 |
| 未对齐 | `0x2fac40` `_unaligned` | `:single` | 同上（p2/p3 门控） |
| 未对齐 · sparsity | `0x2fa9c0` `_unaligned_sparsity` | `:single` | 同上 |

**四条路径无一例外都是"一串 MAC 累进单累加器 → 单次 `cvt.uh=acc:2x2` 排空"。**
没有"每个 byte-product 各排空一次再合并"的多 drain 结构。

---

## 4. 成本机理：为什么 ≈ 6× u8i8，且 777 是伪地板

原生实测（`example/qnn_matmul_profile/output_*_*_256/optrace`，256³ HMX-compute）：

| kernel | byte-prod | drain（含 2x2=int16 act） | 权重喂数 | 256³ cyc | per-64³ | vs u8i8 |
|---|---|---|---|---|---|---|
| u8i8 (`convbbb`) | 1 | `cvt.ub`(1B, 无 2x2) | `:deep` | 12,435 | 194 | 1× |
| w8a16 (`convhbh`) | 2 | `cvt.uh`(2B):2x2 | `:deep` | 30,182 | 472 | 2.4× |
| **w16a16 (`convhhh`)** | **4** | `cvt.uh`(2B):2x2 | **`:dilate`** | **74,670** | **1167** | **6.0×** |

把 w16a16 拆解：
- "4 个裸 byte-MAC" 的朴素地板 = 4 × 194 = **777/64³**。
- 实测 = **1167/64³**，多出的 **390（33%）** = `:dilate` 喂数 + `2x2` 累加排布 + `cvt.uh` 2 字节排空的**固有机器成本**。

**关键结论（反编译证实）**：这 390 **不是**可消的"多 drain 浪费"——
QNN 内核**已经是单累加器 + 单 drain 最优解**（§2–3）。所谓 "把 1167 压到 777" 的前提
（消掉多余 drain）**在硅片上不存在**：HMX 做 int16×int16 只有 `dilate`+`2x2`+`cvt.uh` 这一条通路，
1167/64³ 就是它的真实地板。**777 是个朴素 byte-count 伪地板，硬件做不到。**

> 推论（GDN 纯-HMX 求逆裁定）：纯 HMX w16a16 对角 matmul 的 per-64³ 卡死在 1167，
> 即便 merge 走 u8i8，32-head 总量 ≈ 1.74M ≈ **打平** HVXMixHMX(1.78M)，无净胜空间。
> 详见 `Agent/current/gdn_solve.md` §4。

---

## 5. 复现

```bash
OBJD=tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump
SO=tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
# 全函数反汇编（含 HMX 包，--show-packet-decode-fail 解出普通槽）
"$OBJD" -d --mattr=+hvxv75,+hmx,+hvx-length128b \
  --start-address=0x2fa740 --stop-address=0x2fae48 --show-packet-decode-fail "$SO"
# 原生 256³ HMX-compute cycle（by_htp_type_cycles 里的 HmxU16I16ToU16MatMul / ConvLayer_s1.opt）
python3 -c "import json;print(json.load(open('example/qnn_matmul_profile/output_w16a16_accepted_256/optrace/summary.json'))['by_htp_type_cycles'])"
```

相关：`docs/w16a16_is_two_w8a16.md`（w16a16=2×w8a16 字节分解 + drain 是 2 的幂移位）、
`Agent/current/gdn_solve.md` §4（纯 HMX 裁定）、`Agent/qnn_re/hmx_v73_convhhh1x1_w16a16.disasm`（全反汇编）。
