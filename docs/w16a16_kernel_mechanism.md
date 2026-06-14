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
（消掉多余 drain）**在硅片上不存在**：HMX 做 int16×int16 只有 `dilate`+`2x2`+`cvt.uh` 这一条通路。

> ⚠️ **口径修正（2026-06-13，权威 `Agent/current/int16_matmul_cycle_model.md`）**：这里的 **1167 是 `by_htp_type`
> 吞吐口径（HMX-busy 总周期 / 64）**，反映 4 byte-pass 的总 MAC 工作量。但 native 单 `q::ConvLayer_s1.opt`
> 的 **latency（dominant-path）只有 256/64³（u8i8 176，仅 1.45×）**——4 个 byte-pass **流水**,关键路 ≪ 吞吐。
> 两个都对、口径不同:**算力占满(throughput-bound)时用 1167;关键路/低占用(latency-bound)时用 256。**
> 所以"1167 是真实地板"只在 throughput 口径成立;**777 朴素 byte-count 仍是伪地板**,但真正的 latency 地板是 256。

> 推论修正（GDN 求逆）：旧推论"纯 HMX w16a16 对角卡在 1167 → 打平无净胜"**用了吞吐口径**。GDN 求逆
> **HMX 仅 7% 占用(producer-bound)→ merge 该按 latency 1.45× 算,不是吞吐 6×**。详见 `gdn_solve.md` §4 顶部
> 2026-06-13 修正脚注 + `int16_matmul_cycle_model.md` Decision 段:**int16 merge 求逆未否决,待 S1 测 producer pack delta。**

---

## 5. 吞吐量设计：n_tiles 卡最小 + 批摊销（per-call 成本模型）

§4 是**单 tile 的 MAC 成本**(optrace HMX-busy/latency 口径)。这一节是**整次 kernel 调用**的成本
(bare-metal `C15:14` per-call wall,口径④,VTCM-resident,back-to-back warm —— 即喂数已藏好后的纯
kernel 钱)。决定吞吐的两个旋钮 = **`n_tiles`** 和 **批的 M**。

**`n_tiles` = kernel 实际走的 tile 遍数**,必须 = 精确最小值:
```
n_tiles_min = ceil(M/32) × ceil(N/32) × byte_pass     （w16a16 的 byte_pass = 2）
```
HMX tile 是 32×32;n_tiles 设大了**线性多做冗余 MAC pass**(纯浪费),设小了漏 tile 算错。
> 教训:旧 M=256 载体曾设 `n_tiles=256`(实际只需 32)= **8× 过切**,per-call 42333 cyc;
> 砍到 32(每个 64-行块 8 tile × 4 块)后 = 5547,**4× 提升**,且已贴地板。

**实测拟合(本机 v75 TURBO,K=N=64,resident warm):**

| M | n_tiles | per-call cyc | per-64³-matmul |
|---|---|---|---|
| 64 | 8 | 1576 | 1576 |
| 256 | 32 | 5547 | 1387 |
| 256 | 256(过切) | 42333 | 10583 |

→ 线性模型 **`per-call ≈ 262 + 164 × n_tiles`**(`164×n_tiles` = 走 tile 的 MAC 钱,`262` =
每次调用固定开销:descriptor 读取 + array 配置 + drain)。换成每个 64³ matmul(M 行 = M/64 个):
**`per-matmul ≈ 16768/M + 1312`**,M→∞ 地板 **1312**。

**吞吐最优 recipe(任意 shape):**
1. **`n_tiles` 卡 `ceil(M/32)×ceil(N/32)×2`**,绝不过切。
2. **把共享同一权重的多个独立 matmul 拼成一个大-M 调用**摊销那 262 固定开销。
   `M=64` 离地板 +20%、`M=128` +10%、**`M=256` +6%(98% 地板,甜点)**,再大收益 <1%。
   (GDN solve 的 4-head fan-out 把 4 个 64³ 拼成 M=256,就是这个。)
3. **操作数必须 VTCM-resident** —— 上表都是 warm 数。**冷调用(每次从 DDR 搬+填 mxmem array)
   ≈ 10K+/call**,远超 kernel 本身 → **真瓶颈是 producer 喂数,不是 matmul kernel**(kernel 已贴地板)。
4. **K(收缩维)**:上面常数(164/tile、地板 1312)是 **K=64** 实测;K 更大 → 每 tile 累加更多 →
   per-tile 常数上升(`n_tiles` 公式不含 K,K 在 tile 内 accumulate),K=128 需重测常数。

> 注:`byte_pass=2` 是 w16a16 的**固定属性**(int16 = 2 遍 int8,`docs/w16a16_is_two_w8a16.md`),
> **不是可调旋钮**。本节的吞吐优化全部在 **w16a16 精度不变**的前提下(n_tiles + 批 + 喂数)。
> **不要把"降到 w8a16"当默认/回退手段** —— 那是改数值精度的独立决策,只在对应算子**单独**论证
> 精度余量充分后才可用,与吞吐设计分开评估,绝不默认触发。

> ⚠️ **口径**:本节 per-call wall(口径④,resident)≠ §4 的 optrace HMX-busy(1167/64³)/latency(256/64³)。
> 别拿本节的 1312-1576 和 §4 的 256/1167 直接比 —— 一个是整次调用 wall(含固定开销),一个是单 tile 的 MAC。

**复现(per-call 成本模型 + bit-exact)**:
```bash
cd example/gdn_native/baremetal
EXTRA_DEFS="-DGDNBM_GDN_PURE_SOLVE -DGP_ALIGN -DGP_ALIGN_NATCTRL=1 -DGP_ALIGN_NCASE=4" bash build.sh
# 部署跑 → raw stats[0]=M64-dense-n8 per-call, stats[1]=M256-n256, stats[2]=M256-n32 (cyc)
# 同时 4 case 输出 bit-exact 对 native ConvLayer_s1（见 Agent/current/pure_hmx_solve_build.md 末节）
```

---

## 6. 量化配置：act 必须 uint16 zp32768（对称量化无益,反而多 glue）

**HMX 这条 conv 路硬性要 unsigned uint16 激活**(mxmem 激活面是无符号)。native `ConvLayer_s1.opt` 的真实
effective 量化(lowered 图 scalar_params 实测)永远是:

| 张量 | dtype | zp | scale(mm1ex) |
|---|---|---|---|
| act | **uint16** | **32768** | sA=7.3243e-6 |
| weight | signed int16(喂进核前拆 SFIXED8 2 byte-pass) | 0 | sB=9.1556e-6 |
| out | **uint16** | **32768** | sC=2.2522e-5 |

`out_code = round(Σ_k(act_code−32768)·wt_code · sA·sB/sC) + 32768`。act 的 `−32768`(zp)修正由 bias
`eff=−colsum/2`(=`−zp·colsum` 经 2^16 drain)吸收;weight 若声明 int16,QNN 用一个 **Cast(+32768→uint16)**
转成核要的形态(我们手写路用 `pack_wt_kmajor` 的 dilated byte 拆,等价)。

**把 act/weight/out 都声明成对称 int16(zp0)= 无益且更慢**(设备实测,单 64³ 图 optrace):

| op | 非对称(act u16 / wt i16,**推荐**) | 对称(全 int16) |
|---|---|---|
| **ConvLayer_s1.opt(HMX matmul)** | 1970/inst | 2274/inst（**基本相同**,单图冷跑噪声） |
| q::Cast | 2 inst(Σ10264) | **6 inst(Σ27240)** |
| q::Reshape | **0** | **3 inst(Σ4057)** |

⇒ **HMX matmul op 与对称/非对称无关(永远 uint16-act zp32768)**;声明对称只是让 QNN 在外面套
**A→u16 / B→u16 / out→i16 三类 Cast + Reshape** 把 int16 转进/转出这个 uint16-only 的路 = 纯多 ~20K glue cyc。
**∴ 用非对称(act uint16 zp32768 + weight int16 + out uint16 zp32768)= HMX 原生形态,glue 最少。** 我们手写
solve 已是此配置(standalone 量化契约),无需改;换对称只会多一步偏移。复现:`example/gdn_native/solve_op/standalone/mm_sym/`
(对称 ovr,全 int16)vs `mm_1x1x64x64/`(非对称),`qairt-converter`+`qnn-context-binary-generator`+`qnn-net-run --profiling_option optrace`,`scripts/decode_qnn_optrace.py` 解 per-op。

---

## 7. 复现（反编译 + 256³ 算力）

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
