# W16A16 脱离-QNN 手写 kernel：使用说明 + 原理 + 对齐方法论

w16a16 = uint16 激活 × int16 权重 → uint16 的 HMX matmul。本文档讲的是它的
**“脱离 QNN 裸调用”手写实现**：直接驱动 byte-verified 的 kernel body（`hm_w16a16_v73_kernel`
= native `hmx_v73_convhhh1x1_stride1`），全程**零 QNN runtime**，输出**逐字节对齐 QNN native**。

- **状态**：byte-exact，**hexagon-sim 与真设备 CDSP 双向验证通过**（`diff=0`）。
- **shape 覆盖**：**M=256 × 任意 K,N（均 32 倍数）** 全部 byte-exact（split 路 N%128==0、
  single-call 路 N%128≠0、K∈{32,64,128,256} 都验过）。~~M<256 暂未覆盖~~；productized QNN op 则
  M 一直下探到 64（`scripts/w16a16_shape_sweep.sh`）。
  - **更新（cron#66–67, 2026-06-14）**：**M=64 单块 64³ 已 byte-exact 到 native `ConvLayer_s1`（max|d|=0,
    4 case）**，用 native 的 M=64 描述符（out_y=4/m_total=8/n_tiles=8/act_y=4, 4 tile @mod4）+ **闭式
    `crouton_pos(r,c)` bit-置换布局**（act/out 共用）+ native control `0x804035F3/0x4000023E`。"M<256 缺
    per-shape Crouton padding" 的旧限制 = 之前没逆出 M=64 的 crouton 布局公式而已,现已解。详见
    `Agent/current/pure_hmx_solve_build.md` 末节「对齐 native 64³ ✅ SOLVED」。
- QNN custom-op 出货路线（任意 32-multiple shape）见 `Agent/current/w16a16_production_kernel_plan.md`；
  kernel 本身机理见 `docs/w16a16_kernel_mechanism.md`、`docs/w16a16_is_two_w8a16.md`。

---

## 1. 使用说明

### 1.1 准备 artifact（一次）

```bash
# 生成 owned prepared-state（activation crouton 面 / weight·bias sidecar / 描述符 / mask）
uv run python example/handwritten_hmx_matmul/run_owned_smoke.py \
    --family w16a16 --out-dir /path/to/artifact
```

### 1.2 在 hexagon-sim 跑（零 QNN，最快）

```bash
# 专用 op-faithful driver：单输出 crouton 面 + per-split stride-4 sub-table
uv run python scripts/run_w16a16_standalone_kernel.py \
    --artifact /path/to/artifact --json-out /tmp/w16.json
#   -> "w16a16 standalone (QNN-free): BYTE-EXACT  diff_bytes=0"

# 也可走通用 body-sim runner（同样 byte-exact）
uv run python scripts/run_handwritten_artifact_body_sim.py \
    --family w16a16 --artifact /path/to/artifact --json-out /tmp/g.json
```

### 1.3 在真设备 CDSP 跑（确认非 sim artifact）

```bash
# 经 run_main_on_hexagon 把同一配置跑在真 HMX 上，FARF 回报多种 deblock 的 diff
uv run python scripts/run_w16a16_standalone_device.py --artifact /path/to/artifact
#   -> "[HM_DEVICE] w16dev ... row4=0 ..."   (row4 deblock diff=0 = byte-exact)
```
设备 = `ssh oneplus`（perf-burst 偶发假失败，重跑即可）。

### 1.4 多 shape 扫（M=256 × 任意 K,N）

```bash
# 每个 shape:profile_all 生成 native ref → build_w16a16_standalone_prepared.py 造 prepared
# → standalone byte-exact 断言。需设备(出 native ref)。
bash scripts/w16a16_standalone_shape_sweep.sh
#   或子集:SHAPES="256,256,256 256,256,96" bash scripts/w16a16_standalone_shape_sweep.sh
```

### 1.5 CI 门（已接入 `run_all.sh`）

| 门 | 脚本 | 需设备 |
|---|---|---|
| sim byte-exact（256³，通用 body-sim + 专用 driver） | `test_w16a16_standalone.sh` | 否 |
| **设备** byte-exact（真 CDSP） | `test_w16a16_standalone_device.sh` | 是 |
| **多 shape** byte-exact（split + single-call + K-var） | `test_w16a16_standalone_sweep.sh` | 是 |
| **值分布 + seed** byte-exact（uniform×N / signs / zeros） | `test_w16a16_standalone_dist.sh` | 是 |

与其他 active family 对等:sim + 设备 + 多 shape + 值分布 四类门齐全。

### 1.6 scale / 值分布覆盖说明（重要）

- **scale 不是自由维度**:native-a16 量化契约把 act/weight/output 的 scale 全部钉死在
  `1/32767`（16-bit 对称），drain 指数（`extra={1,1536}` + bias 控制字 `0x00404420`，exp=17→×2）
  是**常量,不随数据/scale 变**。其他 kernel（u8i8/w8a16/w4a16）**也不 sweep scale**。
- **真正的正确性维度 = 值分布**。`gen_onnx.py` 现支持 `--dist {uniform,signs,sparse,zeros,
  extreme,impulse}` + `--seed` + 独立 `--weight-dist/--act-dist`（也可 `GEN_DIST/GEN_SEED/
  GEN_WDIST/GEN_ADIST` env）。`scripts/w16a16_standalone_dist_sweep.sh` 扫多分布/seed,断言 byte-exact。
- **全部 7 种分布 byte-exact**:uniform（多 seed）、signs、sparse、zeros、**extreme(±max)**、**impulse**。
- **值分布覆盖抓到并修了一个真 bug**(详见 [[project_w16a16_effective_bias_gap_2026-06-10]] memory):
  `generate_w16a16_weight_sidecar.py` 把 int16 权重 high 字节存成 `(q16+128)>>8` 的 **signed int8**;
  `q16 ∈ [32640,32767]`（W≳0.996）时 high=128=`0x80`=−128 → 硬件重建出大负权重 → 整列算错。
  **256³ uniform oracle 的 `|q16|≤16384` 从没碰到 → 隐藏很久**。修复 = q16 clip 到 **32639**（可表示上界
  `127*256+127`,native 也 saturate 到这）。**op 和 standalone 共用该 sidecar,一并修复**。
  （int32 累加器溢出是红鲱鱼:byte-identical kernel wrap 一致,extreme 修完权重 clip 就 byte-exact。）

---

## 2. 原理：QNN-free 路怎么对齐 native

native QNN 把一次 w16a16 matmul 拆成两个节点：**custom HMX op**（写 crouton 输出）+ 一个
**ForceFormat**（crouton→linear）。脱离 QNN 就是要自己复刻这两步：

1. **kernel body** = `kernels/w16a16/v73deep_conv1x1_kernel.inc`，与 native convhhh @0x2fa740
   **逐字节一致**（`.codex/skills/hmx-inline-asm/scripts/verify_hexagon_inline_asm.py` 验过）。
2. **描述符**（与 custom-op 的 FORMULA_DESC 完全一致）：`out_y_stride = n_tiles_pow2 = M`、
   `m_total_minus_step = 1`、per-split `k_total_bytes = 128`、`act_table_y_stride = K_t·64`、
   `out_table_stride = 4`、`n_act_pairs = K_t`。
3. **N128 split**：op 默认 `#define HMX_W16A16_INTERNAL_SPLIT_N128`，所以是**一个**完整
   crouton 输出面 + 每个 128-列 split 用 stride-4 sub-table 写自己的 4 个 N-tile；
   weight/bias 每 split 各前进一段（`split_weight_bytes = K_t·4·2048`、`split_bias_bytes = 4·512`）。
4. **mask**：`set_hmx_params_conv1x1(buf, 0x70b, 0, 0, 0, 0x80)`，`mask[14]=&extra`，
   `extra={1, 1536}`（1536 = drain 增益指数）。
5. **crouton16-row4 → linear deblock**（= w8a16 同一个 proven 逆，marker run 实证）：
   输出块按 `(rg&7)·N_t + nt` **连续**排布；32×32 tile 内 `row0 col c → pos 2c`、
   `row1 → 2c+1`；col-half1（col 16–31）在同 block 的 `pos 32+`（即 kernel 的 `r10+0x40`）。

> 自洽性：standalone 自己分配输出块（连续 `B·2048`，与 QNN 实测一致）并自己 deblock；
> 只要 kernel/描述符/weight/bias/**mask** 与 op 一致，结果必然对齐 native。

---

## 3. 对齐方法论 / 经验沉淀（可复用）

这次对齐踩了很久，结论性教训如下：

### 3.1 不能信的是“复刻件”，不是 sim

- **sim 是可信的**：坏 mask 时 sim 与真设备**同样 corrupt**；好 mask 时**同样 byte-exact**。
  两个方向都对得上 → hexagon-sim 的 HMX `:2x2` store 建模准确。中途“sim bug”假说上设备**排除了**。
- **真正不可信 = 对闭源 helper / 硬件布局的手写复刻**：本例根因是
  `scripts/emulate_hmx_conv1x1_params.py`（`set_hmx_params_conv1x1` 的 Python 复刻）
  **只在 w8a16(arg5=0x20) 验过**，被当通用函数复用到 w16a16(arg5=0x80, dilate) 时
  **word6 算错**（仿真 `0x7ff` vs 真值 `0x3ff`）。
  → **原则：任何复刻/仿真，只要没在“目标参数点”对过 ground truth，就不能默认它对，哪怕它在别的参数点 byte-exact。**

### 3.2 根因机理

word6 = kernel 里 `r9 = memw(r4+#0x18)`，直接做 `weight.b = mxmem(r8,r9):dilate` 的 rt span。
dilate（int16 权重 = 2 字节平面）时激活成对读，rt span 减半 → `r9` 不做倍增（`0x400→0x3ff`）；
`:deep`（int8，arg5=0x20）才倍增（`0x800→0x7ff`）。**修复**：`conv1x1_words` 在
`arg5 & 0x80` 时跳过 `r9` 的 span-doubling。w8a16/u8i8 不受影响。

误导点：**col-half0（前 16 列）恰好不依赖这个 bit，一直 byte-exact**，让人误判“输入都对、是输出布局问题”，
绕了很多弯（activation / layout / sim 都怀疑过）。

### 3.3 拿 ground truth 的关键招：**marker run**

QNN 不肯吐 custom-op 的物理 crouton 输出（输出永远被 ForceFormat 转 linear；
`--debug`/`--set_output_tensors` 不能配 `--retrieve_context`；DLC 路 QNN validator SIGSEGV；
op 内 FARF 也不透出 logcat）。突破口是**借 QNN 自己的 ForceFormat 把布局“拍”出来**：

1. 给 op 加 gated flag（`HMX_W16A16_FILL_BLOCK_INDEX` 等），**跳过 matmul**、把每个输出
   crouton block 填 `(logical_block<<10) | pos_in_block`；
2. 让 graph 的 ForceFormat 正常跑；
3. linear 输出直接反推 `(row,col) → (block, pos)` = **精确 deblock 映射**。

同法填 `block0 = 真 skel mask words`，对比仿真 → 一眼抓到 word6 差异。
（这些 flag 是诊断 scaffolding，已从生产 op 移除；要复用时按上面思路重加即可。）

**一句话**：拿不到 ground truth 时别硬推；想办法让被测系统自己把真值“打印”出来，再逐字对比。
