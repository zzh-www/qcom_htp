# V8 架构完成 + formula 部分校准 — HMX semantics 比文档更复杂 (2026-04-24)

> **架构已完成**: V8 复刻 QNN `ConvLayer_s1.opt`：3-op graph, 4-asm HMX-only kernel, 无 HVX requant.
> **性能**: 32³ V8 总 20K cyc vs QNN 17K (1.18×). HMX 内核 6162 cyc vs QNN 2926 (2.1×).
> **未达 bit-exact**: `:cm:sat.ub` + fp16 bias 语义比文档复杂; 有隐式 offset/scale 行为尚未完全解码.

## 0. 已完成

### 架构（P3 + P4）
- `HmxMatMulV8Op.cpp`: 纯 4 asm 指令（bias + clracc + K × pair + `:after:cm:sat.ub`）
- `pack_act_rm_hvx.c`: 新 HVX MT=4 op 产出 row-major 1 KB tile 供 `:cm` 消费
- `run_matmul_v8_graph.cpp`: 3-node graph (pack_act_rm + pack_wt_v3 + mmv8)
- 注册 + build + 上设备 ✓

### Silicon RE（P1）
`probe_sat_ub.c` 6 个 test 跑通：
- ✓ `:after:cm:sat.ub` 填满 **32 行**（不继承 dual-scale 的 16 行问题）
- ✓ Per-col bias 有效（T3）
- ⚠ 公式 **不是** `out = sat_u8(round(acc × bias_fp16 / 512) + 128)` — 那是 T2 表面现象，**在真实 matmul 场景不成立**

## 1. 未解码的 HMX semantics（关键 blocker）

探针 T5（act=all-1, wt[0][n]=n+1, acc=(n+1)）各种 bias 扫描发现：

```
bias_fp16   | out 观测  | 预期（per T2 公式）
0x3800 (0.5)| 112      | 128
0x3C00 (1.0)| 120      | 128
0x4000 (2.0)| 128      | 128
0x4400 (4.0)| 136      | 128
0x4800 (8.0)| 144      | 128
```

pattern: `out = 120 + 8 × log2(bias_fp16)` —— 与 acc 几乎无关（!）且按 log2 bias 线性。这不符合任何简单乘法公式。

V8 DIAG1（act=wt=all-1, acc=32, bias=2.0）得 **112 而非预期 128**。

**可能原因（全部需要进一步 probe）**：
1. `:cm` activation **可能被当作 signed (u8-128)**: 那 K=32 下 acc = -127×32 = -4064, out = round(-4064×2/512) + 128 = 112 ✓（匹配 DIAG1！）但这不能同时解释 T5 的 log2 关系.
2. **bias 某种 log-encoding**：如 HMX 把 fp16 解读为带 log scale，T5 结果可能自然浮现
3. **bias lane mapping 不是 bias[n]↔col n**：T6 显示 bias[0] 改动影响多列. 也许实际是 bias[128] 对 32 cols 的多对一映射
4. **上述几点组合**

## 2. 还需要做的 probe（~1-2 day）

- **T7**: act=k (row-ramp), wt=1, 单 col 输出 — 看 acc 是否 = Σ(k-128) 或 Σk
- **T8**: bias sweep 更 fine (0x4000 vs 0x4400 之间各 exp/mant 点) — pin down 编码
- **T9**: bias[c] 逐位反扫 — mapping bias index → col output 关系  
- **T10**: 用 probe_cm_singlecell.c 模式，单 (m,n) cell 观测 acc 是否和 scalar 预期一致

## 3. 本次实现的完整性能数据（32³, 非 bit-exact）

```
架构 / op           cycles   相对 V6    相对 QNN w8a8
---                 ------   --------   -------------
V6 monolithic       ~32K     —          1.9×
V7 split (w/req)    ~60K     1.9×       3.5× 
V8 (本轮)           ~20K     0.63×      1.18×
QNN w8a8            ~17K     0.53×      1.0×

mmv8 HMX 内核        6K cyc  -          2.1× (vs ConvLayer_s1.opt 2.9K)
```

V8 是 **迄今最接近 QNN 的总 cycle**，但 bit-exact 还没拿到.

## 4. Files 本次改动

```
新:
  example/hmx_matmul_device/probe_sat_ub.c  （T1..T6 probe + runner）
  example/hmx_matmul_device/run_sat_ub_probe.sh
  example/hmx_matmul_phase3/kernel/pack_act_rm_hvx.c
  example/hmx_matmul_phase3/src/HmxMatMulV8Op.cpp
  example/hmx_matmul_phase3/src/run_matmul_v8_graph.cpp
  example/hmx_matmul_phase3/run_v8_graph_on_device.sh

改:
  HmxMatMulPhase3Interface.cpp  （注册 V8 + pack_act_rm）
  build.sh                      （加 V8 / pack_act_rm 到 link）
```

## 5. Resume

```bash
source scripts/env.sh

# Silicon probe
cd example/hmx_matmul_device && bash run_sat_ub_probe.sh

# V8 graph
cd ../hmx_matmul_phase3 && bash build.sh
bash run_v8_graph_on_device.sh --shape 32,32,32              # current: 20K cyc, not bit-exact
# DIAG modes (4th arg = 999/998/997 for uniform/col-ramp/all-neg):
ssh oneplus "cd ~/qnn_run && ./run_matmul_v8_graph 32 32 32 999"

# 之前的 working V6 仍可用：
bash run_v6_graph_on_device.sh --shape 1024,1024,1024        # 0.0015, 1.9× QNN, bit-exact
```

## 6. 建议下轮

**走 probe 路线**: T7-T10 系列 targeted probe 直接反解 HMX `:cm:sat.ub` + bias 实际语义。**不要试图从文档/intrinsic 头文件推**——已经验证它们在这里不完整。拿到 HMX 黑盒的精确 transfer function 后才能写 bit-exact V8 reference + bias lookup。

一旦 bit-exact：V8 当前 20K cyc（32³） → 目标 ~100K cyc（512³）约 1.5× QNN。可能还能通过 tile pipeline 再降到 1.2× QNN。
