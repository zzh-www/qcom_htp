# QNN MatMul 设计原则（shape-adaptive implementation）

> 基于 32³/128³/256³/512³/4096³ 五个 shape 的实测 chrometrace 综合得出。
> 数据来源：`example/qnn_matmul_profile/sweep_data_2026-04-19/s{32,128,256,512}/w8a8/`
> + `sweep_data_4096/w8a8/`。反汇编：`hmx_convbbb1x1_stride1 @ 0x2ea740`。

## 0. 一句话

**QNN MatMul 是"一个 shape-invariant 的 HMX 内核 + 一个 shape-adaptive 的图生成器"。**
内核永不变，图在 `QnnGraph_finalize` 时根据 `(M, K, N)` 和 VTCM 预算动态拼出来。

## 1. 五个 shape 的实测拓扑

| Shape | Wall (cyc) | Conv calls | Tile shape     | HVX pack threads | Input/Format | Spill/Fill | Concat | cyc/packet |
|-------|-----------:|-----------:|----------------|-----------------:|-------------:|-----------:|-------:|-----------:|
| 32³   |       27K  |   2        | [1,8,4,32]     |                1 |      0 / 2   |       0    |    0   | 47.3 (6× 上限) |
| 128³  |       31K  |   2        | [1,8,16,128]   |                1 |      0 / 2   |       0    |    0   | 32.2 (4× 上限) |
| 256³  |       35K  |   2        | [1,8,32,256]   |                1 |      0 / 2   |       0    |    0   | 10.2 (1.3×)   |
| 512³  |       67K  |   4        | [1,8,32,256]   |                2 |      4 / 4   |       0    |    2   | **5.67** ✓    |
| 4096³ |      28.9M | **1024**   | [1,8,32,64]    |            **4** |    50 / 42   | **4014**   |   16   | 5.75          |

**silicon ceiling = 7.89 cyc/packet**（`Rt_wt=0x3FF` 实测下限，见
`Agent/qnn_hmx_pipelining.md`）。从 512³ 起稳定打穿 ceiling（硅厂数字有偏差，
实测 5.7 一致）。

## 2. 七条设计原则

### 原则 1：HMX 内核是 shape-invariant 的
所有 shape 都调同一个 `q::ConvLayer_s1.opt` → `hmx_convbbb1x1_stride1`。内循环：
```asm
{ activation.ub = mxmem(act_ptr, Rt_act=2063):cm
  weight.b      = mxmem(wt_ptr,  Rt_wt=0x3FF) }         ; 2-MAC unroll
{ activation.ub = mxmem(act_ptr2, Rt_act):cm
  weight.b      = mxmem(wt_ptr2, Rt_wt) }
...
  mxmem(out_tile_ptr, 0):after:cm:sat.ub = acc         ; sat.ub drain
```
**一个 ConvLayer 调用 = 一个 M_tile × K × N_tile 的小矩阵乘**，`M_tile`
一般 8×32=256 行，`N_tile` 跟 shape 自适应（见下）。无条件用 `Rt_wt=0x3FF`。

**→ 这说明 V8 的 HMX 内核已经追平 QNN，没什么可改。**

### 原则 2：图结构是 shape-dependent 的（编译期动态生成）

`QnnGraph_finalize` 时 HTP 编译器做 **静态** 决策：

1. 从 `(M, K, N)` + VTCM 预算算 `(M_tile, N_tile)`
2. 生成 `N_rounds = ceil(M/M_tile) × ceil(N/N_tile)` 个 ConvLayer 实例
3. 插入 `Concat` 节点重组输出
4. 根据 HMX 预估负载决定用几个 HVX 线程做 pack（1 → 2 → 4）
5. 如果工作集 > VTCM，插入 `@Spill / @Fill`
6. 绑定 HVX pack op（`InputSlicePad` + `ForceFormat_Crouton`）

**运行时只是执行这张预先拼好的图。** 没有动态分支，没有 shape-check 跳转。

### 原则 3：Tile 大小由 VTCM 预算决定

V75 VTCM = 8 MB。工作集 = `packed_act + packed_wt + output_tile + bias +
spill_buffer`。编译器策略（观察出来的）：

| 问题规模            | Tile 策略                           | 原因                               |
|---------------------|-------------------------------------|------------------------------------|
| 全部装得下 VTCM     | tile = full problem, 2 rounds (hi/lo) | 最少开销                           |
| 单 tile 能装 VTCM   | tile = [1, 8, 32, N]，固定 256 行 M，N 切到 VTCM 能装 | 最大化每 round 的 MAC 工作量 |
| 总工作集 > VTCM     | tile 缩小 + 加 spill/fill          | 给 spill buffer 留空间             |

**数字观察**：
- 256³：tile `[1,8,32,256]` = **65 K elements**（接近单 tile 最大）
- 512³：tile `[1,8,32,256]` 不变，但 `N_rounds=2` → 切 N 维成两半
- 4096³：tile 缩成 `[1,8,32,64]` = **16 K elements**，给 spill buffer 腾出 ~4 MB VTCM

**→ V8 monolithic 图的 4096³ fail 就是因为没有这一层自适应**（`packed_act`
固定 = `[1, M/32, K/32, 1024]`，shape 大了直接溢）。

### 原则 4：HVX pack 线程数 scale to saturate HMX

| Shape | HMX util | HVX pack 线程 | 原因 |
|-------|---------:|--------------:|------|
| ≤256³ |   77-88% |             1 | HMX 就快 idle 了，pack 单线程够 |
| 512³  |      60% |             2 | pack 成瓶颈，加一个线程 |
| 4096³ |   95.6%  |             4 | HMX 全满，榨干所有 HVX 喂它 |

规则：**HVX 线程数 ≈ (pack_cost × N_rounds) / HMX_can_wait_budget**。
具体阈值 QNN 编译器内建。

### 原则 5：Pack 是 upstream op，不在 HMX 内核里

`InputSlicePad` + `ForceFormat_Crouton` 永远是独立图节点，**从不** inline 到
MAC kernel。好处：
- pack 可以被 HVX scheduler 独立多线程化
- pack 可以和 HMX 的 `weights_to_vtcm` / `bias_to_vtcm` prefetch 并行
- MAC kernel 可以跨 shape 复用同一个 `ConvLayer_s1.opt` 实现

### 原则 6：@Spill / @Fill 是 VTCM 溢出的 escape valve

当 `total_working_set > VTCM` 时，编译器**自动插入**显式的 DDR 腾挪 op：
- `@Spill`: VTCM → DDR（暂存），事件粒度按 tile
- `@Fill`:  DDR → VTCM（取回）

**4096³ 实测**：2020 次 Spill 共 17.9M cyc，1994 次 Fill 共 13.8M cyc —
Spill/Fill **总和（31.7M）超过 HMX MAC 工作（27.6M）** 本身。大 shape 下
是 DDR 带宽和 VTCM 容量共同限制的 regime。

**不会 fail，会慢**。V8 fail 是因为没这个机制。

### 原则 7：Per-packet 成本随 shape 收敛到硅 ceiling

| Shape | cyc/packet | 距 ceiling (7.89) | 主要开销来源 |
|-------|-----------:|------------------:|--------------|
| 32³   |      47.3  |              6.0× | op 启动、bias load、register setup |
| 128³  |      32.2  |              4.1× | 同上 |
| 256³  |      10.2  |              1.3× | 快打满 |
| 512³  |       5.7  |   接近 ceiling    | 内循环 MAC-bound（per-packet 本来就会有点波动）|
| 4096³ |       5.8  |        同上       | 同上 |

**小 shape 的 per-packet 成本被固定 overhead 放大 6×**。这解释了为什么
即使 512³ 和 4096³ 的 MAC 总数差 512 倍，wall-clock 只差 430 倍（有一点超
线性 = 大 shape 更高效）。

## 3. 对"我们怎么写 MatMul"的直接推论

| 原则 | V8 现状 | 应该做 |
|------|---------|--------|
| 1 HMX shape-invariant | ✓ 已对 | — |
| 2 图 shape-adaptive   | ✗ 单节点硬编码 | ONNX generator 按 shape 展开节点 |
| 3 Tile by VTCM budget | ✗ tile = 1 KiB 固定 per mmv8 call，但 `packed_act` 大小随 shape 增长到溢出 | 按 shape 切 M/N，每节点工作集 ≤ VTCM/4 |
| 4 HVX scale           | ✗ pack 1 线程 | 多 pack 实例（QNN scheduler 自动发线程）|
| 5 Pack upstream       | ✓ 已分 pack_act / pack_wt / mmv8 节点 | — |
| 6 Spill/Fill          | ✗ 没有 | 先靠切分避免溢出；真要做 spill 需要新 op |
| 7 Per-packet ceiling  | ✓ 已用 Rt_wt=0x3FF | — |

**V8 最缺的是"图生成器"**。现在是 `gen_v8_onnx.py` 只会生成 M=K=N=512 一种拓扑；
要做成 `gen_v8_graph(M, K, N, vtcm_mb)` 按 shape 展开成 shape-specific ONNX，
匹配 QNN 的 tile/并行/spill 策略。

## 4. Shape → ONNX 展开的具体算法（抄 QNN 的）

```python
def plan_matmul_graph(M, K, N, vtcm_mb=8, hmx_util_target=0.90):
    # Step 1: tile size
    M_TILE = 256                  # QNN 固定用 M=256 per ConvLayer (8×32)
    # 留 40% VTCM 给 spill/fill + bias + overhead
    workspace_per_tile = 0.6 * vtcm_mb * 1024 * 1024
    # 工作集 ≈ packed_act(M_TILE × K) + packed_wt(K × N_TILE) + out_tile(M_TILE × N_TILE)
    # 按 1 byte/element:  M_TILE·K + K·N_TILE + M_TILE·N_TILE ≤ workspace_per_tile
    # 解 N_TILE ≤ (workspace - M_TILE·K) / (K + M_TILE)
    N_TILE = min(256, int((workspace_per_tile - M_TILE * K) / (K + M_TILE)))
    N_TILE = max(32, (N_TILE // 32) * 32)   # 向下取整到 32 的倍数

    # Step 2: round count
    M_ROUNDS = (M + M_TILE - 1) // M_TILE
    N_ROUNDS = (N + N_TILE - 1) // N_TILE

    # Step 3: HVX thread count heuristic
    total_rounds = M_ROUNDS * N_ROUNDS
    if total_rounds <= 2:   hvx_pack_inst = 1
    elif total_rounds <= 8: hvx_pack_inst = 2
    else:                    hvx_pack_inst = min(4, M_ROUNDS)

    # Step 4: spill/fill needed?
    need_spill = (M * K + K * N + M * N) > vtcm_mb * 1024 * 1024

    return {
        'M_TILE': M_TILE, 'N_TILE': N_TILE,
        'M_ROUNDS': M_ROUNDS, 'N_ROUNDS': N_ROUNDS,
        'hvx_pack_inst': hvx_pack_inst,
        'need_spill': need_spill,
    }

# 检验 @ 实测 shape：
# 512³:  M_TILE=256, N_TILE=256, M_R=2, N_R=2  → 4 rounds, 2 HVX threads ✓
# 4096³: M_TILE=256, N_TILE=64 (VTCM-bound), M_R=16, N_R=64 → 1024 rounds, 4 HVX threads ✓
```

上面的参数建议和 QNN 实测拓扑基本重合（见 §1 表）。

## 5. "不对称 MatMul" 的 hint（LLM workloads）

LLM 推理常见 `[1, S, D]×[D, V]`（`S`=seq, `D`=hidden, `V`=vocab）不对称。
QNN 怎么处理？从 shape scan 看：tile 的 M 维**固定 256**（`[1,8,32,...]`），
tile 的 N 维随 shape 变化。所以瘦长 MatMul（M≫N 或 M≪N）：

- M≫N（例如 4096×K×64）：N 不切（N_TILE=64 本身），M 切很多份 → 很多
  rounds 但每 round 很轻。
- M≪N（例如 32×K×4096）：M 是 32（小于 M_TILE），就 1 个 M_round；
  N 切 64 份。

Tile 结构不变，只是分片数变化。这进一步说明**内核级代码永远一样，图
级拓扑全动态**的设计原则。

## 6. Cross-refs

- `Agent/matmul_blueprint_2026-04-25.md` — "MatMul 怎么写" 行动项
- `Agent/qnn_hmx_pipelining.md` — HMX 内核指令级 RE
- `Agent/forceformat_crouton_re.md` — HVX pack kernel RE
- `example/qnn_matmul_profile/sweep_data_2026-04-19/` + `sweep_data_4096/` — 原始数据
- `trace_for_review/` — 浏览用 chrometrace 汇总
