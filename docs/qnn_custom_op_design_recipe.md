# QNN Custom Op 设计 Recipe —— 怎么写才能吃到 QNN 的调度红利

> 基于 V8 matmul + shape sweep (32³→4096³) + `libQnnHtpV75Skel.so` 反汇编
> 合成的可执行设计规则。2026-04-25。
>
> 配套文档：
> - `docs/qnn_custom_op_sop.md` —— 构建 + 部署流程（怎么做）
> - 本文 —— op 设计约束（怎么写才对）

## TL;DR

QNN 会**免费**帮你做的事（前提：你的 op 签名写对了）：
- 按 `multithreaded=true` 把 op 自动 fan-out 到 4 个 HVX 线程
- 在 graph 边界自动插 `InputSlice` / `OutputSlice` / `weights_to_vtcm` staging
- STATIC tensor 自动 DMA 预载到 VTCM

QNN **不会**帮你做的事（需要你在 ONNX 里手工处理）：
- 把一个 op 的大 tensor 切成多个小 instance
- 在 custom op 之间插 `@Spill` / `@Fill` VTCM 溢出管理
- 理解你 op 的 M/N/K 维语义（所以它不知道怎么拆）

所以规则一句话：**op kernel 小而专；ONNX 图 shape-adaptive；签名把能声明的全声明。**

---

## 1. kernel 层规则（每个 op 的 C 代码）

### 1.1 **能标 `multithreaded=true` 就必标**

唯一例外：HMX 资源 op（HMX 只有一个 unit，多线程无意义）。

```c
static QHPI_Kernel_v1 sg_kernels[] = {{
    THIS_PKG_NAME_STR "::MyPackOp",
    my_pack_kernel,
    QHPI_RESOURCE_HVX,
    /* source_destructive */ false,
    /* multithreaded       */ true,   // ← 免费 4× HVX 并行
    /* variable_inputs     */ false,
    /* variable_outputs    */ false,
    ...
}};
```

kernel 必须用 `qhpi_num_slices(handle)` + `qhpi_slice_number(handle)` 自切：

```c
static uint32_t my_pack_kernel(QHPI_RuntimeHandle *handle, ...) {
    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
    // 按 M 维切
    const uint32_t mt_start = (M_tiles * slice_idx)     / num_slices;
    const uint32_t mt_end   = (M_tiles * (slice_idx+1)) / num_slices;
    // 只处理自己的 [mt_start, mt_end)
    for (uint32_t mt = mt_start; mt < mt_end; mt++) { ... }
    return QHPI_Success;
}
```

**实测效果**：V8 @ 512³，一个 `PackActivationU8RowMajor` 节点被 QNN 自动
切成 **6 次** 调用发到 **4 个 HVX 线程**。我们没在 ONNX 写多实例，全自动。

### 1.2 kernel 资源标 **精确**

```c
enum {
    QHPI_RESOURCE_MAIN = 1,   // 标量线程（普通 CPU 代码）
    QHPI_RESOURCE_HVX  = 2,   // HVX 向量线程（4 个）
    QHPI_RESOURCE_HMX  = 4,   // HMX 矩阵单元（1 个）
};
```

一个 kernel 只能标 **一种主要资源**：
- Pack / unpack / format conversion → `HVX`
- MAC / 点乘 → `HMX`
- scalar loops over metadata → `MAIN`

标错了 scheduler 会发错线程（例：HMX op 标 HVX 会被发到 HVX 线程，那里
根本没有 HMX，直接 fail）。

### 1.3 tensor signature 把 MemLoc 写对

```c
static QHPI_Tensor_Signature_v1 sig_inputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct,
     QHPI_MemLoc_DDR_OR_TCM},                 // 从 DDR 拿可以
};
static QHPI_Tensor_Signature_v1 sig_outputs[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct,
     QHPI_MemLoc_TCM_Only},                    // 必须 VTCM 驻留
};
```

| 标志                     | 含义                                     | 用途                           |
|--------------------------|------------------------------------------|--------------------------------|
| `QHPI_MemLoc_DDR_Only`   | 必须 DDR                                 | 超大 tensor（>几 MB）          |
| `QHPI_MemLoc_DDR_OR_TCM` | 哪个方便都行                             | graph 边界（INPUT/OUTPUT）     |
| `QHPI_MemLoc_TCM_Only`   | 必须 VTCM                                | kernel 要 HMX/HVX 直接消费     |

Compiler 看到 signature **不一致的相邻 op** 会自动插 staging 原语
（`*InputSlice` / `weights_to_vtcm` 等）。写对了，staging 免费。

### 1.4 数据类型声明对齐 kernel 实际期望

一个坑：`QHPI_QUInt8` ≠ `QNN_DATATYPE_UINT_8`，前者 = UFIXED_POINT_8
（带 quant encoding）。如果 kernel 签名要 `QHPI_QUInt8`，ONNX 侧必须
是 `QNN_DATATYPE_UFIXED_POINT_8` 或有 quant encoding 的 UINT_8。
不一致 → graph finalize 报 `node validating failed`.

### 1.5 **不要** 把多个 stage 塞进一个 kernel

反例（V8 早期的 monolithic mmv8，后来拆了）：
```c
static uint32_t bad_all_in_one_kernel(...) {
    pack_activation();   // HVX 工作
    pack_weight();       // HVX 工作
    hmx_matmul();        // HMX 工作
    unpack_to_ddr();     // HVX 工作
}
```

问题：kernel 标一种 resource，这个函数里混了 HVX + HMX 工作，
**其中一种必然单线程**。分成 4 个 op：
- `PackActivation` (HVX, MT=true)
- `PackWeight` (HVX, MT=true)
- `MatMul` (HMX, MT=false)
- `Unpack` (HVX, MT=true)

这样 3 个 HVX op 各自自动并行到 4 线程，HMX op 单独跑。而且相邻 op 如果
都是 HVX+VTCM 驻留，compiler 看到会尝试 pipeline 它们。

## 2. ONNX / 图层规则

### 2.1 按 shape 动态生成图（shape-adaptive）

不要写死"4 节点固定图"。写 `gen_graph(M, K, N, vtcm_mb)` 按 shape 展开：

```python
def gen_graph(M, K, N, vtcm_mb=8):
    # 抄 QNN 的 tile planner
    M_TILE = 256                                       # QNN 固定
    workspace = int(0.6 * vtcm_mb * 1024 * 1024)       # 留 40% 给 compiler staging
    N_TILE = max(32, min(256,
        ((workspace - M_TILE * K) // (K + M_TILE) // 32) * 32))

    M_ROUNDS = (M + M_TILE - 1) // M_TILE
    N_ROUNDS = (N + N_TILE - 1) // N_TILE

    for mr in range(M_ROUNDS):
        for nr in range(N_ROUNDS):
            emit_node("MatMulV8",
                inputs=[
                    f"packed_act_M{mr}",      # 每个 M_ROUND 一个 pack_act 节点
                    f"packed_wt_N{nr}",       # 每个 N_ROUND 一个 pack_wt 节点
                    f"bias_N{nr}", "scratch",
                ],
                outputs=[f"out_M{mr}_N{nr}"])

    # 按 N 拼 row stripe，再按 M 拼全图
    for mr in range(M_ROUNDS):
        emit_node("Concat", axis=N_AXIS,
            inputs=[f"out_M{mr}_N{nr}" for nr in range(N_ROUNDS)],
            outputs=[f"out_M{mr}"])
    emit_node("Concat", axis=M_AXIS,
        inputs=[f"out_M{mr}" for mr in range(M_ROUNDS)],
        outputs=["out"])
```

### 2.2 每个 instance 的工作集 ≤ **~40% VTCM**

QNN 在编译时会在 VTCM 里同时放：
- 我们 op 声明的 tensor
- `weights_to_vtcm` 的 DMA staging buffer
- `InputSlice` 的 staging buffer
- Scheduler 的元数据

留 40% 空间是经验值（QNN @ 4096³ 用 N_TILE=64 正好给 spill 留约 40%）。

### 2.3 STATIC 用在**真的不变**的数据上

- 权重（inference 之间不变）→ STATIC initializer
- bias 如果是 learned 的 → STATIC；如果是每 inference 计算的 → APP_WRITE
- scratch / workspace → APP_WRITE（让 compiler 每次 allocate）

STATIC 的好处：compiler 自动插 `weights_to_vtcm` DMA 预载，发到 HMX
thread 跑，完美 overlap HVX pack。写成 APP_WRITE 就得每次 inference
重新 DDR→VTCM，浪费。

### 2.4 layout 标 `NONTRIVIAL`

qairt-converter 默认对 rank-4 tensor 做 NHWC 强制转换，对 custom domain
op 是灾难。XML 里每个 Shape 加 `<Layout>NONTRIVIAL</Layout>` + CLI 四连：

```bash
qairt-converter \
    --source_model_input_layout <name> NONTRIVIAL \
    --desired_input_layout       <name> NONTRIVIAL \
    --source_model_output_layout <name> NONTRIVIAL \
    --desired_output_layout      <name> NONTRIVIAL \
    ...
```

(详见 `docs/qnn_custom_op_sop.md` §11 FAQ)

### 2.5 用 QNN 内置 `Concat` 不用自己写

合并子 tile 用 `qti.aisw::Concat`。QNN 有高度优化的 N-to-1 Concat
（支持 VTCM 驻留输入 + 多线程），比我们自己写一个 `MergeTiles` custom op
快得多。

## 3. 决策矩阵：何时切多实例

| Shape 场景                                | 切吗  | 为什么                                          |
|-------------------------------------------|-------|-------------------------------------------------|
| 整体 working set < 4 MB (≤ 1024³ u8)      | 否    | 一个 instance 够，不切省节点调度开销            |
| working set ∈ [4, 8) MB (≈ 2048³ u8)      | 勉强不切 | VTCM 刚够，但无安全 margin                    |
| working set ≥ 8 MB (≥ 4096³ u8)           | **必须切** | compiler 不会帮拆，不切直接 fail             |
| 单 inference HMX 工作 > 100K cyc          | 考虑切 | 切开后 HMX 和 HVX 有 overlap 机会              |
| 批量小 batch/ LLM 的 decoder（M=1）       | **不切** | 本来就小，切反而是 overhead                    |

## 4. Anti-patterns（别踩坑）

### ❌ 一个超大 custom op 塞所有工作
```
Bad:   MegaMatMul([M,K]×[K,N]→[M,N])  一个 op 搞定一切
```
后果：compiler 无法切；≥ 某 shape 一定 fail；HMX/HVX 资源互相等待。

### ❌ `multithreaded=false` on HVX op
```c
QHPI_RESOURCE_HVX, /* multithreaded */ false,  // ← 白白浪费 4× 并行
```

### ❌ MemLoc 乱标
```c
{QHPI_QUInt8, ..., QHPI_MemLoc_DDR_Only},  // 但 kernel 里用 HMX mxmem() 读
```
HMX mxmem 必须从 VTCM 读，DDR tensor 会导致 runtime fault。

### ❌ 手写 tile reassembly
```
Bad:   custom op MergeMTiles + MergeNTiles
Good:  qti.aisw::Concat axis=-1 / axis=-2
```

### ❌ 一次生成固定 shape 的 ONNX
```python
# Bad
M = K = N = 512  # hardcoded
```
shape 一变就要改代码。参数化 + planner。

### ❌ 漏标 Layout=NONTRIVIAL
不标，rank-4 tensor dims 被 converter 自动 permute，kernel 读错 shape，
HTP fault 或 ssh 掉线。

## 5. 检查清单

写完一个 custom op 前自问：

- [ ] kernel 标了对的 `QHPI_RESOURCE_*` 吗？
- [ ] HVX kernel 标了 `multithreaded=true` 吗？`qhpi_num_slices`/`qhpi_slice_number`
      实现了自切吗？
- [ ] tensor signature 的 `MemLoc` 和 kernel 实际访问模式对吗？
- [ ] XML OpDef 每个 Shape 加了 `<Layout>NONTRIVIAL</Layout>` 吗？
- [ ] 接受的 dataType 和 kernel QHPI 类型对齐了吗（UFIXED vs UINT 易错）？
- [ ] 每个 op 只做一件事（不混 HVX+HMX 工作）？
- [ ] STATIC / APP_WRITE / NATIVE 用对地方了吗？
- [ ] ONNX 生成脚本按 shape 参数化了吗？
- [ ] 大 shape 下 tile 工作集 ≤ VTCM × 40% 吗？
- [ ] 用的是 `qti.aisw::Concat` 而不是自己的 merge op 吗？

## 6. 最小正确样例 skeleton

```python
# gen_<op>_onnx.py
def gen(M, K, N, vtcm_mb=8):
    plan = plan_matmul_graph(M, K, N, vtcm_mb)   # §4 of qnn_matmul_design_principles doc

    act = Input("act", [1,1,M,K], UINT8)
    wt  = Static("wt", [1,1,K,N], UFIXED_POINT_8)   # let compiler DMA-preload

    outs = []
    for mr in range(plan.M_ROUNDS):
        pa = Node("PackAct", act[mr*256:(mr+1)*256], layout=NONTRIVIAL, MT=true)
        for nr in range(plan.N_ROUNDS):
            pw = Node("PackWt", wt[:, nr*N_TILE:(nr+1)*N_TILE], layout=NONTRIVIAL, MT=false)
            y  = Node("MatMul", [pa, pw, bias_N[nr], scratch], layout=NONTRIVIAL, MT=false)
            outs.append(y)

    out_stripes = [Concat(axis=-1, [outs[mr*N_ROUNDS:(mr+1)*N_ROUNDS]])
                   for mr in range(M_ROUNDS)]
    out = Concat(axis=-2, out_stripes)
```

```c
// pack_<op>_hvx.c
static uint32_t pack_kernel(QHPI_RuntimeHandle *handle, ...) {
    const uint32_t num_slices = qhpi_num_slices(handle);
    const uint32_t slice_idx  = qhpi_slice_number(handle);
    const uint32_t mt_start = (M_tiles * slice_idx)     / num_slices;
    const uint32_t mt_end   = (M_tiles * (slice_idx+1)) / num_slices;
    hvx_pack_body(/*slice=*/mt_start, mt_end, ...);
    return QHPI_Success;
}

static QHPI_Tensor_Signature_v1 in[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_DDR_OR_TCM}};
static QHPI_Tensor_Signature_v1 out[] = {
    {QHPI_QUInt8, QHPI_Layout_Flat4, QHPI_Storage_Direct, QHPI_MemLoc_TCM_Only}};

static QHPI_Kernel_v1 k[] = {{
    THIS_PKG_NAME_STR "::PackOp",
    pack_kernel,
    QHPI_RESOURCE_HVX,
    /*source_destructive*/ false,
    /*multithreaded     */ true,       // ← 关键
    /*variable_inputs   */ false,
    /*variable_outputs  */ false,
    1, in, 1, out,
    nullptr, 0, 0, nullptr, nullptr, nullptr,
}};
```

## 7. 文档交叉参考

- `Agent/qnn_matmul_design_principles_2026-04-25.md` — QNN 自身的设计原则（抄对象）
- `Agent/qnn_custom_op_scheduling_2026-04-25.md` — custom op 能拿到/拿不到哪些自动调度（证据）
- `Agent/qnn_matmul_as_composition_2026-04-25.md` — HTP 原语组合的分层视角
- `Agent/matmul_blueprint_2026-04-25.md` — V8 具体行动项（shape-adaptive 图生成）
- `docs/qnn_custom_op_sop.md` — 构建 + 部署流程 SOP
