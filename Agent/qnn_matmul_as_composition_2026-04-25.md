# QNN MatMul 其实是"高层 op"，在 graph-prepare 时 **lower 成一组 HTP 原语**

> 这是对 `qnn_matmul_design_principles_2026-04-25.md` 的重要补充/修正。
> 之前说"shape-invariant 内核 + shape-adaptive 图生成器"抓到了对的方向，
> 但还少了一层 —— **QNN MatMul 本身就不是一个 HTP op，是一个会展开的 meta-op**。

## 1. 分层模型

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L3 · User / ONNX / qairt frontend                                       │
│     qti.aisw::MatMul   or  onnx::MatMul     ← 用户写的一个节点           │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ qairt-converter (不展开，只转 IR)
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L2 · QNN IR (DLC / QnnModel 里的 op 图)                                 │
│     matmul_1 : MatMul [1,M,K] × [K,N] → [1,M,N]                          │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ QnnGraph_finalize ⇒ HTP graph optimizer LOWERS
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L1 · HTP primitive op graph (chrometrace 里带 q:: 前缀的)                │
│     q::*InputSlicePad × N                                                │
│     q::ForceFormat_Crouton × N        （HVX 打包）                       │
│     q::ConvLayer.opt.weights_to_vtcm  （HMX prefetch DMA）               │
│     q::ConvLayer.opt.bias_to_vtcm     （HMX prefetch DMA）               │
│     q::ConvLayer_s1.opt × N_rounds    （HMX MAC core）                   │
│     @Spill / @Fill                    （VTCM↔DDR 溢出管理）              │
│     q::Concat × N_concat              （tile 拼接）                      │
│     q::*OutputSlice × N               （格式逆变换 + 写回 DDR）           │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ hnnx scheduler 发到硬件资源
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  L0 · Hardware execution                                                 │
│     HMX unit (tid=256)                                                   │
│     HVX units (tid=512,513,514,515)                                      │
│     DMA / VTCM                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

**关键**：L2 → L1 的 lowering 是 **QNN compiler 内部**的行为，对用户不可见。
用户的 ONNX 里只有一个 `MatMul` 节点；设备上跑的却是几百~几千个 HTP 原语。

## 2. 我们实际观察到的 HTP 原语清单（5 shape 综合）

基于 32³/128³/256³/512³/4096³ 全部 chrometrace 提取出现过的 `q::*` / `@*`
op 类型，以及出现频次随 shape 的变化：

| HTP 原语                              | 32³ | 128³| 256³| 512³|4096³| 角色 |
|---------------------------------------|----:|----:|----:|----:|----:|------|
| `q::ConvLayer_s1.opt`                 |   2 |   2 |   2 |   8 |2048 | **HMX MAC core**（真正的乘加） |
| `q::ConvLayer.opt.weights_to_vtcm`    |   2 |   4 |   4 |   8 |2174 | 权重 DMA 到 VTCM                |
| `q::ConvLayer.opt.bias_to_vtcm`       |   4 |   2 |   2 |   4 |2222 | 偏置 DMA 到 VTCM                |
| `q::ForceFormat_Crouton`              |   2 |   2 |   2 |   4 |  42 | **HVX pack**（flat→Crouton 布局）|
| `q::*InputSlicePad`                   |   0 |   0 |   0 |   4 |  50 | HVX 切 + zero-pad input         |
| `q::*InputSlice`                      |   4 |   4 |   4 |   0 |   0 | 小 shape 直切（不 pad）         |
| `q::*OutputSlice`                     |   2 |   2 |   2 |   4 |  32 | HVX 解 pack + 写回 DDR          |
| `q::Concat`                           |   0 |   0 |   0 |   2 |  16 | 子 tile 拼接                    |
| `@Spill`                              |   0 |   0 |   0 |   0 |2020 | VTCM → DDR 溢出                 |
| `@Fill`                               |   0 |   0 |   0 |   0 |1994 | DDR → VTCM 取回                 |
| `DmaCheckpointSet`                    |   4 |   4 |   4 |   4 |   0 | DMA 完成同步                    |
| `ChunkPreload`                        |   0 |   0 |   0 |   0 |  70 | chunk-level 预取                |
| `$Const` / `$Shape`                   |  ~7 |  ~7 |  ~7 | ~11 | ~149| 元数据（非执行）                 |
| `SyncOp` / `SystemService`            |   2 |   2 |   2 |   2 |  16 | 同步控制                        |

**每条原语都是 `libQnnHtpV75Skel.so` 里的一个内部函数**（在 hnnx op-def
registry 注册），不属于公开 QnnOpDef API。反汇编可见：
- `q::ForceFormat_Crouton` → `convert_to_crouton_b/h @ 0x237700/0x237c80`
- `q::ConvLayer_s1.opt`    → `hmx_convbbb1x1_stride1 @ 0x2ea740`

## 3. 为什么这个分层对"怎么写 MatMul"至关重要

### 3.1 QNN 的 MatMul 为什么快 —— 因为 lowering 免费做了几件事

`qti.aisw::MatMul` 的 compiler pass 负责：

1. **看 shape → 选 tile 尺寸**：靠 `(M, K, N)` 和 VTCM 预算跑一个 planner。
2. **展开成 N_rounds 个 `q::ConvLayer_s1.opt`**：这些全是内置原语，调用
   开销几乎为零（内部 vtable dispatch）。
3. **自动插入 pack/unpack 原语**：`InputSlicePad` → `ForceFormat_Crouton` →
   （MAC）→ `OutputSlice`。用户不用写一行 pack 代码。
4. **自动多线程调度**：hnnx scheduler 看到多个 `q::ForceFormat_Crouton`
   实例会自动发到不同 HVX 线程。
5. **VTCM 溢出时自动插 `@Spill`/`@Fill`**：工作集超 8 MB 时编译器主动加
   溢出节点，图仍能跑。

**这些全是 L2→L1 lowering 的 hard-coded 能力**。不是 runtime 调度决策，
是编译期固化的 rewrite rules。

### 3.2 我们自定义 op 拿不到这层能力

当我们写 `HmxMatMulPhase3Package::MatMulV8`：

- **我们的 op 停在 L2**。QNN compiler 看到陌生的 custom-domain op，
  **不做任何 lowering**，只是把它作为一个 opaque 节点，finalize 时
  直接调我们的 QHPI kernel 函数。
- **我们无法调用 `q::` 原语**。它们不在公开 QnnOpDef API 里，
  QnnOpPackage 也不能 `reference` 它们。
- **我们自己的"L1"只有 QHPI kernel 函数 + HVX/HMX 指令直写**。

所以 V8 当前的结构是：

```
L3 / L2：1 个 custom MatMulV8 节点（我们在 ONNX 里手写）
         ↓
L2 "lowering"（我们自己在 gen_v8_onnx.py 里手拼）：
  PackActivationU8RowMajor  →  PackWeightToHmxTileV3  →  MatMulV8  →  TcmDramCopy
         ↓
L1：4 个 QHPI kernel 函数（我们自己的 C 代码）
         ↓
L0：HMX / HVX 指令（我们自己写的 asm）
```

**所有 L2→L1 的工作都是我们 gen_v8_onnx.py 手拼的 custom-op 子图。**
QNN 只负责调度、不负责改写。

### 3.3 直接后果

**原则 A：写 MatMul = 写一个 gen_graph(shape, vtcm) 函数**

不是"写一个更好的 kernel"，而是"**写一个 shape-aware 的子图生成器**"。
compiler 该做的 lowering 现在要我们自己做。参考算法在
`qnn_matmul_design_principles_2026-04-25.md` §4。

**原则 B：每个 HTP 原语都要 custom op 对应**

想要 QNN 拓扑的每个原语都要有我们的 custom 对应：

| QNN HTP 原语                  | 我们 V8 对应               | 状态     |
|-------------------------------|----------------------------|----------|
| `q::*InputSlicePad`           | （没做）                   | 🔴 缺    |
| `q::ForceFormat_Crouton`      | `PackActivationU8RowMajor` | 🟡 布局不同 |
| `q::ConvLayer.opt.weights_to_vtcm` | `PackWeightToHmxTileV3`| 🟡 不是 HMX thread |
| `q::ConvLayer.opt.bias_to_vtcm` | bias 在 MatMulV8 里读     | 🟡 没独立节点 |
| `q::ConvLayer_s1.opt`         | `MatMulV8`                 | ✅ 近似    |
| `@Spill` / `@Fill`            | （没做）                   | 🔴 缺    |
| `q::Concat`                   | QNN 内置 Concat（可直接用）| ✅        |
| `q::*OutputSlice`             | `TcmDramCopy` / `UntileToRowMajor` | 🟡 |

🔴 缺的两个是 4096³ 跑不了的主因：没 `InputSlicePad` 就不能 pad 到 32 对齐
（当前靠 shape 必须是 32 倍数的硬约束绕过），没 `@Spill/@Fill` 就不能处理
VTCM 溢出。

**原则 C：L1 原语可以是多种实现（HMX / HVX / CPU）的分派**

注意 `q::ConvLayer.opt.weights_to_vtcm` 运行在 tid=256（HMX thread）但做的
是 DMA，不是 MAC。这说明 HTP 原语绑定的是**资源类型**（HMX / HVX / DMA），
不等于具体算子类型。compiler 的 planner 会根据调度压力选最优资源。

我们的 QHPI op 可以通过 `QHPI_RESOURCE_HMX` / `QHPI_RESOURCE_HVX` /
`QHPI_RESOURCE_DMA` 标记资源亲和性，但 scheduler 是否真的按我们想的
那样发线程需要实测（V7 split 架构就是这么试过的，部分有效）。

## 4. 实现建议（重拟 blueprint）

之前的 blueprint 主要说"切图 + HVX 改 vshuff"；加上 composition 视角后
更完整的路线：

### 阶段 A：把 gen_v8_onnx.py 升级成 gen_v8_graph(M, K, N)
- 实现 §4 的 `plan_matmul_graph(M,K,N,vtcm_mb)` 决策函数
- 输出 M_ROUNDS × N_ROUNDS 个 MatMulV8 实例 + 对应 pack + Concat
- 512³ 开始能看到多线程并行；2048³+ 仍能跑

### 阶段 B：补 custom 版的 `InputSlicePad` + `@Spill/@Fill`
- **InputSlicePad**: `PackActivationU8RowMajor` 扩展成能 `M_offset, M_len`
  两个参数 + zero-pad to 32。
- **@Spill/@Fill**: 两个新 QHPI DMA op，`VtcmSpill([NHWC] in VTCM) → DDR`
  和 `VtcmFill([NHWC] in DDR) → VTCM`。planner 检测到工作集超 VTCM 时
  在子图里显式插入。

### 阶段 C：HVX pack 改用 vshuff 拓扑
- 如 blueprint §5 #4 —— `V6_vshuffvdd(Vu,Vv,-32)` 改写 pack_act。
- 但只在阶段 A 把并行度打开之后做，否则单线程加速的 ROI 有限。

### 阶段 D：bias/weights prefetch 走 HMX resource
- 标 `QHPI_RESOURCE_DMA`，让 scheduler 在 HVX pack 窗口里并发预载。
- 验证 scheduler 确实能 overlap，再推广。

## 5. 为什么 V8 的 4096³ fail 本质上是"missing lowering"

`packed_act = [1, 128, 128, 1024]` = 16 MB 超 VTCM — 这只是表象。

**根本原因**：compiler 把 V8 当 opaque 节点，不会自动插入 `@Spill/@Fill`
也不会切子图。如果 V8 是 QNN built-in MatMul，同样 shape 下 compiler 会：

1. 看到 packed_act 超 VTCM → 选小 tile（`[1,8,32,64]` 而不是 `[1,128,128,1024]`）
2. 展开 1024 个 `ConvLayer_s1.opt` 节点
3. 在 tile 之间插 4014 个 `@Spill`/`@Fill`
4. 图能正常跑，28.9M cyc 完成

V8 把这些 lowering 步骤全**压到用户侧**（= 我们自己的 gen_v8_onnx.py 要
做）。没做，4096³ 就崩。

## 6. 和 ML 编译器类比

这个架构和其他 ML 编译器完全同构：

| ML 编译器  | L3 (frontend) | L2 (IR)       | L1 (primitive) | L0 (hw)   |
|------------|---------------|---------------|----------------|-----------|
| XLA        | TF/PyTorch    | HLO           | LLO            | GPU ptx   |
| TVM        | Relay         | TE / Relax    | TIR tensor intrinsics | GPU SASS |
| MLIR       | linalg dialect | tosa/linalg   | scf + memref + vector | LLVM IR  |
| **QNN HTP**| **ONNX MatMul** | **QNN IR MatMul** | **q::/hnnx primitives** | **HMX/HVX asm** |

QNN 的特色：L1 原语是 Qualcomm 内部闭源集合，不对外 export；custom op 
作者只能在 L2 层做事。这就是为什么自定义 MatMul 比内置 MatMul 难很多 —
**我们要亲手补上编译器原本会做的 lowering**。

## 7. Cross-refs

- `Agent/qnn_matmul_design_principles_2026-04-25.md` — shape-adaptive 原则
- `Agent/matmul_blueprint_2026-04-25.md` — 具体行动项
- `Agent/forceformat_crouton_re.md` — `ForceFormat_Crouton` 原语反汇编
- `Agent/qnn_hmx_pipelining.md` — `ConvLayer_s1.opt` 原语反汇编
- `docs/qnn_custom_op_sop.md` — 自定义 op 流程（L2 用户侧）
