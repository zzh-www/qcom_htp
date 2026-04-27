---
name: QNN primitive alignment Phase 0/1 results 2026-04-26
description: Symbol catalog for QNN HTP primitives + validation that QNN compiler auto-inserts DMA/Slice/Concat for custom op TCM_Only tensors. Greenlights the 1:1 alignment plan.
type: project
---

# Phase 0 + 1 — 1:1 QNN primitive 对齐计划验证（2026-04-26）

目标：确认能否把 V8 自定义 op 集合改造成与 QNN HTP primitive 1:1 对应，让 QNN 调度器把它们当 native ConvLayer 一样调度。

## Phase 0 — DSP-skel symbol catalog（dlsym 可调清单）

| QNN primitive | skel symbol | 地址 | 状态 |
|---------------|-------------|------|------|
| `q::ConvLayer_s1.opt` | `hmx_convbbb1x1_stride1` | (已 RE) | ✅ 单 tile 已通 |
| `q::ForceFormat_Crouton` | `convert_to_crouton_b` | 0x237700 | ✅ PackActCrouton 13-shape 已通 |
| `q::ForceFormat_Flat` | `convert_from_crouton_b` | 0x212e60 | ⏳ 待 wrapper |
| `q::*InputSlicePad` | `extract_tile_vmemu_u8` | 0x236cc0 | ⏳ 待 wrapper |
| `q::SlicePad_shape_inplace` | `scatter_tile_u8` 或 vmemcpy_2d | 0x303d40 | ⏳ |
| `q::Concat` | `concat_depth_crouton_b` 等 | 0x207f20 | （内置可用，不用动） |

**额外可用变体**：
- `convert_from_crouton_b_align` (0x2ea300), `convert_from_crouton_b_wrapper` (0x212ea0)
- `convert_from_crouton_h` (h-axis)
- `align_crouton_b/h` (alignment)
- `extract_tile_vmemu_u16/u32`, `scatter_tile_u16/u32`
- `vmemcpy_2d_gather_asm` (hnnx::vmemcpy_2d_gather_asm)

**不能 dlsym 的**：
- `q::ConvLayer.opt.{bias,weights,activations}_to_vtcm` —— host runtime DMA (`runtime_graph2::dma_memcpy` 系列在 host 端，不是 skel symbol)
- 但 Phase 1 验证表明这个**不需要我们做**，QNN 编译器会自动插入。

## Phase 1 — TCM_Only 声明 → QNN 自动 DMA 插入验证

**实验**：对当前 V8 graph 跑 `qnn-context-binary-generator --save_backend_op_mapping`，查看 `model_bottom_mapping.json` 里 QNN 编译器实际 emit 的节点。

### V8 256³ → 8 节点（与 native 一致）

```
HmxMatMulPhase3Package::PackActivationU8RowMajor   uses_hvx
HmxMatMulPhase3Package::PackWeightToHmxTileV3      uses_hvx
HmxMatMulPhase3Package::MatMulV8                   uses_hmx
HmxMatMulPhase3Package::TcmDramCopy                uses_hvx
q::*InputSlice                                     dma  ← 自动插入
q::ConvLayer.opt.weights_to_vtcm  (×3, 给 wt+bias+scratch) ← 自动插入
```

### V8 2048³ → 192 节点
```
 64 HmxMatMulPhase3Package::MatMulV8               (8 M × 8 N rounds)
 64 q::SlicePad_shape_inplace                       ← 自动插入（每个 tile 输出后）
 16 q::Concat                                       ← 自动插入
 16 q::ConvLayer.opt.weights_to_vtcm                ← 自动插入
  8 HmxMatMulPhase3Package::PackActivationU8RowMajor
  8 HmxMatMulPhase3Package::PackWeightToHmxTileV3
  8 HmxMatMulPhase3Package::TcmDramCopy
  8 q::*InputSlicePad                               ← 自动插入
```

### 对比 native 2048³（446 节点）

| op_type | native | V8 ours |
|---------|-------:|--------:|
| ConvLayer_s1.opt / MatMulV8 | 285 | 64 |
| Concat | 32 | 16 |
| ForceFormat_Flat | 30 | 0（用 TcmDramCopy 代替）|
| SlicePad_shape_inplace | 30 | 64 |
| *InputSlicePad | 19 | 8 |
| ForceFormat_Crouton | 19 | 0（用 PackActivation 代替）|
| ConvLayer.opt.bias_to_vtcm | 15 | 0（bias 走 weights_to_vtcm 一并 DMA）|
| ConvLayer.opt.weights_to_vtcm | 15 | 16 |
| ConvLayer.opt.activations_to_vtcm | 1 | 0 |

## 关键结论

1. **TCM_Only 声明工作正常** — `QHPI_MemLoc_TCM_Only` 让 QNN 编译器把对应 tensor 当 VTCM-resident，自动插入 InputSlice/InputSlicePad/SlicePad_shape_inplace/Concat/weights_to_vtcm。
2. **架构上 1:1 对齐方案完全可行** —— 我们换掉自家 4 个 custom op 内核（PackActivation / MatMulV8 / TcmDramCopy / PackWeight）的实现细节为 dlsym wrapper，QNN 调度器会按 native 同样的方式拼装图。
3. **3 大差距已可解释**：
   - **MAC 数量级**（285 vs 64）：因为我们 M_TILE=256/N_TILE=256，native 用更小的 tile（推算 ~64）。改 `gen_v8_graph.py` 的 plan_matmul_graph 即可对齐。
   - **bias_to_vtcm 缺失**：我们的 bias 作为 MatMulV8 input 走的是同一种 weights_to_vtcm，行为上等价（DMA 各自独立），不算 bug。
   - **activations_to_vtcm ×1 缺失**：QNN 在图首把整块 act 一次性进 VTCM，我们没这个层级。需要 ActToVtcmCache 节点。

## 推进顺序（更新）

- ✅ Phase 0 — symbol catalog
- ✅ Phase 1 — TCM_Only 验证
- **Phase 4** — 最快收益：在 `gen_v8_graph.py` 把 M_TILE/N_TILE 缩到 native 量级（推算 64），ConvLayer 数量从 64 涨到 ~256，给 QNN 调度器更多并行机会。
- **Phase 2** — `MatMulV8` 默认走 `V8_USE_DLSYM_PER_TILE` 路径（dlsym hmx_convbbb1x1_stride1，已通）→ 单 instance 设置开销可能下降。
- **Phase 3** — 把 `PackActivationU8RowMajor` 替换为 `ForceFormatCrouton`（dlsym convert_to_crouton_b）。已有 PackActCrouton wrapper 可复用。
- **Phase 5** — `TcmDramCopy` 替换为 `ForceFormatFlat`（dlsym convert_from_crouton_b）+ 拆出独立 SlicePadShapeInplace。

## Artefacts

- `phase1_validation/ctx/v8_model_bottom_mapping.json` (256³)
- `phase1_validation/ctx_2048/v8_model_bottom_mapping.json` (2048³)
- 复现：`qnn-context-binary-generator --save_backend_op_mapping ...`
