---
name: V8 C8 alignment Phase 1 — Crouton_8 sig + auto ForceFormat_Crouton WORKS 2026-04-27
description: First successful experiment — declaring Crouton_8 + Storage_Indirect on a custom op input triggers QNN compiler to auto-insert q::ForceFormat_Crouton. V8 op-pkg can now match native ConvLayer_s1.opt's input lowering structure.
type: project
---

# V8 C8 alignment Phase 1 (2026-04-27)

## 目标

User 通过比较 256³ optrace `*_htp.json` 发现 native ConvLayer_s1.opt 输入是 `[1, 8, 32, 256]` (Crouton_8 layout)，而我们 MatMulV8 输入是 `[1, 8, 8, 1024]` (tile-array Flat4)。指令：按这个方向对齐 256³。

## Phase 1 结果 — 框架对齐验证 PASSED

修改 V9 (BbbKMajor) op sig：
```cpp
{QHPI_QUInt8, QHPI_Layout_Crouton_8, QHPI_Storage_Indirect, QHPI_MemLoc_TCM_Only}  // act
```

XML 同步：`<Text>[1, M/32, 32, K]</Text>` (native ConvLayer 形状)。

**关键发现 —— Crouton_8 必须配 `Storage_Indirect`**（不是 Direct）。
- 第一次试 Direct 报错 `graph_prepare.cc:1644::ERROR:Op preparation failed with err:-1`
- 改成 Indirect 后 ctxgen 一次过
- 在 `tools/qnn-sdk/examples/QNN/OpPackage/HTP/QHPI/ExampleOpPackageMaxPool.cpp` 验证：所有 Crouton_8/16 sig 都用 Indirect

## ctxgen 输出（256³，gen_v8c8_test.py + BbbKMajor + NOOP body）

```
op_types: BbbKMajor, q::*InputSlice, q::ConvLayer.opt.weights_to_vtcm, q::ForceFormat_Crouton

6 nodes:
  q::*InputSlice              (auto-inserted, dma)
  q::ForceFormat_Crouton      (auto-inserted, uses_hvx)  ← THE WIN
  q::ConvLayer.opt.weights_to_vtcm × 3 (auto, dma) — for static wt+bias+scratch
  HmxMatMulPhase3Package::BbbKMajor   (kernel, uses_hmx)
```

对比 native 256³（8 节点）：
```
q::*InputSlice / bias_to_vtcm / weights_to_vtcm / ConvLayer_s1.opt /
ForceFormat_Crouton / ForceFormat_Flat / Reshape × 2
```

**结构基本对齐**。我们这边少了 ForceFormat_Flat + Reshape（output 还没改成 Crouton_8）和 bias_to_vtcm（bias 走 weights_to_vtcm 一并）。

## Phase 1 改动清单

- `src/HmxMatMulV9SkelOp.cpp:309-314`: sig in[0] → Crouton_8 + Indirect
- `standard_flow/phaseB_v8/MatMulV8Package.xml`: BbbKMajor in[0] shape `[1, M/32, 32, K]`、in[1] shape `[1, 1, K, N]`
- `standard_flow/phaseB_v8/gen_v8c8_test.py`: 新建，最小验证脚本

## Phase 2 — Device runtime FAILED, Qemu sim 不支持 custom op

实测路径：

### 尝试 1：Crouton_8 + Indirect 在 in[0]（act），其他 Flat4 + Direct + TCM_Only
- ctxgen: ✓ 6 节点, 自动插 ForceFormat_Crouton + 3× weights_to_vtcm
- device: ✗ `QnnDsp <E> Internal error handing: Dma execution failed on the skel side. result = 6006`
- 解读：`q::ConvLayer.opt.weights_to_vtcm` 自动 DMA 失败

### 尝试 2：wt/bias/scratch 改成 DDR_OR_TCM（避免强制 VTCM 分配）
- ctxgen: ✓ 3 节点（weights_to_vtcm 不再被插入），简化为 InputSlice + ForceFormat_Crouton + BbbKMajor
- device: ✗ `QnnDsp <E> skelExecute call failed with err 1003` + `QnnDsp <E> DspTransport call failed, error 0x00000010`
- 解读：DMA 错误消失但 skelExecute 失败。可能是 kernel body 跑到访问 `qhpi_tensor_raw_data(inputs[0])` 在 Indirect 输入上 segfault

### 尝试 3：kernel body 改成只 zero output（不访问 inputs），output 也改 Crouton_8 + Indirect
- ctxgen: ✓
- device: ✗ 仍然 Graph Execution failure，且 logcat 显示 SSR (subsystem restart)，errors 被 power config 噪声盖住没有具体信息

### Qemu 模拟器不可用
- `libQnnHtpQemu.so` 报错：**`HTP QEMU does not support custom op packages`**
- 用模拟器调试这条路走不通。需要 Hexagon Simulator (不同工具，应该支持 custom op)。

### 已 REVERT 全部修改
V9 sig + XML 都回到 baseline，V8 production sweep 256³ 仍正常 (32507 cyc)。

## 下次重启 Phase 2 的策略

1. **build & run SDK 例子 ExampleOpPackageMaxPool**（用同样的 Crouton_8 + Indirect 模式）—— 拿到一个已知可工作的基线对比
2. **加 FARF logging** 在 kernel body 里 `farf("...");` 看哪一步先失败（runtime API 调用 vs op 注册）
3. **试 Hexagon Simulator** (`hexagon-sim` 在 tools/hexagon-sdk 里) —— 这个支持 custom op，比 Qemu 强
4. **写一个最最简单的 Crouton_8 op**（一个输入一个输出，没有 weights/bias initializer），先验证最小框架是否 OK，再叠加复杂度
5. 看 ForceFormat_Crouton 对 Source ONNX shape 的具体约束（可能必须是 [1, H, W, C] NHWC，而不是任意 rank-4）

## 验证命令

```bash
cd example/hmx_matmul_phase3/standard_flow/phaseB_v8
python gen_v8c8_test.py --M 256 --K 256 --N 256 -o phase1_validation/v8c8_test/v8c8.onnx
# convert + ctxgen 流程见上文
python3 -c "
import json
d = json.load(open('phase1_validation/v8c8_test/ctx/v8c8_bottom_mapping.json'))
from collections import Counter
print(Counter(n['type'] for n in d['graph']['nodes'].values()))
"
```

## 副作用 — V9 baseline 已变

V9 (BbbKMajor) 不再向后兼容老的 Flat4 tile-array 用法。如果要保留旧 V9 路径，需要分支为新 op `MatMulV8C8` 或加版本号。当前实现：BbbKMajor 是 C8-aligned 的演进版。
