# w16a16 生产算子：shape-general byte-exact HMX MatMul（现状 + 实现 + 复现）

w16a16 = uint16 激活 × int16 权重 → uint16 的 HMX matmul，作为可部署的 **QNN HTP custom op**
(`example/qnn_hmx_matmul_w16a16`)。本文是 w16a16 **生产化（任意 shape）** 的唯一权威状态文档。
kernel 本身的原理/机理见 `docs/w16a16_is_two_w8a16.md`（= 2×w8a16 字节分解）与
`docs/w16a16_kernel_mechanism.md`（convhhh 的 dilate+2x2+cvt.uh 机理 + 全反汇编 `Agent/qnn_re/hmx_v73_convhhh1x1_w16a16.disasm`）。

---

## 0. 现状（device 实测，逐字节对 QNN native）

| 范围 | 状态 |
|---|---|
| **任意 32-multiple shape，M≥64（K、N 可到 32）** | ✅ **byte-exact**（`exact==M*N, maxdiff==0`），方阵/矩形/N%128≠0/N>128/小 K,N 全验 |
| **M=32（输出行维 = 32）** | ⏳ 待 graph-pad（见 §4；K=32/N=32 都已 byte-exact，唯 M=32 撞 kernel 64-行下限） |

已验 shape（CI sweep）：256³、128³、96³、64³、256×256×128、128×128×96、128×128×160、64×64×32、64×32×64 —— 全 `maxdiff=0`。

---

## 1. 三个让它跑通的关键

1. **`MODE=chain_qdq`**（不是 `chain`）：act 以 FLOAT 入图 + `QuantizeLinear(scale=1, zp=32768)` → 量化成
   `SFIXED_POINT_16`（字节与原 uint16 完全相同）。HTP `Reshape` 只接受量化 16-bit（拒 plain UINT16/INT16），
   故必须走 qdq。（这是 256³ accepted 一直在用的方式；用 `chain` 会 ctxgen compose-fail。）
2. **formula descriptor**（`HMX_W16A16_FORMULA_DESC`，已是 `native_record_256` profile 默认）：op 不再写死
   256-record，而是 `f(M_t,K_t)`：`out_y_stride=n_tiles_pow2=M_t·32`(=M)、`act_y_stride=K_t·64`(=2K)、
   `m_total=1`、per-128-split `k_total=128` / 非-split `k_total=N_t·32`。256³ 精确复现 native record（仍 byte-exact），非-256 按 M/K/N 缩放。
3. **partial-128 packer**（`scripts/generate_w16a16_weight_sidecar.py`）：weight/bias 支持 N%32（非%128），
   最后一个 128-block 只打包实有 tile / group（bias 按 `present//16` 组）。

外加：gate 放宽到 `M,K,N>=32 且 %32`；小-M 的 `mt_per_block==0` 误门控已去（该值仅诊断用）。

---

## 2. 实现（文件 → 职责）

| 文件 | 职责 |
|---|---|
| `example/qnn_hmx_matmul_w16a16/src/HmxU16I16ToU16MatMulOp.cpp` | op：tile_counts、QHPI precompute、formula descriptor、N//128 split + 非-split 路 |
| `example/qnn_hmx_matmul_w16a16/src/v73deep_conv1x1_kernel.inc` | byte-exact HMX kernel body（convhhh；可读版见 handwritten 副本） |
| `example/qnn_hmx_matmul_w16a16/build.sh` | profile：`native_record_256` = shape-general（formula + computed sidecar，默认开 FORMULA_DESC） |
| `scripts/generate_w16a16_weight_sidecar.py` | computed weight/bias sidecar（任意 N%32） |
| `scripts/w16a16_shape_sweep.sh` | **CI gate**：扫多 shape，断言 `exact==M*N && maxdiff==0` |
| `example/qnn_hmx_matmul_common/gen_quant_chain.py` | 链路生成（**未改**；用 `MODE=chain_qdq` 即可，是调用方式） |

> 注：曾试过 act 声明 INT16 + XML 加 INT16 来过 reshape —— **错的，已 revert**（plain INT16 也不被 HTP Reshape 支持，须 SFIXED_POINT_16，由 chain_qdq 的 QuantizeLinear 产生）。仓库里 `gen_quant_chain.py` / `QnnHmxMatMulW16A16Package.xml` 干净。

---

## 3. 复现

```bash
# 1) build shape-general op
cd example/qnn_hmx_matmul_w16a16
W16A16_KERNEL_PROFILE=native_record_256 bash build.sh && \
W16A16_KERNEL_PROFILE=native_record_256 bash build_x86.sh

# 2) CI sweep（自动生成 native ref + 跑 custom op + 断言 byte-exact）
bash scripts/w16a16_shape_sweep.sh
#   或单 shape：SHAPES="128,256,96" bash scripts/w16a16_shape_sweep.sh

# 3) 单 shape 手动（native oracle + 对比）
cd example/qnn_matmul_profile && CONFIGS=w16a16 bash profile_all.sh --shape M,K,N --out-dir /tmp/o
cd ../qnn_hmx_matmul_w16a16/standard_flow/custom_w16a16
M=.. K=.. N=.. CHAIN=1 MODE=chain_qdq W16A16_KERNEL_PROFILE=native_record_256 \
  W16A16_NATIVE_ORACLE_DIR=/tmp/o/w16a16 bash run_w16a16_chain.sh
#   -> analysis/w16a16_custom_compare.txt: native-exact M*N/M*N maxdiff=0
```
设备 = `ssh oneplus`（perf-burst 偶发假失败，重跑即可）。

---

## 4. M=32 唯一缺口（根因 + native 修法 + 待做）

- **根因**：v73deep HMX kernel 输出行维 M 有 **64-行硬下限**；M=32（M_t=1）→ kernel 行数不足 → 执行 fault。
  K=32、N=32 都 byte-exact，**只有 M（输出行/token 维）受限**。
- **native 怎么绕的**（optrace 实证）：`ForceFormat_Crouton`（act 的 M pad 到 64）→ `ConvLayer` 算 64 行（写 64-行 VTCM 面）→ `OutputSlice` 切出前 32 行。**graph 级 pad-64 + slice。**
- **op 内 discard 行不通**：试过让 op 跑 64 行、padding 行写 DDR discard buffer → fault，因 **HMX mxmem 只能写 VTCM**，DDR 缓冲非法。已 revert。
- **正确待做**：在 `gen_quant_chain.py` 对 w16a16 且 M<64：act 补零到 M=64、op 输出 M=64、加 `Slice` 切到 M=32（op 对 M=64 本就 byte-exact）。共享文件改动，需回归其他 family。

---

## 5. 集成
- **QNN 图**：直接部署该 HTP custom op（`example/qnn_hmx_matmul_w16a16`）。
- **baremetal / 脱离-QNN 裸调用**：手写 kernel **byte-exact（sim+真设备）已完成** —— 见
  `docs/w16a16_standalone_handwritten.md`（使用/原理/对齐方法论）。driver =
  `scripts/run_w16a16_standalone_kernel.py`（sim）/`run_w16a16_standalone_device.py`（设备）；
  CI = `tests/qnn_kernel_e2e/handwritten_hmx_matmul/test_w16a16_standalone.sh`。
  根因坑：`emulate_hmx_conv1x1_params.py` 的 mask word6 对 arg5=0x80(dilate) 曾算错（已修）。
- **GDN solve 用 w16a16 的性能裁定** 见 `Agent/current/gdn_solve.md` §4（纯 HMX w16a16 ≈ 打平 HVXMixHMX）。
