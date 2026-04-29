# Next Steps — 复刻 QNN 在 shape=256 的 matmul kernel

## 🎯 目标

逆向并**复刻**出 QNN `q::ConvLayer_s1.opt` 在 shape=256³ 的 matmul kernel —
不是 dlsym 借用 QNN binary, 而是**完全自己拥有 + 自己写**。Deliverable:

1. ✅ **Phase A — 完整 RE 已就绪** (前期 sessions): 整个 dispatch 链 disasm
   完成 (`hmx_v73_convbbb1x1deep_stride1` 0x2ebe40 / wrapper 0x3dc2a8 /
   builder 0x3d7920) + descriptor 字段语义全部解码 + native 运行时参数
   通过 v3 patch dump 拿到 ground truth
2. ✅ **Phase B step 1 — own the kernel bytes** (2026-04-29):
   1132 字节嵌入 `libQnnHmxMatMulPhase3_htp.so`, runtime 不再 dlsym QNN binary
3. 🟡 **Phase B step 2 — packet-by-packet inline asm rewrite** (next sessions):
   把 `.byte` 块逐 packet 替换成手写 Hexagon inline asm, bit-exact 验证每步
4. 🔴 **Phase C — kernel-level cpp gap** (low ROI, paused):
   1.10× cpp ratio gap 来源是 HMX state pre-config 等环境差异, 不是 VTCM /
   binary 内容. 关掉它需 RE 整个 dispatcher + 多天工作. 256³ wall time 已超
   native (15us vs 17us), 暂搁

## 📊 当前最佳实测 (V9_OWN_KERNEL + 全 native 描述符)

256³ chain8 hot:

| 路径 | hot dur | pkts | cpp | bit-exact | vs native |
|---|---:|---:|---:|---:|---:|
| **V9_OWN_KERNEL (embedded byte-replica)** | **1684** | **468** | **3.60** | **100%** | **1.50× cyc / 1.35× pkts** |
| dlsym (V9_NATIVE_V73DEEP, 早) | 1659 | 471 | 3.52 | 100% | 1.48× cyc / 1.36× pkts |
| 旧 baseline (descs 错) | 3042 | 747 | 4.07 | 100% | 2.71× / 2.16× |
| Native q::ConvLayer_s1.opt | 1120 | 346 | 3.24 | ref | 1.0× |

跨 shape wall-time (steady-state, Accelerator excl wait):

| Shape | Native us | Ours us | Wall ratio | bit-exact |
|---|---:|---:|---:|---:|
| 256³ chain8 | 17 | **15** | **0.88×** (我们快 12%) | ✓ |
| 512³ chain8 | 62 | **60** | 0.97× | ✓ |
| 1024³ chain8 | 1975 | **361** | **0.18×** (快 5.5×, native 切 75 sub-ops + Spill/Fill) | ✓ |
| 2048³ chain8 | 9797 | ✗ | (graph_finalize VTCM overflow w/ TCM_Only) | n/a |

256-1024³ 区间已经**追平或超过 native wall time**. Chrometrace 单 op
"1.36× pkts gap" 是误导 (HMX-engine cyc/cpp 派生指标, 不是 PMU 原始 packet 数).

## 🔧 Build flags (current best)

```bash
DEFS='-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST'
DEFS="$DEFS -DV73D_N_TILES_POW2=(M_t*4) -DV73D_OUT_Y_STRIDE=(M_t*4) -DV73D_AD_ACT_Y_STRIDE=(M_t*4)"
DEFS="$DEFS -DV73DEEP_ARG1=0x700 -DV73D_MASK_38_EXTRA_PTR=1 -DV73D_NATIVE_LAYOUT=1"
DEFS="$DEFS -DV9_OWN_KERNEL"  # 字节内嵌, 不 dlsym
EXTRA_DEFS="$DEFS" bash build.sh && bash build_x86.sh
```

⚠️ **N_TILES_POW2 / OUT_Y_STRIDE / AD_ACT_Y_STRIDE 必须用 `(M_t*4)` 表达式**
(不是常数 32) — 256³ M_t=8 → 32, 但 512³ M_t=16 → 64. 早期硬编码 32 在 512³
导致 pkts 翻倍 + 50% bit-exact.

## 📁 已就绪的资产

### RE 反汇编 (Agent/qnn_re/)
- `hmx_v73_convbbb1x1deep_stride1_2ebe40.S` — kernel 主体 1132 bytes / 289 行 disasm
- `wrapper_3dc2a8.S` + `wrapper_3dc2a8_TRACE.md` — 调用 wrapper trace
- `descriptor_builder_3d7920_full.S` — desc builder (vtable 卡点)
- `hmx_convbbb1x1_stride1_FULL_decoded.S` — 老版本完整 decoded
- `p1_4_kernel_patch_2026-04-29.md` — native runtime r0..r5 dump
- `p1_5_native_layout_2026-04-29.md` — act_tbl/out_tbl 32-entry × 0x800 layout

### 复刻的 kernel
- `example/hmx_matmul_phase3/src/v73deep_replica_bytes.h` — naked function shell
- `example/hmx_matmul_phase3/src/v73deep_replica_bytes.inc` — 1132 字节
  `.byte` 块, 每 4-byte word 一行 + 解码 asm 注释 (283 行)
- `scripts/extract_v73deep_bytes.py` — 重新生成工具

### 测试基础设施
- `scripts/perf_v8c8.py` — 单 op chrometrace 比较 (注意: pkts 是 HMX-engine
  cyc/cpp 派生, 不是 PMU 原始)
- `qnn-profile-viewer ... | grep "Accelerator (execute excluding wait)"` —
  wall-time 真实比较 (取 3 次 inference 中间值)
- `scripts/parse_v73deep_desc_dump.py` — V9_DESC_DUMP 解码
  (mask/od/ad/extra_param/act_tbl/out_tbl + VTCM 地址)
- V9_PMU_PROBE — kernel-internal vs op-total PMU packet 计数
- V9_PROBE_REGIONS — 4-variant 线性拟合 per-region cost

### 验证手段
1. **bit-exact**: `perf_v8c8.py --compare <native>` 报 100%
2. **byte-replica 等价**: V9_DESC_DUMP 验证 mask/od/ad 与 native 一致
3. **packet 等价**: `hexagon-objdump --mattr=+hvxv75,+hmx,+hvx-length128b`
   解码我们生成的 .so kernel section, 与 disasm 比对

---

## 🟡 Phase B step 2 — packet-by-packet inline asm 重写 (next sessions)

### 工作流

每个 Hexagon "packet" 是 `{ ... }` 一组 1-4 条并行指令 (4-16 字节).
Kernel 主路径约 63 packets. 替换流程:

1. 选一个 packet (e.g., 起始 prologue stack-save)
2. 在 `v73deep_replica_bytes.inc` 把对应的 `.byte` 行替换成等价的
   Hexagon inline asm:
   ```c
   "{ r29 = add(r29, #-0x28); nop; r7:6 = memw(r4+#0x8); r11:10 = memw(r4+#0x0) }\n"
   ```
3. `hexagon-clang++ -c` 编译 + `hexagon-objdump -d` 验证生成字节 ==
   原始字节 (确保等价)
4. Build + 跑 256³ chain8 + `perf_v8c8.py` + 100% bit-exact 验证 + cpp 在 noise
5. 加 inline 注释写明 packet 角色 (e.g., "stack alloc + read mask[8..15] r7:6,
   read mask[0..7] r11:10")
6. 下一个 packet

### 推荐顺序

1. **Prologue (0x2ebe40-0x2ebec8, ~25 packets)** — 描述符读 + 寄存器 setup,
   语义最清晰
2. **Epilogue (0x2ec294-0x2ec2a8, ~5 packets)** — stack 恢复 + jumpr r31
3. **Main path body (0x2ebed0-0x2ebf9c, ~28 packets)** — 我们 production 实际
   走的路径; 包含 K-MAC inner loop + drain
4. **Alt-A/B/C 路径 (0x2ebfa0-0x2ec280, ~30 packets)** — 我们生产不会进 (n_act_pairs=8 + ep[0]=1 → 走 main). 可保留 `.byte` 占位最后再写

### 每个 packet 重写的 risks + mitigations

| Risk | Mitigation |
|---|---|
| 寄存器分配冲突 (compiler 重用同名 reg) | naked function + 明确 register names (r0..r29) |
| Branch 目标地址变化 (asm 生成的字节不等于原始) | 用 `1:`/`2f` 命名 label, objdump 比对字节 |
| inline asm constraint 写错 | 编译期 check, runtime 100% bit-exact 验证 |
| HVX/HMX 修饰符语法 (`:deep:cm`, `:after`) hexagon-clang 不接受 | 已在 V8 inline 验证 hexagon-clang 接受 |
| 删 packet 边界 (`{ }`) 编译器自动重组 | 用 `{`/`}` 显式分组, 验证 packet 数不变 |

### 完成判据

- 所有 .byte 替换为可读的 Hexagon inline asm
- `hexagon-objdump` 字节 byte-for-byte 等于原始 0x2ebe40 主路径
- 100% bit-exact 全 shape (256/512/1024³)
- perf 在 V9_OWN_KERNEL byte-version 的 noise 内

预计 6-10 个工作 session, 每 session 处理 5-10 个 packets.

---

## 🔴 Phase C — cpp gap 关闭 (paused, low ROI)

剩余 1.10× cpp ratio (3.60 vs 3.24 native) 不来自:
- ❌ Kernel binary 内容 (V9_OWN_KERNEL = 字节级 same as native)
- ❌ 描述符字段 (V9_DESC_DUMP 验证 mask/od/ad/ep 全 match native)
- ❌ VTCM 排布 (Phase C-1 实证: 改 input order 让 VTCM bias-wt-act 紧密
  排布到 native pattern, cpp 不变)
- ❌ 表布局 (V73D_NATIVE_LAYOUT 32-entry × 0x800 stride 已 match)
- ❌ wt+bias VTCM 位置 (V9_DESC_DUMP: 我们 wt+bias 已在 VTCM)

**剩下唯一候选**:
1. **HMX state pre-config** (最可能): native dispatcher 在 kernel 调用前发了
   写 HMX 控制寄存器的 mxmem packet, 改变 HMX 内部 fan-out / drain 模式.
   要找需 RE 整个 dispatcher 链:
   - q::ConvLayer_s1.opt 的真实入口 (dispatcher cpp 函数, 在 `libHtpPrepare.so`
     或 `libQnnHtpV75Skel.so` 的某个 mangled C++ symbol)
   - 该函数在调用 0x2ebe40 之前发的所有 HMX packets
2. **TLB/L2 cache 状态**: native 持续访问让 TLB 暖, 我们 cold (但 hot
   chain calls 都应该热, 边际)
3. **L1d miss on od/ad descriptor reads**: stack-allocated 但每 op 重建,
   可能每次 cold miss

**Why paused**:
- 关 cpp gap 多天 work, 风险 (kernel patch 可能破 correctness)
- 256³ wall time 已经超 native (15us vs 17us 我们快 12%)
- 1024³ wall time 我们快 5.5× (native 在大 shape 切 75 sub-ops + Spill/Fill)
- Phase B step 2 (字节 → inline asm) 才是 "复刻" 目标的核心 deliverable

可选 Phase C 验证手段 (如未来想关):
- 在 Phase B step 2 完成后, 加 PMU probes 围绕每个 packet, 找具体 stall 位置
- v4 patch: dump native 在 0x2ebe40 entry 之前的 HMX state register 值
- Disasm dispatch chain 找 mxmem 写控制寄存器的 packet

---

## 🚧 已知不可行 / 已 disprove (don't retry)

### Phase B 之前
- ❌ Multi-call (4 calls k_total=64 / 8 calls per-M-tile) — call 开销 dominant
- ❌ OLD kernel 单 call + V73DEEP 描述符 — crash
- ❌ V73 non-deep + Lane A v2 — 1.74× 慢于 V73DEEP
- ❌ unaligned kernel 替换 — 1003 SIGSEGV (输入要求不同)
- ❌ 4 alt-paths 触发 (alt-A/B/C 都 partial bit-exact)

### Phase B/C 期间
- ❌ V73D_TABLE_VTCM (act_tbl/out_tbl in VTCM scratch) — cpp 飙到 14.5×
  (VTCM bank 与 wt/bias 冲突). Don't retry.
- ❌ 4-input scratch 为 VTCM 表 backing — 但保留 4-input 基础设施
  (无 perf 损失, QNN scheduling 略有改善)
- ❌ Phase C-1 VTCM 重排作为 cpp gap fix — VTCM 排成 native 模式但 cpp 不变
  (重排本身保留, 因为 native-like 利于后续等价性验证)

### 2048³+ 限制
- ❌ TCM_Only sig 在 ≥2048³ 让 VTCM 超 8MB → graph_finalize error 1002
- 解法 (未做): 改 wt sig 为 DDR_OR_TCM 让 QNN 自管 spill, 或 op 层手切 K
  方向. 不在 "复刻 256³ kernel" 目标范围内.

---

## 📦 测量速查

```bash
# build current best (V9_OWN_KERNEL embedded)
cd example/hmx_matmul_phase3
EXTRA_DEFS='-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST -DV73D_N_TILES_POW2=(M_t*4) -DV73D_OUT_Y_STRIDE=(M_t*4) -DV73D_AD_ACT_Y_STRIDE=(M_t*4) -DV73DEEP_ARG1=0x700 -DV73D_MASK_38_EXTRA_PTR=1 -DV73D_NATIVE_LAYOUT=1 -DV9_OWN_KERNEL' \
    bash build.sh && bash build_x86.sh

# run 256³
WT_LAYOUT=kmaj M=256 K=256 N=256 CHAIN=8 \
    OUT_DIR="$(pwd)/standard_flow/phaseB_v8/phase1_validation/test" \
    bash standard_flow/phaseB_v8/run_v8c8_chain.sh

# single op chrometrace compare
source ../../scripts/env.sh
python3 ../../scripts/perf_v8c8.py \
    standard_flow/phaseB_v8/phase1_validation/test \
    --compare standard_flow/phaseA_native/s256_chain8_compare

# wall-time真实比较 (取 3 次 inference 中间值)
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
    --input_log standard_flow/phaseB_v8/phase1_validation/test/device_out/qnn-profiling-data_0.log \
    | grep "Accelerator (execute excluding"

# descriptor / VTCM dump (验证字段 + 地址)
EXTRA_DEFS='... -DV9_DESC_DUMP' bash build.sh && bash build_x86.sh
WT_LAYOUT=kmaj M=256 K=256 N=256 CHAIN=8 OUT_DIR=... bash standard_flow/phaseB_v8/run_v8c8_chain.sh
python3 ../../scripts/parse_v73deep_desc_dump.py <out>/device_out/out.raw

# 跨 shape sweep
for SIZE in 256 512 1024; do
  WT_LAYOUT=kmaj M=$SIZE K=$SIZE N=$SIZE CHAIN=8 \
    OUT_DIR=".../v73d_s${SIZE}" \
    bash standard_flow/phaseB_v8/run_v8c8_chain.sh
done

# extract / regenerate v73deep bytes
python3 scripts/extract_v73deep_bytes.py
```
