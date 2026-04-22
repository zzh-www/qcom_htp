# QNN 三路 matmul — 后续计划（2026-04-22）

三个 kernel（w4a8 / w4a16 / w16a16）已在 SM8650 v75 真机 + hexagon-sim 上全部 bit-exact。基础设施完整。**当前所有的未闭合工作都围绕关掉 500-2900× 的性能差距**，因为 Phase 1 已经确认：HMX 自身不是瓶颈（7.9 cyc/packet silicon ceiling 和 QNN 同水平），scalar DDR 路径 + 图层并行才是。

**先读这三份文档再动手**：
1. `Agent/three_kernel_status_final.md` — 整体交付表
2. `Agent/qnn_hmx_pipelining.md` — Phase 1 RE 全部结论（Rt_wt=0x3FF / `:above` / `:retain` 语义）
3. `Agent/int4_matmul_optimization_log.md` — 迭代 P0-P11 的历史记录（知道哪些路走过了）

---

## 任务清单（按 impact × effort 排序）

### T1 — HVX 化 pack_weight_32x32（高 impact，中等 effort）

**目的**：消除每 K-iter 的 scalar byte transpose 开销（当前占 ~8%，HVX-后 ≤1%）。

**先前尝试（失败了，供参考）**：
- `Q6_V_vdelta_VV` + 4×32→32×4 butterfly permutation：硅片 4% 速度提升但**输出全错**——cyclic-shift-by-2 字节排列不满足 vdelta butterfly 约束。见 `example/hmx_matmul_w4a8/kernel/hmx_int4xint8_matmul.c` 第 73-86 行注释。

**建议路线**：
1. 用 2-stage `Q6_W_vshuff_VVR` 实现 byte transpose（不是 vdelta）：
   - 第一阶段：vshuff Vu(r0||r2) Vv(r1||r3) 在 R=32 处，得到 {r0[0], r1[0], r0[1], r1[1], ...} pair
   - 第二阶段：二次 vshuff 结合 r0/r1 和 r2/r3 的 interleave
   - 验证：用 debug probe 做 1 个 K-group（128 bytes）输出对照 scalar
2. 或用 `Q6_Vb_vlut32_VbVbR` 查表法（32 字节 LUT per lane，需要 4 次查询）

**估算**：全部 HVX 化后 w4a16 从 2.08 → ~1.9 cyc/MAC（~8% 收益）。主要意义是验证 HVX pack 可行，为 T3 打基础。

**关键文件**：
- `example/hmx_matmul_w4a8/kernel/hmx_int4xint8_matmul.c:74-90` — 当前 scalar pack_weight
- `example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c:131-147` — 同一函数 w4a16 版本（字节完全一致）
- `tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hvx_hexagon_protos.h` — intrinsic 声明

**验证**：
```sh
bash tests/test_hmx_matmul_w4a8.sh    # sim bit-exact
bash tests/test_hmx_matmul_w4a16.sh
bash example/hmx_matmul_w4a8/run_on_device.sh --shape 512,512,512    # 真机 cyc/MAC
bash example/hmx_matmul_qnn/run_on_device.sh  --shape 512,512,512
```

---

### T2 — 解码 QNN ForceFormat_Crouton_f2c framework op（极高 impact，高 effort）

**目的**：QNN built-in kernel 没有 HVX pack 在 hot loop 里——因为 `ForceFormat_Crouton_f2c` 把 tile 预处理成 HMX-ready Crouton 布局，作为一个**独立的上游 HVX op**插入 graph。理解这个 op 就能消掉我们 kernel 里几乎全部的 scalar pack。

**已知线索**（见 `Agent/int4_matmul_optimization_log.md` P5 章节）：
- Crouton_8：8×8×32 chunks = 2 KiB each，完美匹配 HMX activation tile 大小
- QNN 自动插入：签名声明 `QHPI_Layout_Crouton_16` + `Storage_Indirect` 就会触发
- Block table：`qhpi_tensor_block_table()` 返回 block 指针数组
- 之前的 Crouton probe 失败（P9）：block 布局是 "adjacent-M-pair interleaved along K"，不是 HMX 直接可用

**路线**：
1. 用 `qnn-net-run --save_tensor_data_dir` dump 一个 512³ w8a16 的 ForceFormat_Crouton 输入和输出张量
2. 逐字节对比 pre-/post-ForceFormat 数据，反推 transpose 规则
3. 在我们的 kernel 签名里声明 Crouton 输入，让 QNN 自动插入这个 op
4. 改写 kernel 用 `qhpi_tensor_block_table()` + block 指针数组替代 scalar pack

**估算**：关闭 500× 差距中的 ~100-200×。3-5 天工作量。

**关键文件**：
- `Agent/int4_matmul_optimization_log.md:378-420` — P9 Crouton probe 的失败分析
- `tools/qnn-sdk/include/QNN/HTP/core/memory_layout.h:311` — `R4CroutonLayout` 定义
- `example/hmx_matmul_qnn/src/HmxInt4MatMulOp.cpp:186-189` — 当前 sig_inputs（Flat4 + Direct）改成 Crouton + Indirect

**验证**：同 T1 命令 + `qnn-profile-viewer` 读 chrometrace 检查 `ForceFormat_Crouton_f2c` 节点出现在我们 op 上游。

---

### T3 — Host graph 端 2×2 M/N tiling（高 impact，中 effort）

**目的**：QNN w8a8 512³ 跑 4 个 `ConvLayer_s1.opt` sub-kernel 并发在 4 个 HVX thread 上——这是它比我们快的另一半原因。我们也要这么干。

**背景**：QHPI 的 `multithreaded=true` 用 `self_slicing` 机制，**但只支持 HVX-resource op**。我们的 HMX-resource op 被硬拒。解决：在 host 端（`run_int4_matmul.cpp` 的 graph-build 里）就拆成 4 个独立的 op 节点。

**路线**：
1. 在 host harness 的 `Qnn_OpConfig_t` 创建循环里：当 M ≥ 64 且 N ≥ 64 时，emit 4 个 sub-op：
   - sub00: act[0:M/2] × wt[:, 0:N/2] → out[0:M/2, 0:N/2]
   - sub01: act[0:M/2] × wt[:, N/2:N] → out[0:M/2, N/2:N]
   - sub10, sub11 类似
2. 4 个 sub-op 数据独立，QNN scheduler 应该自动并发
3. 每个 sub-op 用它自己的 VTCM scratch slot（当前 MAX_SLICES=4 留的 pool 正好用）

**估算**：4× 真机加速。实现 1-2 天。

**关键文件**：
- `example/hmx_matmul_qnn/src/run_int4_matmul.cpp:144-156` — 当前的单 op 构造
- `example/hmx_matmul_qnn/src/HmxInt4MatMulOp.cpp:118-130` — MAX_SLICES scratch pool 已经准备好了
- `Agent/int4_matmul_optimization_log.md:227-265` — QNN 自己如何做 4-way tiling 的 RE

**验证**：
```sh
bash example/hmx_matmul_qnn/run_on_device.sh --shape 512,512,512
# 期望：cyc/MAC 从 2.08 → ~0.5（~4×）
```

---

### T4 — Phase 2B 实现：w16a16 match-QNN 48-tile 复刻（低 impact，高 effort）

**目的**：用户 plan 的 literal 要求——复刻 QNN w16a16 的 exact 48-tile 布局。

**状态**：仅 design doc。见 `Agent/phase2b_w16a16_match_qnn.md` 的完整路线。

**路线**：
1. `qnn-net-run --save_tensor_data_dir` 抓 512³ w16a16 built-in 的每一个 HMX tile 中间数据
2. 从 `chrometrace_htp.json` 的每个 HMX event 的 shape 里读出 48 个 (m_range, k_range, n_range)
3. 建 `static const tile_t qnn_512_tiles[48]` 查找表
4. 新 kernel 按这个 table 顺序 enumerate，每个 tile 调 32×32×32 底层 MAC

**估算**：1-2 天。但**实用价值低**——我们的 w16a16 已经过了 correctness；matching 48-tile 只是让 literal 参数对齐，perf 同样被 scalar 瓶颈限制。

**建议**：除非用户明确要，否则优先 T1-T3。

---

### T5 — 修剩下的小事（每个 < 1 天）

**T5a — 为 sim harness 加 per-scenario pcycle 标记**（P5 nice-to-have）
- 用 `__builtin_readcyclecounter()` 或 `h2_perf_counter_*` 包住 scenario 循环
- 输出 per-scenario 精确 pcycles，替代当前 "总 pcyc / n_scenarios" 近似
- 完善 `Agent/sim_vs_device_cycles.md` 的数字

**T5b — HVX 化 gather_w_col**（P2b 剩余）
- 每 k-row 32 字节连续读 + `Q6_Vb_vsub_VbVb`(-128) + 32 字节连续写
- 为 T1 的 HVX pack 配套（gather 产物→packed tile 流水线起来）
- 当前 scalar gather 不算 hot：只是 ~10% overall

**T5c — 写 `profile_all.sh --custom-ops all` 统一 CSV**（P0 extension）
- 现在每个 custom op 在自己的 `run_on_device.sh` 里报 cyc/MAC
- 统一到 `example/qnn_matmul_profile/profile_all.sh` 输出单张 dtype × shape CSV
- 方便和 QNN built-in 并排对比

---

## 推荐执行顺序

```
Day 1-2: T3 (host 端 2x2 tiling)   ← 最大 ROI，改动最小
Day 3-4: T2 第一步 (Crouton dump + 逆向布局规则)
Day 5:   T1 (HVX pack_weight via vshuff)
Day 6-7: T2 第二步 (kernel 消费 block table)
Day 8+:  视情况 T4 / T5
```

**三件事全部做完** 后真机 cyc/MAC 预期：
- w4a8:  1.92 → ~0.2（10× from T3 + T1）
- w4a16: 2.08 → ~0.3
- w16a16: 12.22 → ~2.0

这能接近 QNN built-in 的同 dtype 性能 2-5× 的范围（而不是当前的 500-2900×）。

---

## 快速上手命令

```bash
# 环境
source scripts/env.sh

# 三个 kernel 真机 sanity check（都应该 0 mismatches）
bash example/hmx_matmul_w4a8/run_on_device.sh   --shape 512,512,512
bash example/hmx_matmul_qnn/run_on_device.sh    --shape 512,512,512
bash example/hmx_matmul_w16a16/run_on_device.sh --shape 512,512,512

# Sim 侧（都应该 3/3 PASS）
bash tests/test_hmx_matmul_{int16,w4a8,w4a16,w16a16}.sh

# 硅探针（当你需要新的 RE）
bash example/hmx_matmul_device/build.sh
bash example/hmx_matmul_device/run_pipeline_probe.sh    # Rt + :cm
bash example/hmx_matmul_device/run_dualacc_probe.sh     # :above + mxswapacc

# 反汇编 QNN built-in（Phase 1 起点）
hexagon-llvm-objdump -d --mattr=+hmxv75,+hvxv75,+hvx-length128b \
    tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
    | less
```

---

## 关键知识点 cheat-sheet

（如果新 session 开始，把这部分塞进 prompt 就能立刻动工）

```
HMX ISA 语义（2026-04-22 硅片验证过）:
  Rt_act = 2047 (0x7FF, 2KB activation tile mask)
  Rt_wt  = 0x3FF (1KB weight tile mask) ← 必须用这个，2047 慢 2.5×
  :above 是 no-op for accumulator routing (跟 plain 一样)
  mxswapacc 真的交换 current/other
  store :after.uh 不加 :retain 会清空两个 acc
  output tile 必须 2KB-aligned (acc:2x1 store 约束)

代码模式:
  kernel .c 必须用 extern "C" header guards (hexagon-clang++ 会 mangling)
  VTCM 布局: act_prepack at 8KB+, out tiles 4KB/6KB (2KB-aligned)
  scalar pack interleaved with HMX MAC (不是 all-prepacked) — VTCM bank 更友好

设备:
  ssh oneplus （termux $HOME 可写，/data/local/tmp/ 只 adb 可写）
  QNN 运行时在 ~/qnn_run/ 里已经推好
  DCVS_PERFORMANCE_MODE + HMX power-on 已在 probe 里封好
```

---

**所有 commit 已推到 main**。代码库是干净的可启动状态。
