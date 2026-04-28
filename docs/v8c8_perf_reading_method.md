# V8C8 perf reading + comparison method (固化)

固化下来的 perf 读取/对比流程。所有 V8C8 BbbKMajor vs native q::ConvLayer_s1.opt
的对比都走这一套，避免再被 chrometrace metric 语义模糊带偏。

## 三个并行测量手段

1. **chrometrace（QNN 出的 optrace 解码）** — 默认，每次跑都有
   - 每个 op event 报告 `dur` (cycles)、`Cycles per Packet` (cpp)、隐含 `pkts = dur/cpp`
   - 单位实测 ≈ Hexagon PMU `COMMITTED_PKT_ANY` 真实执行包数（差异 ~30%，原因见下文）
2. **C15:14 pcycle counter（`Q6_R_pcycle_R` 内联汇编）** — 任意 build 都能用
   - 任何代码点都能读，不需要权限；直接给硬件 cycle delta
   - 对照 chrometrace `dur` 用，验证 chrometrace 不是平均后产物
3. **QURT PMU API（`qurt_pmu_set/get/enable`）** — 需要 V9_PMU_PROBE 编译开关
   - 真实硬件计数器：committed pkts、dispatched pkts、AXI traffic、HVX active 等
   - 是 chrometrace pkts 的"ground truth"参考

## chrometrace 单位实测确认（2026-04-28）

跑 V73DEEP 256³ chain hot 同一个 op，三种测量并列：

| 测量 | dur (cyc) | pkts | cyc/pkt |
|------|----:|-----:|--------:|
| chrometrace (no probe) | 2808 | 747 | 3.76 |
| pcycle delta (C15:14, V9_PROBE_KERNEL_CYC) | ~3700 (whole op) | — | — |
| PMU COMMITTED_PKT_ANY (V9_PMU_PROBE, whole op) | 3237 | 1805 | 1.79 |
| PMU COMMITTED_PKT_ANY kernel-only | 1885 | 1253 | 1.50 |

观察：
- chrometrace `dur` ≈ PMU cyc delta — 同一回事（HW pcycle counter）✓
- chrometrace `pkts` ≠ PMU `COMMITTED_PKT_ANY`（PMU 高 ~40%）
  - 原因：PMU "ANY" 计 **所有 HW thread** 的 committed packets。Hexagon V75 跑 5 个 HW thread，闲置 thread 也在 commit NOP，被 ANY 全部计入
  - chrometrace 的 pkts 像是 **目标 op 那个 thread 单独** 的真实执行包数
- 二者 **同量纲、同方向**：op A 比 op B 慢，两个指标会一致告诉你

**结论：chrometrace `pkts` 可以信。当看到 native 346 vs ours 747，差距 2.16× 是真实包数比。**

## 标准对比工具：`scripts/perf_v8c8.py`

```bash
# 单点：解码我们的 V73DEEP run
python3 scripts/perf_v8c8.py example/hmx_matmul_phase3/standard_flow/phaseB_v8/phase1_validation/<run_name>

# 加 PMU 数据（要求 build 时带 -DV9_PMU_PROBE）
python3 scripts/perf_v8c8.py <run_dir> --pmu

# 对比 native（自动算 ours/native ratio）
python3 scripts/perf_v8c8.py <run_dir> \
    --compare example/hmx_matmul_phase3/standard_flow/phaseA_native/s256_chain8_compare
```

工具做的事：
1. 找 `<run_dir>/device_out/qnn-profiling-data*.log` + `<run_dir>/ctx/*schematic.bin`
2. 调 `qnn-profile-viewer` 出 optrace.txt
3. 正则匹配每个 op event 的 `dur` / `Cycles per Packet` / `QNN Op Name`
4. 算冷/热区分（chain0 是 cold，1..7 是 hot）+ 平均 dur/pkts/cpp
5. 如有 `--pmu`，从 `device_out/out.raw[0..31]` 反序列化 V9_PMU_PROBE 写入的 8×uint32 marker
6. 如有 `--compare`，并排同 op 找 native 同样统计 + 算 cyc/pkts/cpp ratio

## 编译开关

`example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp` 暴露的 perf-相关 toggles:

| 开关 | 用途 |
|------|------|
| `V9_PROBE_KERNEL_CYC` | 4-phase pcycle 拆分（kernel/desc/table/setup），写 out_buf[0..15] |
| `V9_PMU_PROBE` | PMU counter 拆分（pkt_ANY/T0/insts/disp + cyc，kernel + whole-op），写 out_buf[0..31] |
| `V73DEEP_ARG{1,4,5}` | mask 描述符 args sweep |
| `V73D_{N_TILES_POW2,M_TOTAL_MINUS_STEP,K_TOTAL_BYTES,N_ACT_PAIRS,EXTRA_PARAM_0,EXTRA_PARAM_1}` | kernel 描述符 sweep |
| `V9_KERNEL_{V73_UNALIGNED,OLD_UNALIGNED,V73_BBB_NXN,V73_BBB_ALIGNED,V73DEEP_PER_M,V73DEEP_SPLIT,OLD_V73DESC}` | 替换 kernel entry |

`V9_PROBE_KERNEL_CYC` + `V9_PMU_PROBE` 都和 `V73DEEP_*` 互斥（kernel call 处只插一段探测代码）。需要分别 build/run。

## Marker 字节布局（V9_PMU_PROBE）

写到 `out_blocks[0]` 的前 32 字节，全部小端 uint32：

```
[ 0..3 ]  any_kernel    - PMU COMMITTED_PKT_ANY delta around kernel call
[ 4..7 ]  t0_kernel     - PMU COMMITTED_PKT_T0 delta (opcode 0x0a — 实际是不是 thread 0 待确认)
[ 8..11]  insts_kernel  - PMU COMMITTED_INSTS delta (opcode 0x25 — 实测返回 0，opcode 可能错)
[12..15]  disp_kernel   - PMU DISPATCHED_PKTS delta (opcode 0x2f — 实测返回 0，opcode 可能错)
[16..19]  cyc_kernel    - C15:14 pcycle delta around kernel call
[20..23]  any_op        - PMU COMMITTED_PKT_ANY delta whole op
[24..27]  t0_op         - PMU COMMITTED_PKT_T0 delta whole op
[28..31]  cyc_op        - C15:14 pcycle delta whole op
```

经过 UntileToRowMajor + Reshape，前 32 字节落在 `out.raw` 的前 32 个 fp32 cell（u8 round 后）。

## 用法实例：256³ V73DEEP gap 实测（2026-04-28）

```bash
# 1) baseline 跑出来
EXTRA_DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST" \
    bash example/hmx_matmul_phase3/build.sh
# (同样的 EXTRA_DEFS 跑 build_x86.sh)
WT_LAYOUT=kmaj CHAIN=8 \
    OUT_DIR=example/hmx_matmul_phase3/standard_flow/phaseB_v8/phase1_validation/baseline \
    bash example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v8c8_chain.sh

# 2) compare 一行
source scripts/env.sh
python3 scripts/perf_v8c8.py example/.../phase1_validation/baseline \
    --compare example/hmx_matmul_phase3/standard_flow/phaseA_native/s256_chain8_compare

# 输出：
#  chrometrace (BbbKMajor): n=8  hot avg dur=2808  pkts=747  cpp=3.76
#  bit-exact: 100.00%
#  chrometrace (ConvLayer_s1.opt): n=8  hot avg dur=1120  pkts=346  cpp=3.24
#  GAP: ours/native  cyc=2.51×  pkts=2.16×  cpp_ratio=1.16×
```

## 附录 — PMU event opcodes

来自 `tools/hexagon-sdk/libs/itrace/inc/itrace_dsp_events_pmu.h`（itrace 0x80XX
ID 的低字节 = 真实 PMU opcode；与 `examples/itrace/src_app/advanced_run.c`
注释验证一致）：

| opcode | 名字 |
|-------:|------|
| 0x03 | COMMITTED_PKT_ANY ★ 主用 |
| 0x04 | COMMITTED_PKT_BSB |
| 0x07 | COMMITTED_PKT_B2B |
| 0x08 | COMMITTED_PKT_SMT |
| 0x0a | COMMITTED_PKT_T0 (待验证) |
| 0x10 | ICACHE_DEMAND_MISS |
| 0x11 | DCACHE_DEMAND_MISS |
| 0x25 | COMMITTED_INSTS (待验证 opcode) |
| 0x2f | DISPATCHED_PKTS (待验证) |
| 0x3f | AXI_LINE128_READ_REQUEST |
| 0x42 | AXI_WRITE_REQUEST |
| 0x50 | COMMITTED_FPS |
| 0xCC | HVX_ACTIVE (9-bit, 不能塞 8-bit PMUEVTCFG slot) |

`PMUEVTCFG` 32-bit 寄存器布局：低字节 → cnt0、次字节 → cnt1、… 4 个 8-bit slot。
`PMUCFG=0x400`（bit 10 必须设）才会真正 enable counting。`qurt_pmu_enable(1)`
之后还要写 `PMUCFG=0x400` 才出数。
