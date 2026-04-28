# Next Steps — Closing V73DEEP → Native Gap @ 256³ (2026-04-28)

## 当前状态

| Path | cyc | packets | cyc/pkt | vs native |
|------|-----|---------|---------|-----------|
| V8C8 inline (V9_KERNEL_HMX) | 8033 | 1959* | 4.55 | 7.6× |
| V73 dlsym (non-deep) | 5436 | 1537* | 3.53 | 5.1× |
| **V73DEEP (今天 win)** | **3360** | **1107** | **3.22** | **3.16×** |
| Native q::ConvLayer_s1.opt | 1063 | 346 | 3.08 | 1.0× |

`*` 来自 pickup/memory 之前的测量。

## ★ Step 1+2+ Lane A/B 测量结果

### Step 1+2 (已完成)

V73DEEP 256³ chain hot 内部分解（pcycle 实测）:

| Component | cyc | est. packets | % of (V73DEEP-Native) gap |
|-----------|-----|--------------|--------------------------|
| **C wrapping** (qhpi calls + act/out table build) | **1132** | ~351 | **49%** |
| **HMX kernel body** (`hmx_v73_convbbb1x1deep_stride1`) | **2228** | ~692 | **51%** |
| **V73DEEP total** | **3360** | 1107 | — |
| Native (whole op, includes its own setup) | 1063 | 346 | — |

**Native s256_chain8_compare 实测确认 hot=1063 cyc / 346 pkts / 3.08 cyc/pkt**（chain instances 2..7 稳态）。

两半各贡献约一半 gap：减半 C wrapping ~ 减半剩余 packets，收益相近，要并行做。

### Lane A 第一轮已完成 ✓

restructured table-building loop (rg outer, inb middle, kt/nt inner; pre-load act_src/out_src base ptrs per rg, avoid div/mod):

| Stage | Total cyc (chrometrace) | Kernel pcycle | Pre-kernel pcycle | vs native |
|-------|--------------------------|---------------|-------------------|-----------|
| Pre-Lane A | 3360 | 2228 | 1132 | 3.16× |
| **+ Lane A loop opt** | **3013** (-10%) | **1846** | **1826** | **2.84×** |

(pcycle 和 chrometrace dur 单位不完全一致 — pcycle 是 Hexagon HW 处理器周期, chrometrace dur 单位待考证, 不直接相加。)

Pre-kernel 1826 pcycle 占比约 50%, 跟 kernel 半半。要再缩 wrapping 必须深度优化 (HVX vectorize table-building 或重写 op-pkg 接口绕过 qhpi 部分调用)，单纯算法改动效果有限。

### Lane B 第一轮 — DEAD-END

`hmx_v73_convbbb1x1deep_stride1_sparsity` 跑了, **device crash (SIGSEGV)** —— sparsity 变体需要 sparse-format 字节 (compressed mask + values), 我们 dense wt 直接喂会读越界。Disasm 也显示 sparsity 变体的 loop 结构跟 non-deep 相同 (`:cm` act 而不是 `:deep:cm`, 1 drain/loop1 而不是 2 drain), 即便能跑也不会减少 packet count。

### 目前格局

- V73DEEP at 256³ chain hot: **3013 cyc** (pre-Lane A 3360, savings 10%)
- vs V73 (5436 cyc): **1.80×** 加速
- vs V8C8 inline (8033 cyc): **2.67×** 加速
- vs native (1063 cyc): **2.84× off**, 还差近 3 倍。

剩余 gap source 已定位：
1. C wrapping ~1826 pcycle (~50% gap)
2. HMX kernel 多用 packets ~692 vs native ~346 (~50% gap)

## 已排除

- ❌ **graph 调度 / multi-instance overhead**: native ConvLayer_s1.opt 单 op 1063 cyc 总共，**没有任何 graph-level overlap 可言**。差距 100% 在 kernel-internal。
- ❌ **多 HMX core 并行**: SoC 单 HMX，confirmed by user。
- ❌ **cyc/pkt 差异**: 3.22 vs 3.08，基本对齐 HW ceiling，不是瓶颈。

## 待解之谜

**Disasm 计算的 V73DEEP body packet 数 vs 实测**:
- 按 disasm 数（loop1 = 12 MAC + 5 drain = 17 pkts，M_t=8 loop1，4 outer = 568 + ~30 entry = ~600 pkts）
- 实测: 1107 pkts
- **多了 ~500 pkts，来源未知** — 可能是 C-wrapping、可能是 kernel body 还有我没数到的分支

**Native 346 pkts 怎么来的** — 比我们 deep 估算 600 还少 1.7×。可能用更高级 fanout 变体、sparsity 变体、或完全不同的 kernel 入口。

---

## 计划（按 ROI × 确定性排序）

### Step 1: 量 C-wrapping vs kernel body 占比 ★ 必做先
**Effort**: 30 min  
**Why**: 1107 pkts 里如果 C-wrapping 占 200-400, 真实 kernel body 就 ~700-900, 跟 disasm 估算 ~600 接近。如果 wrapping 占很小，那 kernel body 自己就 1100 pkts, 比 disasm 估算多近一倍 → kernel 内部有我没数到的 packet。

**做法**: 在 `HmxMatMulV9SkelOp.cpp` V73DEEP 调用点前后插 `Q6_R_pcycle_R()`（hexagon 提供的 pcycle 计数器，已用过），用一个未用的 byte 输出存差值：
```c
uint64_t cyc_before = Q6_R_pcycle_R();  // or HEXAGON_V62_PERFCOUNT_R(c)
hmx_v73_convbbb1x1deep_stride1(&od, &ad, wt_pack, bias_bytes, md, extra_param);
uint64_t cyc_after = Q6_R_pcycle_R();
// stash (cyc_after - cyc_before) into out_buf[16..23] for retrieval
```
然后跑 chain=8 256³，pull 输出读出每个 chain 实例的 kernel-only cyc，跟 chrometrace 的总 cyc 对比。

**结论**:
- 如果 kernel-only ~ 2400-2700 → C wrapping ~700-900 cyc，那是 op-pkg 这边的优化点
- 如果 kernel-only ~ 3000+ → wrapping 不占大头，gap 真在 HMX kernel 内

### Step 2: 拉 native ConvLayer_s1.opt 256³ optrace ★
**Effort**: 30 min  
**Why**: 确认 1063 cyc / 346 pkts 数字。可能 native 把某些前后处理算在 ConvLayer 里也可能不算。看 events tree 有无子事件、`Cycles per Packet`、是不是真单一节点。

**做法**: `phaseA_native/s256_w8a8/` 已有 ctx-binary 和 profile。如果没 optrace decode 过，跑：
```sh
cd example/hmx_matmul_phase3/standard_flow/phaseA_native/s256_w8a8
LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang \
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
    --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so \
    --input_log device_out/qnn-profiling-data_2.log \
    --schematic model_schematic.bin \
    --config /tmp/optrace_config.json --output optrace.txt
```
看 `chrometrace.json` 里的 ConvLayer_s1.opt event：duration、packets、有无 sub-events、是否被多个 op type 共享 thread。

### Step 3: 试 sparsity variant ★
**Effort**: 45 min  
**Why**: 0x2ebb00 的 `hmx_v73_convbbb1x1deep_stride1_sparsity` 才 820 B（vs `_deep_stride1` 1132 B = 28% 更小）。如果 inner loop 紧凑、无条件分支少，packet 数可能少 ~30-50%。Native 在 wt 不全零但有规律时可能优先选 sparsity。

**做法**:
1. dlsym `hmx_v73_convbbb1x1deep_stride1_sparsity` 入口（已在 disasm 里看到 0x2ebb00）。Symbol 名应该类似 `_Z..._sparsity_...` —— 跟现有 entry 一致的方法。
2. Forward declare in `HmxMatMulV9SkelOp.cpp`。
3. 加 `V9_NATIVE_V73DEEP_SPARSITY` build flag, 调它代替 `_deep_stride1`。其他参数同 V73DEEP（mask 0x70b, arg5=0x20, K-major wt, extra_param={1,0,...}）。
4. 跑 256³ chain=8。看 (a) bit-exact (b) packet count。
5. 如果 bit 不一致, 看 sparsity 变体可能需要额外 sparse-mask 或 wt 不同 pack。disasm `hmx_v73_convbbb1x1deep_stride1_sparsity` 的入口看它额外读什么参数。

### Step 4: Re-RE 9 个 caller，全部 args
**Effort**: 60 min  
**Why**: 上一轮我们只看了 `arg1` (0x700 vs 0x70b)。可能 native 还在 arg2/arg3/arg4 传特定值。我们当前传 `(0x70b, 0, 0, 0, 0x20)` 只对了 arg1+arg5；arg2/3/4 是不是 0 不确定。

**做法**:
1. 重新 disasm `libHtpPrepare.so::0xd998f0` 的 9 个 caller（已识别在 0x14f0a00..0x14f4500）。
2. 对**每个 deep call site (用 0x70b)** 提取 arg2-4 的 register source（rdx, rcx, r8d）, trace 回去看是从 stack 还是 reg 来的、最终值。
3. 找 256³ 时 native 实际传的那一套，跟 (0x70b, 0, 0, 0, 0x20) 对比。
4. 测试不同 args 组合，看 bit-exact 和 packet count 怎么变。

**风险**: arg2-4 可能是 shape-dependent（不是常量），那需要按 256³ 输入参数推算。set_hmx_params_conv1x1 disasm 可以告诉我们这些 arg 怎么用：
- arg2 (edx): `lea -0x1(%rdx),%ebx; shr $0x2,%ebx; and $0x7,%ebx` → arg2 编码某个 K 相关 count，再 `cmp $0x21,%edx` 跟 0x21 比。
- arg3 (ecx) → `mov %ecx,%r10d` → 后面 `mov %r10d,%eax; shl $0x5,%eax; and $0x7e0,%edx; mov %edx,(%rdi)` → arg3 << 5 写到 mask[+0x00] (out_check)
- arg4 (r8d) → `neg %r8d; and $0x7,%r8d; ...; shl %cl,%r8d; and %r14d,%r8d` → 影响 mask[+0x08]
- arg5 (r9d) 已知，写到 mask[+0x30]

### Step 5: 找其他没识别的 kernel 变体
**Effort**: 30 min  
**Why**: 我们已知的变体在 0x2ea740-0x2ec2ac 范围。可能 v75 SoC 还有另一组没看到的（比如名字不带 v73 前缀的 v75 专用）。

**做法**: 
```sh
hexagon-llvm-objdump --dynamic-syms libQnnHtpV75Skel.so | grep -iE "conv|bbb|hmx|matmul" | sort
```
看有没有 `hmx_v75_*` 或 `hmx_*1x1*v76*` 之类的。

### Step 6 (LATE): Multi-instance graph split for ≥2048³
**Effort**: 高（半 session）  
**Why**: 单 instance 的 V73DEEP 已经搞定，把 V73DEEP + multi-instance 切片应用到 ≥2048³ 是下一波 wins。但 256³ gap 没全收齐前别开这条。

**做法**: port `gen_v8_graph.py M_TILE=128` recipe 到 `gen_v8c8_chain.py`。需要先把 `HmxMatMulV9SkelOp.cpp` 的 shape table（4 处 cap 1024³）扩到 ≥2048³ + `act_tbl_all[1024]` → `[4096]`。

---

## 推荐执行顺序

**Step 1 + 2 已完成 — 见上面表格。**

### 接下来并行干 (高 ROI)：

**Lane A (减 wrapping)**: cache act_tbl_all + 跳过冗余 qhpi 调用 → 目标 1132 → ~300 cyc
- 第一次调用建 act_tbl_all/out_tbl_all 后存到 static buf
- 后续调用只更新跟 instance 有关的指针（输入 act_blocks 不变, 不用重建; 不过 chain 模式每个 BbbKMajor 输入是不同的, 所以这个 cache 不能 across-instance, 只能 per-instance）
- 实际上每个 BbbKMajor 都拿 4 次 qhpi_tensor_block_table 和 _length。可能可以缓存 length（一定）。
- Disasm 我们的 op-pkg 看 1132 cyc 具体花在哪（pcycle 在更细粒度上插）。

**Lane B (减 kernel body)**: 试 sparsity variant — Step 3 (45 min)
- dlsym `hmx_v73_convbbb1x1deep_stride1_sparsity` (0x2ebb00)
- replace V73DEEP call site, 同样 mask args + K-major wt + extra_param
- 测 (a) bit-exact (b) packet count
- sparsity body 820 B 比 deep 1132 B 小 28%，inner loop 紧凑些

如果 Lane A 砍掉 800+ cyc + Lane B 把 kernel body 减半到 ~1100 cyc → V73DEEP 总 ~1400 cyc, 离 native 1063 还 1.32×, 已基本追平。

### Step 4 (兜底): RE 完整 args
如果 Lane A/B 没显著效果，回去 RE caller 全部 args （不只 arg1 和 arg5）。

### Step 5 (cheap explore): 找其他 kernel 变体
30 min 工作。

### Step 6 (LATE): Multi-instance ≥2048³
等 256³ 真追平再开。

---

## 文件状态（dirty/untracked，等 commit）

```
modified:   example/hmx_matmul_phase3/src/HmxMatMulV9SkelOp.cpp           (+150/-49)
modified:   example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_v8c8_chain.py  (+33/-7)
modified:   example/hmx_matmul_phase3/standard_flow/phaseB_v8/run_v8c8_chain.sh  (+6/-2)

新增 (untracked):
  Agent/qnn_re/hmx_v73_convbbb1x1_stride1_2eadc0.S
  Agent/qnn_re/hmx_v73_convbbb1x1deep_stride1_2ebe40.S
  Agent/qnn_re/v73deep_analysis_2026-04-28.md
  Agent/qnn_re/v73deep_wt_layout_DECODED_2026-04-28.md
  example/hmx_matmul_phase3/standard_flow/phaseA_native/gen_marker_wt_256.py
  example/hmx_matmul_phase3/standard_flow/phaseA_native/s256_marker_K/   (~110KB ctx-binary)
  example/hmx_matmul_phase3/standard_flow/phaseA_native/s256_marker_N/   (~110KB ctx-binary)
```

V73 (1.48× win) 和 V73DEEP (2.39× win) 都未 commit. user 决策什么时候 commit。
