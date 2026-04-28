# Next Steps — V73DEEP → Native Gap @ 256³

## 当前状态（2026-04-28 EOD）

| Path | dur (cyc) | pkts | cpp | bit-exact | vs native |
|------|----:|-----:|----:|:---:|:---:|
| **V73DEEP + Lane A v2 + HVX**（当前最佳） | **2808** | **747** | 3.76 | 100% | **2.51× cyc / 2.16× pkts** |
| Native q::ConvLayer_s1.opt | 1120 | 346 | 3.24 | ref | 1.0× |

**测量方法已固化**：`scripts/perf_v8c8.py <run> --compare <native>`，详见
`docs/v8c8_perf_reading_method.md`。

## 已确认（不要再重做）

### 已 ruled out（empirical sweeps）

| 方向 | 试过的范围 | 结论 |
|------|------------|------|
| mask args (set_hmx_params_conv1x1) | arg1 ∈ {0x700, 0x70b, 0x73b, 0x780}; arg4 ∈ 0..0x40; arg5 ∈ {0x20, 0x21, 0x60, 0xa0, 0x120} | 全 PASS 但 pkts 永远 747 |
| 描述符字段缩小 | n_tiles_pow2 / k_total_bytes / n_act_pairs ↓ | pkts 等比下降，输出覆盖率也等比下降，无 magic 组合 |
| m_total_minus_step ↑ | 16, 32 | 反而 K 多算一遍，PASS 但 pkts 上涨 |
| VTCM 显式 8MB | vtcm_mb=8 | 无变化 |
| 替换 kernel entry | unaligned / NxN-bbb / OLD-unaligned | SIGSEGV (ABI 不同) |
| 多 call 切分 | 4 calls k_total=64 / 8 calls per-M-tile | 反而更慢（call 开销 dominant） |
| OLD kernel 单 call + V73DEEP 描述符 | — | crash |
| V73 non-deep + Lane A v2 | — | 1.74× 慢于 V73DEEP |

### 已 verified

- 同一个 dlsym kernel symbol：`hmx_v73_convbbb1x1deep_stride1` (0x2ebe40)
- 我们走 path 1（n_act_pairs > 1, extra_param[0] == 1）— 最 dense 的路径
- 我们的 mask_desc 字段值与 native 已通过 V9_PARAMS_PROBE 对齐（`Agent/qnn_re/set_hmx_params_conv1x1_probe_2026-04-28.md`）
- chrometrace `pkts` ≈ PMU 真实 committed packets（同方向同量纲）

### 已知 gap 不在哪里

- 不在 mask 描述符
- 不在我们写的 out_desc / act_desc loop 字段
- 不在 wrapping（已经 ~50 pkts，Lane A 砍到底了）
- 不在 cpp（4.0 vs 3.24，差异只占 16%；剩余 84% 在 pkt count）

## 下一步：Wrapper 静态 RE（option B）

### 目标

破解 `libQnnHtpV75Skel.so::wrapper_at_0x3dc2a8`（u8×i8 case 的 ConvLayer_s1.opt
入口）的描述符构造逻辑，找出它在 256³ 时给 kernel 传的实际参数。

### 关键 entry

| 地址 | 名字 | 说明 |
|------|------|------|
| 0x3d7620 | `code_to_type_name<...pkWeightsF16_TCM>` | FP16 weights 的 wrapper（不是我们的 case，但是结构参考） |
| **0x3dc2a8** | （u8 case wrapper，估计） | 调 0x3dc440 → 0x2eadc0 v73 dispatcher |
| 0x3d7920 | descriptor builder | 填 12-field param struct at r16 |
| 0x2eadc0 | hmx_v73_convbbb1x1_stride1 dispatcher | 检查 mask flag → 0x2ebe40 deep |
| 0x2ebe40 | hmx_v73_convbbb1x1deep_stride1 | 实际 kernel body |

### 关键已 RE 出的事实

`wrapper_at_0x3dc2a8` 在 0x3dc440 处 call kernel，**外层有 loop**：
```
3dc4a0  r24 = add(r24, #1)        ; loop counter ++
        memw(r29+#0x40) += r26     ; advance some pointer by r26
        memw(r29+#0x28) = r2.new   ; advance another pointer by r25*4
3dc4a8  cmp.eq(r24, r23); if (!eq) jump 3dc394   ; loop while r24 != r23
```

每次 call 用 **不同的 (act_blocks, out_blocks, wt_pack, bias) 子集**。

### 待解的问题

1. **`r23` (loop trip) 在 256³ 时是多少**？
   - 候选：M_t=8（按 M-tile 切）/ N_t=8（按 N-tile 切）/ 其它
2. **`r25`（per-iter pointer stride）** 怎么算的？
   - 提示：源自 `r6=r7*r6=lsr(memw(arg1[0x8]+0x1c),3) * lsr(memw(arg1[0x8]+0x20),5)`，再 `r25 = mpyi(r6, r3)`
3. **`r26`（per-iter accumulator advance）** 怎么算的？
4. **每次 call 的 `n_tiles_pow2 / k_total_bytes` 是什么值**？
   - 重要：如果每次 call 描述符更小（比如 `n_tiles_pow2=8`），单次 call 的 pkts ~ 100-200
   - 8 次 × 50 pkts = 400 ≈ native 的 346 ✓ 量纲对得上

### Static RE 步骤

1. 完整反汇编 `0x3dc2a8 → 0x3dc4a8` 整个 wrapper（约 ~2KB 代码）
   ```bash
   $HEXAGON_TOOLS_ROOT/bin/hexagon-llvm-objdump -d \
       --start-address=0x3dc2a8 --stop-address=0x3dc4b0 \
       $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so \
       > Agent/qnn_re/wrapper_3dc2a8.S
   ```
2. 同步反汇编 0x3d7920 descriptor builder（已有 `Agent/qnn_re/descriptor_builder_3d7920.S`）
3. 自下而上 trace 每个寄存器的来源（focus on r23, r25, r26 at 0x3dc4a0）
4. 每个寄存器追到 caller-supplied tensor descriptor 的某个字段偏移
5. 实际 256³ 的 act/wt/out tensor descriptor 字段值需要从 ctx-binary 或 runtime
   probe 拿（option A 备用）

### Risks

- wrapper 入口 args 可能依赖 ConcreteTensor C++ 对象的 vtable 调用（看到 `callr r2` at 0x3d7b1c），**虚函数会让静态 trace 断**
- caller-context 的 tensor object format 我们不知道；可能要再翻 `libHtpPrepare.so` 才能看见上层
- 即使 RE 出 r23/r25/r26 公式，套到我们 op 里**调用同一个 kernel** 但每次 call 仍要付 ~30 pkts prologue（= 8 × 30 + 256 work = 496 pkts，比 747 好但比 346 还差）
- 真要逼到 346 pkts，可能必须 **HMX state 在 call 之间共享** —— 这要 **patch kernel binary** 跳过 prologue（option A 范畴）

### 备用：Option A（runtime hook）

如果 option B trace 不通（被 vtable 阻断 / 字段语义猜不出），就上 runtime hook：

1. 在 op-pkg 里 dlopen libQnnHtpV75Skel.so，dlsym `hmx_v73_convbbb1x1deep_stride1`
2. 用 QURT API（`qurt_mem_region_attr_set`）把目标函数 .text 页改可写
3. 写 trampoline：保存 r0..r5 + 描述符 64B + extra_param 64B 到 buffer
4. 改 kernel 第一个 packet 跳到 trampoline
5. 跑一个 graph：先 native q::MatMul 再我们的 op；native 的 kernel call 经过 trampoline 被 dump
6. 我们的 op 读 buffer，把字节贴到 V73DEEP 调用点

工作量 1-2 天。风险：QURT 可能拒绝 mprotect .so 的 .text；trampoline 在 Hexagon
asm 里要小心 ABI（caller-saved regs / R31 链）。

## 推荐执行顺序

1. **Option B static RE** — 先做（半 session）：
   - 反汇编 `wrapper_3dc2a8.S`
   - 找 r23/r25/r26 公式
   - 看公式有没有"机器自己看输入算出来"的简单形式
   - 如果有，套进 V9_KERNEL_V73DEEP_NATIVE_LOOP 实现 + 测
   - 如果不行（vtable 拦路 / 公式依赖未知字段），转 A
2. **Option A runtime hook** — option B 失败后做（1-2 天）

## 备选退路

如果 option A/B 都黄了 / 投入产出不划算：
- **接受 2.51× gap，转去做 ≥2048³ 多实例切片**（NEXT_STEPS Step 6 路线）
  - 已有 V8 4096³ M_TILE=128 = 4.6× 加速的成功经验可移植
  - 需要：把 `gen_v8_graph.py M_TILE=128` 配方 port 到 `gen_v8c8_chain.py`
  - 256³ 单 instance 的 2.51× 在大 shape 下被多实例 overlap 摊薄

## 测量速查（不要再凭印象）

```bash
# 1. build + run
cd example/hmx_matmul_phase3
EXTRA_DEFS="-DV9_USE_NATIVE_KERNEL -DV9_NATIVE_SINGLE_CALL -DV9_NATIVE_V73DEEP -DV9_C8_ALIGNMENT_TEST" \
    bash build.sh && \
EXTRA_DEFS="..." bash build_x86.sh
WT_LAYOUT=kmaj CHAIN=8 OUT_DIR="$(pwd)/standard_flow/phaseB_v8/phase1_validation/<name>" \
    bash standard_flow/phaseB_v8/run_v8c8_chain.sh

# 2. compare
source ../../scripts/env.sh
python3 ../../scripts/perf_v8c8.py \
    standard_flow/phaseB_v8/phase1_validation/<name> \
    --compare standard_flow/phaseA_native/s256_chain8_compare
```

输出格式：
```
GAP: ours/native  cyc=X.XX×  pkts=X.XX×  cpp_ratio=X.XX×
```
