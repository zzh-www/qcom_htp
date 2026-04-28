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

### 已知 obstacles 和绕过方案

| Obstacle | 绕过 |
|----------|------|
| wrapper 入口 args 依赖 ConcreteTensor 虚函数（`callr r2` at 0x3d7b1c） | trace 到 vtable 处停下，**记录 vtable slot 偏移**，然后从 ctx-binary 反向找它指向哪个 concrete impl（每个 tensor 类型一个，static dispatch 在 prepare 期已确定） |
| caller-context tensor object format 未知 | 反汇编 `libHtpPrepare.so` 的 ConvLayer node 构造代码（在 prepare-side），那里有完整结构体写入 |
| 即使 RE 出公式，仍要付 ~30 pkts/call prologue | **必上 Option A**：patch kernel 的 prologue 跳过（`mxclracc` + `r29 -= 0x28` + 寄存器保存 = ~8 packet，跨 call 复用） |
| QURT 可能拒绝 mprotect .so 的 .text | 备用：直接 **dlopen 一份新副本**（mmap 自己 RWX 区域），把 kernel 字节复制过来，patch 后 dlsym 到副本。Hexagon QURT 的 `mmap(PROT_READ\|PROT_WRITE\|PROT_EXEC)` on `MAP_PRIVATE` 是允许的（FastRPC PD 受信） |
| trampoline ABI 复杂 | 用 hexagon-clang 把 trampoline 写成 `__attribute__((naked))` C 函数，编译器帮我们处理 packet 边界；只用 caller-saved regs (r0..r7, r28..r31)，不踩 callee-saved |

## 执行顺序（不退）

### Phase 1: Option B static RE（半 session）

目标：拿到 native 在 256³ 时给 kernel 的 (r23 loop trip / r25 stride / r26 stride
/ per-iter descriptor) 公式。即使做不到 346 pkts，也能进一步缩 gap。

1. 反汇编 `0x3dc2a8 → 0x3dc4b0` 整个 wrapper 出 `Agent/qnn_re/wrapper_3dc2a8.S`
2. 自下而上 trace r23/r25/r26 至寄存器源头，标注每个偏移对应哪个 tensor 字段
3. 把追到 vtable 边界的 callr r2 也记下来（不能跳过就先标记）
4. 把公式套进 V9_KERNEL_V73DEEP_NATIVE_LOOP 实现，测 cyc/pkts/bit-exact
5. 测出值如果 ≤ 600 pkts，已经超过当前 747；继续 phase 2 推到 346
6. 如果 vtable 阻断 trace，跳到 phase 2

### Phase 2: Option A runtime kernel-prologue patching（1-2 天）

目标：达到/逼近 native 的 346 pkts。

1. 设计 trampoline：在我们 op 第一次跑时，把 kernel 入口的前 8 packet 替换成
   "save state once → jump to body"，body 起点重定向到 prologue 之后
2. dlopen + dlsym 拿 kernel 函数地址；用 QURT API 改 .text 页权限（首选
   `qurt_mem_region_attr_set`；不行用 mmap 副本路线）
3. 第一次 call 时执行完整 prologue 把 HMX state 装好；后续 N-1 次直接跳 body，
   省下 (N-1) × ~30 packet
4. 配合 phase 1 的 r23/r25/r26 公式，按 native 的 loop count 重复 call 同一
   修改后入口
5. 实测 ≤ 400 pkts 就算大胜利；300-360 pkts 视为对齐 native

### Phase 3: 如果 Phase 2 还差最后一截，写 V8C8 自己的 HMX inline kernel

目标：bit-exact + ≤ native pkts。

1. 抄 `hmx_v73_convbbb1x1deep_stride1` 0x2ebe40 的字节（约 1132 B），inline 到
   我们 op-pkg 的 `.text`（确保自己的 .text 是可执行的，no PD trust 问题）
2. 修 prologue：去掉 stack frame setup（我们调用 site 已经在合适 frame）
3. 直接 jump 到我们 inlined kernel body —— 0 prologue 开销，0 dlsym 开销，
   call 是 absolute jump
4. multi-call 一份 body 跑 N 次 —— 一次性付 prologue，N-1 次 reset 内部 acc state

成本：写 ~300 行 hexagon asm + bit-exact 调试 ~1 天，但每次 call 直接 ~30
packet（kernel body 本身），总 N×30 ≈ 240 packets for N=8。**这是逼到 native
346 的真正路径**。

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
