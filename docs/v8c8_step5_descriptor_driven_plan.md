# V8C8 Step 5 — descriptor-driven HMX kernel plan

**Status**: PLANNED. 详细执行计划。
**Goal**: V8C8 BbbKMajor 单 matmul cyc 8,000 → 1,500 (5.3× 加速); 256³ wall 65 µs → ~25 µs.
**Reference**: `docs/hmx_dsp_vs_descriptor_driven.md` (核心架构洞见)。

## 上下文

V8C8 Step 1+2 完成后（commits `fba391b`、`55915cb`），ONNX→DLC→ctx-binary 全套
flow 跑通，wt VTCM 字节布局与 native byte-1:1，bit-exact at 256/512/1024³。

但 chrometrace 单事件级测量（`optrace_compare_256_indep/`）暴露 V8C8 BbbKMajor 内核
比 native ConvLayer_s1.opt 慢 **5.3×**（8000 vs 1500 cyc/matmul at 256³）。根因不是
V8C8 内核 inner asm 写得差—— cyc/packet 双方都是 ~4.2 (HMX HW 节拍)——而是
V8C8 是 **DSP-driven**（每个 K-tile 软件发 mxmem packet），native 是
**descriptor-driven autonomous**（一次 HMX state-config 让 HMX 沿 M 轴自动 fan-out
8 个 tile）。

软件 MAC packet 对：V8C8 = 512，native = 64 (8× fewer, M 轴 HMX 内部扫)。

## Step 5 = 切到 descriptor-driven

### 路径 A (RECOMMENDED): dlsym 调 native `hmx_convbbb1x1_stride1`

复用 native 的整个 inner kernel。我们 SkelOp 只负责构造 0x40 字节 descriptor +
发起 dlsym 调用。HMX fan-out / autonomous tile sweep 全是 native 函数 + HMX HW
自己的事。

#### Step 5.1 — 反逆 `set_hmx_params_conv1x1` 5 个参数

**File**: `Agent/qnn_re/set_hmx_params_conv1x1.S`

ABI 推断：
```c
void set_hmx_params_conv1x1(
    void   *hmx_params,  // r0 — 0x40 byte output buffer
    uint32_t arg1,       // r1 — packed mode + count fields
    uint32_t arg2,       // r2 — N_count (output cols / 32)
    uint32_t arg3,       // r3 — output_stride or M-related
    uint32_t arg4,       // r4 — M_count or step
    uint32_t arg5);      // r5 — packed Rt mode + flags
```

需要做的：
1. 阅读 `set_hmx_params_conv1x1.S` 全部 ~70 行 disasm
2. 识别每个参数的 bit 字段（看 `extractu`、`bitsclr`、`mux` 模式）
3. 识别每个 descriptor 字段的写入位置（`memw(r0+#0x??)`）
4. 写 C 等价实现 `our_set_hmx_params_conv1x1(...)`，per-shape 调用得到 0x40
   字节 descriptor

**验证**: 跑 host x86 build, 调我们的 C 实现 + 对比 dlsym 调 native 实现, 字节级 diff。

**预期 effort**: 中等 (~半天 reverse engineering + 半天 C 实现 + 验证)。

**已有线索** (Agent/qnn_re/):
- `set_hmx_params_conv1x1.S` 完整 disasm 已存
- `descriptor_builder_3d7920.S` / `descriptor_builder_full.S` 周边代码 (内层 caller)
- `hmx_convbbb1x1_stride1_full.S` 内核 disasm — 显示了字段使用模式 (`r17 = memw(r0+#0x4)` etc.)

#### Step 5.2 — 在 SkelOp 中 per-call 构造 descriptor

`HmxMatMulV9SkelOp.cpp` 修改：
1. 在 V9_KERNEL_HMX 路径里，per-call 算出 descriptor 输入参数
   (M=K=N=S, M_count, K_count, N_count, output_stride, base addrs from VTCM)
2. 调 `our_set_hmx_params_conv1x1()` 填 0x40 字节 buffer (栈 buffer)
3. 同时构造 act/wt/bias 的指针描述符 (`hmx_conv_out_desc_t`、`hmx_conv_act_desc_t`、
   `hmx_conv_mask_desc_t`，已在源码 line 43-66 定义)
4. dlsym `hmx_convbbb1x1_stride1` 并调用

**已有基础设施**:
```c
// HmxMatMulV9SkelOp.cpp line 43-74
typedef struct {
    int32_t *out_tile_ptr_table;
    uint32_t out_table_stride_dwords;
    ...
} hmx_conv_out_desc_t;

#if defined(__hexagon__) && defined(V9_USE_DLSYM)
extern "C" void hmx_convbbb1x1_stride1(
    const hmx_conv_out_desc_t  *out_desc,
    const hmx_conv_act_desc_t  *act_desc,
    const void                 *weight_base,
    const void                 *bias_base,
    const hmx_conv_mask_desc_t *mask_desc);
#endif
```

5 个参数已知，结构体字段命名已知 (源自 `Agent/sig_hmx_convbbb1x1_stride1_2026-04-25.md`)。

#### Step 5.3 — 调通 256³ bit-exact

**Reference output**: 与 Step 2 相同 (`gen_v8c8_chain.py --mode chain` 的 ref_out)。

**对照**: 现行 V8C8 (Step 2 末态) 在 256³ bit-exact 65536/65536。Step 5 后必须
**仍** 65536/65536。

**预期问题**:
- VTCM stride / wt byte layout 必须严格对齐 native kernel 的期望
- 1024-MAC spike test (`Agent/dlsym_spike_PASS_2026-04-25.md`) 已验证 dlsym
  调用本身能跑, 但完整 256³ sweep 会暴露其他对齐约束
- 我们的 wt VTCM (Step 2 N-outer `[1, N_t, K_t, 1024]`) 与 native 期望已 byte-1:1，
  这一面应没问题
- act VTCM 是 Crouton_8 + Indirect (block_table)，native kernel 期望连续 VTCM；
  可能要先把 act block_table flatten 进连续区域 (额外 HVX copy) 或修改 sig 让
  QNN 给我们 Direct 布局

#### Step 5.4 — Sweep 验证

bit-exact 测：256, 512, 1024 (与现行同)
perf 测：chrometrace 单事件 dur 应降到 ~native level

### 路径 B (备选): 自实现 HMX state-config 指令

不调 native 函数, 直接发 mxmem-encoded 指令配置 HMX state machine。

风险：HMX state-config 编码大多无文档（`0xfd094718`、`0xe2198028`、
`0x2ea800` 处那个 16-byte packet 的字节级语义）。硬写极易踩硅级 bug。
**不推荐**。

### 路径 C (兜底): 优化现有 DSP-driven kernel

如果 Path A 受阻，仍可减少软件开销：
1. 移除 act_ptrs / wt_ptrs prebake，inline 寻址
2. 用更密集的 hardware loop0 / loop1 嵌套包更多 (mt, nt) tile
3. mxmem2 一次装多个 K-tile？(若硬件支持)

预期收益 ~12% (~1K cyc), 不接近 5× gap。仅作 Path A 失败时的 fallback。

---

## Profile 验证方法

任何 Step 5 中间态都用以下方法验证 perf, **避免 profile_text bucket 误读**。

### 方法 1: chrometrace 单事件 dur (PRIMARY)

```sh
# 跑 V8C8 chain 8 instance independent
cd example/hmx_matmul_phase3/standard_flow/phaseB_v8
M=256 K=256 N=256 CHAIN=8 MODE=independent OUT_DIR=phase1_validation/v8c8_step5_test \
  bash run_v8c8_chain.sh

# 拉 profile + decode chrometrace
ssh oneplus "cat qnn_run/phaseB_c8/out/qnn-profiling-data_0.log" \
  > phase1_validation/v8c8_step5_test/profile_dev.log

PV=$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer
LDIR=$QNN_SDK_ROOT/lib/x86_64-linux-clang
echo '{"htp_json": true, "runtrace": true, "memory_info": true}' \
  > /tmp/optrace_config.json
LD_LIBRARY_PATH=$LDIR $PV \
  --config /tmp/optrace_config.json \
  --reader $LDIR/libQnnHtpOptraceProfilingReader.so \
  --input_log phase1_validation/v8c8_step5_test/profile_dev.log \
  --schematic phase1_validation/v8c8_step5_test/ctx/v8c8_schematic.bin \
  --output phase1_validation/v8c8_step5_test/chrometrace.json

# 提取 BbbKMajor 单事件 dur (chain 1..7 平均 = steady-state)
python3 << 'PY'
import json
d = json.load(open('phase1_validation/v8c8_step5_test/chrometrace.json'))
xs = [e for e in d.get('traceEvents',[]) if e.get('ph')=='X']
seen = set()
durs = []
for e in xs:
    if 'BbbKMajor' not in e.get('name',''): continue
    args = e.get('args',{})
    qname = args.get('QNN Op Name','') if isinstance(args,dict) else ''
    key = (qname, e.get('ts'))
    if key in seen: continue
    seen.add(key)
    durs.append((qname, e.get('dur',0)))
durs.sort()
for q, dur in durs:
    print(f'  {q}: {dur} cyc')
warm = [d for q, d in durs if 'indep0' not in q and 'chain0' not in q]
print(f'\nWarm avg (skipping instance 0): {sum(warm)/len(warm):.0f} cyc')
print(f'TARGET: ~1,500 cyc (native ConvLayer_s1.opt level)')
PY
```

**判定**:
- Warm avg <= 2,000 cyc: ✓ Step 5 success (native parity)
- 2,000 < x <= 5,000: △ partial success (descriptor 部分起作用，但有未对齐字段)
- > 5,000: ✗ 仍是 DSP-driven, descriptor 没生效

**为什么不用 profile_text**: profile_text 把 ForceFormat_Crouton (HVX) 时间也归到
`bbb_indep_X` bucket，看到的不是纯 HMX kernel cyc。chrometrace 单事件 dur 才是
本 op 在 HMX thread 上实际占用时间。

### 方法 2: chrometrace `Cycles per Packet` 字段 (DIAGNOSTIC)

`cyc/pkt` ≈ 4.2 是 HMX HW 节拍下限。如果 V8C8 chrometrace BbbKMajor 事件 args 里
`Cycles per Packet` 接近 ~4.2 (与 native 同), 说明 packet 生成已优化到位; 总 dur
低 (~1500) = 5×加速达成。

`cyc/pkt` 远 > 4.2 (e.g. 10+) 说明软件 packet 之间有 stall, descriptor 模式没启动。

### 方法 3: DSP-issued packet 数估算

```python
# from chrometrace event args
total_cyc = event['dur']
cyc_per_pkt = event['args']['Cycles per Packet']
packets = total_cyc / cyc_per_pkt
# Step 5 target: ~350 packets per 256³ matmul (matching native)
# Step 2 baseline: ~1,950 packets
```

### 方法 4: 跨实例 stability 检查

8 个 chain instance 中, instance 1..7 的 dur 应该方差很小 (±5%)，证明 steady-state。
Instance 0 通常贵 (cold setup), skip 它。如果 instance 1..7 也大幅波动，可能有
HMX state contamination 或 VTCM bank conflict。

### 方法 5: bit-exact regression

Step 5 任何中间态都必须保持 bit-exact at 256/512/1024:

```sh
python3 -c "
import numpy as np
b = np.fromfile('phase1_validation/v8c8_step5_test/device_out/out.raw', dtype=np.float32)
out = np.round(b).astype(int).clip(0,255).astype(np.uint8).reshape(256,256)
ref = np.load('phase1_validation/v8c8_step5_test/v8c8.onnx.out_ref_u8.npy')
m = (out==ref).sum(); t=256*256
assert m == t, f'Step 5 broke bit-exact: {m}/{t}'
print(f'OK 256³: {m}/{t}')"
```

### 方法 6: native 平行对照

每次 Step 5 中间态测完 V8C8 后, 也跑 native 同 shape 的 indep mode 作 baseline:

```sh
cd example/hmx_matmul_phase3/standard_flow/phaseA_native
SIZE=256 CHAIN=8 MODE=independent SHARED_W=0 OUT_NAME=s256_indep_baseline \
  bash run_native_chain.sh
```

确保 native warm avg (instance 1..7) ≈ 1,500 cyc 没漂移 (HW thermal / SoC state
变化可能影响)。

---

## 风险评估

| 风险 | 概率 | 应对 |
|---|---|---|
| `set_hmx_params_conv1x1` 反逆字段错 1 个 → HMX fault | 高 | byte-level diff 我们 vs dlsym 输出 |
| Native kernel 期望 act 连续 VTCM, 我们是 Crouton_8 indirect → 需 layout 转换 | 高 | act sig 改回 Direct + 单独 ForceFormat 节点 (回到 V8 prod 模式) |
| dlsym 资源 (寄存器, VTCM offsets) 跨 ABI 边界 mismatch | 中 | 在 SkelOp 进入 dlsym 前后 dump HMX 状态比对 |
| HMX state pollution 跨 instance | 低 | 每 instance 重新调 set_hmx_params + 完整 mxclracc |
| 256³ pass 但 512/1024 fail (VTCM 容量) | 中 | per-shape 重新算 descriptor，不是常量 |

## 时间估算 (粗)

- Step 5.1 反逆 + C 实现: 0.5-1 day
- Step 5.2 SkelOp 集成: 0.5 day
- Step 5.3 256³ 调通 bit-exact: 0.5-1 day (踩 layout 对齐坑)
- Step 5.4 sweep 验证 + perf 报告: 0.5 day
- 总计: 2-3 days 实施 (假设无大坑)

## 不在 Step 5 scope 内

- Step 4 (`build_tile`): native 256³ 自己也不切 tile, 不属复刻
- Step 3 (`do_precomputation`): NOT VIABLE (架构不兼容)
- 多 dtype (w8a16, w16a16): 当前只追 u8×i8 path 的 native parity
- ≥2048³ 实现: 当前 V8C8 在 1024³ 已 bit-exact, 4096³ V9 sweep 已 done; 不在
  本 step 主线

## Deliverables

1. `set_hmx_params_conv1x1.c` 等价实现 + byte-level test (host)
2. 修改 `HmxMatMulV9SkelOp.cpp` 加 V9_USE_NATIVE_KERNEL build flag
3. 跑通 256³ bit-exact + chrometrace dur ≈ 1,500 cyc
4. Sweep 报告 256/512/1024³ V8C8/native ratio
5. 更新 `optrace_compare_256_indep/` bundle (Step 5 后版本)
6. Memory entry 标记 Step 5 完成

---

## 接手清单

下一 session 接手者：
1. 读 `docs/hmx_dsp_vs_descriptor_driven.md` 理解架构区别
2. 读本文 §"路径 A" + §"Profile 验证方法"
3. 看 disasm `Agent/qnn_re/set_hmx_params_conv1x1.S`
4. 开始 Step 5.1 反逆
