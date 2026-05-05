# 06 调试和性能怎么看

HMX custom op 出错或变慢时，先把问题分层：

```text
data preparation
descriptor ABI
HMX compute body
QHPI / profiling boundary
```

不要一上来就怀疑所有层。越底层的东西越难改，越应该先排除上层输入组织问题。

## 先看正确性

当前 custom MatMul 的基本正确性信号是：

```text
bit-exact: 65536/65536
```

如果不 bit-exact，按这个顺序查：

```text
1. weight packing
2. folded bias record
3. activation/output Crouton pointer table
4. M_t/N_t/K_t tile count
5. descriptor field unit
6. mask state
7. owned body bytes
```

`HMX_U8I8_DESC_DUMP` 是最有用的第一层诊断宏。它会把 descriptor 和 mask 的派生状态写进 output block，并跳过真正 compute。这样可以先确认 wrapper 有没有给错 ABI。

## 再看性能数字

当前主要看三个指标：

```text
dur / cycles
  HTP timeline 上的最终耗时

pkts
  执行了多少 Hexagon packet，用来判断是不是多做了工作

cpp
  cycles per packet，用来观察 stall、locality、issue 效率
```

对当前 256 chain8：

```text
custom hot op:       1165 cycles / 337 packets / cpp 3.458
native kernel-only:  1140 cycles / 346 packets / cpp 3.296
native aggregate:    1452 cycles / 451 packets / cpp 3.220
```

直接解释就是：

```text
custom 没有比 native kernel-only 多跑 packet
custom 比 native aggregate 更快
剩下 +25 cycles 不是额外 MatMul 计算
```

## probe 怎么切开时间

打开 probe：

```bash
EXTRA_DEFS=-DHMX_U8I8_PROBE_CYCLES \
  bash example/qnn_hmx_matmul_u8i8/build.sh
```

解码：

```bash
python scripts/perf_hmx_u8i8_matmul.py <out_dir> --probe-cycles
```

当前拆分：

```text
kernel = 1074 cycles
desc   =   29 cycles
table  =    0 cycles
qhpi   =    0 cycles
```

probe mode 会向 output 写 marker，所以 bit-exact 会失败。这是预期行为。

## 怎么解释 gap

如果 `pkts` 高：

```text
大概率是多做了工作
重点查 lookup、copy、pack、patch、branch
```

如果 `pkts` 不高但 `cycles` 高：

```text
大概率是 locality、issue、alignment 或 framework boundary 成本
重点查 descriptor 位置、VTCM/TCM 访问、profiling 边界
```

如果 native kernel-only 看起来明显更快：

```text
一定要同时看 native aggregate
native 可能把准备工作藏在 graph-load 或 sidecar HTP event 里
```

这也是本项目最重要的经验：不能只盯最终 HMX event，要弄清楚哪些工作被谁做了、算在哪个 profile event 里。

## 已经试过但不该反复重开的方向

没有新证据时，不要重新打开这些分支：

```text
old row-major kernel
dlsym/native-kernel swap path
VTCM scratch descriptor/mask/extra storage
precomputed full descriptor record
static mask/extra variants
kernel alignment above 64 bytes
explicit table dcfetch as production path
PMU-heavy probe as normal measurement
```

当前保留的有效改动是：

```text
QHPI precompute
pointer-table copy in precomputed_data
mask pre-initialization
64-byte aligned stack descriptor
small extra_param[2]
owned V73DEEP inline asm
```

## 常用命令

Build：

```bash
bash example/qnn_hmx_matmul_u8i8/build.sh
bash example/qnn_hmx_matmul_u8i8/build_x86.sh
```

Run：

```bash
OUT_DIR=example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/out/u8i8_chain_256 \
DEVICE=oneplus M=256 K=256 N=256 CHAIN=8 MODE=chain \
bash example/qnn_hmx_matmul_u8i8/standard_flow/custom_u8i8/run_u8i8_chain.sh
```

Compare：

```bash
python scripts/perf_hmx_u8i8_matmul.py <custom_out_dir> \
  --compare <native_out_dir>
```
