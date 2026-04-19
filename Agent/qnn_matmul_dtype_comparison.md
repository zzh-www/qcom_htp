# QNN MatMul 在 HTP v75 上的 dtype 速度对比

**测量环境**: 32×32×32 / 128³ / 256³ / 512³ 方阵 MatMul,SM8650 (Pineapple) cDSP v75,QNN 2.45.0,`pd_session=unsigned`,`perf_profile=burst`。数据源:每次 QNN inference 后 optrace 自动生成的 `chrometrace_qnn_htp_analysis_summary.json`(简称 QHAS)。

**重要:正确的 metric 是 `timeline_cycles` / `graph_execute_us`**,不是 `matmul_1:OpId_17 cycles`(后者只计某个资源上 matmul op 的 cycles,漏了并行的 HVX 工作,导致 dtype 比较失真)。脚本:`example/qnn_matmul_profile/parse_qhas.py`。

## 实测数据(QHAS 提取)

### timeline cycles(整个 graph 的 wall-clock 周期)

| config | 32³ | 128³ | 256³ | **512³** | 32³→512³ |
|--------|-----|------|------|----------|----------|
| fp16   | 21,037 | 33,455 | 59,408 | **169,382** | 8.1× |
| w16a16 | 26,351 | 42,732 | 125,739 | **677,092** | **25.7×** |
| w8a16  | 23,192 | 35,369 | 59,623 | **108,573** | 4.7× |
| w8a8   | 27,075 | **30,808** | **34,532** | **66,513** | **2.5×** |

### graph_execute_us(微秒)

| config | 32³ | 128³ | 256³ | **512³** |
|--------|-----|------|------|----------|
| fp16   | 31  | 47   | 80   | **216**  |
| w16a16 | 38  | 59   | 161  | **581**  |
| w8a16  | 34  | 50   | 79   | **141**  |
| w8a8   | 39  | 44   | 48   | **89**   |

### 512³ 排名:**w8a8 > w8a16 > fp16 > w16a16**

| rank | config | graph_execute_us | inf/sec |
|------|--------|------------------|---------|
| 1 | **w8a8** | 89us | **1209** |
| 2 | w8a16 | 141us | 1116 |
| 3 | fp16 | 216us | 1057 |
| 4 | w16a16 | 581us | 750 |

符合 HMX 硬件直觉:**u8·i8 原生阵列最快**,f16 次之,int16 最慢(尤其大矩阵下掉 HMX 路径)。

## 关键发现 1:w16a16 **HMX 高利用率 + tile 数量爆炸**(不是"掉 HMX")

最初版本写过"w16a16 在 ≥256³ 掉 HMX",依据是 QHAS 的 `type=UNK`。**这个判断是错的**。`type=UNK` 是 profiler reader 的 demux 失败,不是硬件没用 HMX。从编译后的 HTP graph(`chrometrace_htp.json`)反向验证:所有 dtype / 所有 size 的 MatMul compute 节点都带 `op_flags: "uses_hmx"`。

`example/qnn_matmul_profile/parse_qhas.py` 已修复这个 profiler bug:
- **type 修复**: 用 TID→type 映射(tid=256 = HMX,tid=512-515 = HVX;HTP v75 上稳定一致)把 UNK 替换回真值
- **I/O counter 修复**: QHAS 报 0 I/O 时,从 `chrometrace_htp.json` 的每节点 `mem_{dram,vtcm}_{read,write}` 重建

修复后的真实数据:

### HMX cycles_used(实际 HMX 工作量,数字越大说明 HMX 越忙)

| size | fp16 | w16a16 | w8a16 | w8a8 |
|------|------|--------|-------|------|
| 32³  | 8,102  | 9,393  | 9,857  | 10,497 |
| 128³ | 14,330 | 22,032 | 15,196 | 13,232 |
| 256³ | 29,207 | 94,451¹ | 24,898 | 15,200 |
| 512³ | 86,621 | **579,564¹** | 43,678 | 19,958 |

¹ 需要 TID + I/O 双 fixup 才能看到正确值

### HMX 利用率(%)

| size | fp16 | w16a16 | w8a16 | w8a8 |
|------|------|--------|-------|------|
| 32³  | 38.5 | 35.6 | 42.5 | 38.8 |
| 128³ | 42.8 | 51.6 | 43.0 | 42.9 |
| 256³ | 49.2 | **75.1** | 41.8 | 44.0 |
| 512³ | 51.1 | **85.6** | 40.2 | 30.0 |

w16a16 在大矩阵下 HMX **利用率反而最高**(85.6%),远超其他 config。但整体 wall time 还是最慢 —— 经典的 **HMX-bound**:HMX 跑得飞起,但每个 HMX cycle 做的有效功少。

### HMX compute tile 数量

| size | fp16 | w16a16 | w8a16 | w8a8 |
|------|------|--------|-------|------|
| 32³  | 1 | 1 | 1 | 1 |
| 128³ | 1 | 1 | 1 | 1 |
| 256³ | 1 | **2**  | 1 | 1 |
| 512³ | 2 | **48** | 4 | 4 |

w16a16@512 需要 **48 次 HMX 核调用**,是 fp16 的 24×。

### HMX cycles / nominal MAC(越小越好,反映硬件吞吐密度)

| size | fp16 | w16a16 | w8a16 | w8a8 |
|------|------|--------|-------|------|
| 32³  | 0.247  | 0.287 | 0.301 | 0.320 |
| 128³ | 0.0068 | 0.0105| 0.0072| 0.0063|
| 256³ | 0.0017 | 0.00563| 0.00148| 0.00091|
| **512³** | **0.000645** | **0.00432** | **0.000325** | **0.000149** |

换成 MACs / cycle:

| 512³ | MACs/HMX-cycle | 硬件吞吐评分 |
|------|----------------|-------|
| **w8a8** | **6,711** | 原生 u8·i8 阵列,接近峰值 |
| w8a16 | 3,077 | activation side 需额外处理 |
| fp16 | 1,550 | f16 单元流水深度高 |
| w16a16 | 232 | **28× 比 w8a8 差** — int16 decomp 摊薄吞吐 |

## 关键发现 2:为什么 w16a16 切 48 个 tile?——**compute kernel dims 给出答案**

实测 HMX compute event 的 Dims 字段(来自 chrometrace)揭示两种完全不同的布局:

| config | 32³ Dims | 128³ Dims | 布局类型 |
|--------|----------|-----------|---------|
| fp16   | [1,8,4,32]  | [1,8,16,128]  | **Crouton** (HTP NHWC tiled) |
| w8a16  | [1,8,4,32]  | [1,8,16,128]  | Crouton |
| w8a8   | [1,8,4,32]  | [1,8,16,128]  | Crouton |
| **w16a16** | **[1,1,32,32]** | **[1,1,128,128]** | **Flat 2D matrix** |

**Crouton** 第二维固定 8 —— 是 HMX 8-way 并行的空间展开维度,对应 HMX 里 8 个 phys_row 并发。`H*W` 被折叠进这 8 个 slot,per-tile 元素密度高。

**Flat [1,1,M,N]** 第二维塌成 1 —— 完全回到朴素 2D matmul 视角,HMX 只能按 32×32 基础 tile 算。per-tile 元素密度低,需要更多 tile 覆盖同样工作量。

### 为什么 w16a16 不能用 Crouton?HMX byte-stream 机制决定的

从之前 reverse engineer 的 HMX activation tile 字节布局:
```
A_byte(phys_row, K, stream) @ 128·phys_row + 4·K + (stream ? 3 : 1)
```
每 4 字节有效位只在 offset 1 (stream 0) 和 offset 3 (stream 1)。**两个 stream 天生是"同一 element 的低字节和高字节"** —— 这就是为 int16 byte-decomposition 设计的硬件接口:

- **fp16**:element 2 bytes,整个放进 stream 0,不拆
- **int8**:element 1 byte,打包放 stream 0/1(Crouton 空间展开)
- **w8a16**:activation 是 int16(拆成 stream 0 低字节 + stream 1 高字节),但 weight 是 int8,只单边拆 → 还能用 Crouton
- **w16a16**:activation 和 weight **都** int16 —— 两边都需要字节拆分。HMX 的 2-stream 机制只支持一边做 byte decomposition,**另一边不知道怎么对齐**。

HTP compiler 的解决方案:**放弃 Crouton,退回扁平 2D matmul 布局,手动管理两侧的字节拆分**。代价是失去 8× 空间并行,所以大矩阵要切很多 32×32 子 tile。

### tile 数爆炸的完整根因链

1. HMX 只有 `u8·i8` 原生 MAC → int16 matmul 必须双边字节拆
2. HMX 的 2-stream 机制只支持**单边** byte-decomposition(stream 0 + stream 1 = 同一 element 的 2 字节)
3. **w16a16 双边都要 byte-decompose** → Crouton 布局无解 → 退回 `[1,1,M,N]` Flat 布局
4. Flat 布局无法利用 HMX 8-way 空间并行 → HMX tile 粒度变成最小 32×32
5. 大矩阵切成很多小 tile → pipeline fill/drain 开销 × tile 数 → 慢
6. 结果: HMX util 85.6%(拼命在跑),但吞吐 0.00432 cyc/MAC(比 w8a8 差 29×)

### 每 tile HMX 使用对比(512³)

| | tile 数 | HMX cyc / tile | 总 HMX cyc | 单 tile 效率 |
|--|-------|---------------|-----------|-----|
| fp16   | 2  | **43,310** | 86,621  | 高(Crouton 打包 8×) |
| w8a16  | 4  | 10,919    | 43,678  | 中(activation side Crouton) |
| w8a8   | 4  | **4,989** | 19,958  | 极高(双边原生 u8·i8) |
| w16a16 | 48 | 12,074    | 579,564 | **低**(Flat 布局,tile 切碎) |

## 关键发现 3:profiler 的 UNK bug 触发条件

观察 w16a16 跨 size 的 QHAS:

| size | tile 数 | QHAS type | 需要 fixup? |
|------|-------|-----------|------------|
| 32³  | 1 | HMX ✓ | N |
| 128³ | 1 | HMX ✓ | N |
| 256³ | **2** | UNK ❌ | T + I |
| 512³ | 48 | UNK ❌ | T + I |

**从 2 个 HMX tile 起,profiler reader 就无法 demux** —— 同一 TID 上 2+ 个 HMX event 并发,reader 失败。fp16@512 也有 2 tiles 但 kernel name 不同(`.fp16.s1.tcm` vs `_s1.opt`),reader 对 fp16 的 2-tile case 还能处理。所以 bug 是 int kernel + multi-tile 组合触发的。

修复脚本 `parse_qhas.py` 对这类数据全自动 recovery,输出里打 `fixup` 列(T=type fixup, I=I/O fixup)。

## 关键发现 2:w8a8 大矩阵下 scaling 几乎是平的

从 32³ 到 512³,MAC 数量涨了 4096×,但 w8a8 的 timeline_cycles 只涨 2.5×(27075 → 66513)。HMX utilization 持平 30-44%,cycles_used 只涨 1.9× —— 说明 **HTP 的 u8·i8 MAC 阵列大尺寸下饱和度极高**,每 MAC 近乎 0.0005 cycle。

fp16 也 scaling 好(2.5× 32→256,然后 2.9× 256→512),HMX util 稳在 38-51%。

## 关键发现 3:`matmul_1:OpId_17 cycles` 不能用

之前(错误地)拿这个字段对比 dtype,得出"fp16 和 w16a16 在 32³ 等速"的结论。那是因为:
- `matmul_1:OpId_17` 只算 matmul op 在主资源(HMX)上用的周期
- 但 dtype 转换、tile 变换、requant 跑在**并行的 HVX tid**,不算在里面
- 不同 dtype 的 HVX 负载完全不同(见下表)

### HVX cycles_used 差异巨大(同一 matmul op 里的并行工作)

| size | fp16 HVX | w16a16 HVX | w8a16 HVX | w8a8 HVX |
|------|----------|------------|-----------|----------|
| 32³  | 12,619 | 16,486 | 13,850 | 16,386 |
| 512³ | 90,014 | 0 (fallback) | 162,263 | 107,139 |

w8a16 @ 512³ HVX 负载 162k cycles —— 比 HMX cycles_used (43k) 还多 4×。HVX 在做大量 dtype 转换。`matmul_1 cycles` 字段看不到这个,所以假象是"w8a16 matmul 很便宜"。实际上 graph_execute_us 141us 包含了 HVX 和 HMX 的 wall time。

## 历史偏差记录(供诊断参考)

之前报告过的错误 metric / 结论:

| 声称 | 实际 | 错误根因 |
|------|------|---------|
| "fp16 ≈ w16a16 at 32³" | 32³ 大家都 framework-bound,但 HVX 开销已经有差异 | metric 错(用 OpId_17 而非 timeline) |
| "w16a16 只比 fp16 慢 ~10×" | 确实如此,因为 tile 切太碎 | "tile 数量爆炸"才是对的解释 |
| "w16a16 256³/512³ 掉 HMX 路径" | **错的**。所有 tile 的 `op_flags: uses_hmx`,HMX 全程在用 | 误读 QHAS 的 `type=UNK` 当成"没 HMX",其实是 profiler demux 失败 |
| "w8a8 比 fp16 快 4.3×" | 512³ 下 w8a8 = 89us vs fp16 216us = 2.4× | 用 compute cycles 估算不准 |
| "w8a8 有双峰分布" | `matmul_1:OpId_17 cycles` 有双峰,但 `timeline_cycles` 稳定 | OpId_17 把资源等待算进去了 |

## 跑数据的方式

```sh
cd example/qnn_matmul_profile

# 单尺寸
bash profile_all.sh --shape 512,512,512 --configs "fp16 w16a16 w8a16 w8a8"
python parse_qhas.py output/   # 读 QHAS summary 出 timeline_cycles 表

# 尺寸扫描
bash bench_sweep.sh 32 128 256 512 -- --configs "fp16 w16a16 w8a16 w8a8"
# 每个子目录都有 chrometrace_qnn_htp_analysis_summary.json
```

参考数据存档:
- `example/qnn_matmul_profile/bench_data_2026-04-19/` — 32³ 5-run(旧,基于 matmul_1 cycles,有偏差)
- `example/qnn_matmul_profile/sweep_data_2026-04-19/` — 32/128/256/512 sweep(**推荐**,QHAS JSON 可以直接出真实 timeline)

## 次要说明

### QHAS 只报第一次 inference

和 chrometrace.json 一样,QHAS 的 `htp_overall_summary` 是第 1 次 inference 的快照。要看多次平均,`bench_repeat.sh` 跑 N 次然后平均 QHAS summary。

实测 fp16@512 steady-state 和 QHAS 报告的 warmup 时长,graph_execute_us 基本重合(216us vs 后续 ~150-200us),偏差 ≤30%。

### `matmul_1:OpId_17 cycles` 仍有用途

它是 HTP 给出的 per-op breakdown,适合**单 dtype 内不同大小的 scaling 分析**(因为 metric 定义一致)。但不适合**跨 dtype 对比**,因为每种 dtype 在 HMX/HVX 分工不同。
