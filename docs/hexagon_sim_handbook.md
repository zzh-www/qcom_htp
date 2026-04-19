# hexagon-sim 详细使用指南

本仓库 `tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-sim` 是 Qualcomm Hexagon 架构的指令集模拟器,既能做功能验证,也能做精确的微架构/性能分析。本文聚焦实用场景,按"怎么用"而不是按"所有选项"组织。

---

## 1. 执行模式速览

hexagon-sim 有两种基本执行模式:

| 模式 | 开关 | 特点 | 适用 |
|------|------|------|------|
| Functional (fast) | 默认 | 只保证功能正确,cycle 数是粗略估计 | 回归测试、跑单元测试 |
| Timing (accurate) | `--timing` | 精确流水线、stall、cache 建模 | 性能分析、微架构调优 |

`--timing_nodbc` 是 timing 模式但禁用 data-backed cache,进一步接近硬件。

**经验**: 做性能数字报告前,所有测量必须加 `--timing`,否则 Pcycles 没有参考意义。

---

## 2. 最小可运行命令

### 2.1 裸机 / 独立程序(QuRT 风格)
```bash
hexagon-sim --mv75 -- my_program
```

### 2.2 H2 Hypervisor 引导(需要 HMX、VTCM 的程序)
```bash
hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    -- tools/h2-install/bin/booter \
       --ext_power 1 --use_ext 1 --fence_hi 0xfe000000 \
       my_program
```
- `--mv75`:目标 v75 核(对应本项目 test_hvx_hmx 用的 `-mv75` 编译选项)
- `--mhmx 1`:启用 HMX 协处理器建模
- `--simulated_returnval`:把 guest 程序的 exit code 透传给宿主 shell
- `--` 之后是 guest 命令行,第一个 token 是 booter(由 H2 hypervisor 提供),它加载并启动 `my_program`

支持的架构版本:**68, 69, 71, 73, 75, 77, 79, 81**。查看某架构可用的 core:`hexagon-sim --help --mv75`。

---

## 3. 常用选项分组

### 3.1 I/O 重定向
| 选项 | 作用 |
|------|------|
| `-O`, `--sim_out FILE` | 把 guest stdout 写到文件 |
| `-E`, `--sim_err FILE` | guest stderr 写到文件 |
| `-I`, `--sim_in FILE`  | guest stdin 从文件读 |
| `--usefs PATH`         | guest 中打开文件时的搜索根 |

### 3.2 性能计数输出
| 选项 | 作用 | 产物 |
|------|------|------|
| `--timing` | 精确 timing 模式 | 必加开关,没有产物 |
| `-S`, `--statsfile FILE` | 结构化统计摘要 | 文本,含 IPC、cache 命中、协处理器利用率等 |
| `--pmu_statsfile FILE` | PMU 硬件性能事件 | 类似 `perf stat` 的计数 |
| `--ihist FILE` | 指令直方图 | 每种指令出现次数 |
| `--packet_analyze FILE` | VLIW 包分析 | 看打包密度/slot 利用率 |

### 3.3 追踪 / Trace
| 选项 | 作用 | 备注 |
|------|------|------|
| `-t`, `--pctrace FILE`    | 仅记录控制流变化包 | 轻量 |
| `-u`, `--pctrace_min`      | 更精简的 pctrace   | |
| `--pctrace_nano`          | 最精简 pctrace     | 巨大程序用 |
| `-m`, `--memtrace FILE`    | 所有 load/store    | 体积巨大 |
| `-b`, `--bustrace FILE`    | DDR/总线流量       | 看带宽 |
| `--dcachetrace FILE`       | D-cache 活动       | |
| `--icachetrace FILE`       | I-cache 活动       | |
| `--l2cachetrace FILE`      | L2 cache 活动      | |
| `--stalltrace FILE`        | 每个 stall 的成因  | 性能调优核心工具 |
| `--coproctrace FILE`       | 协处理器 trace     | 看 HMX tile 执行时序 |
| `--uarchtrace FILE`        | commits+dcache+l2+bus 合并 trace | 最全,也最大 |

### 3.4 调试
| 选项 | 作用 |
|------|------|
| `-G`, `--gdbserv PORT` | 启动 gdbserver,挂 `hexagon-gdb` 远程调试 |
| `--interactive` | 进入内置交互式调试器 |
| `--coredump FILE` | 终止时 dump binary core |
| `--symfile FILE` | 加载可重定位文件中的符号 |
| `--debug` | 打开 debug 输出 |
| `--verbose` | 详细日志 |
| `--reconnect` | 调试器前端断开后继续运行 |

### 3.5 范围控制
| 选项 | 作用 |
|------|------|
| `--plimit N`        | 最多跑 N pcycle 就停(防跑飞) |
| `--fastforward N`   | 跳过前 N cycle 的 timing 开销,只在关心区间精确模拟 |
| `--pcfilter "A-B"`  | 只在 PC 落在 [A,B] 区间时采集数据 |
| `--timefilter_ns N` | 从第 N ns 开始采集 |
| `--bypass_idle`     | 所有硬件线程 idle 时跳过 pcycle 计数(加速) |

### 3.6 协处理器 / 扩展
| 选项 | 作用 |
|------|------|
| `--mhmx N`  | HMX 版本/开关 |
| `--archstring "..."` | 传额外参数给底层 arch lib(加引号) |
| `--archsim_opts "..."` | 传额外命令行给底层 archsim |
| `--cosim_file CFG` | Cosim 配置(加载 qtimer、l2vic 等 cosim 模型) |

### 3.7 总线 / 内存行为
| 选项 | 作用 |
|------|------|
| `--buspenalty N`   | 总线响应周期数 |
| `--busratio N`     | Core:Bus 频率比 |
| `--ahbbuspenalty` / `--ahbbusratio` / `--axi2buspenalty` / `--axi2busratio` | 对应 AHB / AXI2 总线 |
| `--memfill 0xNN`   | 内存初始填充字节(默认 0x1f) |
| `--memfill_rand 1` | 随机填充(排查未初始化读) |
| `--dsp_clock MHz`  | 模拟时钟频率 |
| `--nullptr {0,1,2}`| NULL 解引用策略:忽略/警告/报错退出 |

---

## 4. 输出怎么读

### 4.1 尾部统计(永远会打印)

```
Done!
    T0: Insns=1920635 Packets=861023
    T1: Insns=66      Packets=66
    ...
    Total: Insns=2214228 Pcycles=2982582
```

- **Insns**:执行的 Hexagon 标量/矢量指令总数
- **Packets**:VLIW 打包数。一个 packet 最多 4 条指令并行
  - `Insns / Packets` → 平均打包密度,理想接近 4
- **Pcycles**:处理器周期数
  - `Insns / Pcycles` → IPC (instructions per cycle)
  - `Packets / Pcycles` → PPC,考虑 VLIW 的更合适指标
- **T0..T5**:按硬件线程拆分,查看线程负载平衡

### 4.2 statsfile (示例片段)
`--statsfile stats.txt` 后,文件里会有类似:
```
L1D hits/misses         : ...
L2   hits/misses        : ...
HVX packets executed    : ...
HMX MAC ops             : ...
Branch mispredictions   : ...
Stalls by category      : memory / execution / resource / ...
```
这是做性能报告的主要数据源。

### 4.3 PC trace
`--pctrace trace.txt` 输出每个变更控制流的包(分支、跳转、返回),行格式大致是 `cycle pc asm`。配合 `hexagon-profiler` 或 `hexagon-llvm-profdata` 可以做基本块/函数级热点分析。

---

## 5. 推荐工作流

### 5.1 正确性回归(本仓库 `tests/test_hvx_hmx_sim.sh` 即用此)
```bash
hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    -- booter ... my_program
```
stdout grep + 与 golden 文件 diff。**不要**加 timing / trace,跑得快。

### 5.2 性能摸底
```bash
hexagon-sim --mv75 --mhmx 1 --timing --simulated_returnval \
    --statsfile stats.txt \
    --ihist hist.txt \
    -- booter ... my_program
```
看尾部 `Pcycles` 和 `stats.txt` 里的 IPC、HVX/HMX 占比。

### 5.3 定位瓶颈
```bash
hexagon-sim --mv75 --mhmx 1 --timing --simulated_returnval \
    --stalltrace stall.txt \
    --coproctrace coproc.trace \
    -- booter ... my_program
```
- `stall.txt` 告诉你哪类 stall 最多(内存等待?HVX 资源冲突?)
- `coproc.trace` 看 HMX tile 调度是否有气泡

### 5.4 热点函数 profile
```bash
hexagon-sim --mv75 --profile -- my_program
# 产出 gmon.out
hexagon-gprof my_program gmon.out | head -30
```

### 5.5 交互式调试
```bash
# 终端 A
hexagon-sim --mv75 --gdbserv 1234 -- my_program

# 终端 B
hexagon-gdb my_program
(gdb) target remote :1234
(gdb) break main
(gdb) continue
```

### 5.6 只测热点区间
用 `--pcfilter "0x40001000-0x40002000"` 或 `--fastforward N` + `--plimit M` 组合,把精确 timing 只开在关心的代码段上,前面和后面跑快速功能模式。

---

## 6. HVX / HMX 特别说明

- 必须在编译时打开: `hexagon-clang -mhvx -mhvx-length=128B -mhmx -mv75`
- 在模拟器必须: `--mhmx 1`(HMX);HVX 由 `-mv75` 的 core 自带
- HMX 依赖 VTCM,VTCM 又依赖页表/物理映射,这是为什么裸 QuRT 跑 HMX 容易失败,本仓库用 **H2 Hypervisor booter** 来提供正确的内存映射
- `--coproctrace` 能看 HMX 的 tile 级指令流,用于验证:
  - tile 切分是否符合预期(每个 tile 一个 matmul 指令)
  - activation / weight 的 VTCM 地址是否对齐

**常见陷阱**:
- 忘记 `--mhmx 1` → HMX 指令走 unknown opcode 异常
- `-mv75` 和 `--mv75` 不一致 → 可能能跑但指令语义不对
- 用 `--timing` 跑超长程序 → 非常慢,配合 `--fastforward` 或 `--pcfilter` 使用

---

## 7. 与 H2 booter 的关系

本仓库所有需要 HMX/VTCM 的测试都要走 booter:
```
hexagon-sim [sim 选项] -- booter [booter 选项] guest_elf [guest argv...]
```
- **sim 选项**(比如 `--timing`、`--statsfile`):在 `--` 之前,影响模拟器本身
- **booter 选项**(比如 `--ext_power 1`、`--fence_hi 0xfe000000`):在 `--` 之后,影响 hypervisor 初始化
- **guest argv**:跟在 `guest_elf` 之后

典型 booter 选项:
- `--ext_power 1 --use_ext 1`:通过外部寄存器通知 HVX/HMX 上电就绪
- `--fence_hi 0xfe000000`:guest 地址空间上界(booter/host 使用高地址)
- `--subsystem_base 0xfeXX`:某些 cosim 模型需要

---

## 8. 快速排错

| 现象 | 可能原因 |
|------|----------|
| `Unknown opcode` 在 HMX 指令处 | 少了 `--mhmx 1` 或编译时少了 `-mhmx` |
| `HMX VTCM address ...` 报错 | 没用 booter / VTCM 未映射 |
| 程序卡住不退出 | guest 死循环,加 `--plimit` 限 cycle |
| exit code 总是 0 | 没加 `--simulated_returnval` |
| `--timing` 太慢 | 加 `--fastforward` / `--pcfilter` / `--bypass_idle` |
| trace 文件太大 | 用 `--pctrace_min` 代替 `--pctrace`,或加 `--pcfilter` |
| Packets/Insns ≈ 1 | VLIW 没打包,检查编译优化等级 `-O2` / `-O3` |

---

## 9. 参考

- 完整选项: `hexagon-sim --help`
- 某架构的 core 列表: `hexagon-sim --help --mvXX`
- Hexagon SDK 文档: `tools/hexagon-sdk/docs/`(离线 HTML)
- 本仓库 E2E 示例: `tests/test_hvx_hmx_sim.sh`
