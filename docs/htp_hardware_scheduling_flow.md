# HTP 硬件调度流程方法论（手写 Hexagon 算子时如何编排 DMA / HVX / HMX / VTCM）

> 从 `docs/hexagon-tutorial/hmx-tutorial/`（ch03–ch07 + hmx.md）提炼。**核心不是任何具体 op，而是
> llama.cpp / htp-ops-lib / Genie 这套把多个硬件单元编排成重叠流水线的调度方法。** 所有数字均为
> 骁龙 8 Gen 3 (v75) 真机实测。复现见各章 `run_device.sh`。

## 一句话

手写高性能 HTP 算子 = **让 DMA 引擎、HVX、HMX 三个硬件单元同时各干各的**，VTCM 做公共中转台，
靠 **DMA 双缓冲**把数据搬运藏进计算时间里，靠 **dspqueue** 把 ARM↔DSP 通信开销摊销掉，靠 **VTCM
持久 tile 格式**消掉重复的格式转换。瓶颈永远是数据搬运，不是计算。

## 三条不可妥协的首要原则（先套这三条）

1. **一次 RPC = 整张图。** FastRPC 只 `start` 一次，之后**整张图/整条复合计算在 DSP 侧一口气跑完**
   再返回，**绝不一个 op 一次 RPC**。per-op 往返每个 op 都要付通信税（外加一次 DDR 往返）；一次
   dispatch 只摊一次。（Qwen3-0.6B 196 ops：per-op FastRPC 71ms vs 一次 dispatch 12ms。）
2. **数据能放进 VTCM 就全放 VTCM。** 让整个工作集驻留片上——权重、激活、中间结果全放，不只是热/复用
   的张量。收益是消掉 op/stage **之间**的 DDR 往返、让数据全程留在片上（且 HMX 强制要 VTCM、还避免
   cache 驱逐与不确定延迟）。**只有当工作集真的超出 VTCM 容量时**，才退化到 Layer 3 的流式/双缓冲。
3. **不可避免的搬运，全部藏进计算**（DMA 双缓冲，Layer 3）。

> 一个细节（不是例外）：VTCM 对**单次顺序** HVX 不比 DDR 快（L2 预取已饱和带宽）。但"全放 VTCM"
> 依然赢，因为它消掉的是**跨 op 的 DDR 流量**并喂给 HMX——按整张图设计，不要按单个 vadd 评判。

## 四层调度（从外到内）

### Layer 1 — ARM↔DSP 通信：dspqueue（不是 per-call FastRPC）
- FastRPC **只用一次** `start`（传 queue id / n_hvx / use_hmx）和一次 `stop`；所有计算走共享内存队列。
- 零拷贝：`rpcmem_alloc` + `rpcmem_to_fd` + `fastrpc_mmap`；ARM 端 `dspqueue_write`，DSP 端消息循环
  `dspqueue_read_noblock` → `switch(op)` dispatch → `dspqueue_write` 回复。
- **实测开销：dspqueue ~61µs/op vs FastRPC ~364µs/op（6×）。** LLM 每 token 196 ops → 12ms vs 71ms。
- 推论：高频小 op 场景，per-call FastRPC 的通信开销会淹没一切；必须 dspqueue。也可把整条复合计算
  **融成一条消息**（hmx.md 的 `OP_TRAIN_BATCH`：一条消息内跑完前向+反向+SGD）。
- 代价：DSP 必须保持忙碌——空闲时 VTCM 会被 camera/audio 等高优先级客户端回收，数据被静默覆盖
  （"VTCM 消失" bug）；评估/收尾也放在 DSP 上做。

### Layer 2 — 多 HVX 线程池（worker-pool）
- 可按数据维度并行的 HVX 算子用线程池切分（llama.cpp `worker-pool.c`，环境变量 `GGML_HEXAGON_NHVX`）。
- ⚠️ **HMX 不靠多线程并行**（它是单矩阵单元，进程内对锁串行）。HVX 可以多线程；HMX 的并行性来自
  Layer 3 的流水重叠，不是 Layer 2 的线程。把这两件事分清是关键（我之前混淆了）。

### Layer 3 — 单 op 内的 DMA∥HVX∥HMX 流水线 + VTCM 双缓冲（**这套流程的心脏**）
四级流水线（ch04 Part 6 / ch07 matmul-ops / flash-attn-ops）：
```
Stage1 DMA        Stage2 HVX         Stage3 HMX        Stage4 HVX/DMA
DDR→VTCM    →     反量化/重排    →   mxmem matmul  →   反交织/写回
(异步,后台)       (拼 32×32 tile)    (VTCM 常驻)        (VTCM→VTCM 或 →DDR)
```
- **DMA = UDMA 引擎**：`Q6_dmstart_A(desc)` 启动（type0 线性描述符，`srcbypass=1`），`Q6_R_dmwait()`
  阻塞等 / `Q6_R_dmpoll()` 轮询。CPU/HVX/HMX 全程不参与搬运。
- **Cache 陷阱**：DMA 绕过 L2 直读物理内存 → DMA 前必须 `qurt_mem_cache_clean(..., QURT_MEM_CACHE_FLUSH,
  QURT_MEM_DCACHE)` 把 DDR 源刷回；VTCM 不过 cache，作目标不用刷。
- **ping-pong 双缓冲**：两个 VTCM scratch（0/1）。循环里先 `dma_start(下一块→nxt buffer)`（不阻塞），
  再 `hvx+hmx` 处理当前块（cur buffer），最后 `dma_wait()`。搬运被算掩盖。
- **实测：DMA+HVX 重叠 217µs vs 串行 3701µs = 17×。** 这是整套方法的最大杠杆。
- **纯 HMX 计算近乎免费**：数据 VTCM 常驻、cache 热时 ~1µs（28,000 GFLOPS，~4-5 cyc/tile）。
- 瓶颈层级（务必背下来）：DDR↔VTCM I/O ~1000µs ≫ hexkl readback 120-3000µs ≫ HVX vdeal 反交织
  5-126µs ≫ 纯 HMX 计算 ~1µs。所以：**绕过 hexkl readback（用 `hmx_store_acc`+`vdeal`）、中途绝不落 DDR。**

### Layer 4 — VTCM 数据驻留 / 持久 tile 格式（NativeKV）
- VTCM：8MB，1 周期延迟，HMX 强制要求，可作 DMA 目标。无 malloc，用 **bump allocator**（128B 对齐）。
- ⚠️ 反直觉（ch04 Part 5）：**VTCM 对顺序 HVX 不比 DDR 快**（L2 预取很强）。VTCM 的价值只有两个：
  HMX 别无选择 + 能做 DMA 目标实现重叠。别为了"VTCM 更快"而 VTCM，要为这两个理由而用。
- **NativeKV / 持久 tile**：把会被反复使用的数据**永久以 HMX tile 格式存在 VTCM/DDR**，HMX 直接消费，
  零格式转换（ch06：长上下文每步省 512KB 转换）。成立条件：v75+ 才支持 NPU 直接消费 tile-layout I/O；
  element-wise 更新（如 SGD）不关心排列，可直接在 tile 格式上做。tile 尺寸按维度选：长维(K)用大 tile
  减少搬入，短维用小 tile。

## 何时 NPU/HMX 真正值得（清醒线）
- HMX 只在**大矩阵**赢（≥512）：256×1024×4096 时 NPU 比 CPU 快 166×。
- 小矩阵（如 0.6B 模型的 896×896、MNIST 832×128、本仓 GDN 的 64×64 块）：通信 + 格式转换 + 回写的
  **固定开销淹没 HMX 吞吐**，HMX 仅 ~1.1× 于 HVX，甚至更慢。**小块要靠零转换 tile 常驻把固定开销消掉**，
  否则纯 HVX qf32 更优。模型 ≥4B 时 NPU 在 prefill 才真正值回票价。

## 对本仓 GDN solve 的修正（我之前为什么失败 → 修了之后的真实数字）
旧裸机 BR solve(H=8) = 271K、被数据搬运主导。错在没按这套流程,改正后:
1. **VTCM acquire-once 共享**(原来每 worker 各自 acquire → 资源管理器串行化 worker)。
2. **A 经 UDMA 常驻 VTCM**;标量访问的 scratch 留 DDR(VTCM 标量访问病态 7×,但根因是标量本不该有)。
3. **测真实 H=32**(H=8 是 8head/4线程负载不均的假象,会得出相反结论)。
> 改正后(真机 H=32,int16-HVX,VTCM 常驻 A,4 HVX 线程):**~151K cyc/head 4-thread,2.92× 扩展,
> oc 0.28%**。这是调度问题(流程编排),不是换 kernel/dtype。⚠️ **151K 是裸机 wall,出货版 70–83K 是 QNN
> domain cycles,口径未对齐前不下倍数结论**(见 `docs/cycle_metric_alignment.md`)。当前状态详见
> `Agent/current/gdn_solve_handwritten_route.md` 的 CURRENT STATE。
>
> 注:HVX∥HMX overlap 对这条 solve 不是杠杆 —— mxmem(真 HMX)只占 ~6%,solve 是 HVX-bound;
> overlap 只在有大块 HMX 工作可藏时才划算,用前先 profile HMX 占比。

## 复现 / 出处
- ARM↔DSP dspqueue：`docs/hexagon-tutorial/hmx-tutorial/ch03-dspqueue/`（61 vs 364µs）。
- DMA 双缓冲 + 4 级流水线：`ch04-vtcm-memory/`（17× 重叠；UDMA descriptor；cache flush 陷阱）。
- HMX 瓶颈层级（DDR I/O ≫ readback ≫ 纯 HMX 计算）：`ch05-hmx/README.md`。HMX 调用机制（power+VTCM+lock）见 skill `references/asm_building_blocks.md`。
- NativeKV 持久 tile：`ch06-kv-cache/`（零转换、v75+、tile 尺寸选择、对齐 32）。
- 完整模型编排：`ch07-llama-cpp-run/`（worker-pool 多 HVX 线程；MUL_MAT = HVX 反量化→HMX，VTCM 双缓冲）。
- 训练全流程叙述：`hmx-tutorial/hmx.md`。
