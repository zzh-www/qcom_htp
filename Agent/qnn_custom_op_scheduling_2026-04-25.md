# QNN 对 custom op 做不做调度？——实锤修正 (2026-04-25)

> 前一篇 `qnn_matmul_as_composition_2026-04-25.md` 里我说"QNN 对 custom op
> 不做 lowering"，**这个说法不准确**。用 V8 @ 512³ 的实际 chrometrace 验证，
> QNN 对 custom op 做了**部分**调度，但不做**全部**。下面是严格按实测分界。

## 1. 证据：V8 @ 512³ 单次推理的真实调度

把 chrometrace dedup（去掉 pid 0/pid 2 重复），**1 个 inference** 的实际事件：

| 事件                                                      | 实例数 | 线程分布 | 来源        |
|-----------------------------------------------------------|-------:|----------|-------------|
| `HmxMatMulPhase3Package::PackActivationU8RowMajor`        |  **6** | HVX tid 512/513/514/515 | 我们的 op |
| `HmxMatMulPhase3Package::TcmDramCopy`                     |  **6** | HVX tid 513/514/515     | 我们的 op |
| `HmxMatMulPhase3Package::PackWeightToHmxTileV3`           |      1 | HVX tid 515             | 我们的 op |
| `HmxMatMulPhase3Package::MatMulV8`                        |      1 | HMX tid 256             | 我们的 op |
| `q::*InputSlice`                                          |      5 | HMX tid 256             | **compiler 加的** |
| `q::*OutputSlice`                                         |      2 | HMX tid 256             | **compiler 加的** |
| `q::ConvLayer.opt.weights_to_vtcm`                        |      2 | HMX tid 256             | **compiler 加的** |
| DmaCheckpointSet / SyncOp / $Const / $Shape              |  ~20   |                         | 元数据      |

**ONNX 里我们写的**：
```
act_raw → pack_act → packed_act
wt_raw  → pack_wt  → packed_wt
packed_act + packed_wt + bias + scratch → mmv8 → out_tile
out_tile → tcm2ddr → out
```
**4 个节点**。

**实际在硬件上跑的**：**~22 个原语/ops 事件**，分布在 HMX + 4 个 HVX 线程。

## 2. QNN 对 custom op 做了什么 / 没做什么

### ✅ 做了 A：intra-op 自动多线程（`multithreaded=true` 的 op）

我们标了 `multithreaded=true` 的 kernel (`pack_act_rm_hvx.c`,
`tcm_dram_copy_hvx.c`, `untile_to_rowmajor_hvx.c`) 被 QNN scheduler **自动
发到多个 HVX 线程**。机制：
- 同一个 op 实例被**调用 N 次**，每次传不同 `qhpi_num_slices()`/`qhpi_slice_number()`
- 我们的 kernel 用这俩值自切 M_tiles 子集
- 实测 PackActivationU8RowMajor 被切成 **6 个 slice**，跑在 4 个 HVX 线程上

这个**不需要**我们在 ONNX 里写多实例。**QNN 自动帮我们做了 intra-op 并行**。

`multithreaded=false` 的 op（`PackWeightToHmxTileV3` 和 `MatMulV8`）则只跑
**1 次**，单线程。HMX 的 MatMulV8 本身只有 1 个 HMX unit，不能多线程；
pack_wt 我们标成 false 是另一个原因（wu-cache 跨线程麻烦）。

### ✅ 做了 B：graph 边界自动插 InputSlice/OutputSlice

我们 ONNX 的输入 `act_raw` 是 `[1,1,512,512]` 平坦 u8，但 kernel 要的是
VTCM-resident + padded 格式。QNN 自动在 `act_raw` 和 `pack_act` 之间插入
`q::*InputSlice` 原语（5 个），做 DDR→VTCM staging。输出侧 `out` 同理 2 个
`q::*OutputSlice`。

### ✅ 做了 C：STATIC tensor 的自动 DMA 预载

ONNX 里 `wt_raw` 是 STATIC initializer（baked 进 .bin）。QNN 看到 custom
op 的 STATIC 输入 + TCM_Only 签名，**自动插入 2 个 `q::ConvLayer.opt.weights_to_vtcm`**
DMA 原语。这是 compiler 在帮我们做 HMX-thread-parallel 的预载。

### ❌ 没做 D：tensor-level tile 切分

**这是关键的 "没做"。** QNN 不会把 `PackActivationU8RowMajor` 的**输出 tensor**
（`packed_act = [1, M/32, K/32, 1024]`）切成多个小 tensor 配多个 op 实例。
tensor 声明多大就分配多大 VTCM 一次。

@ 4096³ 这个 tensor = 16 MB > VTCM 8 MB → ctxgen 报错：
```
"pack_act" not sufficiently tiled to fit in TCM. Requires 16777216 bytes
```

**不是 scheduler 不够聪明，是 compiler 没有 "拆 custom op 的 tensor" 这个
改写规则**。内置 MatMul 有这个规则（它知道 MatMul 可以按 M/N 维 axis 拆），
我们的 `MatMulV8` 没有，compiler 不知道怎么拆。

### ❌ 没做 E：@Spill / @Fill 插入

即使 tensor 大到 VTCM 装不下，compiler 也不会在 custom op 周围自动插溢出
管理。内置 MatMul 在 4096³ 下自动插 4014 个 `@Spill`/`@Fill` 事件；我们
custom op 直接 ctxgen fail。

## 3. 为什么差异这么大

**区别的本质**：

| 操作类型 | QNN 编译器需要知道的信息 | 内置 MatMul 有？ | 我们 custom op 有？ |
|---|---|---|---|
| 资源分派 (HMX/HVX) | "要跑在哪种硬件" | ✓ | ✓ via `QHPI_RESOURCE_*` |
| 多线程自切 | "这个 kernel 支持切 N 个 slice 并行跑" | ✓ | ✓ via `multithreaded=true` |
| Graph boundary staging | "DDR tensor 进来要先 stage 到 VTCM" | ✓ | ✓ 自动 |
| **Tensor tile 拆分** | "这个 tensor 的哪个 axis 可以拆成 N 段独立算，然后 Concat" | **✓ 内置 MatMul 知道 M/N 可拆** | **✗ 我们声明不了** |
| **VTCM Spill/Fill** | "这个 op 的 IO 哪些中间数据可以暂存 DDR" | **✓ compiler 按图型切分自动推导** | **✗ 我们声明不了** |

**Custom op 缺的两件事，本质上都是"shape 语义"的缺失**：compiler 不知道
我们的 `MatMulV8(packed_act, packed_wt, bias, scratch) → out_tile` 这几个
tensor 在 M/N/K 轴向有什么语义，所以不能自动拆分。

## 4. 这对"怎么写 MatMul"的准确结论

之前说得过粗，修正：

> ~~"QNN 不给 custom op 做调度，所有 lowering 都要自己做"~~

**正确表述**：
> QNN 给 custom op 做 **intra-op 多线程** 和 **graph 边界 staging**，
> 但**不给 custom op 做 tile 拆分和 VTCM 溢出管理**。

所以：
- **≤2048³**（tensor 装得下 VTCM）：`multithreaded=true` 就够了，QNN 自动并行
  pack/unpack，HMX 单线程跑一次 MAC。**不需要**切图。
- **≥4096³**（tensor > VTCM）：必须**用户自己**在 ONNX 里拆多个 MatMulV8 实例，
  因为 compiler 不会帮拆。

## 5. 为什么 512³ V8 已经"看起来"被调度了

你看 chrometrace 会发现 512³ 下 HVX 有 4 个 tid 同时跑 — 这不是因为我们
拆了多个 MatMulV8 节点，而是因为：
- `PackActivationU8RowMajor` 标了 `multithreaded=true`
- QNN 看到这个 flag 后调用 6 次，发到 4 个 HVX 线程
- 这 6 次共同完成一个 tensor 的填充

V8 的实测性能差距主要来源不是"没并行"，而是：
- pack_act 本身单次 ~87K cyc（我们 scalar u64 loop），QNN Crouton 单次 ~20K cyc（vshuffvdd）
- mmv8 是一个 big instance，MAC 完前 HMX 资源全占着；QNN 切成 4 个 ConvLayer_s1.opt，每个小，MAC gaps 可以被其他 HVX 工作 overlap

## 6. 对行动项的影响

**第一优先级还是一样，但理由更精确**：

| 行动               | 原理由（错）                       | 修正后理由                                   |
|--------------------|--------------------------------------|---------------------------------------------|
| ONNX 切多实例       | "QNN 完全不给 custom op 调度"       | "QNN 不给 tile 拆分；小 shape 不用，大 shape 必须" |
| HVX pack vshuff 改写| "QNN 比我们快因为用了 vshuff"       | 同（这个是对的）                             |
| 多 HVX 线程 pack    | "必须切多实例才能用多线程"          | **错，`multithreaded=true` 已经在用 4 线程了** |
| 补 @Spill/@Fill    | "compiler 会给内置 op 加，不给我们" | 同（这个是对的）                             |

**最大的认知修正**：之前推 "阶段 A：切 M_half × N_half = 4 instances" 的主要
benefit 不是 HVX 并行（那个 multithreaded=true 已经在给我们），而是：
- **大 shape 下绕开 "tensor > VTCM" 的硬限制**
- **让 HMX MAC 切成多个小 instance，留 MAC gaps 给 HVX overlap**

## 7. Cross-refs

- 修正对象：`Agent/qnn_matmul_as_composition_2026-04-25.md` §3.2（custom op 的限制）
- `Agent/qnn_matmul_design_principles_2026-04-25.md` §2/§3（shape-adaptive 原则）
- `Agent/matmul_blueprint_2026-04-25.md` §7（行动项）
- QHPI header：`tools/qnn-sdk/include/QNN/HTP/core/qhpi.h:828` (multithreaded flag doc)
