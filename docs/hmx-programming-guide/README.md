# HMX 编程指南（v75）

本指南教你在 Hexagon v75 cDSP 上**用 C + inline asm 直接写 HMX 代码**。
读者画像：**会写 C、对汇编有概念、第一次碰 HMX** 的工程师。

## 这份指南是什么

- **怎么写代码**：如何组合 HMX intrinsic / 汇编指令，让 MAC 阵列跑起来、算对、读回。
- **每一章都带可跑 demo**：打开一份 `.c`，`bash build.sh` 就能编，`hexagon-sim` 跑出 `[PASS]`。
- **所有示例都在 hexagon-sim 上验证过** bit-exact（对照 C reference）。

## 这份指南**不是**什么

- **不是架构解读**。HMX 内部（累加器宽度、sub-byte 展开、cell 结构等）已经在
  `../../Agent/hmx_u8xi8_matmul_layers.md`（u8×i8 全栈 L1→L8）和
  `../../Agent/hmx_int4_combos_analysis.md`（ISA + 微架构 + int4 分解）里详细写过。
  本指南只给"结论句 + 链接"，不重复推导。
- **不是 QNN OpPackage 教程**。本指南只跑 hexagon-sim 下的独立 ELF，不涉及 QNN Skel / FastRPC。
  真机（SM8650）侧走 QNN 的参考：`example/hmx_matmul_qnn/`。

## 章节

| 章 | 主题 | 读了会什么 |
|----|------|-----------|
| [01 mental-model](01-mental-model.md) | HMX 三件套：acc / tile / convert | 能画出一次 MAC 的数据流图 |
| [02 first-demo](02-first-demo.md) | 5 条指令跑通 `A=1, W=1, K=32 → 32` | demo01 的逐行拆解 |
| [03 tile-layout-cheatsheet](03-tile-layout-cheatsheet.md) | act / wt / output 字节布局速查 | 给任意逻辑 `(ir, K, jc)` 写出 VTCM 字节偏移 |
| [04 instr-clracc-swap](04-instr-clracc-swap.md) | `mxclracc` / `mxclracc.hf` / `mxswapacc` | int vs fp acc 区分、double-buffered conv |
| [05 instr-bias](05-instr-bias.md) | `bias = mxmem` / `mxmem2` | 怎么选 scale、mxmem 与 mxmem2 差在哪 |
| [06 instr-activation-load](06-instr-activation-load.md) | `activation.ub / .hf / .f8` + 修饰符 | 各 activation 类型用场景 |
| [07 instr-weight-load](07-instr-weight-load.md) | `weight.b / .n / .c / .ubit / .sbit / .sc / .sm / .hf / .f8` + `:2x` | 选哪种 weight + 风险点 |
| [08 instr-mac-packet](08-instr-mac-packet.md) | VLIW packet 的 act+wt 配对规则 | 合法与非法组合 |
| [09 instr-convert](09-instr-convert.md) | `mxmem(...)=acc` 全家族（`.uh/.ub/.hf` + `:2x1/:2x2`） | dual-scale readback 手法 |
| [11 modifiers](11-modifiers.md) | `:dilate / :deep / :single / :above / :drop / :cm / :sat / :retain / :pos / :2x` | 按使用场景挑修饰符 |
| [12 pitfalls](12-pitfalls-and-debugging.md) | 常见踩坑 + sim 调试技巧 | 写错会看到什么 |
| [13 tutorial int4×i8 matmul](13-tutorial-int4xi8-matmul/) | 从 u8×i8 开始，逐步推到 int4×int8 完整 kernel | 独立写出自己的 matmul |
| [14 HVX integration](14-hvx-integration.md) | HVX 加速 pack/fill/correction 的最佳实践 + 每个 demo 的 CPU/HVX 双路径 | 混合写 CPU+HMX+HVX kernel |
| [附录 A intrinsic index](appendix-A-intrinsic-index.md) | 所有 ≈ 200 条 `Q6_*` intrinsic 索引 | 查"这个 intrinsic 在讲哪章" |
| [附录 B asm ↔ intrinsic](appendix-B-asm-cheatsheet.md) | 两种写法对照 | 在 inline asm 和 Q6 intrinsic 之间翻译 |
| [附录 C debug recipes](appendix-C-debugging-recipes.md) | copy-paste 的调试代码段 | kernel 挂了时的急救包 |
| [附录 D cross-reference](appendix-D-cross-reference.md) | 主题 → 仓库文件索引 | 快速跳转到既有 Agent/example/tests |

## 前置环境

你需要：

1. **Qualcomm Hexagon SDK v5.5 的安装**（本仓库在 `tools/hexagon-sdk/`，通过 `bash scripts/install.sh` 初始化）。
2. **H2 Hypervisor 编译产物**（`tools/h2-install`，通过 `bash scripts/install.sh` 里的
   `make ARCHV=75 TARGET=ref USE_PKW=0` 或 `tools/hexagon-hypervisor/` 的 README）。
3. **hexagon-clang / hexagon-sim** 在 PATH（`source scripts/env.sh` 完成）。

验证环境 OK：

```sh
source scripts/env.sh
bash tests/test_hmx_programming_guide.sh
```

期望输出尾部 `PASS: HMX programming guide demos (1/1)`（v1 只有 demo01，后续会增加）。

## 代码/文档对照

- **本指南文档**：`docs/hmx-programming-guide/*.md`（你正在看的）
- **示例 C 源码**：`example/hmx_programming_guide/demo*.c`
- **构建脚本**：`example/hmx_programming_guide/build.sh`
- **集成测试**：`tests/test_hmx_programming_guide.sh`

所有 demo 遵循统一约定：
- 在 `main()` 里先算 C reference，再算 HMX，对比。
- 失败时打印 `[FAIL]` + 原因，成功时打印 `[PASS] demoNN`。
- `h2_thread_stop(fail_count)` 让 sim exit code 反映结果。

## 证据级别约定

延用仓库现有的分层标注（和 `Agent/` 下文档一致）：

- **(F)** — 由 ISA 头 / sim 反汇编 / 设备二进制 / 硅级探针直接支撑
- **(P)** — 合理推断，可被 RTL 或更深探针证伪

本指南里讲 API 语义的部分基本都是 (F)（ISA 头权威）；讲"为什么这样"或推到底层细节时才会出现 (P)。

## 读者要了解的背景

指南会假设你知道以下概念（不在这里展开，给你参考链接）：

| 概念 | 参考 |
|------|------|
| Hexagon VLIW packet / slot 约束 | `docs/pdf/80-N2040-57_AB_Hexagon_V75_Programmers_Reference_Manual.pdf` |
| HVX 向量寄存器 / `Q6_V_*` intrinsic | `docs/pdf/80-N2040-58_AB_Hexagon_V75_HVX_Programmers_Reference_Manual.pdf` |
| VTCM 是什么 | `docs/hexagon-tutorial/hmx-tutorial/ch04-vtcm-memory/` |
| HMX 累加器、tile 的宏观图 | `Agent/hmx_u8xi8_matmul_layers.md` L1–L5 |
| HMX 在 v75 上没有独立公开文档 | `Agent/hmx_int4_combos_analysis.md` §1 |

## 完成状态

- ✅ **核心章节** (01–12)：所有 intrinsic 族 + 修饰符 + 踩坑
- ✅ **int4×i8 tutorial** (ch13)：5 step 全，从 u8·i8 到 K=128 dual-scale
- ✅ **附录** (A–D)：intrinsic 索引、asm↔intrinsic 对照、debug 配方、交叉引用
- ✅ **10 个 demo**：每个都有 **CPU+HMX / HVX+HMX 双路径**，全部 hexagon-sim bit-exact PASS
  （`tests/test_hmx_programming_guide.sh`）
- ✅ **HVX integration 章节 (ch14)**：说明 HVX 加速的分工原则 + `hvx_shuffe` 踩坑
- ✅ **公共 HVX header** `example/hmx_programming_guide/hmx_hvx_common.h`：
  `hvx_zero`, `hvx_fill_*`, `hvx_pack_*`, `hvx_add_i8_plus_128`,
  `hvx_col_sum_w`, `hvx_apply_col_sum_correction`

~3,500 行文档 + ~1,800 行 C 代码 + 1 个 test runner。

## 感谢与引用

本指南建立在以下仓库内既有工作基础上：

- `Agent/hmx_int4_combos_analysis.md` — v75 HMX 微架构解读（硅级验证过）
- `Agent/hmx_u8xi8_matmul_layers.md` — u8×i8 matmul 全栈分层
- `example/hmx_matmul_int16/` — bit-exact int16 matmul 参考实现 + 各类 sim 探针
- `example/hexagon_hmx_matmul_native_int.md` — tile 字节布局的实测日志

出现不一致时以更下游（硅级探针 > sim > ISA 头）的证据为准。
