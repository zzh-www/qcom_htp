# 附录 D · 交叉引用

本指南**不重复**仓库里既有的架构 / 反汇编 / 优化日志内容。这里给**主题 → 文件**
的速查：下次你想"这个我记得在哪里看过"的时候直接跳。

## 按主题

### HMX ISA / intrinsic 清单
- `tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hmx_hexagon_protos.h`
  —— 权威清单，~200 条 `Q6_*` 宏
- 本指南：[附录 A](appendix-A-intrinsic-index.md)

### Tile 字节布局（act / wt / out）
- **实测日志**：`example/hexagon_hmx_matmul_native_int.md`（single-hot-byte 探针结果）
- **原理解释（per-layer）**：`Agent/hmx_u8xi8_matmul_layers.md` §L3
- 本指南：[ch03 cheatsheet](03-tile-layout-cheatsheet.md)

### HMX 微架构（cell 内部、两套 acc、MAC 基元）
- `Agent/hmx_u8xi8_matmul_layers.md` §L5–L7（8 层分层 L6 是 cell 层）
- `Agent/hmx_int4_combos_analysis.md` §3（硅级验证的 sub-byte MAC 结构）

### HMX 指令在真机的使用频次
- `Agent/hmx_int4_combos_analysis.md` §0.5.6（libQnnHtpV75Skel.so 反汇编）

### int4 / int2 / int1 子字节 MAC 机制
- `Agent/hmx_int4_combos_analysis.md` §3.2–§3.5（硅实测 + 公式推导）

### dual-scale readback 原理 + bit-exact int32 重建
- `example/hmx_matmul_int16/int16_matmul_hmx.c`（生产级参考）
- `example/hmx_matmul_int16/probe_dual_scale.c`（探针数据）
- 本指南：[ch09](09-instr-convert.md)

### int4×int16 的 4-partial 分解（扩展阅读）
- `Agent/hmx_int4_combos_analysis.md` §5.3
- `example/hmx_matmul_qnn/kernel/hmx_int4_matmul.c`（QNN OpPackage 实现，HMX 路径在硅上有 err 1003 bug）

### HMX kernel 优化迭代（性能向）
- `Agent/int4_matmul_optimization_log.md`（K-accumulate、pre-pack、VTCM 带宽优化的日志）

### HMX 真机 power / VTCM 获取 (HAP API)
- `example/hmx_matmul_device/bench_matmul_device.c` 的 `power_on_hvx_hmx()` + `HAP_compute_res_acquire`
- `docs/hexagon-tutorial/ch02-real-device/` 原始教程

### sim 模拟器的 HMX 函数模型
- `tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/lib/iss/libhexagonissv75.so`
  带调试符号的 x86-64 sim。关键符号：
  - `hmx_mult_body_ptr_table @ 0xc809c0`（3 个 MAC 基元）
  - `hmx_unpack_ptr_table @ 0xc809e0`（8 个 sub-byte unpacker）

### 真机探针 + 自动校验器
- `example/hmx_matmul_device/probe_subbyte_device.c`
- `example/hmx_matmul_device/verify_subbyte_results.py`（硅实测 vs sim 预测 20/20 PASS）

### HMX 公开文档缺失事实
- `Agent/hmx_int4_combos_analysis.md` §1 记录：v75 PRM 两份 PDF 共 965 页没 HMX 章节

### QNN OpPackage 接入 HMX（进阶）
- `example/hmx_matmul_qnn/` 完整 QNN 工程
- `Agent/int4_matmul_optimization_log.md` § "Decoded: QNN built-in MatMul"

## 按文件反查

### 本指南目录
- `docs/hmx-programming-guide/README.md` —— 入口
- `docs/hmx-programming-guide/01..12` —— 正文章节
- `docs/hmx-programming-guide/13-tutorial-int4xi8-matmul/` —— 5 step tutorial
- `docs/hmx-programming-guide/appendix-A..D` —— 附录

### 示例代码目录（本指南配套）
- `example/hmx_programming_guide/demo01..09*.c` —— 从简到繁的 sim 可跑 demo
- `example/hmx_programming_guide/demo_probe_n.c` —— int4 tile layout probe
- `example/hmx_programming_guide/build.sh` —— 一键编译
- `tests/test_hmx_programming_guide.sh` —— 一键 build + sim + check

### 既有架构文档（Agent/）
- `Agent/hmx_int4_combos_analysis.md` —— HMX 微架构 + int4×{4,8,16} 分解方案
- `Agent/hmx_u8xi8_matmul_layers.md` —— u8×i8 matmul L1→L8 分层
- `Agent/int4_matmul_optimization_log.md` —— QNN OpPackage int4 kernel 优化日志
- `Agent/qnn_matmul_dtype_comparison.md` —— QNN 各 dtype 性能比对
- `Agent/memory.md` —— session 杂记

### 既有示例代码（按用途）
- `example/hmx_matmul_int16/` —— bit-exact int16 matmul 参考实现 + sim 探针集合
- `example/hmx_matmul_qnn/` —— QNN OpPackage int4×int16（HMX 路径有 bug 待修）
- `example/hmx_matmul_device/` —— 真机 HAP harness + sub-byte probe + 自动校验器
- `example/qnn_matmul_profile/` —— QNN 各 dtype matmul cycle profiling

### 测试脚本（tests/）
- `tests/test_hmx_matmul_int16.sh` —— int16 matmul bit-exact test (sim)
- `tests/test_hmx_programming_guide.sh` —— 本指南 demo 全跑
- `tests/test_hvx_hmx_sim.sh` —— 老的 sim sanity test

### 工具链位置
- `tools/hexagon-sdk/` —— Hexagon SDK（hexagon-clang, hexagon-sim, 头文件）
- `tools/hexagon-hypervisor/` —— H2 源码
- `tools/h2-install/` —— H2 编译产物（符号链接）
- `tools/qnn-sdk/` —— QNN SDK（含 libQnnHtpV75Skel.so）

## 到这里指南完结

回到 [README](README.md)。
