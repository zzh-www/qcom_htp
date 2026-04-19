# Agent memory — qcom_htp

高价值、非显而易见、不易从代码本身推断的信息。供 agent 下次进入这个仓库时快速上手。

---

## 1. 仓库目录约定

```
qcom_htp/
├── scripts/          # 所有工具脚本(install / env / 生成器)都放这里
│   ├── install.sh            # bash scripts/install.sh
│   ├── env.sh                # source scripts/env.sh(不是 root 了)
│   └── gen_compile_commands.py
├── tests/            # 测试脚本 + golden 文件
│   ├── test_hvx_hmx_sim.sh
│   └── golden/sim_output.txt
├── docs/             # 只放文档/参考资料,不放代码
│   ├── hexagon_sim_handbook.md
│   └── hexagon-tutorial/     # 上游内容,只读;有 bug 也别改,在仓库根绕过
├── tools/            # 已安装的 SDK(gitignored)
│   ├── hexagon-sdk/
│   ├── qnn-sdk/
│   ├── android-ndk/
│   ├── hexagon-hypervisor/   # 源码 + 构建产物
│   └── h2-install -> hexagon-hypervisor/install   # 相对符号链接
├── downloads/        # 下载缓存(gitignored)
├── .vscode/          # VSCode 配置 + compile_commands.json(见 §6)
└── Agent/memory.md   # 本文件
```

**强规则**:
- 新工具脚本永远放 `scripts/`,不散落在仓库根。
- `docs/hexagon-tutorial/` 是上游参考,只读。哪怕它里面的脚本写死了错误的版本路径,也不要改 —— 在仓库根写新脚本绕开。
- `docs/` 可以加新的 `.md` 文档/手册,但不放代码(`.c/.cpp/.sh/.py/Makefile` 等)。

---

## 2. 已安装的 SDK 版本 + 下载 URL

这些 URL 不容易搜到(Qualcomm softwarecenter 需要登录),保留原始链接:

| 组件 | 版本 | URL |
|------|------|-----|
| Hexagon SDK | **6.5.0.0** | https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.5.0.0/Hexagon_SDK_Linux.zip |
| QNN / QAIRT | **2.45.0.260326** | https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.45.0.260326/v2.45.0.260326.zip |
| Android NDK | **r27c** | https://dl.google.com/android/repository/android-ndk-r27c-linux.zip |

Hexagon SDK 6.5.0.0 的文件名是 `Hexagon_SDK_Linux.zip`(大写 L),与老版本的 `Hexagon_SDK_lnx.zip` 不同。`scripts/install.sh` 里写死的就是这个。

---

## 3. 版本陷阱:HEXAGON_Tools 子版本

- SDK 6.5.0.0 ships **HEXAGON_Tools 19.0.07**
- 但 `docs/hexagon-tutorial/ch01-simulator-setup/run.sh` 和 `install_tools.sh` 都写死 `19.0.04`,所以这些脚本在当前 SDK 下**直接运行会 fail**。
- 绕法:`scripts/env.sh` 用 glob 解析 `HEXAGON_Tools/*/Tools`,`tests/test_hvx_hmx_sim.sh` 靠 `env.sh` 设好的 PATH 调 `hexagon-clang` / `hexagon-sim`。
- 如果以后要跑 `docs/hexagon-tutorial/` 里其它 chapter 的脚本,优先在仓库根写新脚本,而不是改上游。

---

## 4. H2 Hypervisor 相关

### 4.1 为什么必须用
在模拟器上跑 HMX/VTCM 程序时必须经 H2 booter 引导。QuRT 自己无法通过 HMX 的 VTCM 地址检查(这是硬件约束)。H2 给 guest 提供正确的物理/虚拟地址映射。

`docs/hexagon-tutorial/ch01-simulator-setup/README.md` 第 18-24 行说得最清楚。

### 4.2 构建
不在 `scripts/install.sh` 里 —— 那个脚本故意不碰 H2(clone + make 要 hexagon-clang,跑得慢,且只有模拟器 chapter 需要)。手动构建:
```bash
source scripts/env.sh
cd tools/hexagon-hypervisor
make ARCHV=75 TARGET=ref USE_PKW=0
cd -
ln -sfn hexagon-hypervisor/install tools/h2-install   # 注意符号链接目标
```

### 4.3 符号链接陷阱(踩过)
`tools/h2-install -> hexagon-hypervisor/install`(**不要**加 `../`)。符号链接的相对路径是从符号链接文件所在目录解析的,`tools/h2-install` 所在目录是 `tools/`,写 `../hexagon-hypervisor/install` 会解析到 repo_root/hexagon-hypervisor,不存在。

### 4.4 booter 命令行
```
hexagon-sim [sim 选项] -- booter [booter 选项] guest_elf [guest argv...]
```
- `--` 之前:`--mv75 --mhmx 1 --simulated_returnval --timing ...`(属于模拟器)
- `--` 之后:`--ext_power 1 --use_ext 1 --fence_hi 0xfe000000`(属于 booter),再接 guest ELF

---

## 5. E2E 测试的校验策略

`tests/test_hvx_hmx_sim.sh` 用两层校验:

1. **grep 断言**:正则匹配 `^[[:space:]]*Results:[[:space:]]+3 PASS[[:space:]]+/[[:space:]]+0 FAIL`。这行来自 `docs/hexagon-tutorial/ch01-simulator-setup/test_hvx_hmx.c:257`(`printf("  Results: %d PASS / %d FAIL\n", pass, fail)`)。
2. **golden diff**:用 `sed -n '/Chapter 1: HVX/,/Results: [0-9]* PASS/p'` 截出确定性段,跟 `tests/golden/sim_output.txt` `diff -u`。

截段是为了绕过非确定性内容(booter 的 CORE 0 横幅、尾部 `Insns/Packets/Pcycles` 统计 —— 虽然在同一模拟器下通常稳定,但易受 sim 版本影响)。

失败时日志保留在 `tests/last-failure.log` + `tests/last-failure-actual.txt`(`.gitignore` 已排除)。

---

## 6. VSCode / IntelliSense 配置

用户用的是**官方 Microsoft C/C++ 扩展(ms-vscode.cpptools)**,不是 clangd。

### 6.1 `compile_commands.json` 必须在 `.vscode/`
不是仓库根。`scripts/gen_compile_commands.py` 会自动写到 `.vscode/compile_commands.json`。`.gitignore` 已排除(含绝对路径,不宜提交)。

### 6.2 配置文件
- `.vscode/c_cpp_properties.json`(注意:**不是** `c_cpp_configuration.json`,那是拼错,MS 扩展会静默忽略)—— ms-cpptools 读这个。字段:
  - `compileCommands` 指向 `.vscode/compile_commands.json`
  - `includePath` 兜底(数据库里没的文件,比如打开 `.h` 头文件本身时)
  - `compilerPath` = hexagon-clang,用于推断系统头
- `.vscode/settings.json` —— 同时配了 clangd 参数(无害),主要是 `C_Cpp.default.compileCommands`

### 6.3 改配置后
`Ctrl+Shift+P → Developer: Reload Window` 才生效。

### 6.4 源文件分类
`scripts/gen_compile_commands.py` 按路径模式分 **DSP**(用 hexagon-clang)和 **host**(用 NDK aarch64 clang):
- `src/host/` 或 `src/arm/` → host
- `src/dsp/` → dsp
- `ch01-simulator-setup/`、`ch02-real-device/`、`hmx-tutorial/*/src/` → dsp
- `qnn-tutorial` 下的文件额外加 `-I$QNN/include/QNN`
- ch01 特殊加 H2 头(`-Itools/h2-install/include`)

---

## 7. 常用路径速查

```bash
$HEXAGON_SDK_ROOT                                        # tools/hexagon-sdk
$HEXAGON_TOOLS_ROOT/bin/hexagon-clang                    # 19.0.07
$HEXAGON_SDK_ROOT/incs/HAP_*.h                           # HAP APIs
$HEXAGON_SDK_ROOT/rtos/qurt/computev75/include/qurt      # QuRT 头
$HEXAGON_SDK_ROOT/rtos/qurt/computev75/include/posix     # posix shim

$QNN_SDK_ROOT/include/QNN/Qnn*.h                         # QNN APIs
$QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so           # ARM target
$QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so  # DSP skel
$QNN_SDK_ROOT/lib/x86_64-linux-clang/                    # host

$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang

$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hexagon_types.h
$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.07/Tools/target/hexagon/include/hvx_hexagon_protos.h
```

---

## 8. 跑起来的快速清单

```bash
# 首次(或新机器)
bash scripts/install.sh                                    # 3 个 SDK,幂等
source scripts/env.sh                                      # 每次新 shell

# 要跑模拟器测试的话还要建 H2
source scripts/env.sh
(cd tools/hexagon-hypervisor && make ARCHV=75 TARGET=ref USE_PKW=0)
ln -sfn hexagon-hypervisor/install tools/h2-install

# 日常
bash tests/test_hvx_hmx_sim.sh                             # 回归测试
python3 scripts/gen_compile_commands.py                    # 重建 VSCode DB
```

---

## 9. hexagon-sim 的价值不止正确性

除了跑程序拿 stdout,模拟器还能出:
- `--timing` 开启精确 pcycle;默认 functional 模式的 Pcycles 不可靠
- `--statsfile` 结构化统计(IPC、cache 命中、HVX/HMX 利用率)
- `--stalltrace` 每个 stall 的分类来源,性能调优核心
- `--coproctrace` HMX tile 级执行轨迹
- `--pctrace` / `--profile` 控制流 / 函数热点
- `--gdbserv PORT` gdb 远程调试
- `--plimit` / `--fastforward` / `--pcfilter` 限制采样范围

详见 `docs/hexagon_sim_handbook.md`。

---

## 10. 已知坑

- 模拟器运行时忘加 `--simulated_returnval` → exit code 永远 0,测试总通过
- 忘加 `--mhmx 1` → HMX 指令走 Unknown opcode
- `--timing` 跑完整程序非常慢 → 配合 `--fastforward` + `--pcfilter` 只精确模拟热点
- `Packets/Insns ≈ 1` → VLIW 没打包,检查编译优化(`-O2`/`-O3`)

---

## 11. HMX v75 MAC 数据类型(很容易猜错)

从 `tools/hexagon-sdk/.../target/hexagon/include/hmx_hexagon_protos.h` 的汇编列表看,HMX 的乘加对并**不是对称**的:

**Activation 可用类型**(`activation.X = mxmem(...)`):`.ub`(uint8)、`.hf`(f16)、`.f8`(fp8)
**Weight 可用类型**(`weight.X = mxmem(...)`):`.b`(int8)、`.hf`、`.f8`、`.sbit`/`.ubit`(1-bit)、`.c`/`.n`/`.sc`/`.sm`(压缩/特殊)

→ 整数 MAC 只有 **u8 × i8 → int32 acc** 这一种本征模式。**没有 u8×u8**。

**影响**:任何 u8×u8 partial product 都得改写成:
```
u8 * u8 = u8 * i8_reinterpret + 256 * u8 * top_bit_of_u8
```
第二项用 `weight.ubit` HMX 1-bit 矩阵乘。所以 int16 拆 4 次 u8 matmul 的方案实际上是 **4 次 u8×i8 + 2 次 u8×1bit**,共 6 次 HMX 调用 + CPU 合并。

**Activation + weight load 必须在同一个 VLIW packet**:
```c
asm volatile(
    "{ activation.ub = mxmem(%0, %1)\n"
    "  weight.b     = mxmem(%2, %3) }\n"
    :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
```
第 2 个操作数 2047 = `(TILE_BYTES - 1)` 用于"完整 32×32 tile 从基址开始"的场景。

**清 acc**:整数用 `mxclracc`,f16 用 `mxclracc.hf`,不能混用。

**HMX u8×i8 完整解码**(2026-04,端到端 bit-exact int16 matmul 已就绪):

**Tile 布局**:
- **Activation**(2 KiB):`A_byte(phys_row, K, stream)` 在 `128·phys_row + 4·K + (stream ? 3 : 1)`,`phys_row ∈ 0..15`, `K ∈ 0..31`,`stream ∈ {0,1}`。`4·K + 0/2` 的字节被忽略(必须清零,否则输出会乱)。
- **Weight**(1 KiB):`W_byte(K, col)` 在 `128·(K>>2) + 4·col + (K&3)`,`K ∈ 0..31`, `col ∈ 0..31`。
- **输出**(`:after.uh=acc:2x1`,1024 × u16):`phys_row·64 + 2·col + stream`。
- **逻辑 32×32 映射**:`ir → phys_row = ir & 15, stream = ir >> 4`。stream 0 对应逻辑行 0..15,stream 1 对应 16..31。**一次 HMX 调用可以产生完整的 32×32 logical output**。

**Bias 格式(实测 2026-04)**:
- `bias = mxmem(P)`,P 指向 **128 个 uint16**(每 column 一个 f16 scale,共 256 字节)。
- convert 内部有固定 /2:`OUT = floor(acc · bias_f16/2) mod 2^16`。f16(2.0)=`0x4000` 得 scale 1.0 恒等;f16(2^-7)=`0x2000` 得 scale 2^-8;f16 1.0=`0x3C00` 得 0.5。
- **slot 1/2/3 确认被忽略**(probe: splat 全 512 u16 vs 仅填 slot 0,输出完全相同)。没有 pre-convert offset/shift 机制可用。
- **denormal f16 bias 有 ~1.5× scale 异常**(0x0200 应代表 2^-15,实测生效 scale 是 3·2^-17,不是 2^-16)。应避开 f16 denormal 范围,只用 normal。
- 之前"257/256"观察是污染伪影,clean pack 后没有。

**输出 wrap 限制 & 解法**:
- `:after.uh=acc:2x1` 把 int32 acc 截成 uint16。K=32 u8×i8 MAC 下 `|acc|` 可到 2^20,会 wrap。
- **Dual-scale readback**(已实装):`:after:retain.uh=acc:2x1` 保留 acc 做第一次 convert,再换 bias 做第二次 convert。scale pair:`0x4000`(=1.0)读低 byte + `0x2000`(=2^-8)读中高 bytes;重构 `acc = ((int16_t)OUT_HI << 8) | (OUT_LO & 0xFF)`,覆盖 ±2^23,足够 K=32 partial。
- **每 32×32×32 tile 12 次 HMX packet**:4 partials × (1 full-K MAC + 2 convert)。对比旧 K-slicing 的 128 次,~11×提速且仍 bit-exact。

**实现位置**:`example/hmx_matmul_int16/int16_matmul_hmx.c`,测试 `tests/test_hmx_matmul_int16.sh`,全解码历程 `example/hexagon_hmx_matmul_native_int.md`。

**通过的场景**(全部 bit-exact vs int64 oracle):
- S1 常量小值:A=256, W=256 ✓
- S2 uint16-wrap 范围:A=-10000, W=20000 ✓
- S3 随机 int16 全范围 ✓

---

## 12. example/ 目录

目录 `example/` 放实际的 HMX/HVX 内核示例 + 真机 profile 工具链(跟 `tests/` 区分:tests 是回归测试,example 是教学性代码 + 对应的独立可执行)。

当前:
- `example/hmx_matmul_int16/` — per-tensor 量化 int16 matmul(裸机 bit-exact dual-scale kernel,≤12 HMX packet/tile)
- `example/hmx_matmul_device/` — 真机 bench 对比 fp16 vs 我们的 int16 kernel(HAP_perf_get_pcycles,Unsigned PD,run_main_on_hexagon 加载 .so)
- `example/qnn_matmul_profile/` — QNN 真机 MatMul cycle profiler(fp16/w16a16/w8a16/w8a8/w4a16/w4a8/w4a4),支持 --device/--connect ssh|adb/--arch;跑 5 次取中位数见 `Agent/qnn_matmul_dtype_comparison.md`

`scripts/gen_compile_commands.py` 会扫 `example/` 下所有 `.c/.cpp` 并加到 VSCode 的 compile_commands.json。

## 13. QNN profile 关键坑(2026-04 实测)

- QNN `soc_id` 是 **QnnSocModel 枚举**(`tools/qnn-sdk/include/QNN/QnnTypes.h` 里 SM8650=57),**不是** Android 硬件 revision id(SM8650 对应 577)。填错会让 ctxgen 用默认 v68 能力表拒绝 int16 kernel,运行时 fallback 到 fp16 emulation,**出现 `q::ConvLayer.fp16.s1.tcm` 就是中招了**。填对以后 int16 MAC kernel 是 `q::ConvLayer_s1.opt`,原生 HMX,不走 fp16。
- HTP v75 MatMul op **不支持 int4**(任何形式):compose 阶段直接 `QNN_TENSOR_ERROR_INVALID_TENSOR_PARAM`。HMX 有 `weight.n` nibble 指令,但 QNN 只把它接到 Conv2D(走 LPBQ 块量化)。
- QNN `libDlModelToolsPy.so` **ABI 锁定 CPython 3.10**,所以 pyproject.toml pin `>=3.10,<3.11`。venv 用 3.11/3.12 会报 circular import。
- ctxgen 需要传 `--config_file`(指向带 `socModel` 的 HTP backend_extensions JSON),否则默认 v68。
