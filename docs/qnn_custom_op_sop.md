# QNN Custom-Op 标准化流程 SOP

> 从 ONNX 自定义算子到 on-device optrace 的 full pipeline，2026-04-25 在 SM8650 v75 上 V8 验证通过。
> 以后写自定义算子走这套。参考实现：`example/hmx_matmul_phase3/standard_flow/phaseB_v8/`。

## 0. 管线总览

```
 ┌──────────────┐    ┌───────────────────────┐    ┌───────────────┐
 │ custom-domain│    │ Converter Op Package  │    │  qairt-       │
 │  .onnx       ├───▶│ (shape/dtype infer)   ├───▶│  converter    │
 └──────────────┘    │ libConverterOpPackage.so │    │   → .dlc     │
                     └───────────────────────┘    └───────┬───────┘
                                                          │
  ┌───────────────────────────────────────────────────────┘
  ▼
 ┌─────────────────────────────┐       ┌─────────────────────────────┐
 │ qnn-context-binary-generator │      │ on-device qnn-net-run        │
 │   + x86 op-pkg (prepare)     │─ctx─▶│   + aarch64/hexagon op-pkg   │
 │   + schematic.bin            │      │   + --use_native_input_files │
 │   (--profiling_option optrace│      │   (--profiling_option optrace│
 └─────────────────────────────┘       └───────────────┬─────────────┘
                                                       │ qnn-profiling-data.log
                                                       ▼
                                        ┌──────────────────────────┐
                                        │ qnn-profile-viewer       │
                                        │   + schematic.bin        │
                                        │   + optrace reader .so   │
                                        │   → chrometrace.json     │
                                        │   → QHAS summary.json    │
                                        └──────────────────────────┘
```

一共 **4 种 op-package .so** 要构建：

| Target                | 用途                                                      | 路径约定 |
|-----------------------|----------------------------------------------------------|----------|
| x86_64-linux-clang    | host 端 `qnn-context-binary-generator` 做 HTP prepare     | `build/x86_64-linux-clang/lib<Pkg>.so` |
| hexagon-v75           | 设备 HTP 上真正跑 kernel                                   | `build/hexagon-v75/lib<Pkg>_htp.so` |
| aarch64-android       | 设备 ARM 端做 CPU fallback / on-device prepare             | `build/aarch64/lib<Pkg>_cpu.so` |
| ConverterOpPackage    | qairt-converter 调用的 shape/dtype 推断（x86）             | `gen_out/<Pkg>_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so` |

## 1. 前置准备

```bash
source /home/zzh/work/qcom_htp/scripts/env.sh
# 设好 QNN_SDK_ROOT, HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT, ANDROID_NDK_ROOT
# Python venv (uv-managed, project local)
source .venv/bin/activate
export PYTHONPATH=$QNN_SDK_ROOT/lib/python
export PATH=$ANDROID_NDK_ROOT:$PATH

# 一次性 Python 依赖（check-python-dependency 里列的子集即可）
uv pip install lxml mako numpy onnx pandas pyyaml tabulate scipy packaging six decorator jsonschema attrs aenum psutil pydantic rich
```

## 2. 目录结构模板

每个自定义算子包建议放在 `example/<pkg_name>/` 下。V8 参考实现
（`example/hmx_matmul_phase3/`）现在就是这个形状：

```
example/my_custom_pkg/
├── README.md                         # 一段话说清楚做什么、两种运行方式
├── src/                              # C++ op + interface (V8 path 精简到 3 个文件)
│   ├── MyPkgInterface.cpp            # OpPackage interface provider
│   ├── MyOpA.cpp                     # 每个 op 一个 QHPI kernel 源文件
│   └── run_reference.cpp             # （可选）手搓 harness，ground truth / DIAG 用
├── kernel/                           # HVX/HMX .c 内核（Hexagon intrinsics）
├── build.sh                          # hexagon-v75 + aarch64-android 构建
├── build_x86.sh                      # x86_64 构建（ctxgen 用）
├── run_<op>_on_device.sh             # 快速跑 host harness 的小脚本（可选）
├── standard_flow/                    # SOP 第 5 节以后的所有产物
│   ├── <Pkg>Package.xml              # OpDef
│   ├── gen_<op>_onnx.py              # 生成 custom-domain ONNX
│   ├── quant_overrides.json          # （可选）量化 override
│   ├── htp_config.json               # backend extensions 包装
│   ├── htp_backend_ext.json          # 设备/graph 配置
│   ├── gen_out/                      # qnn-op-package-generator 输出（含 Converter Op Pkg）
│   ├── runtime_inputs_<dtype>/       # 推理时 APP_WRITE 输入 .raw
│   ├── input_list.txt                # qnn-net-run 输入列表
│   ├── <pkg>_model.onnx              # 自定义 domain ONNX
│   ├── <pkg>_model.dlc               # qairt-converter 产物
│   ├── <pkg>_model_schematic.bin     # ctxgen 产物（profile-viewer 必需）
│   ├── ctx_out/                      #   └ 含 <pkg>_ctx.bin
│   └── device_out/                   # 从设备拉下来的 profile.log / chrometrace.json
└── _archive/                         # 迭代过程中用过但不再活跃的代码（可选）
```

已经废弃的迭代代码建议放 `_archive/`（用 `git mv`，保留 blame 历史）而不是删。
V8 的归档样例见 `example/hmx_matmul_phase3/_archive/README.md`。

## 3. 步骤 1 — XML OpDef

`standard_flow/<Pkg>Package.xml`：

关键注意点：
1. `PackageName=` **必须等于** C++ interface 里 `sg_packageName` 的值（即 op-pkg so 通过 `THIS_PKG_NAME` 宏广播的包名）。
2. 每个 Shape 都加 `<Layout>NONTRIVIAL</Layout>`，防止 converter 插入 NHWC Transpose 节点。
3. `BACKEND_SPECIFIC` 允许 HTP 特定 datatype；Supplemental 里再精确列出接受的 QNN_DATATYPE_*。
4. 如果一个 tensor 希望走 `QHPI_QUInt8`（UFIXED_POINT_8）以绕过 QNN Cast +128，但 ONNX 只能标 UINT8，那就在 XML 里同时列 `<Datatype>QNN_DATATYPE_UFIXED_POINT_8</Datatype><Datatype>QNN_DATATYPE_UINT_8</Datatype>`，converter 就接受两边。

模板：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpDefCollection PackageName="MyCustomPackage" Domain="mypkg" Version="1.0">
    <OpDefList>
        <OpDef>
            <Name>MyOpA</Name>
            <Description><Content>...</Content></Description>
            <Input>
                <Name>in[0]</Name>
                <Mandatory>true</Mandatory>
                <Datatype>BACKEND_SPECIFIC</Datatype>
                <Shape>
                    <Rank>4D</Rank>
                    <Layout>NONTRIVIAL</Layout>   <!-- 关键 -->
                    <Text>[1, 1, M, K]</Text>
                </Shape>
            </Input>
            <Output>
                <Name>out[0]</Name>
                <Mandatory>true</Mandatory>
                <Datatype>BACKEND_SPECIFIC</Datatype>
                <Shape>
                    <Rank>4D</Rank>
                    <Layout>NONTRIVIAL</Layout>
                    <Text>[1, M/32, K/32, 1024]</Text>
                </Shape>
            </Output>
            <SupportedBackend>HTP</SupportedBackend>
        </OpDef>
        <!-- 其他 op ... -->
    </OpDefList>

    <SupplementalOpDefList Backend="HTP">
        <SupportedOps>
            <OpName>MyOpA</OpName>
        </SupportedOps>
        <SupplementalOpDef>
            <Name>MyOpA</Name>
            <Input><Name>in[0]</Name>
                <Datatype>QNN_DATATYPE_UINT_8</Datatype>
                <Datatype>QNN_DATATYPE_UFIXED_POINT_8</Datatype>
            </Input>
            <Output><Name>out[0]</Name>
                <Datatype>QNN_DATATYPE_UINT_8</Datatype>
            </Output>
        </SupplementalOpDef>
    </SupplementalOpDefList>
</OpDefCollection>
```

## 4. 步骤 2 — Converter Op Package（shape/dtype 推断）

```bash
cd example/my_custom_pkg/standard_flow
rm -rf gen_out
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-op-package-generator \
    -p MyCustomPackage.xml -cop -o gen_out -f
```

**`-cop`（--converter_op_package）是生成 Converter Op Package 的关键开关。**

输出：
- `gen_out/MyCustomPackage/` — HTP 运行时 skeleton（我们不用它，用自己的 src/）
- `gen_out/MyCustomPackage_Converter_Op_Package/ConverterOpPackage/ConverterOpPackage.cpp` — 要填肉的 shape 推断

**填 shape 推断**（把骨架里的空函数体改成实际逻辑）：

```cpp
EXPORT_API Qnn_ErrorHandle_t MyOpAShapeInference(Qnn_OpConfig_t *op) {
    uint32_t *in  = op->v1.inputTensors[0].v1.dimensions;
    uint32_t *out = op->v1.outputTensors[0].v1.dimensions;
    uint32_t M = in[2], K = in[3];
    out[0] = 1; out[1] = M/32; out[2] = K/32; out[3] = 1024;
    return QNN_SUCCESS;
}
EXPORT_API Qnn_ErrorHandle_t MyOpADataTypeInference(Qnn_OpConfig_t *op) {
    op->v1.outputTensors[0].v1.dataType = QNN_DATATYPE_UINT_8;   // 或 UFIXED_POINT_8
    return QNN_SUCCESS;
}
// Pointers 在生成骨架里已有，不用改
```

**支持 tensor v2 格式**（qairt-converter 会产 v2 tensor）：辅助函数见 `example/hmx_matmul_phase3/standard_flow/phaseB_v8/gen_out/.../ConverterOpPackage.cpp` 里的 `tensor_dims()` / `tensor_rank()` / `set_tensor_dtype()`。

构建：
```bash
cd gen_out/MyCustomPackage_Converter_Op_Package
make cpu   # → ConverterOpPackage/libConverterOpPackage.so
```

## 5. 步骤 3 — C++ Op Package（运行时）

沿用已有的 `src/MyPkgInterface.cpp` 模式（参考 `example/hmx_matmul_phase3/src/HmxMatMulPhase3Interface.cpp`）：

1. `sg_packageName = THIS_PKG_NAME_STR;`（由 `-DTHIS_PKG_NAME=MyCustomPackage` 注入）
2. `sg_opNames[]` **必须** 列全所有 op 名（`PkgValidateOpConfig` 按此校验；漏一个 runtime 会 "node validating failed"）
3. `qhpi_init()` 里调 `register_<op>_op()` 注册每个 kernel
4. **三平台同时构建**：
   - `build.sh`  → hexagon-v75（HTP 运行时）
   - `build.sh`  → aarch64-android（设备 ARM prepare）
   - `build_x86.sh` → x86_64（host ctxgen）

### build_x86.sh 关键要点

```bash
# 只链 libnative（Hexagon intrinsic stubs），不要链 libQnnHtp / libHtpPrepare
# （链了会拉入 libSnpeHtp.so 依赖而 x86 无此 so）
"$CXX" -std=c++17 -shared "${COMMON_FLAGS[@]}" \
    -o "$OUT_X86/libMyCustomPackage.so" \
    src/*.cpp kernel/*.o \
    -Wl,--whole-archive -L "$X86_LIBNATIVE/lib" -lnative -Wl,--no-whole-archive \
    -lpthread
```

kernel .c 文件必须有 `#if !defined(__hexagon__)` 的 fallback（空/scalar 实现），x86 ctxgen 只读 metadata 不实际执行 kernel。

参考：`example/hmx_matmul_phase3/build_x86.sh`。

## 6. 步骤 4 — 写 custom-domain ONNX

Python 脚本生成 ONNX，用 `domain=<XML Domain 值>` 声明自定义节点：

```python
import onnx
from onnx import helper, TensorProto, numpy_helper

DOMAIN = "mypkg"   # must match XML Domain="mypkg"

# 非 STATIC 的 tensor 都是 graph inputs (APP_WRITE)
#   STATIC 的（如 weight）作为 initializer
act_in  = helper.make_tensor_value_info("act",  TensorProto.UINT8, [1,1,M,K])
out_tv  = helper.make_tensor_value_info("out",  TensorProto.UINT8, [1,M_T,N_T,1024])
wt_init = numpy_helper.from_array(weight_u8, name="wt")

node = helper.make_node("MyOpA", ["act", "wt"], ["out"], name="myop_a", domain=DOMAIN)

graph = helper.make_graph([node], "my_graph", [act_in], [out_tv], [wt_init])
model = helper.make_model(graph, opset_imports=[
    helper.make_opsetid("", 13),
    helper.make_opsetid(DOMAIN, 1),
])
model.ir_version = 8
# ⚠️ 不要 onnx.checker.check_model() —— 它会因自定义 op 而报错
onnx.save(model, "my_model.onnx")
```

**Tensor type 铁律**：
- graph input (APP_WRITE)：按 kernel 期望的类型填（通常 UINT8、UINT16）
- graph output (APP_READ)：同上
- STATIC initializer：weight 常用 UFIXED_POINT_8 绕过 Cast +128；bias/scratch 建议走 APP_WRITE 而非 STATIC，避免 VTCM 驻留问题

## 7. 步骤 5 — qairt-converter → DLC

**核心命令**（NONTRIVIAL 四连发缺一不可）：

```bash
CPL=$(pwd)/gen_out/MyCustomPackage_Converter_Op_Package/ConverterOpPackage/libConverterOpPackage.so
qairt-converter \
    -i my_model.onnx \
    --op_package_config MyCustomPackage.xml \
    --converter_op_package_lib $CPL \
    --source_model_input_layout act        NONTRIVIAL \
    --source_model_input_layout bias       NONTRIVIAL \
    --source_model_input_layout scratch    NONTRIVIAL \
    --source_model_output_layout out       NONTRIVIAL \
    --desired_input_layout       act       NONTRIVIAL \
    --desired_input_layout       bias      NONTRIVIAL \
    --desired_input_layout       scratch   NONTRIVIAL \
    --desired_output_layout      out       NONTRIVIAL \
    -o my_model.dlc
```

**为什么 NONTRIVIAL 四连发**：converter 默认对 rank-4 tensor 做 NCHW→NHWC permutation，既会重排 dims（`[1,16,16,1024]` → `[1,16,1024,16]`）也会插入 Transpose 节点。XML 里的 `<Layout>NONTRIVIAL</Layout>` 只阻止 Transpose 节点插入，不阻止 IO dims 重排。CLI 四个标志都必须加：
- `--source_model_input_layout`：声明 ONNX 侧该 tensor 的 layout
- `--desired_input_layout`：声明 QNN 侧该 tensor 的 layout（同样 NONTRIVIAL 才不做 permute）
- output 两个同理

**验证**：
```bash
qairt-dlc-info -i my_model.dlc | grep "tensor dimension"
# 期望：dims 与 ONNX 声明完全一致
```

## 8. 步骤 6 — Context binary + schematic

```bash
X86_PKG=$(pwd)/../build/x86_64-linux-clang/libMyCustomPackage.so

qnn-context-binary-generator \
    --backend $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
    --dlc_path my_model.dlc \
    --op_packages "$X86_PKG:MyCustomPackageInterfaceProvider" \
    --binary_file my_ctx \
    --output_dir ctx_out \
    --config_file htp_config.json \
    --profiling_level detailed --profiling_option optrace \
    --save_backend_op_mapping
```

产出：
- `ctx_out/my_ctx.bin` — context binary
- `my_model_schematic.bin` — **profile-viewer 解 optrace 必需**（drop 在 CWD，会被混在目录里，用完移走）
- `my_model_bottom_mapping.json` — tensor↔op 映射

`htp_config.json`（模板）：
```json
{
  "backend_extensions": {
    "shared_library_path": "libQnnHtpNetRunExtensions.so",
    "config_file_path": "htp_backend_ext.json"
  }
}
```

`htp_backend_ext.json`（模板，`graph_names` 必须匹配 ONNX 里 `make_graph` 的 name）：
```json
{
  "graphs": [{"graph_names": ["my_graph"], "vtcm_mb": 8, "O": 3}],
  "devices": [{
    "dsp_arch": "v75", "soc_id": 57, "pd_session": "unsigned",
    "cores": [{"core_id": 0, "perf_profile": "burst", "rpc_control_latency": 100}]
  }]
}
```

## 9. 步骤 7 — 设备执行

**推上 ctx binary + schematic + input .raw**：
```bash
ssh $DEV "cat > qnn_run/phase/my_ctx.bin"             < ctx_out/my_ctx.bin
ssh $DEV "cat > qnn_run/phase/my_model_schematic.bin" < my_model_schematic.bin
ssh $DEV "cat > qnn_run/phase/runtime_inputs/act.raw" < runtime_inputs/act.raw
# （注意：scp 在 termux 上常 FORTIFY umask 崩，用 ssh cat > file < local 代替）
```

**qnn-net-run 调用**：
```bash
ssh $DEV 'cd ~/qnn_run/phase && rm -rf out && \
  LD_LIBRARY_PATH=../:.:/vendor/lib64 ADSP_LIBRARY_PATH=../ \
  ../qnn-net-run \
    --backend ../libQnnHtp.so \
    --retrieve_context my_ctx.bin \
    --op_packages ../libMyCustomPackage_cpu.so:MyCustomPackageInterfaceProvider:CPU,\
../libMyCustomPackage_htp.so:MyCustomPackageInterfaceProvider:HTP \
    --input_list input_list.txt \
    --profiling_level detailed --profiling_option optrace \
    --config_file htp_config.json \
    --output_dir out \
    --use_native_input_files'
```

**`--use_native_input_files` 必加**：否则 qnn-net-run 把 .raw 当 fp32 解读，对 uint8/uint16 会报 "batch size = 2 does not match ... batch size = 4"（字节比 2/4→ratio 2 ↔ expected 4）。

## 10. 步骤 8 — 导出 chrometrace

**host 端**（需要 schematic.bin + optrace reader .so）：
```bash
ssh $DEV "cat qnn_run/phase/out/qnn-profiling-data_0.log" > device_out/profile.log

cat > device_out/optrace_config.json <<'EOF'
{"enable_input_output_flow_events": false, "enable_sequencer_flow_events": false,
 "htp_json": true, "runtrace": true, "memory_info": true}
EOF

LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang \
qnn-profile-viewer \
    --config device_out/optrace_config.json \
    --reader $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so \
    --input_log device_out/profile.log \
    --schematic my_model_schematic.bin \
    --output device_out/chrometrace.json
```

产物：
- `chrometrace.json` — 用 `chrome://tracing` 或 Perfetto 打开
- `chrometrace_qnn_htp_analysis_summary.json` — QHAS 资源级汇总（HMX/HVX 利用率、VTCM/DRAM 带宽）
- `chrometrace_htp.json` — HTP-only 精简视图
- `chrometrace_runtrace.json` — 事件顺序

## 11. 常见踩坑 FAQ

| 症状                                                                   | 根因                                                                   | 修 |
|-----------------------------------------------------------------------|------------------------------------------------------------------------|----|
| qairt-converter 报 `data_type UINT_8 does not match op package config` | ONNX 声明 UINT8 但 XML 只列 UFIXED_POINT_8（或反之）                   | XML 两种都列 |
| DLC info 里 tensor dims 被转置成 NHWC                                  | 没加 `--source_model_*_layout NONTRIVIAL` 四连                          | 补全四连发 |
| ctxgen: `libSnpeHtp.so: cannot open shared object file`               | x86 op-pkg 链了 libQnnHtp/libHtpPrepare                                | 只链 libnative + libc |
| ctxgen: `Unable to load op pkg ... wrong ELF class: ELFCLASS32`        | 传给 ctxgen 的是 hexagon-v75 .so                                        | 传 x86_64 版 |
| qnn-net-run: `QnnSystemDlc_composeGraphs 1002 nullptr graphsInput`     | op-pkg 没加载 / 版本不匹配 / quantize 元数据缺失                         | 通过 `--op_packages` 传全 CPU + HTP 两份 |
| qnn-net-run: `node validating failed` for `<op>`                       | op-pkg Interface.cpp `sg_opNames[]` / `validateOpConfig` 漏登记该 op   | 补上再 rebuild |
| qnn-net-run: `graph_name not found in graphsConfigInfo`                | `htp_backend_ext.json` 里 `graph_names` 和模型里实际 graph name 不符   | 和 ONNX `make_graph` 第 2 参数对齐 |
| qnn-net-run: `batch size = N does not match expected M`（N < M）       | 没加 `--use_native_input_files`，QNN 按 fp32 解 u8/u16                   | 加上 |
| qnn-profile-viewer: `No Valid Input Schematics`                       | 少了 schematic.bin 或 ctxgen 没加 `--profiling_option optrace`         | 两边都加 |
| graph execute 无任何额外报错就失败（HTP 死机）                          | kernel 收到被 NHWC 重排后的 dims，读越界                                | NONTRIVIAL 四连 + 检查 qhpi_tensor_shape() 输出 |
| scp 到 termux 报 `FORTIFY: umask invalid`                              | termux sshd bug                                                         | 用 `ssh $DEV "cat > X" < local` 替代 |

## 12. 快捷复现（V8 reference impl）

完整可跑参考在：`example/hmx_matmul_phase3/standard_flow/phaseB_v8/`

一键复现：
```bash
cd example/hmx_matmul_phase3
bash build.sh            # hexagon-v75 + aarch64-android
bash build_x86.sh        # x86_64

cd standard_flow/phaseB_v8
python gen_v8_onnx.py
(cd gen_out/HmxMatMulPhase3Package_Converter_Op_Package && make cpu)

# ↓ qairt-converter 全长命令见 Agent/v8_vs_native_optrace_2026-04-25.md 第 "Reproduce" 节
```

## 13. 产物清单 checklist

端到端跑完应有：
- [ ] `standard_flow/<pkg>/my_model.onnx` — custom-domain ONNX
- [ ] `gen_out/<Pkg>_Converter_Op_Package/.../libConverterOpPackage.so`
- [ ] `build/x86_64-linux-clang/libMyCustomPackage.so`
- [ ] `build/hexagon-v75/libMyCustomPackage_htp.so`
- [ ] `build/aarch64/libMyCustomPackage_cpu.so`
- [ ] `standard_flow/<pkg>/my_model.dlc`
- [ ] `standard_flow/<pkg>/ctx_out/my_ctx.bin`
- [ ] `standard_flow/<pkg>/my_model_schematic.bin`
- [ ] `device_out/profile.log`、`device_out/chrometrace.json`、`chrometrace_qnn_htp_analysis_summary.json`

chrometrace 可以用 `chrome://tracing` 打开，和 QNN native baseline（如 `example/qnn_matmul_profile/sweep_data_*/s<dim>/<dtype>/chrometrace.json`）并排对比。

---
*持久化于 2026-04-25，基于 V8 matmul 跑通经验。后续改动请在这里直接更新。*
