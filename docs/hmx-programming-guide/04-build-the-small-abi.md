# 04 C++ wrapper 只搭一个很小的 ABI

数据准备好以后，hot callback 的工作应该很窄：

```text
把已经准备好的 QHPI state
变成 HMX body 期待的几个 native descriptor
```

它不是一个数学函数，也不应该是一个通用 layout adapter。

## QHPI precompute 先存什么

在 graph/context 阶段，`hmx_u8i8_precompute()` 会记录：

```text
bias_bytes
  Direct TCM native bias record

wt_pack
  Direct TCM K-major packed weight

act_qhpi_table
  activation pointer-table copy

out_qhpi_table
  output pointer-table copy

M_t / N_t / K_t
  32-wide tile count

mask state
  已经初始化好的 V73DEEP mask
```

对 256x256x256：

```text
M_t = 8
N_t = 8
K_t = 8
mt_groups = 4

act entries = 32
out entries = 32
```

这样 hot callback 就不用每次再向 QHPI 查询 tensor metadata。

## hot callback 临时搭什么

运行时只需要搭三块小东西：

```text
out_desc
act_desc
extra_param[2]
```

可以把它们理解成 HMX body 的入场券。

`out_desc` 告诉 body 输出怎么走：

```text
out_tile_ptr_table       = output pointer table
out_table_stride_dwords  = N_t
out_y_stride_words       = M_t * 4
n_tiles_pow2             = M_t * 4
m_total_minus_step       = 8
k_total_bytes            = N_t * 32
```

`act_desc` 告诉 body activation 怎么走：

```text
act_ptr_pairs            = activation pointer table
n_act_pairs              = K_t
act_table_y_stride_words = M_t * 4
```

`extra_param` 当前固定是：

```text
extra_param = {1, 0}
```

descriptor 是 stack-local，并且 64-byte aligned。这个对齐不是装饰，它避免 body 读 descriptor 时出现额外风险。

## register ABI

最终 C++ call 会变成这组 native register：

```text
r0 = out_desc
r1 = act_desc
r2 = packed_weight
r3 = folded_bias_record
r4 = mask_desc
r5 = extra_param
```

HMX body 会按固定 offset 读取这些结构。因此 descriptor 字段顺序、单位、对齐，都是 ABI 的一部分。

## hot callback 不该做什么

hot callback 不应该做：

```text
每次恢复 shape
每次复制 QNN pointer table
每次 patch mask
运行时 pack weight
运行时 fold bias/scale
支持一堆没有验证过的 layout 分支
```

这些事情都会把非计算成本塞进 custom op 的 profile event。

## 为什么这一步解决了大部分 gap

之前的路径是：

```text
hot callback
  QHPI lookup
  shape recovery
  pointer-table copy
  descriptor build
  HMX body
```

现在的路径是：

```text
hot callback
  descriptor build
  HMX body
```

probe 结果显示，当前 descriptor glue 大约是 29 cycles。256 chain8 的 owned HMX body 本身大约是 1074 cycles。

这说明剩下的差距已经不是“我们多做了一堆 MatMul 计算”，而是 wrapper/profiling 边界上的小成本。
