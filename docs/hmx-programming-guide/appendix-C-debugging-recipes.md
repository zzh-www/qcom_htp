# 附录 C · Debug 配方

一组 copy-paste-ready 的 debug snippet，在你的 HMX kernel 挂的时候能用上。

## 1. 打印 acc 的低 16 bit

HMX acc 看不见，只能 convert 读回。快速 dump 方法：

```c
/* 在你怀疑的 MAC 后面插入: */
static uint16_t _dbg_tmp[1024];
asm volatile("mxmem(%0,%1):after:retain.uh = acc:2x1"
             :: "r"(_dbg_tmp), "r"(0) : "memory");
printf("acc dump (low 16 bit):\n");
for (int i = 0; i < 32; i++) {
    int pr = i & 15, st = i >> 4;
    for (int j = 0; j < 32; j++) {
        printf("%5u ", _dbg_tmp[pr * 64 + 2 * j + st]);
    }
    printf("\n");
}
```

**`:retain` 很关键**——否则 acc 被消费后面真正的 convert 读到空。

## 2. Single-hot byte probe（pack 公式验证）

```c
memset(act, 0, 2048);
memset(wt,  0, 1024);

/* 只在一个位置放 1 */
int probe_act_off = /* 你怀疑的字节偏移 */;
act[probe_act_off] = 1;
/* wt 某个 byte 也放 1 (或换成全 1 看 act 激活哪些 cell) */
memset(wt, 1, 1024);

/* MAC + readback */
asm volatile("mxclracc" ::: "memory");
asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
asm volatile("{ activation.ub = mxmem(%0,%1)\n"
             "  weight.b      = mxmem(%2,%3) }"
             :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
             :: "r"(out), "r"(0) : "memory");

/* 找所有 nonzero cell */
printf("Nonzero cells:\n");
for (int i = 0; i < 1024; i++)
    if (out[i]) printf("  idx=%4d (phys_row=%d col=%d stream=%d) val=%u\n",
                       i, i/64, (i%64)/2, i%2, out[i]);
```

## 3. Sim exit code 打印

`test_hmx_programming_guide.sh` 已经做了。手动跑时：

```sh
source scripts/env.sh
H2=tools/h2-install
hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    -- "$H2/bin/booter" --ext_power 1 --use_ext 1 --fence_hi 0xfe000000 \
       example/hmx_programming_guide/demoNN
echo "exit=$?"
```

`simulated_returnval` 让 sim 返回程序的实际 exit code（来自 `h2_thread_stop(n)`）。
`exit=0` = PASS。

## 4. 查看编译产物的 HMX 指令

```sh
HLLVM=tools/hexagon-sdk/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump
$HLLVM -d --mcpu=hexagonv75 --mattr=+hmxv75,+hvxv75 your_binary \
    | grep -E "activation\.|weight\.|mxclracc|mxmem|acc:"
```

看：
- 你期望的所有 HMX 指令都出现了吗？
- activation/weight 是同 packet 还是拆开了（看花括号 `{` 和 `}` 位置）？

## 5. Sim 的 PMU stats

```sh
hexagon-sim --mv75 --mhmx 1 --simulated_returnval \
    --statsfile /tmp/sim.stats \
    -- booter --ext_power 1 --use_ext 1 your_binary
grep ":[1-9]" /tmp/sim.stats | head -20   # non-zero counters
```

可以看到总 cycle 数、HMX packet 数等。对比多个版本的 kernel 看吞吐差别。

## 6. 强制 printf flush

sim 有时 buffer 较深，kernel 挂前的 printf 看不到：

```c
printf("before MAC\n");
fflush(stdout);
/* ... your MAC ... */
printf("after MAC\n");
fflush(stdout);
```

## 7. h2 init 失败怎么办

症状：`h2_vecaccess_acquire` / `h2_mxaccess_acquire` 返回但后续 mxmem segfault。

检查：
- 是否先跑了 `bash scripts/install.sh` 装 SDK + H2
- `tools/h2-install` 符号链接是否指向有效编译产物
- `ARCHV=75` 传给 hypervisor make

重建 H2：
```sh
cd tools/hexagon-hypervisor
make ARCHV=75 TARGET=ref USE_PKW=0
cd ../..
ln -sfn hexagon-hypervisor/install tools/h2-install
```

## 8. "Architecture mismatch" 警告

sim log 开头有：`Architecture of executable does not match command-line architecture, main_arch=73, cmdline_cpu=75`.

这是 **无害警告** —— H2 booter 自己是 v73 编的，sim 把它 boot 起来后切 v75 模式
跑你的代码。忽略。

## 9. 检查 VTCM 地址

HMX 的 mxmem 只读 VTCM。想确认 `Rs` 对：

```c
printf("vtcm_base = %p\n", (void *)(unsigned long)vtcm_base);
printf("act = %p, aligned %d\n", act, ((uintptr_t)act & 127) == 0);
printf("wt  = %p, aligned %d\n", wt,  ((uintptr_t)wt & 127) == 0);
```

期望：`act` 和 `wt` 都在 VTCM 范围内（sim 打印的 VTCM base 附近）且 128 B 对齐。

## 10. 比对 C ref 的 corner cells 第一

1024 个 cell 全打印太多。先只打 4 个"角"：

```c
int corners[][2] = { {0,0}, {0,31}, {16,0}, {31,31} };
for (int c = 0; c < 4; c++) {
    int i = corners[c][0], j = corners[c][1];
    printf("  C[%d,%d] hmx=%d ref=%d %s\n",
           i, j, Chmx[i][j], Cref[i][j],
           (Chmx[i][j]==Cref[i][j]) ? "OK" : "MISMATCH");
}
```

如果只角对 → unpack 或 stream 搞反。如果只 (0,0) 对 → pack 循环只覆盖第一个 cell。

## 参考

- 更完整的症状—原因对照见 [ch12 pitfalls](12-pitfalls-and-debugging.md)
- hexagon-sim 手册：`docs/hexagon_sim_handbook.md`
