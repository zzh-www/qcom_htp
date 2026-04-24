/*
 * demo_probe_n.c — single-hot-nibble 探针反推 weight.n tile layout。
 *
 * 策略: A = 1 everywhere，wt 全 0 除一个字节某 nibble 置 0x07，
 * 观察 out 里哪一列被激活，cell 值多少，cells 分布如何。
 *
 * 输出格式: 每个 (byte_off, nibble) 一行，列出 active_col / n_cells / val。
 * 对 byte_off 扫描 512 B (weight.n tile 总字节)，但为了控制输出量，
 * 仅 dump 有激活的条目。
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <h2.h>
#include <h2_common_info.h>
#include <h2_vecaccess.h>
#include <h2_mxaccess.h>
#include <hexagon_types.h>

static void fill_act_ones(uint8_t *act) {
    memset(act, 0, 2048);
    for (int pr = 0; pr < 16; pr++)
        for (int K = 0; K < 32; K++) {
            act[128*pr + 4*K + 1] = 1;
            act[128*pr + 4*K + 3] = 1;
        }
}
static void fill_bias(uint16_t *b, uint16_t v) {
    for (int i = 0; i < 128; i++) b[i] = v;
}

/* 打一 hot nibble，跑 MAC，返回 (active_col, n_cells, val_seen)。*/
static void probe(uint8_t *wt, const uint8_t *act, const uint16_t *bias,
                  uint16_t *out, int byte_off, int hi,
                  int *col, int *cells, int *val)
{
    memset(wt, 0, 1024);
    wt[byte_off] = hi ? 0x70 : 0x07;
    memset(out, 0, 2048);

    asm volatile("mxclracc" ::: "memory");
    asm volatile("bias = mxmem(%0)" :: "r"(bias) : "memory");
    asm volatile("{ activation.ub = mxmem(%0,%1)\n"
                 "  weight.n      = mxmem(%2,%3) }"
                 :: "r"(act), "r"(2047), "r"(wt), "r"(2047) : "memory");
    asm volatile("mxmem(%0,%1):after.uh = acc:2x1"
                 :: "r"(out), "r"(0) : "memory");

    *col = -1; *cells = 0; *val = 0;
    for (int jc = 0; jc < 32; jc++) {
        int cnt = 0;
        uint16_t v = 0;
        for (int ir = 0; ir < 32; ir++) {
            int pr = ir & 15, st = ir >> 4;
            uint16_t x = out[pr * 64 + 2 * jc + st];
            if (x) { cnt++; v = x; }
        }
        if (cnt > 0) {
            if (*col == -1) { *col = jc; *val = v; }
            *cells += cnt;
        }
    }
}

int main(void) {
    unsigned int vtcm = h2_info(INFO_VTCM_BASE);
    if (!vtcm) { printf("[FAIL] no VTCM\n"); h2_thread_stop(1); return 1; }

    h2_vecaccess_state_t va;
    h2_vecaccess_unit_init(&va, H2_VECACCESS_HVX_128, CFG_TYPE_VXU0,
                           CFG_SUBTYPE_VXU0, CFG_HVX_CONTEXTS, 0x1);
    h2_vecaccess_acquire(&va);
    h2_mxaccess_state_t ma;
    h2_mxaccess_unit_init(&ma, CFG_TYPE_VXU0, CFG_SUBTYPE_VXU0,
                          CFG_HMX_CONTEXTS, 0x1);
    h2_mxaccess_acquire(&ma);

    uint8_t  *vt   = (uint8_t *)(unsigned long)vtcm;
    uint8_t  *act  = vt + 0*2048;
    uint8_t  *wt   = vt + 2*2048;
    uint16_t *bias = (uint16_t *)(vt + 4*2048);
    uint16_t *out  = (uint16_t *)(vt + 6*2048);

    fill_act_ones(act);
    fill_bias(bias, 0x4000);

    printf("--- probe_n_layout: single hot nibble in weight.n tile ---\n");
    printf("A=1 everywhere, wt[byte_off] = 0x07 or 0x70, rest zero\n");
    printf("byte  nib  col  cells  val\n");

    /* 扫 128 B = 第一条 128 B 线，这样我们能看清一条线内的结构 */
    for (int off = 0; off < 128; off++) {
        int col, cells, val;
        probe(wt, act, bias, out, off, 0, &col, &cells, &val);
        if (col >= 0)
            printf("%4d  lo   %3d   %4d  %3d\n", off, col, cells, val);
        probe(wt, act, bias, out, off, 1, &col, &cells, &val);
        if (col >= 0)
            printf("%4d  hi   %3d   %4d  %3d\n", off, col, cells, val);
    }
    /* 再扫线 1 的首 8 B，验证跨线的 K-stride */
    printf("--- line 1 head (offsets 128..135) ---\n");
    for (int off = 128; off < 136; off++) {
        int col, cells, val;
        probe(wt, act, bias, out, off, 0, &col, &cells, &val);
        if (col >= 0)
            printf("%4d  lo   %3d   %4d  %3d\n", off, col, cells, val);
        probe(wt, act, bias, out, off, 1, &col, &cells, &val);
        if (col >= 0)
            printf("%4d  hi   %3d   %4d  %3d\n", off, col, cells, val);
    }

    printf("[PASS] probe_n_layout (informational)\n");
    h2_thread_stop(0);
    return 0;
}
