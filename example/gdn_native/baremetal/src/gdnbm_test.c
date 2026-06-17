/* gdnbm_test.c — host driver: load A.raw, run the bare-metal threaded solve, write T.raw + stats. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gdnbm.h"
#include "remote.h"

static long fsize(const char *p) { FILE *f = fopen(p, "rb"); if (!f) return -1; fseek(f, 0, SEEK_END); long n = ftell(f); fclose(f); return n; }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: markers survive SIGKILL (for hang localization) */
    if (argc < 10) {
        printf("usage: %s <nthreads> <A.raw> <T.raw> <H> <C> <zpA> <zpT> <sA> <sT> [probe]\n", argv[0]);
        return 2;
    }
    int nthreads = atoi(argv[1]);
    const char *Apath = argv[2], *Tpath = argv[3];
    int H = atoi(argv[4]), C = atoi(argv[5]), zpA = atoi(argv[6]), zpT = atoi(argv[7]);
    union { float f; int i; } sA, sT; sA.f = strtof(argv[8], 0); sT.f = strtof(argv[9], 0);

    struct remote_rpc_control_unsigned_module data; data.domain = CDSP_DOMAIN_ID; data.enable = 1;
    remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&data, sizeof(data));
    char uri[256]; snprintf(uri, sizeof(uri), "%s%s", gdnbm_URI, "&_dom=cdsp");
    remote_handle64 h = -1;
    int rc = gdnbm_open(uri, &h);
    printf("gdnbm_open rc=0x%x\n", rc); if (rc) return 1;

    if (argc > 10) { /* run the HMX-on-worker probe too */
        int n = 1 + 2 * nthreads, *res = (int *)calloc(n, sizeof(int));
        gdnbm_hmx_probe(h, nthreads, res, n);
        for (int i = 0; i < nthreads; ++i) printf("  probe worker %d: hmx ctx=%d sent=0x%X\n", i, res[1 + 2 * i], res[1 + 2 * i + 1]);
        free(res);
    }

    long abytes = (long)H * C * C * 2;
    if (fsize(Apath) < abytes) { printf("A.raw too small (%ld < %ld)\n", fsize(Apath), abytes); return 3; }
    /* ACVRES (照搬 pure-HMX cv-block A): the A input may be EXTENDED with a host-prepared cv-block region
     * appended after the H natural-A heads. Read the FULL file so the appended region reaches the DSP; the
     * DSP only reads abytes (natural A) for diag + the appended cv-block for the merges. T stays abytes. */
    long afile = fsize(Apath); if (afile < abytes) afile = abytes;
    unsigned char *A = (unsigned char *)malloc(afile), *T = (unsigned char *)calloc(abytes, 1);
    FILE *fa = fopen(Apath, "rb"); fread(A, 1, afile, fa); fclose(fa);

    int stats[32] = {0};
    /* REPS: run the solve N times in ONE FastRPC session (env GDNBM_REPS, default 1). Repeated single-shot
     * processes churn the unsigned-PD FastRPC session and OCCASIONALLY deadlock the cDSP; looping the
     * remote call on the SAME handle does the steady-state measurement with NO session churn. */
    const char *re = getenv("GDNBM_REPS"); int reps = re ? atoi(re) : 1; if (reps < 1) reps = 1;
    for (int r = 0; r < reps; ++r) {
        rc = gdnbm_solve(h, A, (int)afile, H, C, zpA, zpT, sA.i, sT.i, nthreads, T, (int)abytes, stats, 32);
        printf("gdnbm_solve rc=0x%x  P=%d  H=%d  rep=%d/%d  wall=%d scatter=%d wtpack=%d cons=%d\n",
               rc, stats[1], stats[2], r + 1, reps, stats[0], stats[9], stats[7], stats[3]);
        if (stats[15] > 0)   /* GDN_BR_HEADLOAD_PROBE: Σ/per-head head-load copy span vs wall */
            printf("  HEADLOAD: total_cyc=%d per_head=%d heads=%d  vs wall=%d  (Σ%%=%d, /4producers≈%d cyc wall)\n",
                   stats[13], stats[14], stats[15], stats[0],
                   stats[0] ? (int)((long long)stats[13] * 100 / stats[0]) : 0, stats[13] / 4);
        if (stats[25] > 0)   /* GDN_BR_W16_PACKCNT: clean wt-pack vs act-pack Σ-work + per-inst (vs pure-HMX 4026) */
            printf("  PACKCNT: wt_Sigma=%d wt_n=%d wt_perinst=%d | act_Sigma=%d act_n=%d act_perinst=%d\n",
                   stats[24], (stats[28] >> 16) & 0xffff, stats[25],
                   stats[26], stats[28] & 0xffff, stats[27]);
        if (rc) break;
    }
    /* ===== QNN-ALIGNED cycle report (cron#79, skill htp-cycle-metric). Every number is reported in
     * the THREE QNN categories — 真算(MAC,HMX) / 装料(prep,HVX) / 卸料+输入(IO,DMA) — each tagged with
     * its QNN op name + QNN field. This makes cross-impl comparison automatically same-category+same-field,
     * killing the cross-口径 mis-compares (e.g. native ConvLayer batch warm sub-op 263 vs our per-call 1577).
     * The DELETED vocabulary "①②③④ / op-latency / unit-busy / per-call wall" is GONE — say the QNN field.
     * Side-by-side native: scripts/gdn_solve_qnn_aligned_report.py --our-log <this stdout>. ===== */
    /* cron#80 gap#2: explicit PKTPROBE-build marker (stats[31] == "PKTP"). Production NEVER writes stats[31]
     * (its tail stops at stats[29]) and a PKTPROBE build returns BEFORE the production stats are set, so the
     * shared index segment (12..31) means PMU/packets in one build and IO/N_conv/NTSWEEP in the other. The
     * magic disambiguates WITHOUT the old implicit "stats[30]!=0" overload. */
    int is_pktprobe = (stats[31] == 0x504B5450);
    if (rc == 0 && !is_pktprobe) {
        int H_ = stats[2] ? stats[2] : 1;
        int nconv = stats[29] ? stats[29] : H_ * 24;   /* N_conv = H*24 mm @ Newton=0 (8 diag + 16 merge) */
        double cpu = stats[6] ? (double)stats[0] / (double)stats[6] : 0.0;   /* PCYCLE/µs clock self-check */
        printf("  ===== QNN-ALIGNED breakdown (真算 / 装料 / 卸料·IO ; QNN op + field; same-category compare only) =====\n");
        printf("  graph-wall (THE verdict, 32-head TOTAL, field=timeline span) = %d cyc  = %d cyc/conv\n",
               stats[0], nconv ? stats[0] / nconv : 0);
        printf("     basis: per-conv N_conv=H*24=%d (throughput denom; 8 diag + 16 merge, Newton=0) "
               "| per-matmul nmm=H*24=%d @Newton=0 (==N_conv; A^2/A^3 ARE the 2 diag mm, already counted, NOT extra)\n",
               nconv, H_ * 24);
        printf("     clock self-check wall/us = %d/%d = %.1f  (TURBO ~1594; >> => wrong counter, re-measure)\n",
               stats[0], stats[6], cpu);
        printf("  [真算-MAC] q::ConvLayer_s1.opt (HMX): Σcyc(cycles_used)=%d  per-conv=%d  | per-call occupancy=%d cyc\n",
               stats[3], nconv ? stats[3] / nconv : 0, stats[5]);
        printf("            apples-to-apples: native SINGLE ConvLayer cycles=1970 (we=%d, NOT slower); batch warm sub-op cycles=263 is NOT a conv wall (cron#78).\n",
               stats[5]);
        printf("  [装料-wt] convert_weights_to_signed+Cast (HVX): Σcyc=%d  per-conv=%d   (native batch per-conv 装料-wt=3425)\n",
               stats[17], nconv ? stats[17] / nconv : 0);
        printf("  [装料-bias] bias_weight_update+bias_scale_shuff (HVX): Σcyc=%d  per-conv=%d   (native batch per-conv 装料-bias=1703)\n",
               stats[18], nconv ? stats[18] / nconv : 0);
        printf("  [装料-act] q::ForceFormat_Crouton (HVX): Σcyc=%d  per-conv=%d   (native batch per-conv 装料-act=3024)\n",
               stats[19], nconv ? stats[19] / nconv : 0);
        printf("  [装料-alg] renorm/acc (solve-only; NO native op — honestly flagged): Σcyc=%d  per-conv=%d\n",
               stats[10], nconv ? stats[10] / nconv : 0);
        printf("  [卸料-IO] q::*OutputSlice (HVX vxor here): Σcyc=%d  | bulk DDR<->VTCM (q::*Input/OutputSlice, EXCL from wall)=%d\n",
               stats[28], stats[15] + stats[16]);
        printf("  [waste]   SPIN idle-wait (a GAP, not an op): Σcyc=%d   slowest-prod-life(lmax)=%d  PACKCHK=%d(0=ok)\n",
               stats[4], stats[11], stats[8]);
        printf("  THROUGHPUT 口径: graph-wall/N_conv = %d cyc/conv (NEVER cycles_used/N — that OVERSTATES, trap#6).\n",
               nconv ? stats[0] / nconv : 0);
        printf("     packets/cyc-per-pkt: build -DGP_PKTPROBE (consumer MAC packets=130 native @2.04 cyc/pkt; ours nt8 below).\n");
    }
    if (stats[31] == 0x4C45414E)   /* cron#82 LEANCHK build marker ("LEAN"): print lean bit-exact result */
        printf("  LEANCHK max|d|=%d (0 = lean_mm64 bit-exact vs native our_v73deep_kernel_i16)\n", stats[30]);
#ifdef GP_LEANCHK_LIVE
    /* cron#83 LEANCHK_LIVE: device wrote stats[30]=max|d|, stats[31]=checked count (expect H*24=768). */
    printf("  LEANCHK_LIVE max|d|=%d checked=%d (0/768 = lean bit-exact over full live solve)\n",
           stats[30], stats[31]);
#endif
    printf("  raw stats[0..11]: %d %d %d %d %d %d %d %d %d %d %d %d\n",
           stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7], stats[8], stats[9], stats[10], stats[11]);
    printf("  raw stats[12..19]: %d %d %d %d %d %d %d %d\n",
           stats[12], stats[13], stats[14], stats[15], stats[16], stats[17], stats[18], stats[19]);
    if (!is_pktprobe) {   /* production: stats[20..27] = NTSWEEP/DISTINCT-TILE micro-bench (always run) */
        /* P2.3: GP_PMU_UTIL build reuses stats[20..23] for the post-solve PMU真值 pass (NTSWEEP gated off
         * there) and signs stats[26]="PMUU"; host prints the §6 tier-1 PMU line + raw [20..23] then. */
        if (stats[26] == 0x504D5555 /* "PMUU" */) {
            int cyc1 = stats[22];
            printf("  PMU_UTIL (§6 tier-1, separate clean-wall-safe pass): per-call COPROC_BUSY=%d THREAD_IDLE=%d "
                   "CYCLES_1T=%d PKT=%d | HMX-util COPROC/CYC1T=%d%%\n",
                   stats[20], stats[21], stats[22], stats[23], cyc1 ? stats[20] * 100 / cyc1 : 0);
            printf("  raw PMU stats[20..23]: %d %d %d %d  (COPROC THREAD_IDLE CYC1T PKT, per-call)\n",
                   stats[20], stats[21], stats[22], stats[23]);
        } else {
        printf("  NTSWEEP nt8=%d nt16=%d nt32=%d nt64=%d | per-walk=%d pure-conv(8walk)=%d fixed-feed=%d\n",
               stats[20], stats[21], stats[22], stats[23], stats[24], stats[25], stats[26]);
        printf("  DISTINCT-TILE nt8 32-distinct=%d (vs 4-reused nt8=%d; native 370)\n", stats[27], stats[20]);
        }
    } else {   /* cron#80 gap#2: GP_PKTPROBE diagnostic build (explicit magic stats[31] == "PKTP"). Host prints
                * only the EARLY single-final-writer slots that always run (packets/PMU/SMT); the later
                * DILATE/fp16/FANOUT sub-blocks reuse stats[18/19] AND can fault in this from-scratch
                * micro-kernel build, so they are AUTHORITATIVE only in the on-device FARF "GDN_PURE ..." lines
                * (read via the device log, not this host stats echo). */
        printf("  PKTPROBE packets/call: nt8=%d nt16=%d nt32=%d (native nt8=130 pkt; nt64 + cyc/pkt in FARF)\n",
               stats[28], stats[29], stats[30]);
        printf("  PKTPROBE PMU nt8: RUN1=%d IDLE=%d COPROC=%d CU=%d IU_NOPKT=%d SYSBUSY=%d (IDLE~1564 COPROC=0 => thread WAITs on HMX)\n",
               stats[12], stats[13], stats[14], stats[15], stats[16], stats[17]);
        printf("  PKTPROBE SMT spinners 0/1/2/4 = %d/%d/%d/%d (do NOT shrink conv => WAIT is this thread's HMX latency)\n",
               stats[20], stats[21], stats[22], stats[23]);
        printf("  PKTPROBE DILATE micro: SERIAL=%d PIPE=%d cyc/conv, %d pkt/conv (vs convhhh per-call=%d cyc / %d pkt)\n",
               stats[18], stats[19], stats[27], stats[5], stats[28]);
        printf("  raw stats[20..31]: %d %d %d %d %d %d %d %d %d %d %d 0x%08x\n",
               stats[20], stats[21], stats[22], stats[23], stats[24], stats[25],
               stats[26], stats[27], stats[28], stats[29], stats[30], (unsigned)stats[31]);
    }

    FILE *ft = fopen(Tpath, "wb"); fwrite(T, 1, abytes, ft); fclose(ft);
    printf("wrote %s (%ld bytes)\n", Tpath, abytes);
    gdnbm_close(h); free(A); free(T);
    return rc;
}
