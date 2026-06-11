  /* 1132 bytes — replica of tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
   * 0x2ebe40 `hmx_v73_convbbb1x1deep_stride1`.
   *
   * Full kernel body is hand-written Hexagon inline asm:
   *   prologue   P1..P11   @0x2ebe40..0x2ebecc  (stack/desc load + dispatch)
   *   main body  M1..M18   @0x2ebed0..0x2ebf9c  (K-MAC loop + drain + cvt)
   *   alt paths             @0x2ebfa0..0x2ec1fc  (non-production dispatch arms)
   *   final-red  F1..F11   @0x2ec200..0x2ec290  (last-K-pair drain)
   *   epilogue   E1..E4    @0x2ec294..0x2ec2a8  (mxclracc + restore + ret)
   * Numeric labels 1..26 mark internal branch targets:
   *   1=alt-A entry (0x2ec0a0)   2=main K-outer (0x2ebee0)
   *   3=main M-loop body (0x2ebf10)  4=main K-MAC body (0x2ebf20)
   *   5=cvt skip target (0x2ebf5c)   6=alt-B entry (0x2ebfa0)
   *   14..18=alt-B internal loops/drains
   *   19..21=alt-A internal loops, 22..26=alt-C internal loops/drains
   *   9=final-reduce entry (0x2ec200)  10=F3 loop1 (0x2ec220)
   *   11=F5 K-MAC body (0x2ec240)      12=F8 drain (0x2ec260)
   *   13=epilogue (0x2ec294)
   *
   * Each packet was byte-verified against native via /tmp/v73deep_fullasm/.
   * To regen the all-.byte fallback for sanity check:
   *   python3 scripts/extract_v73deep_bytes.py
   *   (would overwrite this file with raw bytes).                       */
  /* P1 @0x2ebe40: stack alloc 0x28 + load mask[0..15] (mask = r4 desc) */
  "{ r29 = add(r29,#-0x28); nop;"
  "  r7:6   = memd(r4+#0x8);"
  "  r11:10 = memd(r4+#0x0) }\n"
  /* P2 @0x2ebe50: load r9=mask[+0x18] (n_act_pairs<<...|arg5),
   *               load r15=ad[+0x8] (act y stride). 2 nops fill slots */
  "{ nop; nop;"
  "  r9  = memw(r4+#0x18);"
  "  r15 = memw(r1+#0x8) }\n"
  /* P3 @0x2ebe60: extract mask[0..10] -> r7 (act_rt), shift y_stride x4,
   *               save r17:16 / r19:18 (callee-saved) via duplex store.
   *               12 bytes (3 words), last word is duplex container. */
  "{ r7  = extractu(r7,#0xb,#0x0);"
  "  r15 = asl(r15,#0x2);"
  "  memd(r29+#0x0) = r17:16;"
  "  memd(r29+#0x8) = r19:18 }\n"
  /* P4 @0x2ebe6c: r22 = (r9 + 1) (=>halved next packet for n_act_pairs),
   *               save r21:20 / r23:22 (duplex). 8 bytes (2 words). */
  "{ r22 = add(r9,#0x1);"
  "  memd(r29+#0x10) = r21:20;"
  "  memd(r29+#0x18) = r23:22 }\n"
  /* P5 @0x2ebe74: r22 = (n_act_pairs+1)>>1 (rounded half),
   *               r4 = ad[+0x4] (n_tiles_pow2),
   *               r1 = ad[+0x0] (act base ptr). Two memw -> duplex.
   *               8 bytes (2 words). */
  "{ r22 = lsr(r22,#0x1);"
  "  r4  = memw(r1+#0x4);"
  "  r1  = memw(r1+#0x0) }\n"
  /* P6 @0x2ebe7c: p1 = n_tiles_pow2.bit0  (odd-tile flag),
   *               p0 = (n_tiles_pow2 > 1),
   *               r24 = ep[0]++4   (extra_param[0]),
   *               save r25:24 to stack. 16 bytes (4 words, no duplex). */
  "{ p1  = tstbit(r4,#0x0);"
  "  p0  = cmp.gt(r4,#0x1);"
  "  r24 = memw(r5++#0x4);"
  "  memd(r29+#0x20) = r25:24 }\n"
  /* P7 @0x2ebe8c: r23 = r1 (act_base saved for outer loop reset),
   *               p2 = (ep[0] == 1) (selects "main" path when also p0),
   *               r17 = od[+0x4], r16 = od[+0x8] (duplex memw).
   *               12 bytes (3 words). */
  "{ r23 = r1;"
  "  p2  = cmp.eq(r24,#0x1);"
  "  r17 = memw(r0+#0x4);"
  "  r16 = memw(r0+#0x8) }\n"
  /* P8 @0x2ebe98: r17:16 = vaslw(.,#2) (od y/x strides x4 in parallel),
   *               r28 = n_tiles_pow2 >> 1 (loop0 trip),
   *               r20 = od[+0xc], r13 = od[+0x14]. 16 bytes (4 words). */
  "{ r17:16 = vaslw(r17:16,#0x2);"
  "  r28 = lsr(r4,#0x1);"
  "  r20 = memw(r0+#0xc);"
  "  r13 = memw(r0+#0x14) }\n"
  /* P9 @0x2ebea8: r13 += 0x1f (round-up for div32 next),
   *               r17 -= 4 (post-decrement once for stride),
   *               r20 += 7 (round-up for div8 next). 8 bytes (2 words). */
  "{ r13 = add(r13,#0x1f);"
  "  r17 = add(r17,#-0x4);"
  "  r20 = add(r20,#0x7) }\n"
  /* P10 @0x2ebeb0: m0 = (od_y_stride*4 - 4) (post-incr modifier for r0++m0),
   *                r13 = ceil(K/32),  r12 = od[+0x10] (M_t),
   *                r0  = od[+0x0]  (out base ptr). 16 bytes (4 words). */
  "{ m0  = r17;"
  "  r13 = lsr(r13,#0x5);"
  "  r12 = memw(r0+#0x10);"
  "  r0  = memw(r0+#0x0) }\n"
  /* P11 @0x2ebec0: r20 = ceil(M_t/8), r19:18 = combine(r1,r0) (saved act+out
   *                base pair), branch to alt-A if n_tiles_pow2 <= 1,
   *                else if ep[0]==1 (p2) preload r25 = ep[1]. 16 bytes. */
  "{ r20    = lsr(r20,#0x3);"
  "  r19:18 = combine(r1,r0);"
  "  if (!p0) jump:nt 1f;"
  "  if (p2) r25 = memw(r5+#0x0) }\n"
  /* M1 @0x2ebed0: arm loop0 (K-MAC trip = r28),
   *               r22 = (n_act_pairs/2)*ceil(K/32) (wt step per K-iter),
   *               if (!p2) jump alt-B (n_act_pairs > 1 path).
   *               Single-insn nop packet @0x2ebedc fills slot wait. */
  "{ loop0(4f,r28);"
  "  r22 = mpyi(r22,r13);"
  "  if (!p2) jump:nt 6f }\n"
  "{ nop }\n"
  /* main K-outer entry */
  "2:\n"
  /* M2 @0x2ebee0: r3 += 0x101 (bias step), then bias = mxmem2(r3). */
  "{ r3 = add(r3,#0x101); nop; nop;"
  "  bias = mxmem2(r3) }\n"
  /* M3 @0x2ebef0: r13 -= 2 (K-iter ctr), r17 = M_t (M-loop ctr reset),
   *               r14 = act_alt + 8 (next-K act ptr base). */
  "{ r13 = add(r13,#-0x2);"
  "  r17 = r12;"
  "  r14 = add(r18,#0x8);"
  "  nop }\n"
  /* M4 @0x2ebf00: r3 += 0xff (= -1 from prev +0x101 → net +2 each K-step),
   *               second bias = mxmem2(r3) (paired bias for fan-out). */
  "{ r3 = add(r3,#0xff); nop; nop;"
  "  bias = mxmem2(r3) }\n"
  /* main M-loop body entry (loop1 start) */
  "3:\n"
  /* M5 @0x2ebf10: arm loop1 with M-trip r20, init act-pair r1:0 = (r19,r18),
   *               r17 -= 8 (M-tile counter step), r8 = wt base. */
  "{ loop1(4f,r20);"
  "  r1:0 = combine(r19,r18);"
  "  r17 = add(r17,#-0x8);"
  "  r8 = r2 }\n"
  /* main K-MAC inner loop body (loop0 start) */
  "4:\n"
  /* M6 @0x2ebf20: load act_tbl[k] (post-incr +8), peek r21 = act_tbl[k+1]. */
  "{ r6  = memw(r1++#0x8);"
  "  r21 = memw(r1+#0x4) }\n"
  /* M7 @0x2ebf28: K-MAC pair #1.
   *   r8 += r22 (advance wt for next pair-element),
   *   activation.ub = mxmem(r6,r7):deep:cm  (load act tile A, deep-cm fan-out),
   *   weight.b = mxmem(r8,r9):deep         (load wt tile A, accumulate). */
  "{ r8 = add(r8,r22);"
  "  activation.ub = mxmem(r6,r7):deep:cm;"
  "  weight.b = mxmem(r8,r9):deep }\n"
  /* M8 @0x2ebf34: K-MAC pair #2 (loop0 body tail).
   *   r8 += r22, act tile B (uses pre-fetched r21 as ptr, same act_rt),
   *   wt tile B. :endloop0 marks loop boundary. */
  "{ r8 = add(r8,r22);"
  "  activation.ub = mxmem(r21,r7):deep:cm;"
  "  weight.b = mxmem(r8,r9):deep }:endloop0\n"
  /* M9 @0x2ebf40: re-arm loop0 for next M-iter, r21 = ep[0] (=1, drain ctr),
   *               if !odd-tile (p1 clear) skip the odd-tile drain MAC,
   *               r10 = out_tbl[m]++4 (output ptr). */
  "{ loop0(4b,r28);"
  "  r21 = r24;"
  "  if (!p1) jump:nt 5f;"
  "  r10 = memw(r0++#0x4) }\n"
  /* M10 @0x2ebf50: odd-tile leftover act ptr (single-insn packet). */
  "{ r6 = memw(r1++#0x4) }\n"
  /* M11 @0x2ebf54: odd-tile final K-MAC pair (no r8 advance — last). */
  "{ activation.ub = mxmem(r6,r7):deep:cm;"
  "  weight.b = mxmem(r8,r9):deep }\n"
  /* main cvt path entry (also branch target when n_act_pairs == 1 odd-tile) */
  "5:\n"
  /* M12 @0x2ebf5c: r25 |= 1<<12 (cvt_rt: set deep-cvt fan-out high bit),
   *                cvt.ub = acc(r25)  (1st-half drain: M-tile pair A). */
  "{ r25 = setbit(r25,#0xc);"
  "  cvt.ub = acc(r25) }\n"
  /* M13 @0x2ebf64: r10 = out_tbl[m]++m0 (advance per modifier),
   *                store cvt to mxmem(r10,r11):cm  (output tile A). */
  "{ r10 = memw(r0++m0);"
  "  mxmem(r10,r11):cm = cvt }\n"
  /* M14 @0x2ebf6c: r25 &= ~(1<<12) (clear high bit → 2nd-half drain),
   *                cvt.ub = acc(r25)  (M-tile pair B). */
  "{ r25 = clrbit(r25,#0xc);"
  "  cvt.ub = acc(r25) }\n"
  /* M15 @0x2ebf74: r8 = r2 (reset wt ptr for next M),
   *                store cvt → output tile B. :endloop1 wraps M-loop. */
  "{ r8 = r2;"
  "  nop;"
  "  mxmem(r10,r11):cm = cvt }:endloop1\n"
  /* M16 @0x2ebf80: advance act/out base by per-K-step strides r15/r16,
   *                if (M-counter > 0) loop back to next M-tile group. */
  "{ r19 = add(r19,r15);"
  "  r18 = add(r18,r16);"
  "  p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 3b }\n"
  /* M17 @0x2ebf8c: r2 += (n_act+1) (wt base advance per K-iter),
   *                r19:18 = combine(act_base_save, act_alt+8) (next-K bases),
   *                if (K-iters left > 1) jump to next K-iter at label 2. */
  "{ r2 += add(r9,#0x1);"
  "  r19:18 = combine(r23,r14);"
  "  p0 = cmp.gt(r13,#0x1); if (p0.new) jump:t 2b }\n"
  /* M18 @0x2ebf9c: K-loop done; jump to common final-reduce. */
  "{ jump 9f }\n"
  /* alt-B path entry (n_act_pairs > 1 but ep[0] != 1; not used by current
   * production graph but kept for ABI completeness). */
  "6:\n"
  /* B1 @0x2ebfa0: alt-B pre-arm K loop, then two pad packets. */
  "{ loop0(15f,r28) }\n"
  "{ nop; nop; nop }\n"
  "{ nop; nop; nop; nop }\n"
  /* alt-B K-outer entry */
  "14:\n"
  /* B2 @0x2ebfc0: same K-outer setup as main, minus paired bias preload. */
  "{ r13 = add(r13,#-0x2);"
  "  r17 = r12;"
  "  r14 = add(r18,#0x8);"
  "  nop }\n"
  /* alt-B M-loop body entry */
  "16:\n"
  /* B3 @0x2ebfd0: arm loop1 over M groups and reset act/weight cursors. */
  "{ loop1(15f,r20);"
  "  r1:0 = combine(r19,r18);"
  "  r17 = add(r17,#-0x8);"
  "  r8 = r2 }\n"
  /* alt-B K-MAC inner loop body */
  "15:\n"
  "{ r6  = memw(r1++#0x8);"
  "  r21 = memw(r1+#0x4) }\n"
  "{ r8 = add(r8,r22);"
  "  activation.ub = mxmem(r6,r7):deep:cm;"
  "  weight.b = mxmem(r8,r9):deep }\n"
  "{ r8 = add(r8,r22);"
  "  activation.ub = mxmem(r21,r7):deep:cm;"
  "  weight.b = mxmem(r8,r9):deep }:endloop0\n"
  /* B4 @0x2ec000: re-arm loop0 and optionally skip odd-tile MAC. */
  "{ loop0(15b,r28);"
  "  r21 = r24;"
  "  if (!p1) jump:nt 17f;"
  "  r10 = memw(r0++#0x4) }\n"
  "{ r6 = memw(r1++#0x4) }\n"
  "{ activation.ub = mxmem(r6,r7):deep:cm;"
  "  weight.b = mxmem(r8,r9):deep }\n"
  /* alt-B first drain loop */
  "17:\n"
  "{ r3 = add(r3,#0x100);"
  "  r21 = add(r21,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 17b;"
  "  cvt.ub = acc(r25) }\n"
  "{ r5 -= asl(r24,#0x2);"
  "  r21 = r24;"
  "  r10 = memw(r0++m0);"
  "  mxmem(r10,r11):cm = cvt }\n"
  /* alt-B second drain loop */
  "18:\n"
  "{ r3 = add(r3,#0x100);"
  "  r21 = add(r21,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 18b;"
  "  cvt.ub = acc(r25) }\n"
  "{ r5 -= asl(r24,#0x2);"
  "  r3 -= asl(r24,#0x9);"
  "  r8 = r2;"
  "  mxmem(r10,r11):cm = cvt }:endloop1\n"
  "{ r19 = add(r19,r15);"
  "  r18 = add(r18,r16);"
  "  p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 16b }\n"
  "{ r3 += asl(r24,#0x9);"
  "  r2 += add(r9,#0x1);"
  "  r19:18 = combine(r23,r14) }\n"
  "{ p0 = cmp.gt(r13,#0x1);"
  "  if (p0.new) jump:t 14b }\n"
  "{ jump 9f }\n"
  "{ nop; nop; nop; nop }\n"
  /* alt-A path entry (n_act_pairs<=1, !ep[0]==1 with p0). */
  "1:\n"
  /* A1 @0x2ec0a0: if p2 is false, fall into alt-C at 22. */
  "{ if (!p2) jump:nt 22f; nop; nop; nop }\n"
  "{ nop; nop; nop; nop }\n"
  /* alt-A K-outer entry */
  "19:\n"
  "{ r3 = add(r3,#0x101);"
  "  r13 = add(r13,#-0x2);"
  "  r17 = r12;"
  "  bias = mxmem2(r3) }\n"
  "{ r3 = add(r3,#0xff);"
  "  r14 = add(r18,#0x8);"
  "  nop;"
  "  bias = mxmem2(r3) }\n"
  /* alt-A M-loop / single-pair MAC entry */
  "21:\n"
  "{ loop0(20f,r20);"
  "  r1 = add(r19,#0x4);"
  "  r17 = add(r17,#-0x8);"
  "  r6 = memw(r19+#0x0) }\n"
  "{ r0 = add(r18,#0x4);"
  "  r21 = r20;"
  "  nop;"
  "  r10 = memw(r18+#0x0) }\n"
  "{ nop }\n"
  "20:\n"
  "{ r21 = add(r21,#-0x1);"
  "  activation.ub = mxmem(r6,r7):deep:cm;"
  "  weight.b = mxmem(r2,r9):deep }\n"
  "{ p0 = cmp.gt(r21,#0x0);"
  "  cvt.ub = acc(r25) }\n"
  "{ r25 = setbit(r25,#0xc);"
  "  r10 = memw(r0++m0);"
  "  mxmem(r10,r11):cm = cvt }\n"
  "{ if (p0) r6 = memw(r1++#0x4);"
  "  cvt.ub = acc(r25) }\n"
  "{ r25 = clrbit(r25,#0xc);"
  "  if (p0) r10 = memw(r0++#0x4);"
  "  mxmem(r10,r11):cm = cvt }:endloop0\n"
  "{ r19 = add(r19,r15);"
  "  r18 = add(r18,r16);"
  "  p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 21b }\n"
  "{ r2 += add(r9,#0x1);"
  "  r19:18 = combine(r23,r14);"
  "  p0 = cmp.gt(r13,#0x1);"
  "  if (p0.new) jump:t 19b }\n"
  "{ jump 9f }\n"
  "{ nop; nop; nop }\n"
  /* alt-C entry (!p2 from alt-A): n_act_pairs <= 1 and ep[0] != 1. */
  "22:\n"
  "{ r13 = add(r13,#-0x2);"
  "  r17 = r12;"
  "  r14 = add(r18,#0x8);"
  "  nop }\n"
  /* alt-C M-loop entry */
  "23:\n"
  "{ loop0(24f,r20);"
  "  r1:0 = combine(r19,r18);"
  "  r17 = add(r17,#-0x8);"
  "  nop }\n"
  "24:\n"
  "{ r6 = memw(r1++#0x4);"
  "  r10 = memw(r0++#0x4) }\n"
  "{ r21 = r24;"
  "  activation.ub = mxmem(r6,r7):deep:cm;"
  "  weight.b = mxmem(r2,r9):deep }\n"
  "25:\n"
  "{ r3 = add(r3,#0x100);"
  "  r21 = add(r21,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 25b;"
  "  cvt.ub = acc(r25) }\n"
  "{ r5 -= asl(r24,#0x2);"
  "  r21 = r24;"
  "  r10 = memw(r0++m0);"
  "  mxmem(r10,r11):cm = cvt }\n"
  "26:\n"
  "{ r3 = add(r3,#0x100);"
  "  r21 = add(r21,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 26b;"
  "  cvt.ub = acc(r25) }\n"
  "{ r5 -= asl(r24,#0x2);"
  "  r3 -= asl(r24,#0x9);"
  "  mxmem(r10,r11):cm = cvt }:endloop0\n"
  "{ r19 = add(r19,r15);"
  "  r18 = add(r18,r16);"
  "  p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 23b }\n"
  "{ r2 += add(r9,#0x1);"
  "  r3 += asl(r24,#0x9);"
  "  r19:18 = combine(r23,r14) }\n"
  "{ p0 = cmp.gt(r13,#0x1);"
  "  if (p0.new) jump:t 22b }\n"
  /* common final-reduce / drain (entry from end of main / alt-A / alt-B / alt-C) */
  "9:\n"
  /* F1 @0x2ec200: skip-K-finalize early-out: if K-iter ctr <=0 jump epilogue. */
  "{ p0 = cmp.gt(r13,#0x0);"
  "  if (!p0.new) jump:t 13f;"
  "  nop; nop }\n"
  /* F2 @0x2ec210: arm loop0 (final K-pair drain, r4 trips),
   *               r9 = wt_rt >> 1 (drop deep-fanout bit for last drain),
   *               r17 = M_t (M-loop ctr reset for final drain). */
  "{ loop0(11f,r4);"
  "  r9 = asr(r9,#0x1);"
  "  r17 = r12;"
  "  nop }\n"
  /* F3 @0x2ec220 [label 10]: arm loop1 (M-loop), reload act-pair,
   *                          step M-counter, reset wt ptr. */
  "10:\n"
  "{ loop1(11f,r20);"
  "  r1:0 = combine(r19,r18);"
  "  r17 = add(r17,#-0x8);"
  "  r8 = r2 }\n"
  /* F4 @0x2ec230: 4-nop slot pad before final K-MAC body. */
  "{ nop; nop; nop; nop }\n"
  /* F5 @0x2ec240 [label 11]: final K-MAC body — single act ptr load. */
  "11:\n"
  "{ r6 = memw(r1++#0x4) }\n"
  /* F6 @0x2ec244: final K-MAC pair (no deep wt — non-deep weight load). */
  "{ r8 = add(r8,r22);"
  "  activation.ub = mxmem(r6,r7):deep:cm;"
  "  weight.b = mxmem(r8,r9) }:endloop0\n"
  /* F7 @0x2ec250: re-arm loop0 for inner-drain (reloads), r8 = wt base,
   *               r21 = drain ctr (= ep[0]), r10 = out_tbl++m0. */
  "{ loop0(11b,r4);"
  "  r8  = r2;"
  "  r21 = r24;"
  "  r10 = memw(r0++m0) }\n"
  /* F8 @0x2ec260 [label 12]: drain-bias staircase. */
  "12:\n"
  "{ r3 = add(r3,#0x100);"
  "  r21 = add(r21,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  /* F9 @0x2ec270: drain back-loop test + cvt acc. */
  "{ p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 12b;"
  "  cvt.ub = acc(r25) }\n"
  /* F10 @0x2ec278: r5 -= ep_step*4, r3 -= bias_back<<8, r0 += 4 (next out tbl),
   *                store cvt. :endloop1 wraps M-loop. */
  "{ r5 -= asl(r24,#0x2);"
  "  r3 -= asl(r24,#0x8);"
  "  r0 = add(r0,#0x4);"
  "  mxmem(r10,r11):cm = cvt }:endloop1\n"
  /* F11 @0x2ec288: act/out base advance, jump to next M-iter if more. */
  "{ r19 = add(r19,r15);"
  "  r18 = add(r18,r16);"
  "  p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 10b }\n"
  /* epilogue label */
  "13:\n"
  /* E1 @0x2ec294: clear HMX accumulator on exit. 4 bytes (1 word). */
  "{ mxclracc }\n"
  /* E2 @0x2ec298: restore r25:24. 4 bytes (1 word). */
  "{ r25:24 = memd(r29+#0x20) }\n"
  /* E3 @0x2ec29c: restore r23:22 + r21:20 via duplex. 4 bytes (1 word). */
  "{ r23:22 = memd(r29+#0x18);"
  "  r21:20 = memd(r29+#0x10) }\n"
  /* E4 @0x2ec2a0: stack reclaim 0x28, return, restore r19:18 + r17:16
   *               via duplex (read uses pre-update r29). 12 bytes (3 words). */
  "{ r29 = add(r29,#0x28);"
  "  jumpr r31;"
  "  r19:18 = memd(r29+#0x8);"
  "  r17:16 = memd(r29+#0x0) }\n"
