  /* 804 bytes -- replica of tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
   * 0x2fdb80 `hmx_v73_convhnh1x1deep_stride1`.
   *
   * Full kernel body is hand-written Hexagon inline asm:
   *   prologue   P1..P11   @0x2fdb80..0x2fdc1c  (stack/desc load + dispatch)
   *   main body  M1..M19   @0x2fdc20..0x2fdd18  (deep K-MAC + HNH drain)
   *   alt path   A1..A15   @0x2fdd20..0x2fdde4  (single-act-pair path)
   *   final-red  F1..F10   @0x2fdde8..0x2fde84  (last-K-pair drain)
   *   epilogue   E1..E4    @0x2fde88..0x2fdea0  (mxclracc + restore + ret)
   *
   * Numeric labels:
   *   2=main K-outer (0x2fdc20)  3=main M-loop (0x2fdc30)
   *   4=main K-MAC (0x2fdc40)    5/6=main HNH drain loops
   *   9=final-reduce entry       10=alt setup     18=alt M-loop
   *   11=alt MAC
   *   12/13=alt HNH drain loops  14=final M-loop  15=final K-MAC
   *   16=final HNH drain loop    17=epilogue
   *
   * HNH-specific weight loads use weight.n. The all-.byte fallback can be
   * regenerated with:
   *   python3 scripts/extract_hmx_kernel_bytes.py \
   *     --symbol hmx_v73_convhnh1x1deep_stride1 \
   *     --out example/qnn_hmx_matmul_w4a16/src/v73deep_conv1x1_kernel.inc
   * That command overwrites this readable form with raw bytes. */

  /* P1 @0x2fdb80: stack alloc 0x30 + load mask routing words. */
  "{ r29 = add(r29,#-0x30);"
  "  r7:6   = memd(r4+#0x8);"
  "  r11:10 = memd(r4+#0x0) }\n"
  /* P2 @0x2fdb8c: load mask[+0x18] and activation y stride. */
  "{ nop; nop;"
  "  r9  = memw(r4+#0x18);"
  "  r15 = memw(r1+#0x8) }\n"
  /* P3 @0x2fdb9c: scale activation y stride and save r17:16/r19:18. */
  "{ r15 = asl(r15,#0x2);"
  "  nop;"
  "  memd(r29+#0x0) = r17:16;"
  "  memd(r29+#0x8) = r19:18 }\n"
  /* P4 @0x2fdba8: save r21:20/r23:22. */
  "{ nop;"
  "  memd(r29+#0x10) = r21:20;"
  "  memd(r29+#0x18) = r23:22 }\n"
  /* P5 @0x2fdbb0: extract activation route and load activation table desc. */
  "{ r7 = extractu(r7,#0xb,#0x0);"
  "  r4 = memw(r1+#0x4);"
  "  r1 = memw(r1+#0x0) }\n"
  /* P6 @0x2fdbb8: tile predicates and save r25:24/r27:26. */
  "{ p0 = cmp.gt(r4,#0x1);"
  "  memd(r29+#0x20) = r25:24;"
  "  memd(r29+#0x28) = r27:26 }\n"
  /* P7 @0x2fdbc4: odd-tile predicate and output strides. */
  "{ p1  = tstbit(r4,#0x0);"
  "  r17 = memw(r0+#0x4);"
  "  r16 = memw(r0+#0x8) }\n"
  /* P8 @0x2fdbcc: scale output strides and load output loop spans. */
  "{ r17:16 = vaslw(r17:16,#0x2);"
  "  r28 = lsr(r4,#0x1);"
  "  r20 = memw(r0+#0xc);"
  "  r13 = memw(r0+#0x14) }\n"
  /* P9 @0x2fdbdc: m0 base and saved activation table. */
  "{ r17 = add(r17,#-0x4);"
  "  r23 = r1 }\n"
  /* P10 @0x2fdbe0: compute K/output trip helpers. */
  "{ m0 = r17;"
  "  r20 = add(r20,#0x3);"
  "  r13 = add(r13,#0x1f);"
  "  r22 = add(r9,#0x1) }\n"
  /* P11 @0x2fdbf0: finish descriptor setup. */
  "{ r13 = lsr(r13,#0x5);"
  "  r22 = lsr(r22,#0x1);"
  "  r12 = memw(r0+#0x10);"
  "  r0  = memw(r0+#0x0) }\n"
  /* P12 @0x2fdc00: select main vs single-act-pair path and load drain count. */
  "{ r20 = lsr(r20,#0x2);"
  "  r19:18 = combine(r1,r0);"
  "  if (!p0) jump:nt 10f;"
  "  r24 = memw(r5++#0x4) }\n"
  /* P13 @0x2fdc10: arm main K loop and precompute weight step. */
  "{ loop0(4f,r28);"
  "  r22 = mpyi(r22,r13);"
  "  nop; nop }\n"

  /* main K-outer entry */
  "2:\n"
  /* M1 @0x2fdc20: initialize K and M counters for this K-pair group. */
  "{ r13 = add(r13,#-0x2);"
  "  r17 = r12;"
  "  r14 = add(r18,#0x8);"
  "  nop }\n"
  /* main M-loop entry */
  "3:\n"
  /* M2 @0x2fdc30: arm M loop and reset act/weight cursors. */
  "{ loop1(4f,r20);"
  "  r1:0 = combine(r19,r18);"
  "  r17 = add(r17,#-0x8);"
  "  r8 = r2 }\n"
  /* main deep K-MAC loop */
  "4:\n"
  "{ r6  = memw(r1++#0x8);"
  "  r21 = memw(r1+#0x4) }\n"
  "{ r8 = add(r8,r22);"
  "  activation.ub = mxmem(r6,r7);"
  "  weight.n = mxmem(r8,r9):deep }\n"
  "{ r8 = add(r8,r22);"
  "  activation.ub = mxmem(r21,r7);"
  "  weight.n = mxmem(r8,r9):deep }:endloop0\n"
  /* M3 @0x2fdc60: re-arm K loop, optionally handle odd activation tile. */
  "{ loop0(4b,r28);"
  "  r26 = r24;"
  "  if (!p1) jump:nt 5f;"
  "  r10 = memw(r0++#0x4) }\n"
  "{ r6 = memw(r1++#0x4) }\n"
  "{ activation.ub = mxmem(r6,r7);"
  "  weight.n = mxmem(r8,r9):deep }\n"
  /* main first HNH drain */
  "5:\n"
  "{ r3 = add(r3,#0x100);"
  "  r26 = add(r26,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r26,#0x0);"
  "  r25 = memw(r5++#0x4);"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r3 = add(r3,#0x100);"
  "  bias = mxmem2(r3) }\n"
  "{ if (p0) jump:nt 5b;"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r5 -= asl(r24,#0x3);"
  "  r26 = r24;"
  "  r10 = memw(r0++m0);"
  "  mxmem(r10,r11) = cvt }\n"
  /* main second HNH drain */
  "6:\n"
  "{ r3 = add(r3,#0x100);"
  "  r26 = add(r26,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r26,#0x0);"
  "  r25 = memw(r5++#0x4);"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r3 = add(r3,#0x100);"
  "  bias = mxmem2(r3) }\n"
  "{ if (p0) jump:nt 6b;"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r5 -= asl(r24,#0x3);"
  "  r3 -= asl(r24,#0xa);"
  "  r8 = r2;"
  "  mxmem(r10,r11) = cvt }:endloop1\n"
  /* M4 @0x2fdcf4: next M group or next K group. */
  "{ r19 = add(r19,r15);"
  "  r18 = add(r18,r16);"
  "  p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 3b }\n"
  "{ r3 += asl(r24,#0xa);"
  "  r2 += add(r9,#0x1);"
  "  r19:18 = combine(r23,r14) }\n"
  "{ p0 = cmp.gt(r13,#0x1);"
  "  if (p0.new) jump:t 2b }\n"
  "{ jump 9f }\n"
  "{ nop; nop }\n"

  /* single-act-pair path */
  "10:\n"
  /* A1 @0x2fdd20: setup for n_act_pairs <= 1. */
  "{ r13 = add(r13,#-0x2);"
  "  r17 = r12;"
  "  r14 = add(r18,#0x8);"
  "  nop }\n"
  /* A2 @0x2fdd30: one K-MAC per M tile. */
  "18:\n"
  "{ loop0(11f,r20);"
  "  r1:0 = combine(r19,r18);"
  "  r17 = add(r17,#-0x8);"
  "  nop }\n"
  "11:\n"
  "{ r6 = memw(r1++#0x4);"
  "  r10 = memw(r0++#0x4) }\n"
  "{ r26 = r24;"
  "  activation.ub = mxmem(r6,r7);"
  "  weight.n = mxmem(r2,r9):deep }\n"
  /* alt first HNH drain */
  "12:\n"
  "{ r3 = add(r3,#0x100);"
  "  r26 = add(r26,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r26,#0x0);"
  "  r25 = memw(r5++#0x4);"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r3 = add(r3,#0x100);"
  "  bias = mxmem2(r3) }\n"
  "{ if (p0) jump:nt 12b;"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r5 -= asl(r24,#0x3);"
  "  r26 = r24;"
  "  r10 = memw(r0++m0);"
  "  mxmem(r10,r11) = cvt }\n"
  /* alt second HNH drain */
  "13:\n"
  "{ r3 = add(r3,#0x100);"
  "  r26 = add(r26,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r26,#0x0);"
  "  r25 = memw(r5++#0x4);"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r3 = add(r3,#0x100);"
  "  bias = mxmem2(r3) }\n"
  "{ if (p0) jump:nt 13b;"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r5 -= asl(r24,#0x3);"
  "  r3 -= asl(r24,#0xa);"
  "  mxmem(r10,r11) = cvt }:endloop0\n"
  "{ r19 = add(r19,r15);"
  "  r18 = add(r18,r16);"
  "  p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 18b }\n"
  "{ r2 += add(r9,#0x1);"
  "  r3 += asl(r24,#0xa);"
  "  r19:18 = combine(r23,r14) }\n"
  "{ p0 = cmp.gt(r13,#0x1);"
  "  if (p0.new) jump:t 10b }\n"

  /* common final-reduce / last-K drain */
  "9:\n"
  /* F1 @0x2fdde8: skip final drain when no K remainder remains. */
  "{ p0 = cmp.gt(r13,#0x0);"
  "  if (!p0.new) jump:t 17f;"
  "  nop; nop }\n"
  /* F2 @0x2fddf8: arm final K loop and drop deep fanout from weight route. */
  "{ loop0(15f,r4);"
  "  r9 = asr(r9,#0x1);"
  "  r17 = r12;"
  "  nop }\n"
  /* F3 @0x2fde08: final M loop. */
  "14:\n"
  "{ loop1(15f,r20);"
  "  r1:0 = combine(r19,r18);"
  "  r17 = add(r17,#-0x8);"
  "  r8 = r2 }\n"
  "{ nop; nop }\n"
  /* F4 @0x2fde20: final non-deep K-MAC. */
  "15:\n"
  "{ r6 = memw(r1++#0x4) }\n"
  "{ r8 = add(r8,r22);"
  "  activation.ub = mxmem(r6,r7);"
  "  weight.n = mxmem(r8,r9) }:endloop0\n"
  /* F5 @0x2fde30: re-arm final K loop and start HNH drain. */
  "{ loop0(15b,r4);"
  "  r8 = r2;"
  "  r26 = r24;"
  "  r10 = memw(r0++m0) }\n"
  "16:\n"
  "{ r3 = add(r3,#0x100);"
  "  r26 = add(r26,#-0x1);"
  "  r25 = memw(r5++#0x4);"
  "  bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r26,#0x0);"
  "  r25 = memw(r5++#0x4);"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r3 = add(r3,#0x100);"
  "  bias = mxmem2(r3) }\n"
  "{ if (p0) jump:nt 16b;"
  "  cvt.uh = acc(r25):2x2 }\n"
  "{ r5 -= asl(r24,#0x3);"
  "  r3 -= asl(r24,#0x9);"
  "  r0 = add(r0,#0x4);"
  "  mxmem(r10,r11) = cvt }:endloop1\n"
  /* F6 @0x2fde7c: next final M group. */
  "{ r19 = add(r19,r15);"
  "  r18 = add(r18,r16);"
  "  p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 14b }\n"

  /* epilogue */
  "17:\n"
  "{ mxclracc }\n"
  "{ r27:26 = memd(r29+#0x28);"
  "  r25:24 = memd(r29+#0x20) }\n"
  "{ r23:22 = memd(r29+#0x18);"
  "  r21:20 = memd(r29+#0x10) }\n"
  "{ r29 = add(r29,#0x30);"
  "  jumpr r31;"
  "  r19:18 = memd(r29+#0x8);"
  "  r17:16 = memd(r29+#0x0) }\n"
