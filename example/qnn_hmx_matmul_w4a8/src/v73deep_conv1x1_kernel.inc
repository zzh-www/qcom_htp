  /* 2624 bytes -- hybrid readable asm/word replica of tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
   * 0x2f0780 `hmx_v73_convbnb1x1_stride1 wrapper/deep contiguous slice`.
   * This slice spans the native wrapper and selected following body bytes;
   * branch targets may point outside the embedded range; byte-proven local
   * labels or `.set` symbols preserve the native branch encodings.
   * Comments retain decoded instructions where available. The remaining raw
   * words are mixed HMX-store/control tails whose readable spelling is not
   * accepted by the current assembler.
   * Residual raw .word audit: all known W4A8 branch, loop, jump, padding,
   * HMX load, accumulator drain, and mixed HMX-store/control tails have
   * been promoted to byte-proven readable asm. No actual .word directives
   * remain in this slice.
   * Source: scripts/extract_hmx_kernel_bytes.py + public decode comments. */
  /* asm packet from comment; words preserved. */
  "{ r28 = #12;\n"
  "nop;\n"
  "nop;\n"
  "r12 = memw(r4+#48) }\n"
  /* decoded packet @0x2f0790..0x2f07a0
   *   0x2f0790: { p0 = bitsclr(r12,r28)
   *   0x2f0794: if (!p0.new) jump:nt 0x2f02c0
   *   0x2f0798: r7:6 = memd(r4+#8)
   *   0x2f079c: r11:10 = memd(r4+#0) }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".set Lhmx_2f02c0, . - 0x4d0\n"
  "{ p0 = bitsclr(r12,r28);\n"
  "if (!p0.new) jump:nt Lhmx_2f02c0;\n"
  "r7:6 = memd(r4+#8);\n"
  "r11:10 = memd(r4+#0) }\n"
  /* decoded packet @0x2f07a0..0x2f07ac
   *   0x2f07a0: { p1 = tstbit(r12,#5)
   *   0x2f07a4: immext(#1664)
   *   0x2f07a8: if (p1.new) jump:t 0x2f0e40 }
   * Purpose: long-range dispatch/guard branch; byte-proven local label preserves target reach.
   */
  "{ p1 = tstbit(r12,#5);\n"
  "immext(#1664);\n"
  "if (p1.new) jump:t Lhmx_2f0e40 }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "r9 = memw(r4+#24);\n"
  "r15 = memw(r1+#8) }\n"
  /* asm packet from comment; words preserved. */
  "{ r29 = add(r29,#-64);\n"
  "memd(r29+#-64) = r17:16 }\n"
  /* asm packet from comment; words preserved. */
  "{ r15 = asl(r15,#2);\n"
  "memd(r29+#8) = r19:18; memd(r29+#16) = r21:20 }\n"
  /* asm packet from comment; words preserved. */
  "{ r7 = extractu(r7,#11,#0);\n"
  "r4 = memw(r1+#4); r1 = memw(r1+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ p0 = cmp.gt(r4,#1);\n"
  "memd(r29+#24) = r23:22;\n"
  "memd(r29+#32) = r25:24 }\n"
  /* asm packet from comment; words preserved. */
  "{ p1 = tstbit(r4,#0);\n"
  "r17 = memw(r0+#4); r16 = memw(r0+#8) }\n"
  /* asm packet from comment; words preserved. */
  "{ r17:16 = vaslw(r17:16,#2);\n"
  "r28 = lsr(r4,#1);\n"
  "r13 = memw(r0+#20);\n"
  "memd(r29+#40) = r27:26 }\n"
  /* asm packet from comment; words preserved. */
  "{ m0 = r17;\n"
  "r20 = memw(r0+#12);\n"
  "memd(r29+#48) = r31:30 }\n"
  /* asm packet from comment; words preserved. */
  "{ r20 = add(r20,#7);\n"
  "r12 = memw(r0+#16);\n"
  "r0 = memw(r0+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ r20 = lsr(r20,#3);\n"
  "r14 = add(r0,#8);\n"
  "r19:18 = combine(r1,r0);\n"
  "memw(r29+#56) = r1 }\n"
  /* decoded packet @0x2f0820..0x2f0830
   *   0x2f0820: { if (!p0) jump:nt 0x2f0980
   *   0x2f0824: nop
   *   0x2f0828: nop
   *   0x2f082c: r30 = memw(r5++#4) }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "{ if (!p0) jump:nt Lhmx_2f0980;\n"
  "nop;\n"
  "nop;\n"
  "r30 = memw(r5++#4) }\n"
  /* decoded packet @0x2f0830..0x2f0840
   *   0x2f0830: { loop0(0x2f0860,r28)
   *   0x2f0834: r26 = r4
   *   0x2f0838: r24 = or(r7,#28)
   *   0x2f083c: r25 = #1023 }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   */
  "{ loop0(Lhmx_2f0860,r28);\n"
  "r26 = r4;\n"
  "r24 = or(r7,#28);\n"
  "r25 = #1023 }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0840:\n"
  "{ r17 = r12;\n"
  "nop;\n"
  "nop;\n"
  "nop }\n"
  /* decoded packet @0x2f0850..0x2f0860
   *   0x2f0850: { loop1(0x2f0860,r20)
   *   0x2f0854: r1:0 = combine(r19,r18)
   *   0x2f0858: r17 = add(r17,#-8)
   *   0x2f085c: r8 = r2 }
   * Purpose: initialize or restart hardware loop1 for the current output-row tile body.
   */
  "Lhmx_2f0850:\n"
  "{ loop1(Lhmx_2f0860,r20);\n"
  "r1:0 = combine(r19,r18);\n"
  "r17 = add(r17,#-8);\n"
  "r8 = r2 }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0860:\n"
  "{ p0 = cmp.eq(r26,#2);\n"
  "r26 = add(r26,#-2);\n"
  "r6 = memw(r1++#8);\n"
  "r23 = memw(r1+#4) }\n"
  /* native packet group @0x2f0870..0x2f0890
   *   0x2f0870: <unknown>
   *   0x2f087c: <unknown>
   *   0x2f0880: <unknown>
   *   0x2f088c: <unknown>
   * Purpose: inner MAC feed; load activation/weight HMX operands for the current K step.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r8 = add(r8,#0x400);\n"
  "if (p0) r25:24 = combine(r9,r7);\n"
  "activation.ub = mxmem(r6,r24):cm;\n"
  "weight.n = mxmem(r8,r25):2x }\n"
  "{ r8 += add(r25,#1);\n"
  "nop;\n"
  "activation.ub = mxmem(r23,r24):cm;\n"
  "weight.n = mxmem(r8,r25):2x }:endloop0\n"
  /* decoded packet @0x2f0890..0x2f08a0
   *   0x2f0890: { loop0(0x2f0860,r28)
   *   0x2f0894: r24 = or(r7,#28)
   *   0x2f0898: if (!p1) jump:nt 0x2f08b0
   *   0x2f089c: nop }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ loop0(Lhmx_2f0860,r28);\n"
  "r24 = or(r7,#28);\n"
  "if (!p1) jump:nt Lhmx_2f08b0;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "{ r6 = memw(r1++#4) }\n"
  /* native packet group @0x2f08a4..0x2f08b0
   *   0x2f08a4: <unknown>
   *   0x2f08ac: <unknown>
   * Purpose: inner MAC feed; load activation/weight HMX operands for the current K step.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r8 += add(r9,#1);\n"
  "activation.ub = mxmem(r6,r7):cm;\n"
  "weight.n = mxmem(r8,r9):2x }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f08b0:\n"
  "{ r31 = asl(r30,#8);\n"
  "r27:26 = combine(r8,r4);\n"
  "r8 = r2;\n"
  "p0 = cmp.gt(r13,#32) }\n"
  /* decoded packet @0x2f08c0..0x2f08cc
   *   0x2f08c0: { r25 = r30
   *   0x2f08c4: if (!p0) jump:nt 0x2f0900
   *   0x2f08c8: if (p0) r3 = add(r3,r31) }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "{ r25 = r30;\n"
  "if (!p0) jump:nt Lhmx_2f0900;\n"
  "if (p0) r3 = add(r3,r31) }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "Lhmx_2f08cc:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r25 = add(r25,#-0x1);\n"
  "r31 = memw(r5++#0x4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r31 = or(r31,#1) }\n"
  /* native packet group @0x2f08e0..0x2f08ec
   *   0x2f08e0: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r25,#0);\n"
  "if (p0.new) jump:nt Lhmx_2f08cc;\n"
  "cvt.ub = acc(r31):sc1 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r30,#2);\n"
  "r3 -= asl(r30,#9);\n"
  "r25 = r30;\n"
  "r10 = memw(r0+#4) }\n"
  /* partially decoded HMX packet group @0x2f08fc..0x2f091c
   *   0x2f08fc: <unknown>
   *   0x2f0900: <unknown>
   *   0x2f0910: <unknown>
   * Purpose: packet has been split below into byte-proven readable asm; this header describes the original native HMX packet group.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: output writeback; commit converted accumulator data to the current output tile.
   */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* asm packet split from known decoded raw words.
   * Purpose: output-column setup; advance bias/window state before the following cvt store.
   */
  "Lhmx_2f0900:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r25 = add(r25,#-0x1);\n"
  "r31 = memw(r5++#0x4);\n"
  "bias = mxmem2(r3) }\n"
  /* byte-proven conversion tail after bias/window setup.
   * Purpose: handles predicate/address update and cvt writeback for this output tile;
   * assembled as one packet so the predicate branch and cvt word stay byte-identical.
   */
  "{ p0 = cmp.gt(r25,#0);\n"
  "if (p0.new) jump:nt Lhmx_2f0900;\n"
  "cvt.ub = acc(r31):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r30,#2);\n"
  "r3 -= asl(r30,#8);\n"
  "r10 = memw(r0++m0) }\n"
  /* native packet group @0x2f0928..0x2f0934
   *   0x2f0928: <unknown>
   * Purpose: restore the output-store lane/mask limit before the following
   * cvt writeback; byte-proven as an :endloop1 packet with the HMX store.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r25 = #1023;\n"
  "nop;\n"
  "mxmem(r10,r11):cm = cvt } :endloop1\n"
  /* decoded packet @0x2f0934..0x2f0940
   *   0x2f0934: { r19 = add(r19,r15)
   *   0x2f0938: r18 = add(r18,r16)
   *   0x2f093c: p0 = cmp.gt(r17,#0); if (p0.new) jump:nt 0x2f0850 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ r19 = add(r19,r15);\n"
  "r18 = add(r18,r16);\n"
  "p0 = cmp.gt(r17,#0); if (p0.new) jump:nt Lhmx_2f0850 }\n"
  /* asm packet from comment; words preserved. */
  "{ r3 += asl(r30,#9);\n"
  "r13 = add(r13,#-64);\n"
  "r18 = r14;\n"
  "r19 = memw(r29+#56) }\n"
  /* decoded packet @0x2f0950..0x2f0960
   *   0x2f0950: { r2 = r27
   *   0x2f0954: r14 = add(r14,#8)
   *   0x2f0958: p0 = cmp.gt(r13,#0)
   *   0x2f095c: if (p0.new) jump:t 0x2f0840 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ r2 = r27;\n"
  "r14 = add(r14,#8);\n"
  "p0 = cmp.gt(r13,#0);\n"
  "if (p0.new) jump:t Lhmx_2f0840 }\n"
  /* decoded packet @0x2f0960..0x2f0964
   *   0x2f0960: { jump 0x2f0a50 }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "{ jump Lhmx_2f0a50 }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0980:\n"
  "{ r17 = r12;\n"
  "nop;\n"
  "nop;\n"
  "nop }\n"
  /* decoded packet @0x2f0990..0x2f09a0
   *   0x2f0990: { loop0(0x2f09a0,r20)
   *   0x2f0994: r1:0 = combine(r19,r18)
   *   0x2f0998: r17 = add(r17,#-8)
   *   0x2f099c: nop }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "Lhmx_2f0990:\n"
  "{ loop0(Lhmx_2f09a0,r20);\n"
  "r1:0 = combine(r19,r18);\n"
  "r17 = add(r17,#-8);\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f09a0:\n"
  "{ r31 = asl(r30,#8);\n"
  "p0 = cmp.gt(r13,#32);\n"
  "r25 = r30;\n"
  "r6 = memw(r1++#4) }\n"
  /* native packet group @0x2f09b0..0x2f09d0
   *   0x2f09b0: <unknown>
   *   0x2f09bc: <unknown>
   *   0x2f09c0: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ if (!p0) jump:nt Lhmx_2f09f4;\n"
  "if (p0) r3 = add(r3,r31);\n"
  "activation.ub = mxmem(r6,r7):cm;\n"
  "weight.n = mxmem(r2,r9):2x }\n"
  /* asm packet split from known decoded raw words. */
  "Lhmx_2f09c0:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r25 = add(r25,#-0x1);\n"
  "r31 = memw(r5++#0x4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r31 = or(r31,#1) }\n"
  /* native packet group @0x2f09d4..0x2f09e0
   *   0x2f09d4: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r25,#0);\n"
  "if (p0.new) jump:nt Lhmx_2f09c0;\n"
  "cvt.ub = acc(r31):sc1 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r30,#2);\n"
  "r3 -= asl(r30,#9);\n"
  "r25 = r30;\n"
  "r10 = memw(r0+#4) }\n"
  /* partially decoded HMX packet group @0x2f09f0..0x2f0a10
   *   0x2f09f0: <unknown>
   *   0x2f09f4: <unknown>
   *   0x2f0a04: <unknown>
   * Purpose: packet has been split below into byte-proven readable asm; this header describes the original native HMX packet group.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: output writeback; commit converted accumulator data to the current output tile.
   */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* asm packet split from known decoded raw words.
   * Purpose: output-column setup; advance bias/window state before the following cvt store.
   */
  "Lhmx_2f09f4:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r25 = add(r25,#-0x1);\n"
  "r31 = memw(r5++#0x4);\n"
  "bias = mxmem2(r3) }\n"
  /* byte-proven conversion tail after bias/window setup.
   * Purpose: handles predicate/address update and cvt writeback for this output tile;
   * assembled as one packet so the predicate branch and cvt word stay byte-identical.
   */
  "{ p0 = cmp.gt(r25,#0);\n"
  "if (p0.new) jump:nt Lhmx_2f09f4;\n"
  "cvt.ub = acc(r31):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r30,#2);\n"
  "r3 -= asl(r30,#8);\n"
  "r10 = memw(r0++m0) }\n"
  /* native packet group @0x2f0a1c..0x2f0a24
   *   0x2f0a1c: <unknown>
   * Purpose: native packet padding before the following cvt writeback path;
   * byte-proven as an :endloop0 nop plus HMX store packet.
   */
  "{ nop;\n"
  "mxmem(r10,r11):cm = cvt } :endloop0\n"
  /* decoded packet @0x2f0a24..0x2f0a30
   *   0x2f0a24: { r19 = add(r19,r15)
   *   0x2f0a28: r18 = add(r18,r16)
   *   0x2f0a2c: p0 = cmp.gt(r17,#0); if (p0.new) jump:nt 0x2f0990 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ r19 = add(r19,r15);\n"
  "r18 = add(r18,r16);\n"
  "p0 = cmp.gt(r17,#0); if (p0.new) jump:nt Lhmx_2f0990 }\n"
  /* asm packet from comment; words preserved. */
  "{ r3 += asl(r30,#9);\n"
  "r13 = add(r13,#-64);\n"
  "r18 = r14;\n"
  "r19 = memw(r29+#56) }\n"
  /* decoded packet @0x2f0a40..0x2f0a50
   *   0x2f0a40: { r2 += add(r9,#1)
   *   0x2f0a44: r14 = add(r14,#8)
   *   0x2f0a48: p0 = cmp.gt(r13,#0)
   *   0x2f0a4c: if (p0.new) jump:t 0x2f0980 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ r2 += add(r9,#1);\n"
  "r14 = add(r14,#8);\n"
  "p0 = cmp.gt(r13,#0);\n"
  "if (p0.new) jump:t Lhmx_2f0980 }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "Lhmx_2f0a50:\n"
  "{ r31:30 = memd(r29+#0x30);\n"
  "mxclracc }\n"
  /* asm packet from comment; words preserved. */
  "{ r27:26 = memd(r29+#40);\n"
  "r25:24 = memd(r29+#32) }\n"
  /* asm packet from comment; words preserved. */
  "{ r23:22 = memd(r29+#24); r21:20 = memd(r29+#16) }\n"
  /* decoded packet @0x2f0a64..0x2f0a70
   *   0x2f0a64: { r29 = add(r29,#64)
   *   0x2f0a68: jumpr r31
   *   0x2f0a6c: r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }
   */
  /* asm packet from decoded raw words; byte-proven. */
  "{ r29 = add(r29,#64);\n"
  "jumpr r31;\n"
  "r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }\n"
  /* decoded packet @0x2f0a70..0x2f0a80
   * Purpose: native padding/alignment between contiguous embedded code slices.
   */
  ".space 16, 0\n"
  /* asm packet from comment; words preserved. */
  "{ r29 = add(r29,#-64);\n"
  "r7:6 = memd(r4+#8);\n"
  "r11:10 = memd(r4+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ r9 = memw(r4+#24);\n"
  "r15 = memw(r1+#8) }\n"
  /* asm packet from comment; words preserved. */
  "{ r15 = asl(r15,#2);\n"
  "memd(r29+#0) = r17:16; memd(r29+#8) = r19:18 }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "memd(r29+#16) = r21:20; memd(r29+#24) = r23:22 }\n"
  /* asm packet from comment; words preserved. */
  "{ r7 = extractu(r7,#11,#0);\n"
  "r4 = memw(r1+#4); r1 = memw(r1+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ p1 = tstbit(r4,#0);\n"
  "p0 = cmp.gt(r4,#1);\n"
  "memd(r29+#32) = r25:24;\n"
  "memd(r29+#40) = r27:26 }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "r17 = memw(r0+#4); r16 = memw(r0+#8) }\n"
  /* asm packet from comment; words preserved. */
  "{ r17:16 = vaslw(r17:16,#2);\n"
  "r28 = lsr(r4,#1);\n"
  "r20 = memw(r0+#12);\n"
  "r13 = memw(r0+#20) }\n"
  /* asm packet from comment; words preserved. */
  "{ r25 = r1;\n"
  "r26 = memw(r5++#4);\n"
  "memd(r29+#48) = r31:30 }\n"
  /* asm packet from comment; words preserved. */
  "{ r31 = asl(r26,#8);\n"
  "r24 = and(r9,#-128);\n"
  "r30 = add(r13,#127) }\n"
  /* asm packet from comment; words preserved. */
  "{ r30 = lsr(r30,#7);\n"
  "r20 = add(r20,#7);\n"
  "r12 = memw(r0+#16);\n"
  "r0 = memw(r0+#0) }\n"
  /* decoded packet @0x2f0b04..0x2f0b14
   *   0x2f0b04: { r20 = lsr(r20,#3)
   *   0x2f0b08: r24 = add(r24,#128)
   *   0x2f0b0c: r19:18 = combine(r1,r0)
   *   0x2f0b10: if (!p0) jump:nt 0x2f0cc0 }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "{ r20 = lsr(r20,#3);\n"
  "r24 = add(r24,#128);\n"
  "r19:18 = combine(r1,r0);\n"
  "if (!p0) jump:nt Lhmx_2f0cc0 }\n"
  /* decoded packet @0x2f0b14..0x2f0b24
   *   0x2f0b14: { loop0(0x2f0b60,r28)
   *   0x2f0b18: r24 = mpyi(r24,r30)
   *   0x2f0b1c: nop
   *   0x2f0b20: nop }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ loop0(Lhmx_2f0b60,r28);\n"
  "r24 = mpyi(r24,r30);\n"
  "nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0b40:\n"
  "{ r30 = r12;\n"
  "r14 = add(r18,#16);\n"
  "r21 = r3;\n"
  "nop }\n"
  /* decoded packet @0x2f0b50..0x2f0b60
   *   0x2f0b50: { loop1(0x2f0b60,r20)
   *   0x2f0b54: r1:0 = combine(r19,r18)
   *   0x2f0b58: r30 = add(r30,#-8)
   *   0x2f0b5c: r8 = r2 }
   * Purpose: initialize or restart hardware loop1 for the current output-row tile body.
   */
  "Lhmx_2f0b50:\n"
  "{ loop1(Lhmx_2f0b60,r20);\n"
  "r1:0 = combine(r19,r18);\n"
  "r30 = add(r30,#-8);\n"
  "r8 = r2 }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0b60:\n"
  "{ r6 = memw(r1++#8);\n"
  "r23 = memw(r1+#4) }\n"
  /* native packet group @0x2f0b68..0x2f0b80
   *   0x2f0b68: <unknown>
   *   0x2f0b70: <unknown>
   *   0x2f0b74: <unknown>
   *   0x2f0b7c: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r8 = add(r8,r24);\n"
  "activation.ub = mxmem(r6,r7):cm;\n"
  "weight.n = mxmem(r8,r9):2x:deep }\n"
  "{ r8 = add(r8,r24);\n"
  "activation.ub = mxmem(r23,r7):cm;\n"
  "weight.n = mxmem(r8,r9):2x:deep }:endloop0\n"
  /* decoded packet @0x2f0b80..0x2f0b90
   *   0x2f0b80: { loop0(0x2f0b60,r28)
   *   0x2f0b84: r23 = r26
   *   0x2f0b88: p0 = cmp.gt(r13,#32)
   *   0x2f0b8c: if (!p1) jump:nt 0x2f0b9c }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   */
  "{ loop0(Lhmx_2f0b60,r28);\n"
  "r23 = r26;\n"
  "p0 = cmp.gt(r13,#32);\n"
  "if (!p1) jump:nt Lhmx_2f0b9c }\n"
  /* asm packet from comment; words preserved. */
  "{ r6 = memw(r1++#4) }\n"
  /* native packet group @0x2f0b94..0x2f0b9c
   *   0x2f0b94: <unknown>
   *   0x2f0b98: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   */
  "{ activation.ub = mxmem(r6,r7):cm;\n"
  "weight.n = mxmem(r8,r9):2x:deep }\n"
  /* decoded packet @0x2f0b9c..0x2f0ba4
   *   0x2f0b9c: { if (!p0) jump:nt 0x2f0bd4
   *   0x2f0ba0: if (p0) r3 = add(r3,r31) }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "Lhmx_2f0b9c:\n"
  "{ if (!p0) jump:nt Lhmx_2f0bd4;\n"
  "if (p0) r3 = add(r3,r31) }\n"
  /* native packet group @0x2f0ba4..0x2f0bb4
   *   0x2f0ba4: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "Lhmx_2f0ba4:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r27 = or(r27,#1) }\n"
  /* native packet group @0x2f0bb8..0x2f0bc0
   *   0x2f0bb8: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0ba4;\n"
  "cvt.ub = acc(r27):sc1 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "r3 -= asl(r26,#9);\n"
  "r23 = r26;\n"
  "r10 = memw(r0+#4) }\n"
  /* partially decoded HMX packet group @0x2f0bd0..0x2f0bec
   *   0x2f0bd0: <unknown>
   *   0x2f0bd4: <unknown>
   *   0x2f0be4: <unknown>
   * Purpose: packet has been split below into byte-proven readable asm; this header describes the original native HMX packet group.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: output writeback; commit converted accumulator data to the current output tile.
   */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0bd4:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0bd4;\n"
  "cvt.ub = acc(r27):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "p0 = cmp.gt(r13,#64);\n"
  "if (p0.new) r3 = add(r3,r31);\n"
  "r10 = memw(r0+#0) }\n"
  /* native packet group @0x2f0bfc..0x2f0c1c
   *   0x2f0bfc: <unknown>
   *   0x2f0c0c: <unknown>
   * Purpose: branch-sensitive HMX/control packet; selects the next inner-loop or epilogue path.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r23 = r26;\n"
  "if (!p0) r3 = sub(r3,r31);\n"
  "if (!p0) jump:nt Lhmx_2f0c40;\n"
  "mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0c0c:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r27 = or(r27,#1) }\n"
  /* native packet group @0x2f0c20..0x2f0c28
   *   0x2f0c20: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0c0c;\n"
  "cvt.ub = acc(r27):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "r23 = r26;\n"
  "r10 = memw(r0+#8) }\n"
  /* native packet group @0x2f0c34..0x2f0c58
   *   0x2f0c34: <unknown>
   *   0x2f0c40: <unknown>
   *   0x2f0c50: <unknown>
   * Purpose: predicate/address update around the HMX output-store path.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r13,#96);\n"
  "if (!p0.new) r3 = sub(r3,r31);\n"
  "mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0c40:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0c40;\n"
  "cvt.ub = acc(r27):sc1 }\n"
  /* decoded packet @0x2f0c58..0x2f0c60
   *   0x2f0c58: { p0 = cmp.gt(r13,#96)
   *   0x2f0c5c: if (!p0.new) jump:nt 0x2f0c68 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ p0 = cmp.gt(r13,#96);\n"
  "if (!p0.new) jump:nt Lhmx_2f0c68 }\n"
  /* asm packet from comment; words preserved. */
  "{ r10 = memw(r0+#12) }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* decoded packet @0x2f0c68..0x2f0c74
   *   0x2f0c68: { r5 -= asl(r26,#2)
   *   0x2f0c6c: r8 = r2
   *   0x2f0c70: r3 = r21; r0 = add(r0,r17) } :endloop1
  */
  /* asm packet from decoded raw words; byte-proven. */
  "Lhmx_2f0c68:\n"
  "{ r5 -= asl(r26,#2);\n"
  "r8 = r2;\n"
  "r3 = r21; r0 = add(r0,r17) } :endloop1\n"
  /* decoded packet @0x2f0c74..0x2f0c84
   *   0x2f0c74: { r19 = add(r19,r15)
   *   0x2f0c78: r18 = add(r18,r16)
   *   0x2f0c7c: p0 = cmp.gt(r30,#0)
   *   0x2f0c80: if (p0.new) jump:nt 0x2f0b50 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ r19 = add(r19,r15);\n"
  "r18 = add(r18,r16);\n"
  "p0 = cmp.gt(r30,#0);\n"
  "if (p0.new) jump:nt Lhmx_2f0b50 }\n"
  /* asm packet from comment; words preserved. */
  "{ r3 += asl(r31,#2);\n"
  "r19:18 = combine(r25,r14);\n"
  "r13 = add(r13,#-128);\n"
  "memw(r29+#56) = r26 }\n"
  /* asm packet from comment; words preserved. */
  "{ r26 = and(r9,#-128) }\n"
  /* asm packet from comment; words preserved. */
  "{ immext(#128);\n"
  "r2 += add(r26,##128);\n"
  "r26 = memw(r29+#56) }\n"
  /* decoded packet @0x2f0ca4..0x2f0cac
   *   0x2f0ca4: { p2 = cmp.gt(r13,#0)
   *   0x2f0ca8: if (p2.new) jump:t 0x2f0b40 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ p2 = cmp.gt(r13,#0);\n"
  "if (p2.new) jump:t Lhmx_2f0b40 }\n"
  /* decoded packet @0x2f0cac..0x2f0cb0
   *   0x2f0cac: { jump 0x2f0e00 }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "{ jump Lhmx_2f0e00 }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0cc0:\n"
  "{ r30 = r12;\n"
  "r14 = add(r18,#16);\n"
  "r21 = r3;\n"
  "nop }\n"
  /* decoded packet @0x2f0cd0..0x2f0ce0
   *   0x2f0cd0: { loop0(0x2f0ce0,r20)
   *   0x2f0cd4: r1:0 = combine(r19,r18)
   *   0x2f0cd8: r30 = add(r30,#-8)
   *   0x2f0cdc: nop }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "Lhmx_2f0cd0:\n"
  "{ loop0(Lhmx_2f0ce0,r20);\n"
  "r1:0 = combine(r19,r18);\n"
  "r30 = add(r30,#-8);\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0ce0:\n"
  "{ p0 = cmp.gt(r13,#32);\n"
  "r23 = r26;\n"
  "r6 = memw(r1++#4) }\n"
  /* native packet group @0x2f0cec..0x2f0d0c
   *   0x2f0cec: <unknown>
   *   0x2f0cf8: <unknown>
   *   0x2f0cfc: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ if (p0) r3 = add(r3,r31);\n"
  "if (!p0) jump:nt Lhmx_2f0d2c;\n"
  "activation.ub = mxmem(r6,r7):cm;\n"
  "weight.n = mxmem(r2,r9):2x:deep }\n"
  "Lhmx_2f0cfc:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r27 = or(r27,#1) }\n"
  /* native packet group @0x2f0d10..0x2f0d18
   *   0x2f0d10: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0cfc;\n"
  "cvt.ub = acc(r27):sc1 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "r3 -= asl(r26,#9);\n"
  "r23 = r26;\n"
  "r10 = memw(r0+#4) }\n"
  /* partially decoded HMX packet group @0x2f0d28..0x2f0d44
   *   0x2f0d28: <unknown>
   *   0x2f0d2c: <unknown>
   *   0x2f0d3c: <unknown>
   * Purpose: packet has been split below into byte-proven readable asm; this header describes the original native HMX packet group.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: output writeback; commit converted accumulator data to the current output tile.
   */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0d2c:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0d2c;\n"
  "cvt.ub = acc(r27):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "p0 = cmp.gt(r13,#64);\n"
  "if (p0.new) r3 = add(r3,r31);\n"
  "r10 = memw(r0+#0) }\n"
  /* native packet group @0x2f0d54..0x2f0d74
   *   0x2f0d54: <unknown>
   *   0x2f0d64: <unknown>
   * Purpose: branch-sensitive HMX/control packet; selects the next inner-loop or epilogue path.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r23 = r26;\n"
  "if (!p0) r3 = sub(r3,r31);\n"
  "if (!p0) jump:nt Lhmx_2f0d98;\n"
  "mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0d64:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r27 = or(r27,#1) }\n"
  /* native packet group @0x2f0d78..0x2f0d80
   *   0x2f0d78: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0d64;\n"
  "cvt.ub = acc(r27):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "r23 = r26;\n"
  "r10 = memw(r0+#8) }\n"
  /* native packet group @0x2f0d8c..0x2f0db0
   *   0x2f0d8c: <unknown>
   *   0x2f0d98: <unknown>
   *   0x2f0da8: <unknown>
   * Purpose: predicate/address update around the HMX output-store path.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r13,#96);\n"
  "if (!p0.new) r3 = sub(r3,r31);\n"
  "mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0d98:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0d98;\n"
  "cvt.ub = acc(r27):sc1 }\n"
  /* decoded packet @0x2f0db0..0x2f0db8
   *   0x2f0db0: { p0 = cmp.gt(r13,#96)
   *   0x2f0db4: if (!p0.new) jump:nt 0x2f0dc0 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ p0 = cmp.gt(r13,#96);\n"
  "if (!p0.new) jump:nt Lhmx_2f0dc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r10 = memw(r0+#12) }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* decoded packet @0x2f0dc0..0x2f0dc8
   *   0x2f0dc0: { r5 -= asl(r26,#2)
   *   0x2f0dc4: r3 = r21; r0 = add(r0,r17) } :endloop0
  */
  /* asm packet from decoded raw words; byte-proven. */
  "Lhmx_2f0dc0:\n"
  "{ r5 -= asl(r26,#2);\n"
  "r3 = r21; r0 = add(r0,r17) } :endloop0\n"
  /* decoded packet @0x2f0dc8..0x2f0dd8
   *   0x2f0dc8: { r19 = add(r19,r15)
   *   0x2f0dcc: r18 = add(r18,r16)
   *   0x2f0dd0: p0 = cmp.gt(r30,#0)
   *   0x2f0dd4: if (p0.new) jump:nt 0x2f0cd0 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ r19 = add(r19,r15);\n"
  "r18 = add(r18,r16);\n"
  "p0 = cmp.gt(r30,#0);\n"
  "if (p0.new) jump:nt Lhmx_2f0cd0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r3 += asl(r31,#2);\n"
  "r19:18 = combine(r25,r14);\n"
  "r13 = add(r13,#-128);\n"
  "memw(r29+#56) = r26 }\n"
  /* asm packet from comment; words preserved. */
  "{ r26 = and(r9,#-128) }\n"
  /* asm packet from comment; words preserved. */
  "{ immext(#128);\n"
  "r2 += add(r26,##128);\n"
  "r26 = memw(r29+#56) }\n"
  /* decoded packet @0x2f0df8..0x2f0e00
   *   0x2f0df8: { p2 = cmp.gt(r13,#0)
   *   0x2f0dfc: if (p2.new) jump:t 0x2f0cc0 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ p2 = cmp.gt(r13,#0);\n"
  "if (p2.new) jump:t Lhmx_2f0cc0 }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "Lhmx_2f0e00:\n"
  "{ r31:30 = memd(r29+#0x30);\n"
  "mxclracc }\n"
  /* asm packet from comment; words preserved. */
  "{ r27:26 = memd(r29+#40);\n"
  "r25:24 = memd(r29+#32) }\n"
  /* asm packet from comment; words preserved. */
  "{ r23:22 = memd(r29+#24); r21:20 = memd(r29+#16) }\n"
  /* decoded packet @0x2f0e14..0x2f0e20
   *   0x2f0e14: { r29 = add(r29,#64)
   *   0x2f0e18: jumpr r31
   *   0x2f0e1c: r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }
   */
  /* asm packet from decoded raw words; byte-proven. */
  "{ r29 = add(r29,#64);\n"
  "jumpr r31;\n"
  "r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }\n"
  /* decoded packet @0x2f0e20..0x2f0e40
   * Purpose: native padding/alignment between contiguous embedded code slices.
   */
  ".space 32, 0\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0e40:\n"
  "{ r29 = add(r29,#-56);\n"
  "r7:6 = memd(r4+#8);\n"
  "r11:10 = memd(r4+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ r9 = memw(r4+#24);\n"
  "r15 = memw(r1+#8) }\n"
  /* asm packet from comment; words preserved. */
  "{ r15 = asl(r15,#2);\n"
  "memd(r29+#0) = r17:16; memd(r29+#8) = r19:18 }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "memd(r29+#16) = r21:20; memd(r29+#24) = r23:22 }\n"
  /* asm packet from comment; words preserved. */
  "{ r7 = extractu(r7,#11,#0);\n"
  "r4 = memw(r1+#4); r1 = memw(r1+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ p1 = tstbit(r4,#0);\n"
  "p0 = cmp.gt(r4,#1);\n"
  "memd(r29+#32) = r25:24;\n"
  "memd(r29+#40) = r27:26 }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "r17 = memw(r0+#4); r16 = memw(r0+#8) }\n"
  /* asm packet from comment; words preserved. */
  "{ r17:16 = vaslw(r17:16,#2);\n"
  "r28 = lsr(r4,#1);\n"
  "r20 = memw(r0+#12);\n"
  "r13 = memw(r0+#20) }\n"
  /* asm packet from comment; words preserved. */
  "{ r25 = r1;\n"
  "r26 = memw(r5++#4);\n"
  "memd(r29+#48) = r31:30 }\n"
  /* asm packet from comment; words preserved. */
  "{ r31 = asl(r26,#8);\n"
  "r24 = add(r9,#1);\n"
  "r30 = add(r13,#127) }\n"
  /* asm packet from comment; words preserved. */
  "{ r30 = lsr(r30,#7);\n"
  "r20 = add(r20,#7);\n"
  "r12 = memw(r0+#16);\n"
  "r0 = memw(r0+#0) }\n"
  /* decoded packet @0x2f0ec4..0x2f0ed0
   *   0x2f0ec4: { r20 = lsr(r20,#3)
   *   0x2f0ec8: r19:18 = combine(r1,r0)
   *   0x2f0ecc: if (!p0) jump:nt 0x2f1040 }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "{ r20 = lsr(r20,#3);\n"
  "r19:18 = combine(r1,r0);\n"
  "if (!p0) jump:nt Lhmx_2f1040 }\n"
  /* decoded packet @0x2f0ed0..0x2f0ee0
   *   0x2f0ed0: { loop0(0x2f0f00,r28)
   *   0x2f0ed4: r24 = mpyi(r24,r30)
   *   0x2f0ed8: nop
   *   0x2f0edc: nop }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ loop0(Lhmx_2f0f00,r28);\n"
  "r24 = mpyi(r24,r30);\n"
  "nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0ee0:\n"
  "{ r30 = r12;\n"
  "r14 = add(r18,#16);\n"
  "r21 = r3;\n"
  "nop }\n"
  /* decoded packet @0x2f0ef0..0x2f0f00
   *   0x2f0ef0: { loop1(0x2f0f00,r20)
   *   0x2f0ef4: r1:0 = combine(r19,r18)
   *   0x2f0ef8: r30 = add(r30,#-8)
   *   0x2f0efc: r8 = r2 }
   * Purpose: initialize or restart hardware loop1 for the current output-row tile body.
   */
  "Lhmx_2f0ef0:\n"
  "{ loop1(Lhmx_2f0f00,r20);\n"
  "r1:0 = combine(r19,r18);\n"
  "r30 = add(r30,#-8);\n"
  "r8 = r2 }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f0f00:\n"
  "{ r6 = memw(r1++#8);\n"
  "r23 = memw(r1+#4) }\n"
  /* native packet group @0x2f0f08..0x2f0f20
   *   0x2f0f08: <unknown>
   *   0x2f0f10: <unknown>
   *   0x2f0f14: <unknown>
   *   0x2f0f1c: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r8 = add(r8,r24);\n"
  "activation.ub = mxmem(r6,r7):cm;\n"
  "weight.n = mxmem(r8,r9):2x:deep }\n"
  "{ r8 = add(r8,r24);\n"
  "activation.ub = mxmem(r23,r7):cm;\n"
  "weight.n = mxmem(r8,r9):2x:deep }:endloop0\n"
  /* decoded packet @0x2f0f20..0x2f0f30
   *   0x2f0f20: { loop0(0x2f0f00,r28)
   *   0x2f0f24: r23 = r26
   *   0x2f0f28: p0 = cmp.gt(r13,#32)
   *   0x2f0f2c: if (!p1) jump:nt 0x2f0f3c }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   */
  "{ loop0(Lhmx_2f0f00,r28);\n"
  "r23 = r26;\n"
  "p0 = cmp.gt(r13,#32);\n"
  "if (!p1) jump:nt Lhmx_2f0f3c }\n"
  /* asm packet from comment; words preserved. */
  "{ r6 = memw(r1++#4) }\n"
  /* native packet group @0x2f0f34..0x2f0f3c
   *   0x2f0f34: <unknown>
   *   0x2f0f38: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   */
  "{ activation.ub = mxmem(r6,r7):cm;\n"
  "weight.n = mxmem(r8,r9):2x:deep }\n"
  /* decoded packet @0x2f0f3c..0x2f0f44
   *   0x2f0f3c: { if (!p0) jump:nt 0x2f0f74
   *   0x2f0f40: if (p0) r3 = add(r3,r31) }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "Lhmx_2f0f3c:\n"
  "{ if (!p0) jump:nt Lhmx_2f0f74;\n"
  "if (p0) r3 = add(r3,r31) }\n"
  /* native packet group @0x2f0f44..0x2f0f54
   *   0x2f0f44: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "Lhmx_2f0f44:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r27 = or(r27,#1) }\n"
  /* native packet group @0x2f0f58..0x2f0f60
   *   0x2f0f58: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0f44;\n"
  "cvt.ub = acc(r27):sc1 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "r3 -= asl(r26,#9);\n"
  "r23 = r26;\n"
  "r10 = memw(r0+#4) }\n"
  /* partially decoded HMX packet group @0x2f0f70..0x2f0f8c
   *   0x2f0f70: <unknown>
   *   0x2f0f74: <unknown>
   *   0x2f0f84: <unknown>
   * Purpose: packet has been split below into byte-proven readable asm; this header describes the original native HMX packet group.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: output writeback; commit converted accumulator data to the current output tile.
   */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0f74:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0f74;\n"
  "cvt.ub = acc(r27):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "p0 = cmp.gt(r13,#64);\n"
  "if (p0.new) r3 = add(r3,r31);\n"
  "r10 = memw(r0+#0) }\n"
  /* native packet group @0x2f0f9c..0x2f0fbc
   *   0x2f0f9c: <unknown>
   *   0x2f0fac: <unknown>
   * Purpose: branch-sensitive HMX/control packet; selects the next inner-loop or epilogue path.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r23 = r26;\n"
  "if (!p0) r3 = sub(r3,r31);\n"
  "if (!p0) jump:nt Lhmx_2f0fe0;\n"
  "mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0fac:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r27 = or(r27,#1) }\n"
  /* native packet group @0x2f0fc0..0x2f0fc8
   *   0x2f0fc0: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0fac;\n"
  "cvt.ub = acc(r27):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "r23 = r26;\n"
  "r10 = memw(r0+#8) }\n"
  /* native packet group @0x2f0fd4..0x2f0ff8
   *   0x2f0fd4: <unknown>
   *   0x2f0fe0: <unknown>
   *   0x2f0ff0: <unknown>
   * Purpose: predicate/address update around the HMX output-store path.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r13,#96);\n"
  "if (!p0.new) r3 = sub(r3,r31);\n"
  "mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f0fe0:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f0fe0;\n"
  "cvt.ub = acc(r27):sc1 }\n"
  /* decoded packet @0x2f0ff8..0x2f1000
   *   0x2f0ff8: { p0 = cmp.gt(r13,#96)
   *   0x2f0ffc: if (!p0.new) jump:nt 0x2f1008 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ p0 = cmp.gt(r13,#96);\n"
  "if (!p0.new) jump:nt Lhmx_2f1008 }\n"
  /* asm packet from comment; words preserved. */
  "{ r10 = memw(r0+#12) }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* decoded packet @0x2f1008..0x2f1014
   *   0x2f1008: { r5 -= asl(r26,#2)
   *   0x2f100c: r8 = r2
   *   0x2f1010: r3 = r21; r0 = add(r0,r17) } :endloop1
  */
  /* asm packet from decoded raw words; byte-proven. */
  "Lhmx_2f1008:\n"
  "{ r5 -= asl(r26,#2);\n"
  "r8 = r2;\n"
  "r3 = r21; r0 = add(r0,r17) } :endloop1\n"
  /* decoded packet @0x2f1014..0x2f1024
   *   0x2f1014: { r19 = add(r19,r15)
   *   0x2f1018: r18 = add(r18,r16)
   *   0x2f101c: p0 = cmp.gt(r30,#0)
   *   0x2f1020: if (p0.new) jump:nt 0x2f0ef0 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ r19 = add(r19,r15);\n"
  "r18 = add(r18,r16);\n"
  "p0 = cmp.gt(r30,#0);\n"
  "if (p0.new) jump:nt Lhmx_2f0ef0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r3 += asl(r31,#2);\n"
  "r2 += add(r9,#1);\n"
  "r19:18 = combine(r25,r14);\n"
  "r13 = add(r13,#-128) }\n"
  /* decoded packet @0x2f1034..0x2f103c
   *   0x2f1034: { p2 = cmp.gt(r13,#0)
   *   0x2f1038: if (p2.new) jump:t 0x2f0ee0 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ p2 = cmp.gt(r13,#0);\n"
  "if (p2.new) jump:t Lhmx_2f0ee0 }\n"
  /* decoded packet @0x2f103c..0x2f1040
   *   0x2f103c: { jump 0x2f1170 }
   * Purpose: branch-sensitive control transfer; byte-proven local label preserves native target encoding.
   */
  "{ jump Lhmx_2f1170 }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f1040:\n"
  "{ r30 = r12;\n"
  "r14 = add(r18,#16);\n"
  "r21 = r3;\n"
  "nop }\n"
  /* decoded packet @0x2f1050..0x2f1060
   *   0x2f1050: { loop0(0x2f1060,r20)
   *   0x2f1054: r1:0 = combine(r19,r18)
   *   0x2f1058: r30 = add(r30,#-8)
   *   0x2f105c: nop }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "Lhmx_2f1050:\n"
  "{ loop0(Lhmx_2f1060,r20);\n"
  "r1:0 = combine(r19,r18);\n"
  "r30 = add(r30,#-8);\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "Lhmx_2f1060:\n"
  "{ p0 = cmp.gt(r13,#32);\n"
  "r23 = r26;\n"
  "r6 = memw(r1++#4) }\n"
  /* native packet group @0x2f106c..0x2f108c
   *   0x2f106c: <unknown>
   *   0x2f1078: <unknown>
   *   0x2f107c: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ if (p0) r3 = add(r3,r31);\n"
  "if (!p0) jump:nt Lhmx_2f10ac;\n"
  "activation.ub = mxmem(r6,r7):cm;\n"
  "weight.n = mxmem(r2,r9):2x:deep }\n"
  "Lhmx_2f107c:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r27 = or(r27,#1) }\n"
  /* native packet group @0x2f1090..0x2f1098
   *   0x2f1090: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f107c;\n"
  "cvt.ub = acc(r27):sc1 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "r3 -= asl(r26,#9);\n"
  "r23 = r26;\n"
  "r10 = memw(r0+#4) }\n"
  /* partially decoded HMX packet group @0x2f10a8..0x2f10c4
   *   0x2f10a8: <unknown>
   *   0x2f10ac: <unknown>
   *   0x2f10bc: <unknown>
   * Purpose: packet has been split below into byte-proven readable asm; this header describes the original native HMX packet group.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: output writeback; commit converted accumulator data to the current output tile.
   */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f10ac:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f10ac;\n"
  "cvt.ub = acc(r27):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "p0 = cmp.gt(r13,#64);\n"
  "if (p0.new) r3 = add(r3,r31);\n"
  "r10 = memw(r0+#0) }\n"
  /* native packet group @0x2f10d4..0x2f10f4
   *   0x2f10d4: <unknown>
   *   0x2f10e4: <unknown>
   * Purpose: branch-sensitive HMX/control packet; selects the next inner-loop or epilogue path.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ r23 = r26;\n"
  "if (!p0) r3 = sub(r3,r31);\n"
  "if (!p0) jump:nt Lhmx_2f1118;\n"
  "mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f10e4:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  /* asm packet from comment; words preserved. */
  "{ r27 = or(r27,#1) }\n"
  /* native packet group @0x2f10f8..0x2f1100
   *   0x2f10f8: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f10e4;\n"
  "cvt.ub = acc(r27):sc0 }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 -= asl(r26,#2);\n"
  "r23 = r26;\n"
  "r10 = memw(r0+#8) }\n"
  /* native packet group @0x2f110c..0x2f1130
   *   0x2f110c: <unknown>
   *   0x2f1118: <unknown>
   *   0x2f1128: <unknown>
   * Purpose: predicate/address update around the HMX output-store path.
   * Encoding note: packet has been promoted to byte-proven readable asm.
   */
  "{ p0 = cmp.gt(r13,#96);\n"
  "if (!p0.new) r3 = sub(r3,r31);\n"
  "mxmem(r10,r11):cm = cvt }\n"
  /* byte-proven mixed HMX-store/control packet.
   * Purpose: advances output/bias state, conditionally rolls back the address update,
   * and controls the next store or loop step.
   */
  "Lhmx_2f1118:\n"
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-1);\n"
  "r27 = memw(r5++#4);\n"
  "bias = mxmem2(r3) }\n"
  "{ p0 = cmp.gt(r23,#0); if (p0.new) jump:nt Lhmx_2f1118;\n"
  "cvt.ub = acc(r27):sc1 }\n"
  /* decoded packet @0x2f1130..0x2f1138
   *   0x2f1130: { p0 = cmp.gt(r13,#96)
   *   0x2f1134: if (!p0.new) jump:nt 0x2f1140 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ p0 = cmp.gt(r13,#96);\n"
  "if (!p0.new) jump:nt Lhmx_2f1140 }\n"
  /* asm packet from comment; words preserved. */
  "{ r10 = memw(r0+#12) }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ mxmem(r10,r11):cm = cvt }\n"
  /* decoded packet @0x2f1140..0x2f1148
   *   0x2f1140: { r5 -= asl(r26,#2)
   *   0x2f1144: r3 = r21; r0 = add(r0,r17) } :endloop0
  */
  /* asm packet from decoded raw words; byte-proven. */
  "Lhmx_2f1140:\n"
  "{ r5 -= asl(r26,#2);\n"
  "r3 = r21; r0 = add(r0,r17) } :endloop0\n"
  /* decoded packet @0x2f1148..0x2f1158
   *   0x2f1148: { r19 = add(r19,r15)
   *   0x2f114c: r18 = add(r18,r16)
   *   0x2f1150: p0 = cmp.gt(r30,#0)
   *   0x2f1154: if (p0.new) jump:nt 0x2f1050 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ r19 = add(r19,r15);\n"
  "r18 = add(r18,r16);\n"
  "p0 = cmp.gt(r30,#0);\n"
  "if (p0.new) jump:nt Lhmx_2f1050 }\n"
  /* asm packet from comment; words preserved. */
  "{ r2 += add(r9,#1);\n"
  "r3 += asl(r31,#2);\n"
  "r19:18 = combine(r25,r14);\n"
  "r13 = add(r13,#-128) }\n"
  /* decoded packet @0x2f1168..0x2f1170
   *   0x2f1168: { p2 = cmp.gt(r13,#0)
   *   0x2f116c: if (p2.new) jump:t 0x2f1040 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  "{ p2 = cmp.gt(r13,#0);\n"
  "if (p2.new) jump:t Lhmx_2f1040 }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "Lhmx_2f1170:\n"
  "{ r31:30 = memd(r29+#0x30);\n"
  "mxclracc }\n"
  /* asm packet from comment; words preserved. */
  "{ r27:26 = memd(r29+#40);\n"
  "r25:24 = memd(r29+#32) }\n"
  /* asm packet from comment; words preserved. */
  "{ r23:22 = memd(r29+#24); r21:20 = memd(r29+#16) }\n"
  /* decoded packet @0x2f1184..0x2f1190
   *   0x2f1184: { r29 = add(r29,#56)
   *   0x2f1188: jumpr r31
   *   0x2f118c: r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }
   */
  /* asm packet from decoded raw words; byte-proven. */
  "{ r29 = add(r29,#56);\n"
  "jumpr r31;\n"
  "r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }\n"
  /* decoded packet @0x2f1190..0x2f11c0
   * Purpose: native padding/alignment between contiguous embedded code slices.
   */
  ".space 48, 0\n"
