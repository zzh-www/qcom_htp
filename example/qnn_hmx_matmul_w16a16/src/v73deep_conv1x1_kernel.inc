  /* 1800 bytes -- hybrid readable asm/word replica of tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so
   * 0x2fa740 `hmx_v73_convhhh1x1_stride1 plus following native slice`.
   * Raw .byte lines were grouped into address-preserving .word directives.
   * Comments retain decoded instructions where available; <unknown> blocks
   * remain raw words until their HMX mnemonics are byte-proven.
   * Residual raw .word audit: all decoded `jumpr` packets that assemble
   * byte-identically were promoted to asm. Remaining raw words are branch/loop
   * packets, native zero padding, unknown HMX packets, or the p2/endloop packet
   * whose readable form assembles with an extra nop word.
   * Source: scripts/extract_hmx_kernel_bytes.py + public decode comments. */
  /* asm packet from comment; words preserved. */
  "{ r28 = #1923;\n"
  "r7:6 = memd(r4+#8);\n"
  "r11:10 = memd(r4+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ p0 = bitsclr(r6,r28);\n"
  "p0 = bitsclr(r10,r28);\n"
  "p2 = cmp.gt(r4,r4);\n"
  "r8 = memw(r4+#48) }\n"
  /* decoded packet @0x2fa75c..0x2fa76c
   *   0x2fa75c: { immext(#1216)
   *   0x2fa760: if (!p0) jump:nt 0x2fac40
   *   0x2fa764: nop
   *   0x2fa768: nop }
   * Purpose: long-range dispatch/guard branch; raw form prevents assembler from re-encoding target reach.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0x00004013, 0x5c204048\n"
  /* asm packet split from known decoded raw words. */
  "{ nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "{ r9 = memw(r4+#24);\n"
  "r15 = memw(r1+#8) }\n"
  /* asm packet from comment; words preserved. */
  "{ r29 = add(r29,#-64);\n"
  "nop;\n"
  "memd(r29+#-64) = r17:16 }\n"
  /* asm packet from comment; words preserved. */
  "{ r15 = asl(r15,#2);\n"
  "r18 = #2;\n"
  "memd(r29+#8) = r19:18; memd(r29+#16) = r21:20 }\n"
  /* asm packet from comment; words preserved. */
  "{ r7 = extractu(r7,#11,#0);\n"
  "r4 = memw(r1+#4); r1 = memw(r1+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ p0 = cmp.gt(r4,#1);\n"
  "r19 = sub(#5,r18);\n"
  "memd(r29+#24) = r23:22;\n"
  "memd(r29+#32) = r25:24 }\n"
  /* asm packet from comment; words preserved. */
  "{ p1 = tstbit(r4,#0);\n"
  "r21 = lsl(#1,r18);\n"
  "r17 = memw(r0+#4); r16 = memw(r0+#8) }\n"
  /* asm packet from comment; words preserved. */
  "{ r17:16 = vaslw(r17:16,#2);\n"
  "r28 = lsr(r4,#1);\n"
  "r13 = memw(r0+#20);\n"
  "memd(r29+#40) = r27:26 }\n"
  /* asm packet from comment; words preserved. */
  "{ m0 = r17;\n"
  "r22 = lsl(#1,r19);\n"
  "r20 = memw(r0+#12);\n"
  "memd(r29+#48) = r31:30 }\n"
  /* asm packet from comment; words preserved. */
  "{ r20 += add(r21,#-1);\n"
  "nop;\n"
  "r12 = memw(r0+#16);\n"
  "r0 = memw(r0+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ r20 = lsr(r20,r18);\n"
  "r14 = r0;\n"
  "r19:18 = combine(r1,r0);\n"
  "memw(r29+#56) = r1 }\n"
  /* decoded packet @0x2fa7f0..0x2fa800
   *   0x2fa7f0: { if (!p0) jump:nt 0x2fa900
   *   0x2fa7f4: nop
   *   0x2fa7f8: nop
   *   0x2fa7fc: r30 = memw(r5++#4) }
   * Purpose: branch-sensitive control transfer; raw form preserves native target encoding.
   */
  ".word 0x5c204088, 0x7f004000, 0x7f004000, 0x9b85c03e\n"
  /* decoded packet @0x2fa800..0x2fa810
   *   0x2fa800: { loop0(0x2fa840,r28)
   *   0x2fa804: r26 = r4
   *   0x2fa808: r24 = or(r7,#112)
   *   0x2fa80c: r25 = #1023 }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   */
  ".word 0x601c4400, 0x7064401a, 0x76874e18, 0x7801fff9\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "r13 = add(r13,#-16);\n"
  "r17 = r12 }\n"
  /* decoded packet @0x2fa830..0x2fa840
   *   0x2fa830: { loop1(0x2fa840,r20)
   *   0x2fa834: r1:0 = combine(r19,r18)
   *   0x2fa838: r17 = sub(r17,r22)
   *   0x2fa83c: r8 = r2 }
   * Purpose: initialize or restart hardware loop1 for the current output-row tile body.
   */
  ".word 0x60344100, 0xf5135200, 0xf3365111, 0x7062c008\n"
  /* asm packet from comment; words preserved. */
  "{ p0 = cmp.eq(r26,#2);\n"
  "r26 = add(r26,#-2);\n"
  "r6 = memw(r1++#8);\n"
  "r23 = memw(r1+#4) }\n"
  /* partially decoded HMX packet group @0x2fa850..0x2fa86c
   *   0x2fa850: <unknown>
   *   0x2fa85c: <unknown>
   *   0x2fa860: <unknown>
   *   0x2fa868: <unknown>
   * Purpose: first inner-loop operand-load group; seeds activation/weight
   * HMX inputs before the loop0 restart below.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: inner MAC feed; issue byte activation/weight mxmem reads for this K tile.
   */
  "{ r8 = add(r8,#0x400);\n"
  "if (p0) r25:24 = combine(r9,r7);\n"
  "activation.ub = mxmem(r6,r24);\n"
  "weight.b = mxmem(r8,r25):dilate }\n"
  /* remaining raw tail of the same HMX operand-load packet.
   * Purpose: completes activation/weight feed for the current MAC step;
   * kept raw because the tail HMX words are not byte-proven asm.
   */
  ".word 0xe2198028, 0x921758ec, 0x9208f9eb\n"
  /* decoded packet @0x2fa86c..0x2fa87c
   *   0x2fa86c: { loop0(0x2fa840,r28)
   *   0x2fa870: r24 = or(r7,#112)
   *   0x2fa874: if (!p1) jump:nt 0x2fa88c
   *   0x2fa878: r10 = memw(r0++m0) }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   */
  ".word 0x601c5d08, 0x76874e18, 0x5c204110, 0x9d80c00a\n"
  /* asm packet from comment; words preserved. */
  "{ r6 = memw(r1++#4) }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ r8 += add(r9,#0x1);\n"
  "activation.ub = mxmem(r6,r7);\n"
  "weight.b = mxmem(r8,r9):dilate }\n"
  /* asm packet from comment; words preserved. */
  "{ r27:26 = combine(r8,r4);\n"
  "r8 = r2;\n"
  "r25 = r30;\n"
  "if (p2) r10 = add(r10,#64) }\n"
  /* partially decoded HMX packet group @0x2fa89c..0x2fa8c8
   *   0x2fa89c: <unknown>
   *   0x2fa8ac: <unknown>
   *   0x2fa8b8: <unknown>
   * Purpose: output-store setup and cvt tail after the main MAC path; pairs
   * bias/window updates with the raw branch/HMX writeback words that follow.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: output-column setup; advance bias/window state before the following cvt store.
   */
  "{ r3 = add(r3,#0x100);\n"
  "r25 = add(r25,#-0x1);\n"
  "r31 = memw(r5++#0x4);\n"
  "bias = mxmem2(r3) }\n"
  /* remaining raw tail after byte-proven bias/window setup.
   * Purpose: handles predicate/address update and cvt writeback for this output tile;
   * kept raw because mixed branch/HMX words re-encode differently as asm.
   */
  ".word 0x75594000, 0x5cdf68f8, 0xa6ffd910, 0x8e1e4245, 0x8e1e8843, 0x78017ff9, 0xa6eacb1a\n"
  /* decoded packet @0x2fa8c8..0x2fa8d4
   *   0x2fa8c8: { r19 = add(r19,r15)
   *   0x2fa8cc: r18 = add(r18,r16)
   *   0x2fa8d0: p0 = cmp.gt(r17,#0); if (p0.new) jump:nt 0x2fa830 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".word 0xf3134f13, 0xf3125012, 0x10b9c0b4\n"
  /* asm packet from comment; words preserved. */
  "{ r3 += asl(r30,#8);\n"
  "r18 = r14;\n"
  "if (!p2) r14 = add(r14,#4);\n"
  "r19 = memw(r29+#56) }\n"
  /* decoded packet @0x2fa8e4..0x2fa8f4
   *   0x2fa8e4: { p2 = not(p2)
   *   0x2fa8e8: r2 = r27
   *   0x2fa8ec: p0 = cmp.gt(r13,#0)
   *   0x2fa8f0: if (p0.new) jump:t 0x2fa820 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".word 0x6bc24002, 0x707b4002, 0x754d4000, 0x5cdff89e\n"
  /* decoded packet @0x2fa8f4..0x2fa8f8
   *   0x2fa8f4: { jump 0x2fa98c }
   * Purpose: branch-sensitive control transfer; raw form preserves native target encoding.
   */
  ".word 0x5800c04c\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "r13 = add(r13,#-16);\n"
  "r17 = r12 }\n"
  /* decoded packet @0x2fa910..0x2fa920
   *   0x2fa910: { loop0(0x2fa920,r20)
   *   0x2fa914: r1:0 = combine(r19,r18)
   *   0x2fa918: r17 = sub(r17,r22)
   *   0x2fa91c: nop }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0x60144100, 0xf5135200, 0xf3365111\n"
  /* asm packet split from known decoded raw words. */
  "{ nop }\n"
  /* asm packet from comment; words preserved. */
  "{ r6 = memw(r1++#4);\n"
  "r10 = memw(r0++m0) }\n"
  /* unknown/HMX raw block @0x2fa928..0x2fa960
   *   0x2fa928: <unknown>
   *   0x2fa934: <unknown>
   *   0x2fa938: <unknown>
   *   0x2fa948: <unknown>
   *   0x2fa954: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0x707e4019, 0x744a480a, 0x920647ec, 0x9202e9eb\n"
  /* asm packet split from known decoded raw words.
   * Purpose: output-column setup; advance bias/window state before the following cvt store.
   */
  "{ r3 = add(r3,#0x100);\n"
  "r25 = add(r25,#-0x1);\n"
  "r31 = memw(r5++#0x4);\n"
  "bias = mxmem2(r3) }\n"
  /* remaining raw tail after byte-proven bias/window setup.
   * Purpose: handles predicate/address update and cvt writeback for this output tile;
   * kept raw because mixed branch/HMX words re-encode differently as asm.
   */
  ".word 0x75594000, 0x5cdf68f8, 0xa6ffd910, 0x8e1e8245, 0x8e1e4843, 0xa6eacb1a\n"
  /* decoded packet @0x2fa960..0x2fa96c
   *   0x2fa960: { r19 = add(r19,r15)
   *   0x2fa964: r18 = add(r18,r16)
   *   0x2fa968: p0 = cmp.gt(r17,#0); if (p0.new) jump:nt 0x2fa910 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".word 0xf3134f13, 0xf3125012, 0x10b9c0d8\n"
  /* asm packet from comment; words preserved. */
  "{ r3 += asl(r30,#8);\n"
  "r18 = r14;\n"
  "if (!p2) r14 = add(r14,#4);\n"
  "r19 = memw(r29+#56) }\n"
  /* unknown/HMX raw block @0x2fa97c..0x2fa994
   *   0x2fa97c: <unknown>
   *   0x2fa98c: <unknown>
   * Purpose: branch-sensitive HMX/control packet; selects the next inner-loop or epilogue path.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0xe2094022, 0x6bc24002, 0x754d4000, 0x5cdff8c2\n"
  /* asm packet split from known decoded raw words. */
  "{ r31:30 = memd(r29+#0x30);\n"
  "mxclracc }\n"
  /* asm packet from comment; words preserved. */
  "{ r27:26 = memd(r29+#40);\n"
  "r25:24 = memd(r29+#32) }\n"
  /* asm packet from comment; words preserved. */
  "{ r23:22 = memd(r29+#24); r21:20 = memd(r29+#16) }\n"
  /* decoded packet @0x2fa9a0..0x2fa9ac
   *   0x2fa9a0: { r29 = add(r29,#64)
   *   0x2fa9a4: jumpr r31
   *   0x2fa9a8: r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }
   */
  /* asm packet from decoded raw words; byte-proven. */
  "{ r29 = add(r29,#64);\n"
  "jumpr r31;\n"
  "r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }\n"
  /* decoded packet @0x2fa9ac..0x2fa9c0
   * Purpose: native padding/alignment between contiguous embedded code slices.
   */
  ".word 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000\n"
  /* asm packet from comment; words preserved. */
  "{ r29 = add(r29,#-64);\n"
  "nop;\n"
  "r7:6 = memd(r4+#8);\n"
  "r11:10 = memd(r4+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ r9 = memw(r4+#24);\n"
  "memd(r29+#0) = r17:16 }\n"
  /* asm packet from comment; words preserved. */
  "{ r20 = #1983;\n"
  "r21 = #2047;\n"
  "memd(r29+#8) = r19:18; memd(r29+#16) = r21:20 }\n"
  /* asm packet from comment; words preserved. */
  "{ r18 = r5;\n"
  "nop;\n"
  "r15 = memw(r1+#8);\n"
  "r5 = memw(r1+#12) }\n"
  /* asm packet from comment; words preserved. */
  "{ r15 = asl(r15,#2);\n"
  "r5 = add(r5,#3);\n"
  "r14 = memw(r1+#16);\n"
  "r4 = memw(r1+#4) }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 = lsr(r5,#2);\n"
  "r24 = #8;\n"
  "memd(r29+#24) = r23:22;\n"
  "memd(r29+#32) = r25:24 }\n"
  /* asm packet from comment; words preserved. */
  "{ r4 = add(r4,#-1);\n"
  "r22 = #131;\n"
  "r1 = memw(r1+#0);\n"
  "r25 = memw(r0+#12) }\n"
  /* asm packet from comment; words preserved. */
  "{ p2 = bitsclr(r10,r22);\n"
  "if (!p2.new) r25 = add(r25,#4);\n"
  "r16 = memw(r0+#8); r17 = memw(r0+#4) }\n"
  /* asm packet from comment; words preserved. */
  "{ r13 = r1;\n"
  "nop;\n"
  "r28 = memw(r0+#20);\n"
  "r12 = memw(r0+#16) }\n"
  /* asm packet from comment; words preserved. */
  "{ r25 -= asl(r5,#2);\n"
  "r28 = add(r28,#15);\n"
  "r0 = memw(r0+#0);\n"
  "memd(r29+#40) = r27:26 }\n"
  /* asm packet from comment; words preserved. */
  "{ r28 = asr(r28,#4);\n"
  "r30 = r7;\n"
  "r19 = memw(r18++#4);\n"
  "memd(r29+#48) = r31:30 }\n"
  /* asm packet from comment; words preserved. */
  "{ r7:6 = and(r7:6,r21:20);\n"
  "r28 = add(r28,#-1);\n"
  "r12 = add(r12,#8);\n"
  "r31 = #1024 }\n"
  /* asm packet from comment; words preserved. */
  "{ r21 = and(r9,#127) }\n"
  /* decoded packet @0x2faa74..0x2faa7c
   *   0x2faa74: { p3 = cmp.eq(r21,#0)
   *   0x2faa78: if (p3.new) jump:nt 0x2faa94 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".word 0x75154003, 0x5c00cb10\n"
  /* asm packet from comment; words preserved. */
  "{ r21 = add(r21,#1) }\n"
  /* asm packet from comment; words preserved. */
  "{ r21 = asr(r21,#4) }\n"
  /* asm packet from comment; words preserved. */
  "{ r21 = add(r21,#1) }\n"
  /* asm packet from comment; words preserved. */
  "{ r31 = +mpyi(r21,#128);\n"
  "r21 = and(r9,#127) }\n"
  /* asm packet from comment; words preserved. */
  "{ r31 = or(r31,r21) }\n"
  /* asm packet from comment; words preserved. */
  "{ r31 = add(r31,#-128);\n"
  "r21 = #2047 }\n"
  /* asm packet from comment; words preserved. */
  "{ r17:16 = vaslw(r17:16,#2);\n"
  "r26 = r0;\n"
  "r27 = #0;\n"
  "nop }\n"
  /* asm packet from comment; words preserved. */
  "{ nop }\n"
  /* asm packet from comment; words preserved. */
  "{ nop;\n"
  "nop;\n"
  "nop;\n"
  "nop }\n"
  /* decoded packet @0x2faac0..0x2faad0
   *   0x2faac0: { loop1(0x2faae0,r5)
   *   0x2faac4: r23 = #0
   *   0x2faac8: r22 = #131
   *   0x2faacc: nop }
   * Purpose: initialize or restart hardware loop1 for the current output-row tile body.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0x60254200, 0x78004017, 0x78005076\n"
  /* asm packet split from known decoded raw words. */
  "{ nop }\n"
  /* asm packet from comment; words preserved. */
  "{ p2 = bitsclr(r10,r22);\n"
  "p0 = cmp.gt(r4,#0);\n"
  "p1 = cmp.gt(r14,r24);\n"
  "memd(r29+#56) = r1:0 }\n"
  /* decoded packet @0x2faae0..0x2faaf0
   *   0x2faae0: { loop0(0x2fab0c,r4)
   *   0x2faae4: r7 = or(r7,#112)
   *   0x2faae8: r22 = memw(r1++#4)
   *   0x2faaec: if (p1) r23 = memw(r1+r15<<#0) }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   */
  ".word 0x60044218, 0x76874e07, 0x9b814036, 0x3081cf37\n"
  /* asm packet from comment; words preserved. */
  "{ r21 = and(r31,#-128) }\n"
  /* asm packet from comment; words preserved. */
  "{ r8 = sub(r2,r21);\n"
  "r21 = #2047 }\n"
  /* decoded packet @0x2faafc..0x2fab0c
   *   0x2faafc: { if (p1) r23 = sub(r23,r22)
   *   0x2fab00: if (!p0) r7 = add(r30,#0)
   *   0x2fab04: if (!p0) jump:nt 0x2fab3c
   *   0x2fab08: r8 = add(r8,#-128) }
   * Purpose: branch-sensitive control transfer; raw form preserves native target encoding.
   */
  ".word 0xfb365737, 0x749e4007, 0x5c204020, 0xbfe8f008\n"
  /* asm packet from comment; words preserved. */
  "{ r7:6 = or(r7:6,r23:22);\n"
  "r21 = and(r31,#-128);\n"
  "r22 = memw(r1++#4);\n"
  "if (p1) r23 = memw(r1+r15<<#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ immext(#128);\n"
  "r8 += add(r21,##128);\n"
  "r21 = #2047 }\n"
  /* unknown/HMX raw block @0x2fab28..0x2fab38
   *   0x2fab28: <unknown>
   *   0x2fab34: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0xd3e69406, 0xfb365737, 0x920647f0, 0x9208ffe6\n"
  /* asm packet from comment; words preserved. */
  "{ r7 = r30 }\n"
  /* asm packet from comment; words preserved. */
  "{ r21 = and(r31,#-128) }\n"
  /* asm packet from comment; words preserved. */
  "{ immext(#128);\n"
  "r8 += add(r21,##128);\n"
  "r21 = #2047 }\n"
  /* asm packet from comment; words preserved. */
  "{ r7:6 = or(r7:6,r23:22);\n"
  "r10 = and(r10,r20);\n"
  "r23 = r19 }\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ r7:6 = and(r7:6,r21:20);\n"
  "r21 = and(r9,#-0x80);\n"
  "activation.ub = mxmem(r6,r7):single;\n"
  "weight.b = mxmem(r8,r9):single }\n"
  /* asm packet from comment; words preserved. */
  "{ immext(#128);\n"
  "r8 += add(r21,##128);\n"
  "r21 = #2047 }\n"
  /* partially decoded HMX packet group @0x2fab74..0x2fab94
   *   0x2fab74: <unknown>
   *   0x2fab84: <unknown>
   * Purpose: single-lane output-store setup; prepares bias/window state and
   * raw HMX writeback words for the scalar tail path.
   */
  /* asm packet split from known decoded raw words.
   * Purpose: output-column setup; advance bias/window state before the following cvt store.
   */
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-0x1);\n"
  "r22 = memw(r18++#0x4);\n"
  "bias = mxmem2(r3) }\n"
  /* remaining raw tail after byte-proven bias/window setup.
   * Purpose: handles predicate/address update and cvt writeback for this output tile;
   * kept raw because mixed branch/HMX words re-encode differently as asm.
   */
  ".word 0x75574003, 0x5cdf6bf8, 0x47805816\n"
  /* asm packet split from known decoded raw words. */
  "{ cvt.uh = acc(r22):2x2 }\n"
  /* asm packet from comment; words preserved. */
  "{ r18 -= asl(r19,#2);\n"
  "r3 -= asl(r19,#8);\n"
  "r22 = add(r22,r27) }\n"
  /* decoded packet @0x2faba0..0x2faba8
   *   0x2faba0: { r10 = or(r10,r22)
   *   0x2faba4: if (!p2) jump:nt 0x2fabac }
   * Purpose: branch-sensitive control transfer; raw form preserves native target encoding.
   */
  ".word 0xf12a560a, 0x5c20c206\n"
  /* unknown/HMX raw block @0x2faba8..0x2fabac
   *   0x2faba8: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   */
  ".word 0xa6eacb1a\n"
  /* decoded packet @0x2fabac..0x2fabb8
   *   0x2fabac: { p2 = or(p2,!p2)
   *   0x2fabb0: if (p2) r0 = add(r0,r17)
   *   0x2fabb4: nop } :endloop1
   * Purpose: hardware-loop tail; raw form preserves endloop packet encoding exactly.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0x6be24202, 0xfb009140\n"
  /* asm packet split from known decoded raw words. */
  "{ nop }\n"
  /* decoded packet @0x2fabb8..0x2fabc4
   *   0x2fabb8: { p3 = cmp.gt(r25,#0)
   *   0x2fabbc: if (!p3.new) jump:t 0x2fabd8
   *   0x2fabc0: if (p3.new) r22 = #0 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".word 0x75594003, 0x5c205b10, 0x7e60e016\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ r10 = and(r10,r20);\n"
  "r22 = memw(r0+#0x0);\n"
  "cvt.uh = acc(r22):2x2 }\n"
  /* asm packet from comment; words preserved. */
  "{ r10 |= or(r22,r27) }\n"
  /* unknown/HMX raw block @0x2fabd4..0x2fabd8
   *   0x2fabd4: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   */
  ".word 0xa6eacb1a\n"
  /* asm packet from comment; words preserved. */
  "{ r24 = add(r24,#8);\n"
  "r1:0 = memd(r29+#56) }\n"
  /* decoded packet @0x2fabe0..0x2fabf0
   *   0x2fabe0: { r1 = add(r1,r15)
   *   0x2fabe4: r0 = add(r0,r16)
   *   0x2fabe8: p0 = cmp.gt(r12,r24)
   *   0x2fabec: if (p0.new) jump:nt 0x2faac0 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".word 0xf3014f01, 0xf3005000, 0xf24c5800, 0x5cdfe870\n"
  /* asm packet from comment; words preserved. */
  "{ r1 = r13;\n"
  "r0 = r26;\n"
  "r28 = add(r28,#-1);\n"
  "p3 = cmp.eq(r27,#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ r2 = r8;\n"
  "r27 = mux(p3,#64,#0) }\n"
  /* decoded packet @0x2fac08..0x2fac18
   *   0x2fac08: { if (r28>=#0) jump:t 0x2faac0
   *   0x2fac0c: r3 += asl(r19,#8)
   *   0x2fac10: if (p3) r26 = add(r26,#4)
   *   0x2fac14: r24 = #8 }
   * Purpose: branch-sensitive control transfer; raw form preserves native target encoding.
   */
  ".word 0x617c7f5c, 0x8e1348c3, 0x747a409a, 0x7800c118\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ r31:30 = memd(r29+#0x30);\n"
  "mxclracc }\n"
  /* asm packet from comment; words preserved. */
  "{ r27:26 = memd(r29+#40);\n"
  "r25:24 = memd(r29+#32) }\n"
  /* asm packet from comment; words preserved. */
  "{ r23:22 = memd(r29+#24); r21:20 = memd(r29+#16) }\n"
  /* decoded packet @0x2fac2c..0x2fac38
   *   0x2fac2c: { r29 = add(r29,#64)
   *   0x2fac30: jumpr r31
   *   0x2fac34: r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }
   */
  /* asm packet from decoded raw words; byte-proven. */
  "{ r29 = add(r29,#64);\n"
  "jumpr r31;\n"
  "r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }\n"
  /* decoded packet @0x2fac38..0x2fac40
   * Purpose: native padding/alignment between contiguous embedded code slices.
   */
  ".word 0x00000000, 0x00000000\n"
  /* asm packet from comment; words preserved. */
  "{ r29 = add(r29,#-64);\n"
  "nop;\n"
  "r7:6 = memd(r4+#8);\n"
  "r11:10 = memd(r4+#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ r9 = memw(r4+#24);\n"
  "memd(r29+#0) = r17:16 }\n"
  /* asm packet from comment; words preserved. */
  "{ r20 = #1983;\n"
  "r21 = #2047;\n"
  "memd(r29+#8) = r19:18; memd(r29+#16) = r21:20 }\n"
  /* asm packet from comment; words preserved. */
  "{ r18 = r5;\n"
  "nop;\n"
  "r15 = memw(r1+#8);\n"
  "r5 = memw(r1+#12) }\n"
  /* asm packet from comment; words preserved. */
  "{ r15 = asl(r15,#2);\n"
  "r5 = add(r5,#3);\n"
  "r14 = memw(r1+#16);\n"
  "r4 = memw(r1+#4) }\n"
  /* asm packet from comment; words preserved. */
  "{ r5 = lsr(r5,#2);\n"
  "r24 = #8;\n"
  "memd(r29+#24) = r23:22;\n"
  "memd(r29+#32) = r25:24 }\n"
  /* asm packet from comment; words preserved. */
  "{ r4 = add(r4,#-1);\n"
  "r22 = #131;\n"
  "r1 = memw(r1+#0);\n"
  "r25 = memw(r0+#12) }\n"
  /* asm packet from comment; words preserved. */
  "{ p2 = bitsclr(r10,r22);\n"
  "if (!p2.new) r25 = add(r25,#4);\n"
  "r16 = memw(r0+#8); r17 = memw(r0+#4) }\n"
  /* asm packet from comment; words preserved. */
  "{ r13 = r1;\n"
  "nop;\n"
  "r28 = memw(r0+#20);\n"
  "r12 = memw(r0+#16) }\n"
  /* asm packet from comment; words preserved. */
  "{ r25 -= asl(r5,#2);\n"
  "r28 = add(r28,#15);\n"
  "r0 = memw(r0+#0);\n"
  "memd(r29+#40) = r27:26 }\n"
  /* asm packet from comment; words preserved. */
  "{ r28 = asr(r28,#4);\n"
  "r30 = r7;\n"
  "r19 = memw(r18++#4);\n"
  "memd(r29+#48) = r31:30 }\n"
  /* asm packet from comment; words preserved. */
  "{ r7:6 = and(r7:6,r21:20);\n"
  "r28 = add(r28,#-1);\n"
  "r12 = add(r12,#8);\n"
  "r31 = #1023 }\n"
  /* asm packet from comment; words preserved. */
  "{ r17:16 = vaslw(r17:16,#2);\n"
  "r26 = r0;\n"
  "r27 = #0;\n"
  "nop }\n"
  /* decoded packet @0x2fad00..0x2fad10
   *   0x2fad00: { loop1(0x2fad20,r5)
   *   0x2fad04: r23 = #0
   *   0x2fad08: r22 = #131
   *   0x2fad0c: nop }
   * Purpose: initialize or restart hardware loop1 for the current output-row tile body.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0x60254200, 0x78004017, 0x78005076\n"
  /* asm packet split from known decoded raw words. */
  "{ nop }\n"
  /* asm packet from comment; words preserved. */
  "{ p2 = bitsclr(r10,r22);\n"
  "p0 = cmp.gt(r4,#0);\n"
  "p1 = cmp.gt(r14,r24);\n"
  "memd(r29+#56) = r1:0 }\n"
  /* decoded packet @0x2fad20..0x2fad30
   *   0x2fad20: { loop0(0x2fad40,r4)
   *   0x2fad24: r7 = or(r7,#112)
   *   0x2fad28: r22 = memw(r1++#4)
   *   0x2fad2c: if (p1) r23 = memw(r1+r15<<#0) }
   * Purpose: initialize or restart hardware loop0 for the current K/output tile body.
   */
  ".word 0x60044200, 0x76874e07, 0x9b814036, 0x3081cf37\n"
  /* decoded packet @0x2fad30..0x2fad40
   *   0x2fad30: { if (p1) r23 = sub(r23,r22)
   *   0x2fad34: r8 = add(r2,#-1024)
   *   0x2fad38: if (!p0) r7 = add(r30,#0)
   *   0x2fad3c: if (!p0) jump:nt 0x2fad64 }
   * Purpose: branch-sensitive control transfer; raw form preserves native target encoding.
   */
  ".word 0xfb365737, 0xbfc24008, 0x749e4007, 0x5c20c01a\n"
  /* asm packet from comment; words preserved. */
  "{ r7:6 = or(r7:6,r23:22);\n"
  "r8 = add(r8,#1024);\n"
  "r22 = memw(r1++#4);\n"
  "if (p1) r23 = memw(r1+r15<<#0) }\n"
  /* unknown/HMX raw block @0x2fad50..0x2fad60
   *   0x2fad50: <unknown>
   *   0x2fad5c: <unknown>
   * Purpose: HMX operand load; feed activation/weight/bias data into the next MAC or cvt packet.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0xd3e69406, 0xfb365737, 0x920647f0, 0x9208ffe6\n"
  /* asm packet from comment; words preserved. */
  "{ r7 = r30 }\n"
  /* asm packet from comment; words preserved. */
  "{ r7:6 = or(r7:6,r23:22);\n"
  "r8 = add(r8,#1024);\n"
  "r10 = and(r10,r20);\n"
  "r23 = r19 }\n"
  /* partially decoded HMX packet group @0x2fad74..0x2fada4
   *   0x2fad74: <unknown>
   *   0x2fad80: <unknown>
   *   0x2fad84: <unknown>
   *   0x2fad94: <unknown>
   * Purpose: second scalar-tail output-store group; feeds activation/weight
   * operands and preserves the following raw HMX writeback/control tail.
   */
  /* asm packet split from known decoded raw words. */
  "{ r8 += add(r9,#0x1);\n"
  "r7:6 = and(r7:6,r21:20);\n"
  "activation.ub = mxmem(r6,r7):single;\n"
  "weight.b = mxmem(r8,r9):single }\n"
  /* asm packet split from known decoded raw words.
   * Purpose: output-column setup; advance bias/window state before the following cvt store.
   */
  "{ r3 = add(r3,#0x100);\n"
  "r23 = add(r23,#-0x1);\n"
  "r22 = memw(r18++#0x4);\n"
  "bias = mxmem2(r3) }\n"
  /* remaining raw tail after byte-proven bias/window setup.
   * Purpose: handles predicate/address update and cvt writeback for this output tile;
   * kept raw because mixed branch/HMX words re-encode differently as asm.
   */
  ".word 0x75574003, 0x5cdf6bf8, 0x47805816\n"
  /* asm packet split from known decoded raw words. */
  "{ cvt.uh = acc(r22):2x2 }\n"
  /* asm packet from comment; words preserved. */
  "{ r18 -= asl(r19,#2);\n"
  "r3 -= asl(r19,#8);\n"
  "r22 = add(r22,r27) }\n"
  /* decoded packet @0x2fadb0..0x2fadb8
   *   0x2fadb0: { r10 = or(r10,r22)
   *   0x2fadb4: if (!p2) jump:nt 0x2fadbc }
   * Purpose: branch-sensitive control transfer; raw form preserves native target encoding.
   */
  ".word 0xf12a560a, 0x5c20c206\n"
  /* unknown/HMX raw block @0x2fadb8..0x2fadbc
   *   0x2fadb8: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   */
  ".word 0xa6eacb1a\n"
  /* decoded packet @0x2fadbc..0x2fadc8
   *   0x2fadbc: { p2 = or(p2,!p2)
   *   0x2fadc0: if (p2) r0 = add(r0,r17)
   *   0x2fadc4: nop } :endloop1
   * Purpose: hardware-loop tail; raw form preserves endloop packet encoding exactly.
   * Encoding note: the .word below is the residual raw subset for this native packet/group; byte-proven pieces may be emitted as asm nearby.
   */
  ".word 0x6be24202, 0xfb009140\n"
  /* asm packet split from known decoded raw words. */
  "{ nop }\n"
  /* decoded packet @0x2fadc8..0x2fadd4
   *   0x2fadc8: { p3 = cmp.gt(r25,#0)
   *   0x2fadcc: if (!p3.new) jump:t 0x2fade8
   *   0x2fadd0: if (p3.new) r22 = #0 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".word 0x75594003, 0x5c205b10, 0x7e60e016\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ r10 = and(r10,r20);\n"
  "r22 = memw(r0+#0x0);\n"
  "cvt.uh = acc(r22):2x2 }\n"
  /* asm packet from comment; words preserved. */
  "{ r10 |= or(r22,r27) }\n"
  /* unknown/HMX raw block @0x2fade4..0x2fade8
   *   0x2fade4: <unknown>
   * Purpose: accumulator conversion/writeback tail; stores or finalizes cvt output for this tile.
   */
  ".word 0xa6eacb1a\n"
  /* asm packet from comment; words preserved. */
  "{ r24 = add(r24,#8);\n"
  "r1:0 = memd(r29+#56) }\n"
  /* decoded packet @0x2fadf0..0x2fae00
   *   0x2fadf0: { r1 = add(r1,r15)
   *   0x2fadf4: r0 = add(r0,r16)
   *   0x2fadf8: p0 = cmp.gt(r12,r24)
   *   0x2fadfc: if (p0.new) jump:nt 0x2fad00 }
   * Purpose: loop/variant control; updates predicate and branches to the selected tile path.
   */
  ".word 0xf3014f01, 0xf3005000, 0xf24c5800, 0x5cdfe888\n"
  /* asm packet from comment; words preserved. */
  "{ r1 = r13;\n"
  "r0 = r26;\n"
  "r28 = add(r28,#-1);\n"
  "p3 = cmp.eq(r27,#0) }\n"
  /* asm packet from comment; words preserved. */
  "{ r2 = r8;\n"
  "r27 = mux(p3,#64,#0) }\n"
  /* decoded packet @0x2fae18..0x2fae28
   *   0x2fae18: { if (r28>=#0) jump:t 0x2fad00
   *   0x2fae1c: r3 += asl(r19,#8)
   *   0x2fae20: if (p3) r26 = add(r26,#4)
   *   0x2fae24: r24 = #8 }
   * Purpose: branch-sensitive control transfer; raw form preserves native target encoding.
   */
  ".word 0x617c7f74, 0x8e1348c3, 0x747a409a, 0x7800c118\n"
  /* asm packet from matched decoded packet; words preserved. */
  "{ r31:30 = memd(r29+#0x30);\n"
  "mxclracc }\n"
  /* asm packet from comment; words preserved. */
  "{ r27:26 = memd(r29+#40);\n"
  "r25:24 = memd(r29+#32) }\n"
  /* asm packet from comment; words preserved. */
  "{ r23:22 = memd(r29+#24); r21:20 = memd(r29+#16) }\n"
  /* decoded packet @0x2fae3c..0x2fae48
   *   0x2fae3c: { r29 = add(r29,#64)
   *   0x2fae40: jumpr r31
   *   0x2fae44: r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }
   */
  /* asm packet from decoded raw words; byte-proven. */
  "{ r29 = add(r29,#64);\n"
  "jumpr r31;\n"
  "r19:18 = memd(r29+#8); r17:16 = memd(r29+#0) }\n"
