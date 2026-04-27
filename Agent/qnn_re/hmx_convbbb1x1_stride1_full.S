
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

002ea740 <hmx_convbbb1x1_stride1>:
  2ea740: { 	r28 = #0x7e0
  2ea744:   	r7:6 = memd(r4+#0x8)
  2ea748:   	r11:10 = memd(r4+#0x0) } 
  2ea74c: { 	p0 = bitsclr(r6,r28)
  2ea750:   	p0 = bitsclr(r10,r28)
  2ea754:   	r9 = memw(r4+#0x18)
  2ea758:   	r15 = memw(r1+#0x8) } 
  2ea75c: { 	immext(#0xa00)
  2ea760:   	if (!p0) jump:nt 0x2eb180 <hmx_convbbb1x1_stride1_unaligned> } 
  2ea764: <unknown>
  2ea774: { 	r18 = ct0(r28)
  2ea778:   	r15 = asl(r15,#0x2)
  2ea77c:   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2ea780: { 	r7 = extractu(r7,#0xb,#0x0)
  2ea784:   	r18 = add(r18,#-0x5)
  2ea788:   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2ea78c: { 	r28 = r1
  2ea790:   	p0 = cmp.gt(r4,#0x1)
  2ea794:   	r19 = sub(#0x6,r18)
  2ea798:   	memd(r29+#0x18) = r23:22 } 
  2ea79c: { 	p1 = tstbit(r4,#0x0)
  2ea7a0:   	r21 = lsl(#0x1,r18)
  2ea7a4:   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2ea7a8: { 	r17:16 = vaslw(r17:16,#0x2)
  2ea7ac:   	r5 = lsr(r4,#0x1)
  2ea7b0:   	r13 = memw(r0+#0x14)
  2ea7b4:   	memd(r29+#0x20) = r25:24 } 
  2ea7b8: { 	m0 = r17
  2ea7bc:   	r22 = lsl(#0x1,r19)
  2ea7c0:   	r20 = memw(r0+#0xc)
  2ea7c4:   	memd(r29+#0x28) = r27:26 } 
  2ea7c8: { 	r20 += add(r21,#-0x1)
  2ea7cc:   	r13 = add(r13,#-0x1)
  2ea7d0:   	r12 = memw(r0+#0x10)
  2ea7d4:   	r0 = memw(r0+#0x0) } 
  2ea7d8: { 	r20 = lsr(r20,r18)
  2ea7dc:   	r14 = add(r0,#0x4)
  2ea7e0:   	r19:18 = combine(r1,r0)
  2ea7e4:   	if (!p0) jump:nt 0x2ea8c0 <hmx_convbbb1x1_stride1+0x180> } 
  2ea7e8: { 	loop0(0x2ea820,r5)
  2ea7ec:   	r26 = r4
  2ea7f0:   	r24 = or(r7,#0x1c)
  2ea7f4:   	r25 = #0x3ff } 
  2ea7f8: { 	nop
  2ea7fc:   	nop } 
  2ea800: <unknown>
  2ea810: { 	loop1(0x2ea820,r20)
  2ea814:   	r1:0 = combine(r19,r18)
  2ea818:   	r17 = sub(r17,r22)
  2ea81c:   	r8 = r2 } 
  2ea820: { 	p0 = cmp.eq(r26,#0x2)
  2ea824:   	r26 = add(r26,#-0x2)
  2ea828:   	r6 = memw(r1++#0x8)
  2ea82c:   	r23 = memw(r1+#0x4) } 
  2ea830: <unknown>
  2ea83c: <unknown>
  2ea840: <unknown>
  2ea848: <unknown>
  2ea84c: { 	loop0(0x2ea820,r5)
  2ea850:   	r24 = or(r7,#0x1c)
  2ea854:   	if (!p1) jump:nt 0x2ea86c <hmx_convbbb1x1_stride1+0x12c>
  2ea858:   	r10 = memw(r0++m0) } 
  2ea85c: { 	r6 = memw(r1++#0x4) } 
  2ea860: <unknown>
  2ea868: <unknown>
  2ea86c: <unknown>
  2ea87c: { 	r19 = add(r19,r15)
  2ea880:   	r18 = add(r18,r16)
  2ea884:   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ea810 <hmx_convbbb1x1_stride1+0xd0> } 
  2ea888: { 	if (r13>=#0) jump:t 0x2ea800
  2ea88c:   	r19:18 = combine(r28,r14)
  2ea890:   	r2 = r27
  2ea894:   	r14 = add(r14,#0x4) } 
  2ea898: { 	r27:26 = memd(r29+#0x28)
  2ea89c:   	r25:24 = memd(r29+#0x20) } 
  2ea8a0: { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2ea8a4: { 	r29 = add(r29,#0x30)
  2ea8a8:   	jumpr r31
  2ea8ac:   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
  2ea8b0: { 	nop
  2ea8b4:   	nop
  2ea8b8:   	nop
  2ea8bc:   	nop } 
  2ea8c0: <unknown>
  2ea8d0: { 	loop0(0x2ea8e0,r20)
  2ea8d4:   	r1:0 = combine(r19,r18)
  2ea8d8:   	r17 = sub(r17,r22)
  2ea8dc:   	nop } 
  2ea8e0: { 	r6 = memw(r1++#0x4)
  2ea8e4:   	r10 = memw(r0++m0) } 
  2ea8e8: <unknown>
  2ea8ec: <unknown>
  2ea8f0: <unknown>
  2ea8f8: { 	r19 = add(r19,r15)
  2ea8fc:   	r18 = add(r18,r16)
  2ea900:   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ea8d0 <hmx_convbbb1x1_stride1+0x190> } 
  2ea904: { 	if (r13>=#0) jump:t 0x2ea8c0
  2ea908:   	r2 += add(r9,#0x1)
  2ea90c:   	r19:18 = combine(r28,r14)
  2ea910:   	r14 = add(r14,#0x4) } 
  2ea914: { 	r27:26 = memd(r29+#0x28)
  2ea918:   	r25:24 = memd(r29+#0x20) } 
  2ea91c: { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2ea920: { 	r29 = add(r29,#0x30)
  2ea924:   	jumpr r31
  2ea928:   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002ea940 <hmx_v73_convbbb1x1_stride1_sparsity>:
  2ea940: { 	r28 = #0x7e0
  2ea944:   	r7:6 = memd(r4+#0x8)
  2ea948:   	r11:10 = memd(r4+#0x0) } 
  2ea94c: { 	p0 = bitsclr(r6,r28)
  2ea950:   	p0 = bitsclr(r10,r28)
  2ea954:   	r8 = memw(r4+#0x30) } 
  2ea958: { 	immext(#0xa00)
  2ea95c:   	if (!p0) jump:nt 0x2eb380 <hmx_v73_convbbb1x1_stride1_unaligned_sparsity> } 
  2ea960: { 	p1 = tstbit(r8,#0x5)
  2ea964:   	if (p1.new) jump:t 0x2ebb00 <hmx_v73_convbbb1x1deep_stride1_sparsity>
  2ea968:   	r9 = memw(r4+#0x18)
  2ea96c:   	r15 = memw(r1+#0x8) } 
  2ea970: { 	r29 = add(r29,#-0x50)
  2ea974:   	r28 = and(r28,r7)
  2ea978:   	nop
  2ea97c:   	memd(r29+#-0x50) = r17:16 } 
  2ea980: { 	r18 = ct0(r28)
  2ea984:   	r15 = asl(r15,#0x2)
  2ea988:   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2ea98c: { 	r7 = extractu(r7,#0xb,#0x0)
  2ea990:   	r18 = add(r18,#-0x5)
  2ea994:   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2ea998: { 	p0 = cmp.gt(r4,#0x1)
  2ea99c:   	r19 = sub(#0x6,r18)
  2ea9a0:   	memd(r29+#0x18) = r23:22
  2ea9a4:   	memd(r29+#0x20) = r25:24 } 
  2ea9a8: { 	p1 = tstbit(r4,#0x0)
  2ea9ac:   	r21 = lsl(#0x1,r18)
  2ea9b0:   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2ea9b4: { 	r17:16 = vaslw(r17:16,#0x2)
  2ea9b8:   	r28 = lsr(r4,#0x1)
  2ea9bc:   	r13 = memw(r0+#0x14)
  2ea9c0:   	memd(r29+#0x28) = r27:26 } 
  2ea9c4: { 	m0 = r17
  2ea9c8:   	r22 = lsl(#0x1,r19)
  2ea9cc:   	r20 = memw(r0+#0xc)
  2ea9d0:   	memd(r29+#0x30) = r31:30 } 
  2ea9d4: { 	r20 += add(r21,#-0x1)
  2ea9d8:   	r12 = memw(r0+#0x10)
  2ea9dc:   	r0 = memw(r0+#0x0) } 
  2ea9e0: { 	r20 = lsr(r20,r18)
  2ea9e4:   	r14 = add(r0,#0x4)
  2ea9e8:   	r30 = memw(r5++#0x4)
  2ea9ec:   	memw(r29+#0x38) = r1 } 
  2ea9f0: { 	r26 = r4
  2ea9f4:   	r24 = or(r7,#0x1c)
  2ea9f8:   	r25 = #0x400 } 
  2ea9fc: { 	memw(r29+#0x40) = r30 } 
  2eaa00: { 	r30 = and(r9,#0x7f) } 
  2eaa04: { 	p3 = cmp.eq(r30,#0x0)
  2eaa08:   	if (p3.new) jump:t 0x2eaa24 <hmx_v73_convbbb1x1_stride1_sparsity+0xe4> } 
  2eaa0c: { 	r30 = add(r30,#0x1) } 
  2eaa10: { 	r30 = asr(r30,#0x4) } 
  2eaa14: { 	r30 = add(r30,#0x1) } 
  2eaa18: { 	r25 = +mpyi(r30,#0x80) } 
  2eaa1c: { 	r30 = and(r9,#0x7f) } 
  2eaa20: { 	r25 = or(r25,r30) } 
  2eaa24: { 	r30 = and(r25,#-0x80) } 
  2eaa28: { 	r30 = add(r30,#-0x80)
  2eaa2c:   	r25 = add(r25,#-0x80) } 
  2eaa30: { 	memw(r29+#0x48) = r30 } 
  2eaa34: { 	r30 = memw(r29+#0x40) } 
  2eaa38: { 	r19:18 = combine(r1,r0)
  2eaa3c:   	p2 = cmp.eq(r30,#0x1)
  2eaa40:   	if (!p0) jump:nt 0x2eab44 <hmx_v73_convbbb1x1_stride1_sparsity+0x204>
  2eaa44:   	if (p2.new) r31 = memw(r5+#0x0) } 
  2eaa48: { 	if (!p2) jump:nt 0x2eabf0 <hmx_v73_convbbb1x1_stride1_sparsity+0x2b0> } 
  2eaa4c: { 	loop0(0x2eaa80,r28)
  2eaa50:   	nop
  2eaa54:   	nop
  2eaa58:   	nop } 
  2eaa5c: { 	nop } 
  2eaa60: <unknown>
  2eaa70: { 	loop1(0x2eaa80,r20)
  2eaa74:   	r1:0 = combine(r19,r18)
  2eaa78:   	r17 = sub(r17,r22)
  2eaa7c:   	r8 = r2 } 
  2eaa80: { 	p0 = cmp.eq(r26,#0x2)
  2eaa84:   	r26 = add(r26,#-0x2)
  2eaa88:   	r6 = memw(r1++#0x8)
  2eaa8c:   	r23 = memw(r1+#0x4) } 
  2eaa90: <unknown>
  2eaa98: <unknown>
  2eaa9c: { 	memw(r29+#0x40) = r30 } 
  2eaaa0: { 	r30 = memw(r29+#0x48) } 
  2eaaa4: { 	immext(#0x80)
  2eaaa8:   	r8 += add(r30,##0x80) } 
  2eaaac: <unknown>
  2eaab4: <unknown>
  2eaab8: { 	immext(#0x80)
  2eaabc:   	r8 += add(r30,##0x80)
  2eaac0:   	r30 = memw(r29+#0x40) }  :endloop0
  2eaac4: { 	loop0(0x2eaa80,r28)
  2eaac8:   	r24 = or(r7,#0x1c)
  2eaacc:   	if (!p1) jump:nt 0x2eaaf4 <hmx_v73_convbbb1x1_stride1_sparsity+0x1b4>
  2eaad0:   	r10 = memw(r0++m0) } 
  2eaad4: { 	r6 = memw(r1++#0x4) } 
  2eaad8: <unknown>
  2eaadc: <unknown>
  2eaae0: { 	memw(r29+#0x40) = r30 } 
  2eaae4: { 	r30 = and(r9,#-0x80) } 
  2eaae8: { 	immext(#0x80)
  2eaaec:   	r8 += add(r30,##0x80)
  2eaaf0:   	r30 = memw(r29+#0x40) } 
  2eaaf4: <unknown>
  2eab04: <unknown>
  2eab10: { 	r25 = or(r25,r30)
  2eab14:   	nop
  2eab18:   	r30 = memw(r29+#0x40) }  :endloop1
  2eab1c: { 	r19 = add(r19,r15)
  2eab20:   	r18 = add(r18,r16)
  2eab24:   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2eaa70 <hmx_v73_convbbb1x1_stride1_sparsity+0x130> } 
  2eab28: { 	r18 = r14
  2eab2c:   	r19 = memw(r29+#0x38) } 
  2eab30: { 	r2 = r27
  2eab34:   	r14 = add(r14,#0x4)
  2eab38:   	p0 = cmp.gt(r13,#0x0)
  2eab3c:   	if (p0.new) jump:t 0x2eaa60 <hmx_v73_convbbb1x1_stride1_sparsity+0x120> } 
  2eab40: { 	jump 0x2eabd0 <hmx_v73_convbbb1x1_stride1_sparsity+0x290> } 
  2eab44: { 	if (!p2) jump:nt 0x2ead20 <hmx_v73_convbbb1x1_stride1_sparsity+0x3e0> } 
  2eab48: { 	nop
  2eab4c:   	nop } 
  2eab50: { 	nop
  2eab54:   	nop
  2eab58:   	nop
  2eab5c:   	nop } 
  2eab60: <unknown>
  2eab70: { 	loop0(0x2eab80,r20)
  2eab74:   	r1:0 = combine(r19,r18)
  2eab78:   	r17 = sub(r17,r22)
  2eab7c:   	nop } 
  2eab80: { 	r6 = memw(r1++#0x4)
  2eab84:   	r10 = memw(r0++m0) } 
  2eab88: <unknown>
  2eab8c: <unknown>
  2eab90: <unknown>
  2eab94: <unknown>
  2eab9c: { 	r19 = add(r19,r15)
  2eaba0:   	r18 = add(r18,r16)
  2eaba4:   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2eab70 <hmx_v73_convbbb1x1_stride1_sparsity+0x230> } 
  2eaba8: { 	r18 = r14
  2eabac:   	r19 = memw(r29+#0x38)
  2eabb0:   	memw(r29+#0x40) = r30 } 
  2eabb4: { 	r30 = and(r9,#-0x80) } 
  2eabb8: { 	immext(#0x80)
  2eabbc:   	r2 += add(r30,##0x80)
  2eabc0:   	r30 = memw(r29+#0x40) } 
  2eabc4: { 	r14 = add(r14,#0x4)
  2eabc8:   	p0 = cmp.gt(r13,#0x0)
  2eabcc:   	if (p0.new) jump:t 0x2eab60 <hmx_v73_convbbb1x1_stride1_sparsity+0x220> } 
  2eabd0: <unknown>
  2eabd8: { 	r27:26 = memd(r29+#0x28)
  2eabdc:   	r25:24 = memd(r29+#0x20) } 
  2eabe0: { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2eabe4: { 	r29 = add(r29,#0x50)
  2eabe8:   	jumpr r31
  2eabec:   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
  2eabf0: { 	loop0(0x2eac20,r28)
  2eabf4:   	nop
  2eabf8:   	nop
  2eabfc:   	nop } 
  2eac00: { 	nop
  2eac04:   	nop
  2eac08:   	r13 = add(r13,#-0x20)
  2eac0c:   	r17 = r12 } 
  2eac10: { 	loop1(0x2eac20,r20)
  2eac14:   	r1:0 = combine(r19,r18)
  2eac18:   	r17 = sub(r17,r22)
  2eac1c:   	r8 = r2 } 
  2eac20: { 	p0 = cmp.eq(r26,#0x2)
  2eac24:   	r26 = add(r26,#-0x2)
  2eac28:   	r6 = memw(r1++#0x8)
  2eac2c:   	r23 = memw(r1+#0x4) } 
  2eac30: <unknown>
  2eac38: <unknown>
  2eac3c: { 	memw(r29+#0x40) = r30 } 
  2eac40: { 	r30 = memw(r29+#0x48) } 
  2eac44: { 	immext(#0x80)
  2eac48:   	r8 += add(r30,##0x80) } 
  2eac4c: <unknown>
  2eac54: <unknown>
  2eac58: { 	immext(#0x80)
  2eac5c:   	r8 += add(r30,##0x80)
  2eac60:   	r30 = memw(r29+#0x40) }  :endloop0
  2eac64: { 	loop0(0x2eac20,r28)
  2eac68:   	r24 = or(r7,#0x1c)
  2eac6c:   	if (!p1) jump:nt 0x2eac94 <hmx_v73_convbbb1x1_stride1_sparsity+0x354>
  2eac70:   	r10 = memw(r0++m0) } 
  2eac74: { 	r6 = memw(r1++#0x4)
  2eac78:   	memw(r29+#0x40) = r30 } 
  2eac7c: { 	r30 = and(r9,#-0x80) } 
  2eac80: <unknown>
  2eac8c: <unknown>
  2eac90: { 	r30 = memw(r29+#0x40) } 
  2eac94: { 	r27:26 = combine(r8,r4)
  2eac98:   	r8 = r2
  2eac9c:   	r25 = r30 } 
  2eaca0: <unknown>
  2eacb0: <unknown>
  2eacbc: <unknown>
  2eaccc: { 	r30 = and(r9,#0x7f)
  2eacd0:   	r25 = memw(r29+#0x48) } 
  2eacd4: { 	r25 = or(r25,r30)
  2eacd8:   	nop
  2eacdc:   	r30 = memw(r29+#0x40) }  :endloop1
  2eace0: { 	r19 = add(r19,r15)
  2eace4:   	r18 = add(r18,r16)
  2eace8:   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2eac10 <hmx_v73_convbbb1x1_stride1_sparsity+0x2d0> } 
  2eacec: { 	r3 += asl(r30,#0x8)
  2eacf0:   	r18 = r14
  2eacf4:   	r19 = memw(r29+#0x38) } 
  2eacf8: { 	r2 = r27
  2eacfc:   	r14 = add(r14,#0x4)
  2ead00:   	p0 = cmp.gt(r13,#0x0)
  2ead04:   	if (p0.new) jump:t 0x2eac00 <hmx_v73_convbbb1x1_stride1_sparsity+0x2c0> } 
  2ead08: { 	jump 0x2eabd0 <hmx_v73_convbbb1x1_stride1_sparsity+0x290> } 
  2ead0c: { 	nop } 
  2ead10: { 	nop
  2ead14:   	nop
  2ead18:   	nop
  2ead1c:   	nop } 
  2ead20: { 	nop
  2ead24:   	nop
  2ead28:   	r13 = add(r13,#-0x20)
  2ead2c:   	r17 = r12 } 
  2ead30: { 	loop0(0x2ead40,r20)
  2ead34:   	r1:0 = combine(r19,r18)
  2ead38:   	r17 = sub(r17,r22)
  2ead3c:   	nop } 
  2ead40: { 	r6 = memw(r1++#0x4)
  2ead44:   	r10 = memw(r0++m0) } 
  2ead48: <unknown>
  2ead50: <unknown>
  2ead54: <unknown>
  2ead64: <unknown>
  2ead70: <unknown>
  2ead7c: { 	r19 = add(r19,r15)
  2ead80:   	r18 = add(r18,r16)
  2ead84:   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ead30 <hmx_v73_convbbb1x1_stride1_sparsity+0x3f0> } 
  2ead88: { 	r3 += asl(r30,#0x8)
  2ead8c:   	r18 = r14
  2ead90:   	r19 = memw(r29+#0x38)
  2ead94:   	memw(r29+#0x40) = r30 } 
  2ead98: { 	r30 = and(r9,#-0x80) } 
  2ead9c: { 	immext(#0x80)
  2eada0:   	r2 += add(r30,##0x80)
  2eada4:   	r30 = memw(r29+#0x40) } 
  2eada8: { 	r14 = add(r14,#0x4)
  2eadac:   	p0 = cmp.gt(r13,#0x0)
  2eadb0:   	if (p0.new) jump:t 0x2ead20 <hmx_v73_convbbb1x1_stride1_sparsity+0x3e0> } 
  2eadb4: { 	jump 0x2eabd0 <hmx_v73_convbbb1x1_stride1_sparsity+0x290> } 
		...
