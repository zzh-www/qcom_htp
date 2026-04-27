
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
