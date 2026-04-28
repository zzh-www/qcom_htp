
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

002ea740 <hmx_convbbb1x1_stride1>:
  2ea740:	1c 7c 03 78	78037c1c { 	r28 = #0x7e0
  2ea744:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2ea748:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2ea74c:	00 5c 86 c7	c7865c00 { 	p0 = bitsclr(r6,r28)
  2ea750:	00 5c 8a c7	c78a5c00   	p0 = bitsclr(r10,r28)
  2ea754:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2ea758:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2ea75c:	28 40 00 00	00004028 { 	immext(#0xa00)
  2ea760:	48 c0 20 5c	5c20c048   	if (!p0) jump:nt 0x2eb180 <hmx_convbbb1x1_stride1_unaligned> } 
  2ea764:	1d 7a fd bf	bffd7a1d { 	r29 = add(r29,#-0x30)
  2ea768:	1c 47 1c f1	f11c471c   	r28 = and(r28,r7)
  2ea76c:	fa 70 dd a7	a7dd70fa   	memd(r29+#-0x30) = r17:16
  2ea770:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2ea774:	92 40 5c 8c	8c5c4092 { 	r18 = ct0(r28)
  2ea778:	4f 42 0f 8c	8c0f424f   	r15 = asl(r15,#0x2)
  2ea77c:	16 0a 0d ea	ea0d0a16   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2ea780:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2ea784:	72 7f f2 bf	bff27f72   	r18 = add(r18,#-0x5)
  2ea788:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2ea78c:	1c 40 61 70	7061401c { 	r28 = r1
  2ea790:	20 40 44 75	75444020   	p0 = cmp.gt(r4,#0x1)
  2ea794:	d3 40 52 76	765240d3   	r19 = sub(#0x6,r18)
  2ea798:	03 d6 dd a1	a1ddd603   	memd(r29+#0x18) = r23:22 } 
  2ea79c:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2ea7a0:	f5 52 80 c6	c68052f5   	r21 = lsl(#0x1,r18)
  2ea7a4:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2ea7a8:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2ea7ac:	25 41 04 8c	8c044125   	r5 = lsr(r4,#0x1)
  2ea7b0:	ad 40 80 91	918040ad   	r13 = memw(r0+#0x14)
  2ea7b4:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2ea7b8:	06 40 31 62	62314006 { 	m0 = r17
  2ea7bc:	f6 53 80 c6	c68053f6   	r22 = lsl(#0x1,r19)
  2ea7c0:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2ea7c4:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2ea7c8:	f4 5f 15 e2	e2155ff4 { 	r20 += add(r21,#-0x1)
  2ea7cc:	ed 7f ed bf	bfed7fed   	r13 = add(r13,#-0x1)
  2ea7d0:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2ea7d4:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2ea7d8:	54 52 54 c6	c6545254 { 	r20 = lsr(r20,r18)
  2ea7dc:	8e 40 00 b0	b000408e   	r14 = add(r0,#0x4)
  2ea7e0:	12 40 01 f5	f5014012   	r19:18 = combine(r1,r0)
  2ea7e4:	74 c0 20 5c	5c20c074   	if (!p0) jump:nt 0x2ea8c0 <hmx_convbbb1x1_stride1+0x180> } 
  2ea7e8:	10 43 05 60	60054310 { 	loop0(0x2ea820,r5)
  2ea7ec:	1a 40 64 70	7064401a   	r26 = r4
  2ea7f0:	98 43 87 76	76874398   	r24 = or(r7,#0x1c)
  2ea7f4:	f9 ff 01 78	7801fff9   	r25 = #0x3ff } 
  2ea7f8:	00 40 00 7f	7f004000 { 	nop
  2ea7fc:	00 c0 00 7f	7f00c000   	nop } 
  2ea800:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2ea804:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2ea808:	11 40 6c 70	706c4011   	r17 = r12
  2ea80c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ea810:	00 41 34 60	60344100 { 	loop1(0x2ea820,r20)
  2ea814:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2ea818:	11 51 36 f3	f3365111   	r17 = sub(r17,r22)
  2ea81c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2ea820:	40 40 1a 75	751a4040 { 	p0 = cmp.eq(r26,#0x2)
  2ea824:	da 7f fa bf	bffa7fda   	r26 = add(r26,#-0x2)
  2ea828:	46 40 81 9b	9b814046   	r6 = memw(r1++#0x8)
  2ea82c:	37 c0 81 91	9181c037   	r23 = memw(r1+#0x4) } 
  2ea830:	08 40 48 b0	b0484008 { 	r8 = add(r8,#0x400)
  2ea834:	18 47 09 fd	fd094718   	if (p0) r25:24 = combine(r9,r7)
  2ea838:	ed 58 06 92	920658ed   	activation.ub = mxmem(r6,r24):cm
  2ea83c:	e0 f9 08 92	9208f9e0   	weight.b = mxmem(r8,r25) } 
  2ea840:	28 80 19 e2	e2198028 { 	r8 += add(r25,#0x1)
  2ea844:	ed 58 17 92	921758ed   	activation.ub = mxmem(r23,r24):cm
  2ea848:	e0 f9 08 92	9208f9e0   	weight.b = mxmem(r8,r25) }  :endloop0
  2ea84c:	08 5d 05 60	60055d08 { 	loop0(0x2ea820,r5)
  2ea850:	98 43 87 76	76874398   	r24 = or(r7,#0x1c)
  2ea854:	10 41 20 5c	5c204110   	if (!p1) jump:nt 0x2ea86c <hmx_convbbb1x1_stride1+0x12c>
  2ea858:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2ea85c:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2ea860:	28 40 09 e2	e2094028 { 	r8 += add(r9,#0x1)
  2ea864:	ed 47 06 92	920647ed   	activation.ub = mxmem(r6,r7):cm
  2ea868:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) } 
  2ea86c:	1a 44 08 f5	f508441a { 	r27:26 = combine(r8,r4)
  2ea870:	08 80 62 70	70628008   	r8 = r2
  2ea874:	f9 7f 01 78	78017ff9   	r25 = #0x3ff
  2ea878:	05 cb ea a6	a6eacb05   	mxmem(r10,r11):after:cm:sat.ub = acc }  :endloop1
  2ea87c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ea880:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ea884:	ca c0 b9 10	10b9c0ca   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ea810 <hmx_convbbb1x1_stride1+0xd0> } 
  2ea888:	bc 7f 6d 61	616d7fbc { 	if (r13>=#0) jump:t 0x2ea800
  2ea88c:	12 4e 1c f5	f51c4e12   	r19:18 = combine(r28,r14)
  2ea890:	02 40 7b 70	707b4002   	r2 = r27
  2ea894:	8e c0 0e b0	b00ec08e   	r14 = add(r14,#0x4) } 
  2ea898:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2ea89c:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2ea8a0:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2ea8a4:	1d 46 1d b0	b01d461d { 	r29 = add(r29,#0x30)
  2ea8a8:	00 40 9f 52	529f4000   	jumpr r31
  2ea8ac:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
  2ea8b0:	00 40 00 7f	7f004000 { 	nop
  2ea8b4:	00 40 00 7f	7f004000   	nop
  2ea8b8:	00 40 00 7f	7f004000   	nop
  2ea8bc:	00 c0 00 7f	7f00c000   	nop } 
  2ea8c0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2ea8c4:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2ea8c8:	11 40 6c 70	706c4011   	r17 = r12
  2ea8cc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ea8d0:	00 41 14 60	60144100 { 	loop0(0x2ea8e0,r20)
  2ea8d4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2ea8d8:	11 51 36 f3	f3365111   	r17 = sub(r17,r22)
  2ea8dc:	00 c0 00 7f	7f00c000   	nop } 
  2ea8e0:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2ea8e4:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2ea8e8:	ed 47 06 92	920647ed { 	activation.ub = mxmem(r6,r7):cm
  2ea8ec:	e0 e9 02 92	9202e9e0   	weight.b = mxmem(r2,r9) } 
  2ea8f0:	00 80 00 7f	7f008000 { 	nop
  2ea8f4:	05 cb ea a6	a6eacb05   	mxmem(r10,r11):after:cm:sat.ub = acc }  :endloop0
  2ea8f8:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ea8fc:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ea900:	ec c0 b9 10	10b9c0ec   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ea8d0 <hmx_convbbb1x1_stride1+0x190> } 
  2ea904:	de 7f 6d 61	616d7fde { 	if (r13>=#0) jump:t 0x2ea8c0
  2ea908:	22 40 09 e2	e2094022   	r2 += add(r9,#0x1)
  2ea90c:	12 4e 1c f5	f51c4e12   	r19:18 = combine(r28,r14)
  2ea910:	8e c0 0e b0	b00ec08e   	r14 = add(r14,#0x4) } 
  2ea914:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2ea918:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2ea91c:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2ea920:	1d 46 1d b0	b01d461d { 	r29 = add(r29,#0x30)
  2ea924:	00 40 9f 52	529f4000   	jumpr r31
  2ea928:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...
