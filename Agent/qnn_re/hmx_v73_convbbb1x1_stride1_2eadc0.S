
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

002eadc0 <hmx_v73_convbbb1x1_stride1>:
  2eadc0:	1c 7c 03 78	78037c1c { 	r28 = #0x7e0
  2eadc4:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2eadc8:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2eadcc:	00 5c 86 c7	c7865c00 { 	p0 = bitsclr(r6,r28)
  2eadd0:	00 5c 8a c7	c78a5c00   	p0 = bitsclr(r10,r28)
  2eadd4:	88 c1 84 91	9184c188   	r8 = memw(r4+#0x30) } 
  2eadd8:	26 40 00 00	00004026 { 	immext(#0x980)
  2eaddc:	50 c0 20 5c	5c20c050   	if (!p0) jump:nt 0x2eb780 <hmx_v73_convbbb1x1_stride1_unaligned> } 
  2eade0:	01 45 08 85	85084501 { 	p1 = tstbit(r8,#0x5)
  2eade4:	30 59 04 5c	5c045930   	if (p1.new) jump:t 0x2ebe40 <hmx_v73_convbbb1x1deep_stride1>
  2eade8:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2eadec:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2eadf0:	1d 78 fd bf	bffd781d { 	r29 = add(r29,#-0x40)
  2eadf4:	1c 47 1c f1	f11c471c   	r28 = and(r28,r7)
  2eadf8:	00 40 00 7f	7f004000   	nop
  2eadfc:	f8 f0 dd a7	a7ddf0f8   	memd(r29+#-0x40) = r17:16 } 
  2eae00:	92 40 5c 8c	8c5c4092 { 	r18 = ct0(r28)
  2eae04:	4f 42 0f 8c	8c0f424f   	r15 = asl(r15,#0x2)
  2eae08:	16 0a 0d ea	ea0d0a16   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2eae0c:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2eae10:	72 7f f2 bf	bff27f72   	r18 = add(r18,#-0x5)
  2eae14:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2eae18:	20 40 44 75	75444020 { 	p0 = cmp.gt(r4,#0x1)
  2eae1c:	d3 40 52 76	765240d3   	r19 = sub(#0x6,r18)
  2eae20:	03 56 dd a1	a1dd5603   	memd(r29+#0x18) = r23:22
  2eae24:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2eae28:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2eae2c:	f5 52 80 c6	c68052f5   	r21 = lsl(#0x1,r18)
  2eae30:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2eae34:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2eae38:	3c 41 04 8c	8c04413c   	r28 = lsr(r4,#0x1)
  2eae3c:	ad 40 80 91	918040ad   	r13 = memw(r0+#0x14)
  2eae40:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2eae44:	06 40 31 62	62314006 { 	m0 = r17
  2eae48:	f6 53 80 c6	c68053f6   	r22 = lsl(#0x1,r19)
  2eae4c:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2eae50:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2eae54:	f4 5f 15 e2	e2155ff4 { 	r20 += add(r21,#-0x1)
  2eae58:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2eae5c:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2eae60:	54 52 54 c6	c6545254 { 	r20 = lsr(r20,r18)
  2eae64:	8e 40 00 b0	b000408e   	r14 = add(r0,#0x4)
  2eae68:	3e 40 85 9b	9b85403e   	r30 = memw(r5++#0x4)
  2eae6c:	0e c1 9d a1	a19dc10e   	memw(r29+#0x38) = r1 } 
  2eae70:	12 40 01 f5	f5014012 { 	r19:18 = combine(r1,r0)
  2eae74:	22 40 1e 75	751e4022   	p2 = cmp.eq(r30,#0x1)
  2eae78:	6e 40 20 5c	5c20406e   	if (!p0) jump:nt 0x2eaf4c <hmx_v73_convbbb1x1_stride1+0x18c>
  2eae7c:	1f d0 85 43	4385d01f   	if (p2.new) r31 = memw(r5+#0x0) } 
  2eae80:	1a 40 64 70	7064401a { 	r26 = r4
  2eae84:	98 43 87 76	76874398   	r24 = or(r7,#0x1c)
  2eae88:	f9 7f 01 78	78017ff9   	r25 = #0x3ff
  2eae8c:	b0 c2 20 5c	5c20c2b0   	if (!p2) jump:nt 0x2eafe0 <hmx_v73_convbbb1x1_stride1+0x220> } 
  2eae90:	00 43 1c 60	601c4300 { 	loop0(0x2eaec0,r28)
  2eae94:	00 40 00 7f	7f004000   	nop
  2eae98:	00 40 00 7f	7f004000   	nop
  2eae9c:	00 c0 00 7f	7f00c000   	nop } 
  2eaea0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2eaea4:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2eaea8:	11 40 6c 70	706c4011   	r17 = r12
  2eaeac:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2eaeb0:	00 41 34 60	60344100 { 	loop1(0x2eaec0,r20)
  2eaeb4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2eaeb8:	11 51 36 f3	f3365111   	r17 = sub(r17,r22)
  2eaebc:	08 c0 62 70	7062c008   	r8 = r2 } 
  2eaec0:	40 40 1a 75	751a4040 { 	p0 = cmp.eq(r26,#0x2)
  2eaec4:	da 7f fa bf	bffa7fda   	r26 = add(r26,#-0x2)
  2eaec8:	46 40 81 9b	9b814046   	r6 = memw(r1++#0x8)
  2eaecc:	37 c0 81 91	9181c037   	r23 = memw(r1+#0x4) } 
  2eaed0:	08 40 48 b0	b0484008 { 	r8 = add(r8,#0x400)
  2eaed4:	18 47 09 fd	fd094718   	if (p0) r25:24 = combine(r9,r7)
  2eaed8:	e1 58 06 92	920658e1   	activation.ub = mxmem(r6,r24):deep:cm
  2eaedc:	e0 f9 08 92	9208f9e0   	weight.b = mxmem(r8,r25) } 
  2eaee0:	28 80 19 e2	e2198028 { 	r8 += add(r25,#0x1)
  2eaee4:	e1 58 17 92	921758e1   	activation.ub = mxmem(r23,r24):deep:cm
  2eaee8:	e0 f9 08 92	9208f9e0   	weight.b = mxmem(r8,r25) }  :endloop0
  2eaeec:	08 5d 1c 60	601c5d08 { 	loop0(0x2eaec0,r28)
  2eaef0:	98 43 87 76	76874398   	r24 = or(r7,#0x1c)
  2eaef4:	10 41 20 5c	5c204110   	if (!p1) jump:nt 0x2eaf0c <hmx_v73_convbbb1x1_stride1+0x14c>
  2eaef8:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2eaefc:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2eaf00:	28 40 09 e2	e2094028 { 	r8 += add(r9,#0x1)
  2eaf04:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2eaf08:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) } 
  2eaf0c:	1a 44 08 f5	f508441a { 	r27:26 = combine(r8,r4)
  2eaf10:	08 40 62 70	70624008   	r8 = r2
  2eaf14:	10 d7 ff a6	a6ffd710   	cvt.ub = acc(r31) } 
  2eaf18:	f9 7f 01 78	78017ff9 { 	r25 = #0x3ff
  2eaf1c:	00 80 00 7f	7f008000   	nop
  2eaf20:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop1
  2eaf24:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2eaf28:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2eaf2c:	c6 c0 b9 10	10b9c0c6   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2eaeb0 <hmx_v73_convbbb1x1_stride1+0xf0> } 
  2eaf30:	12 40 6e 70	706e4012 { 	r18 = r14
  2eaf34:	d3 c1 9d 91	919dc1d3   	r19 = memw(r29+#0x38) } 
  2eaf38:	02 40 7b 70	707b4002 { 	r2 = r27
  2eaf3c:	8e 40 0e b0	b00e408e   	r14 = add(r14,#0x4)
  2eaf40:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2eaf44:	b4 f8 df 5c	5cdff8b4   	if (p0.new) jump:t 0x2eaea0 <hmx_v73_convbbb1x1_stride1+0xe0> } 
  2eaf48:	3c c0 00 58	5800c03c { 	jump 0x2eafc0 <hmx_v73_convbbb1x1_stride1+0x200> } 
  2eaf4c:	ca c2 20 5c	5c20c2ca { 	if (!p2) jump:nt 0x2eb0e0 <hmx_v73_convbbb1x1_stride1+0x320> } 
  2eaf50:	00 40 00 7f	7f004000 { 	nop
  2eaf54:	00 40 00 7f	7f004000   	nop
  2eaf58:	00 40 00 7f	7f004000   	nop
  2eaf5c:	00 c0 00 7f	7f00c000   	nop } 
  2eaf60:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2eaf64:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2eaf68:	11 40 6c 70	706c4011   	r17 = r12
  2eaf6c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2eaf70:	00 41 14 60	60144100 { 	loop0(0x2eaf80,r20)
  2eaf74:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2eaf78:	11 51 36 f3	f3365111   	r17 = sub(r17,r22)
  2eaf7c:	00 c0 00 7f	7f00c000   	nop } 
  2eaf80:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2eaf84:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2eaf88:	e1 47 06 92	920647e1 { 	activation.ub = mxmem(r6,r7):deep:cm
  2eaf8c:	e0 e9 02 92	9202e9e0   	weight.b = mxmem(r2,r9) } 
  2eaf90:	10 d7 ff a6	a6ffd710 { 	cvt.ub = acc(r31) } 
  2eaf94:	00 80 00 7f	7f008000 { 	nop
  2eaf98:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop0
  2eaf9c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2eafa0:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2eafa4:	ea c0 b9 10	10b9c0ea   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2eaf70 <hmx_v73_convbbb1x1_stride1+0x1b0> } 
  2eafa8:	12 40 6e 70	706e4012 { 	r18 = r14
  2eafac:	d3 c1 9d 91	919dc1d3   	r19 = memw(r29+#0x38) } 
  2eafb0:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2eafb4:	8e 40 0e b0	b00e408e   	r14 = add(r14,#0x4)
  2eafb8:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2eafbc:	d8 f8 df 5c	5cdff8d8   	if (p0.new) jump:t 0x2eaf60 <hmx_v73_convbbb1x1_stride1+0x1a0> } 
  2eafc0:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2eafc4:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2eafc8:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2eafcc:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2eafd0:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2eafd4:	1d 48 1d b0	b01d481d { 	r29 = add(r29,#0x40)
  2eafd8:	00 40 9f 52	529f4000   	jumpr r31
  2eafdc:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
  2eafe0:	00 44 1c 60	601c4400 { 	loop0(0x2eb020,r28)
  2eafe4:	00 40 00 7f	7f004000   	nop
  2eafe8:	00 40 00 7f	7f004000   	nop
  2eafec:	00 c0 00 7f	7f00c000   	nop } 
  2eaff0:	00 40 00 7f	7f004000 { 	nop
  2eaff4:	00 40 00 7f	7f004000   	nop
  2eaff8:	00 40 00 7f	7f004000   	nop
  2eaffc:	00 c0 00 7f	7f00c000   	nop } 
  2eb000:	00 40 00 7f	7f004000 { 	nop
  2eb004:	00 40 00 7f	7f004000   	nop
  2eb008:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2eb00c:	11 c0 6c 70	706cc011   	r17 = r12 } 
  2eb010:	00 41 34 60	60344100 { 	loop1(0x2eb020,r20)
  2eb014:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2eb018:	11 51 36 f3	f3365111   	r17 = sub(r17,r22)
  2eb01c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2eb020:	40 40 1a 75	751a4040 { 	p0 = cmp.eq(r26,#0x2)
  2eb024:	da 7f fa bf	bffa7fda   	r26 = add(r26,#-0x2)
  2eb028:	46 40 81 9b	9b814046   	r6 = memw(r1++#0x8)
  2eb02c:	37 c0 81 91	9181c037   	r23 = memw(r1+#0x4) } 
  2eb030:	08 40 48 b0	b0484008 { 	r8 = add(r8,#0x400)
  2eb034:	18 47 09 fd	fd094718   	if (p0) r25:24 = combine(r9,r7)
  2eb038:	e1 58 06 92	920658e1   	activation.ub = mxmem(r6,r24):deep:cm
  2eb03c:	e0 f9 08 92	9208f9e0   	weight.b = mxmem(r8,r25) } 
  2eb040:	28 80 19 e2	e2198028 { 	r8 += add(r25,#0x1)
  2eb044:	e1 58 17 92	921758e1   	activation.ub = mxmem(r23,r24):deep:cm
  2eb048:	e0 f9 08 92	9208f9e0   	weight.b = mxmem(r8,r25) }  :endloop0
  2eb04c:	08 5d 1c 60	601c5d08 { 	loop0(0x2eb020,r28)
  2eb050:	98 43 87 76	76874398   	r24 = or(r7,#0x1c)
  2eb054:	10 41 20 5c	5c204110   	if (!p1) jump:nt 0x2eb06c <hmx_v73_convbbb1x1_stride1+0x2ac>
  2eb058:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2eb05c:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2eb060:	28 40 09 e2	e2094028 { 	r8 += add(r9,#0x1)
  2eb064:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2eb068:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) } 
  2eb06c:	1a 44 08 f5	f508441a { 	r27:26 = combine(r8,r4)
  2eb070:	08 40 62 70	70624008   	r8 = r2
  2eb074:	19 c0 7e 70	707ec019   	r25 = r30 } 
  2eb078:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2eb07c:	f9 7f f9 bf	bff97ff9   	r25 = add(r25,#-0x1)
  2eb080:	3f 40 85 9b	9b85403f   	r31 = memw(r5++#0x4)
  2eb084:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2eb088:	00 40 59 75	75594000 { 	p0 = cmp.gt(r25,#0x0)
  2eb08c:	f8 68 df 5c	5cdf68f8   	if (p0.new) jump:nt 0x2eb078 <hmx_v73_convbbb1x1_stride1+0x2b8>
  2eb090:	10 d7 ff a6	a6ffd710   	cvt.ub = acc(r31) } 
  2eb094:	45 42 1e 8e	8e1e4245 { 	r5 -= asl(r30,#0x2)
  2eb098:	43 88 1e 8e	8e1e8843   	r3 -= asl(r30,#0x8)
  2eb09c:	f9 7f 01 78	78017ff9   	r25 = #0x3ff
  2eb0a0:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop1
  2eb0a4:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2eb0a8:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2eb0ac:	b6 c0 b9 10	10b9c0b6   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2eb010 <hmx_v73_convbbb1x1_stride1+0x250> } 
  2eb0b0:	c3 48 1e 8e	8e1e48c3 { 	r3 += asl(r30,#0x8)
  2eb0b4:	12 40 6e 70	706e4012   	r18 = r14
  2eb0b8:	d3 c1 9d 91	919dc1d3   	r19 = memw(r29+#0x38) } 
  2eb0bc:	02 40 7b 70	707b4002 { 	r2 = r27
  2eb0c0:	8e 40 0e b0	b00e408e   	r14 = add(r14,#0x4)
  2eb0c4:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2eb0c8:	a2 f8 df 5c	5cdff8a2   	if (p0.new) jump:t 0x2eb000 <hmx_v73_convbbb1x1_stride1+0x240> } 
  2eb0cc:	7a ff ff 59	59ffff7a { 	jump 0x2eafc0 <hmx_v73_convbbb1x1_stride1+0x200> } 
  2eb0d0:	00 40 00 7f	7f004000 { 	nop
  2eb0d4:	00 40 00 7f	7f004000   	nop
  2eb0d8:	00 40 00 7f	7f004000   	nop
  2eb0dc:	00 c0 00 7f	7f00c000   	nop } 
  2eb0e0:	00 40 00 7f	7f004000 { 	nop
  2eb0e4:	00 40 00 7f	7f004000   	nop
  2eb0e8:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2eb0ec:	11 c0 6c 70	706cc011   	r17 = r12 } 
  2eb0f0:	00 41 14 60	60144100 { 	loop0(0x2eb100,r20)
  2eb0f4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2eb0f8:	11 51 36 f3	f3365111   	r17 = sub(r17,r22)
  2eb0fc:	00 c0 00 7f	7f00c000   	nop } 
  2eb100:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2eb104:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2eb108:	19 40 7e 70	707e4019 { 	r25 = r30
  2eb10c:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2eb110:	e0 e9 02 92	9202e9e0   	weight.b = mxmem(r2,r9) } 
  2eb114:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2eb118:	f9 7f f9 bf	bff97ff9   	r25 = add(r25,#-0x1)
  2eb11c:	3f 40 85 9b	9b85403f   	r31 = memw(r5++#0x4)
  2eb120:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2eb124:	00 40 59 75	75594000 { 	p0 = cmp.gt(r25,#0x0)
  2eb128:	f8 68 df 5c	5cdf68f8   	if (p0.new) jump:nt 0x2eb114 <hmx_v73_convbbb1x1_stride1+0x354>
  2eb12c:	10 d7 ff a6	a6ffd710   	cvt.ub = acc(r31) } 
  2eb130:	45 82 1e 8e	8e1e8245 { 	r5 -= asl(r30,#0x2)
  2eb134:	43 48 1e 8e	8e1e4843   	r3 -= asl(r30,#0x8)
  2eb138:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop0
  2eb13c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2eb140:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2eb144:	da c0 b9 10	10b9c0da   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2eb0f0 <hmx_v73_convbbb1x1_stride1+0x330> } 
  2eb148:	c3 48 1e 8e	8e1e48c3 { 	r3 += asl(r30,#0x8)
  2eb14c:	12 40 6e 70	706e4012   	r18 = r14
  2eb150:	d3 c1 9d 91	919dc1d3   	r19 = memw(r29+#0x38) } 
  2eb154:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2eb158:	8e 40 0e b0	b00e408e   	r14 = add(r14,#0x4)
  2eb15c:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2eb160:	c6 f8 df 5c	5cdff8c6   	if (p0.new) jump:t 0x2eb0e0 <hmx_v73_convbbb1x1_stride1+0x320> } 
  2eb164:	2e ff ff 59	59ffff2e { 	jump 0x2eafc0 <hmx_v73_convbbb1x1_stride1+0x200> } 
