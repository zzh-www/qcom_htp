
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

002ebe40 <hmx_v73_convbbb1x1deep_stride1>:
  2ebe40:	1d 7b fd bf	bffd7b1d { 	r29 = add(r29,#-0x28)
  2ebe44:	00 40 00 7f	7f004000   	nop
  2ebe48:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2ebe4c:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2ebe50:	00 40 00 7f	7f004000 { 	nop
  2ebe54:	00 40 00 7f	7f004000   	nop
  2ebe58:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2ebe5c:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2ebe60:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2ebe64:	4f 42 0f 8c	8c0f424f   	r15 = asl(r15,#0x2)
  2ebe68:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2ebe6c:	36 40 09 b0	b0094036 { 	r22 = add(r9,#0x1)
  2ebe70:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2ebe74:	36 41 16 8c	8c164136 { 	r22 = lsr(r22,#0x1)
  2ebe78:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2ebe7c:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2ebe80:	20 40 44 75	75444020   	p0 = cmp.gt(r4,#0x1)
  2ebe84:	38 40 85 9b	9b854038   	r24 = memw(r5++#0x4)
  2ebe88:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2ebe8c:	17 40 61 70	70614017 { 	r23 = r1
  2ebe90:	22 40 18 75	75184022   	p2 = cmp.eq(r24,#0x1)
  2ebe94:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2ebe98:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2ebe9c:	3c 41 04 8c	8c04413c   	r28 = lsr(r4,#0x1)
  2ebea0:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2ebea4:	ad c0 80 91	9180c0ad   	r13 = memw(r0+#0x14) } 
  2ebea8:	ed 43 0d b0	b00d43ed { 	r13 = add(r13,#0x1f)
  2ebeac:	7c 20 c9 27	27c9207c   	r17 = add(r17,#-0x4); 	r20 = add(r20,#0x7) } 
  2ebeb0:	06 40 31 62	62314006 { 	m0 = r17
  2ebeb4:	2d 45 0d 8c	8c0d452d   	r13 = lsr(r13,#0x5)
  2ebeb8:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2ebebc:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2ebec0:	34 43 14 8c	8c144334 { 	r20 = lsr(r20,#0x3)
  2ebec4:	12 40 01 f5	f5014012   	r19:18 = combine(r1,r0)
  2ebec8:	f0 40 20 5c	5c2040f0   	if (!p0) jump:nt 0x2ec0a0 <hmx_v73_convbbb1x1deep_stride1+0x260>
  2ebecc:	19 d0 85 41	4185d019   	if (p2) r25 = memw(r5+#0x0) } 
  2ebed0:	00 45 1c 60	601c4500 { 	loop0(0x2ebf20,r28)
  2ebed4:	16 4d 16 ed	ed164d16   	r22 = mpyi(r22,r13)
  2ebed8:	68 c2 20 5c	5c20c268   	if (!p2) jump:nt 0x2ebfa0 <hmx_v73_convbbb1x1deep_stride1+0x160> } 
  2ebedc:	00 c0 00 7f	7f00c000 { 	nop } 
  2ebee0:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ebee4:	00 40 00 7f	7f004000   	nop
  2ebee8:	00 40 00 7f	7f004000   	nop
  2ebeec:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ebef0:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2ebef4:	11 40 6c 70	706c4011   	r17 = r12
  2ebef8:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2ebefc:	00 c0 00 7f	7f00c000   	nop } 
  2ebf00:	e3 5f 03 b0	b0035fe3 { 	r3 = add(r3,#0xff)
  2ebf04:	00 40 00 7f	7f004000   	nop
  2ebf08:	00 40 00 7f	7f004000   	nop
  2ebf0c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ebf10:	00 41 34 60	60344100 { 	loop1(0x2ebf20,r20)
  2ebf14:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2ebf18:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2ebf1c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2ebf20:	46 40 81 9b	9b814046 { 	r6 = memw(r1++#0x8)
  2ebf24:	35 c0 81 91	9181c035   	r21 = memw(r1+#0x4) } 
  2ebf28:	08 56 08 f3	f3085608 { 	r8 = add(r8,r22)
  2ebf2c:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2ebf30:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2ebf34:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2ebf38:	e1 47 15 92	921547e1   	activation.ub = mxmem(r21,r7):deep:cm
  2ebf3c:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2ebf40:	00 5e 1c 60	601c5e00 { 	loop0(0x2ebf20,r28)
  2ebf44:	15 40 78 70	70784015   	r21 = r24
  2ebf48:	0e 41 20 5c	5c20410e   	if (!p1) jump:nt 0x2ebf5c <hmx_v73_convbbb1x1deep_stride1+0x11c>
  2ebf4c:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2ebf50:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2ebf54:	e1 47 06 92	920647e1 { 	activation.ub = mxmem(r6,r7):deep:cm
  2ebf58:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2ebf5c:	19 4c d9 8c	8cd94c19 { 	r25 = setbit(r25,#0xc)
  2ebf60:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ebf64:	0a 40 80 9d	9d80400a { 	r10 = memw(r0++m0)
  2ebf68:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt } 
  2ebf6c:	39 4c d9 8c	8cd94c39 { 	r25 = clrbit(r25,#0xc)
  2ebf70:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ebf74:	08 40 62 70	70624008 { 	r8 = r2
  2ebf78:	00 80 00 7f	7f008000   	nop
  2ebf7c:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop1
  2ebf80:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ebf84:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ebf88:	c8 c0 b9 10	10b9c0c8   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ebf10 <hmx_v73_convbbb1x1deep_stride1+0xd0> } 
  2ebf8c:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2ebf90:	12 4e 17 f5	f5174e12   	r19:18 = combine(r23,r14)
  2ebf94:	20 40 4d 75	754d4020   	p0 = cmp.gt(r13,#0x1)
  2ebf98:	aa f8 df 5c	5cdff8aa   	if (p0.new) jump:t 0x2ebee0 <hmx_v73_convbbb1x1deep_stride1+0xa0> } 
  2ebf9c:	32 c1 00 58	5800c132 { 	jump 0x2ec200 <hmx_v73_convbbb1x1deep_stride1+0x3c0> } 
  2ebfa0:	00 c4 1c 60	601cc400 { 	loop0(0x2ebfe0,r28) } 
  2ebfa4:	00 40 00 7f	7f004000 { 	nop
  2ebfa8:	00 40 00 7f	7f004000   	nop
  2ebfac:	00 c0 00 7f	7f00c000   	nop } 
  2ebfb0:	00 40 00 7f	7f004000 { 	nop
  2ebfb4:	00 40 00 7f	7f004000   	nop
  2ebfb8:	00 40 00 7f	7f004000   	nop
  2ebfbc:	00 c0 00 7f	7f00c000   	nop } 
  2ebfc0:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2ebfc4:	11 40 6c 70	706c4011   	r17 = r12
  2ebfc8:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2ebfcc:	00 c0 00 7f	7f00c000   	nop } 
  2ebfd0:	00 41 34 60	60344100 { 	loop1(0x2ebfe0,r20)
  2ebfd4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2ebfd8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2ebfdc:	08 c0 62 70	7062c008   	r8 = r2 } 
  2ebfe0:	46 40 81 9b	9b814046 { 	r6 = memw(r1++#0x8)
  2ebfe4:	35 c0 81 91	9181c035   	r21 = memw(r1+#0x4) } 
  2ebfe8:	08 56 08 f3	f3085608 { 	r8 = add(r8,r22)
  2ebfec:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2ebff0:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2ebff4:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2ebff8:	e1 47 15 92	921547e1   	activation.ub = mxmem(r21,r7):deep:cm
  2ebffc:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2ec000:	00 5e 1c 60	601c5e00 { 	loop0(0x2ebfe0,r28)
  2ec004:	15 40 78 70	70784015   	r21 = r24
  2ec008:	0e 41 20 5c	5c20410e   	if (!p1) jump:nt 0x2ec01c <hmx_v73_convbbb1x1deep_stride1+0x1dc>
  2ec00c:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2ec010:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2ec014:	e1 47 06 92	920647e1 { 	activation.ub = mxmem(r6,r7):deep:cm
  2ec018:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2ec01c:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2ec020:	f5 7f f5 bf	bff57ff5   	r21 = add(r21,#-0x1)
  2ec024:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2ec028:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ec02c:	f8 40 bd 10	10bd40f8 { 	p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 0x2ec01c <hmx_v73_convbbb1x1deep_stride1+0x1dc>
  2ec030:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ec034:	45 42 18 8e	8e184245 { 	r5 -= asl(r24,#0x2)
  2ec038:	15 40 78 70	70784015   	r21 = r24
  2ec03c:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2ec040:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt } 
  2ec044:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2ec048:	f5 7f f5 bf	bff57ff5   	r21 = add(r21,#-0x1)
  2ec04c:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2ec050:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ec054:	f8 40 bd 10	10bd40f8 { 	p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 0x2ec044 <hmx_v73_convbbb1x1deep_stride1+0x204>
  2ec058:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ec05c:	45 42 18 8e	8e184245 { 	r5 -= asl(r24,#0x2)
  2ec060:	43 89 18 8e	8e188943   	r3 -= asl(r24,#0x9)
  2ec064:	08 40 62 70	70624008   	r8 = r2
  2ec068:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop1
  2ec06c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ec070:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ec074:	b2 c0 b9 10	10b9c0b2   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ebfd0 <hmx_v73_convbbb1x1deep_stride1+0x190> } 
  2ec078:	c3 49 18 8e	8e1849c3 { 	r3 += asl(r24,#0x9)
  2ec07c:	22 40 09 e2	e2094022   	r2 += add(r9,#0x1)
  2ec080:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2ec084:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2ec088:	9e f8 df 5c	5cdff89e   	if (p0.new) jump:t 0x2ebfc0 <hmx_v73_convbbb1x1deep_stride1+0x180> } 
  2ec08c:	ba c0 00 58	5800c0ba { 	jump 0x2ec200 <hmx_v73_convbbb1x1deep_stride1+0x3c0> } 
  2ec090:	00 40 00 7f	7f004000 { 	nop
  2ec094:	00 40 00 7f	7f004000   	nop
  2ec098:	00 40 00 7f	7f004000   	nop
  2ec09c:	00 c0 00 7f	7f00c000   	nop } 
  2ec0a0:	60 42 20 5c	5c204260 { 	if (!p2) jump:nt 0x2ec160 <hmx_v73_convbbb1x1deep_stride1+0x320>
  2ec0a4:	00 40 00 7f	7f004000   	nop
  2ec0a8:	00 40 00 7f	7f004000   	nop
  2ec0ac:	00 c0 00 7f	7f00c000   	nop } 
  2ec0b0:	00 40 00 7f	7f004000 { 	nop
  2ec0b4:	00 40 00 7f	7f004000   	nop
  2ec0b8:	00 40 00 7f	7f004000   	nop
  2ec0bc:	00 c0 00 7f	7f00c000   	nop } 
  2ec0c0:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ec0c4:	cd 7f ed bf	bfed7fcd   	r13 = add(r13,#-0x2)
  2ec0c8:	11 40 6c 70	706c4011   	r17 = r12
  2ec0cc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ec0d0:	e3 5f 03 b0	b0035fe3 { 	r3 = add(r3,#0xff)
  2ec0d4:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2ec0d8:	00 40 00 7f	7f004000   	nop
  2ec0dc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ec0e0:	00 42 14 60	60144200 { 	loop0(0x2ec100,r20)
  2ec0e4:	81 40 13 b0	b0134081   	r1 = add(r19,#0x4)
  2ec0e8:	b6 00 89 47	478900b6   	r17 = add(r17,#-0x8); 	r6 = memw(r19+#0x0) } 
  2ec0ec:	80 40 12 b0	b0124080 { 	r0 = add(r18,#0x4)
  2ec0f0:	15 40 74 70	70744015   	r21 = r20
  2ec0f4:	00 40 00 7f	7f004000   	nop
  2ec0f8:	0a c0 92 91	9192c00a   	r10 = memw(r18+#0x0) } 
  2ec0fc:	00 c0 00 7f	7f00c000 { 	nop } 
  2ec100:	f5 7f f5 bf	bff57ff5 { 	r21 = add(r21,#-0x1)
  2ec104:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2ec108:	e8 e9 02 92	9202e9e8   	weight.b = mxmem(r2,r9):deep } 
  2ec10c:	00 40 55 75	75554000 { 	p0 = cmp.gt(r21,#0x0)
  2ec110:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ec114:	19 4c d9 8c	8cd94c19 { 	r25 = setbit(r25,#0xc)
  2ec118:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2ec11c:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt } 
  2ec120:	26 60 81 9b	9b816026 { 	if (p0) r6 = memw(r1++#0x4)
  2ec124:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ec128:	39 8c d9 8c	8cd98c39 { 	r25 = clrbit(r25,#0xc)
  2ec12c:	2a 60 80 9b	9b80602a   	if (p0) r10 = memw(r0++#0x4)
  2ec130:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop0
  2ec134:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ec138:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ec13c:	d6 c0 b9 10	10b9c0d6   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ec0e0 <hmx_v73_convbbb1x1deep_stride1+0x2a0> } 
  2ec140:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2ec144:	12 4e 17 f5	f5174e12   	r19:18 = combine(r23,r14)
  2ec148:	20 40 4d 75	754d4020   	p0 = cmp.gt(r13,#0x1)
  2ec14c:	c0 f8 df 5c	5cdff8c0   	if (p0.new) jump:t 0x2ec0c0 <hmx_v73_convbbb1x1deep_stride1+0x280> } 
  2ec150:	58 c0 00 58	5800c058 { 	jump 0x2ec200 <hmx_v73_convbbb1x1deep_stride1+0x3c0> } 
  2ec154:	00 40 00 7f	7f004000 { 	nop
  2ec158:	00 40 00 7f	7f004000   	nop
  2ec15c:	00 c0 00 7f	7f00c000   	nop } 
  2ec160:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2ec164:	11 40 6c 70	706c4011   	r17 = r12
  2ec168:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2ec16c:	00 c0 00 7f	7f00c000   	nop } 
  2ec170:	00 41 14 60	60144100 { 	loop0(0x2ec180,r20)
  2ec174:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2ec178:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2ec17c:	00 c0 00 7f	7f00c000   	nop } 
  2ec180:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2ec184:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2ec188:	15 40 78 70	70784015 { 	r21 = r24
  2ec18c:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2ec190:	e8 e9 02 92	9202e9e8   	weight.b = mxmem(r2,r9):deep } 
  2ec194:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2ec198:	f5 7f f5 bf	bff57ff5   	r21 = add(r21,#-0x1)
  2ec19c:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2ec1a0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ec1a4:	f8 40 bd 10	10bd40f8 { 	p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 0x2ec194 <hmx_v73_convbbb1x1deep_stride1+0x354>
  2ec1a8:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ec1ac:	45 42 18 8e	8e184245 { 	r5 -= asl(r24,#0x2)
  2ec1b0:	15 40 78 70	70784015   	r21 = r24
  2ec1b4:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2ec1b8:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt } 
  2ec1bc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2ec1c0:	f5 7f f5 bf	bff57ff5   	r21 = add(r21,#-0x1)
  2ec1c4:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2ec1c8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ec1cc:	f8 40 bd 10	10bd40f8 { 	p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 0x2ec1bc <hmx_v73_convbbb1x1deep_stride1+0x37c>
  2ec1d0:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ec1d4:	45 82 18 8e	8e188245 { 	r5 -= asl(r24,#0x2)
  2ec1d8:	43 49 18 8e	8e184943   	r3 -= asl(r24,#0x9)
  2ec1dc:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop0
  2ec1e0:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ec1e4:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ec1e8:	c8 c0 b9 10	10b9c0c8   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ec170 <hmx_v73_convbbb1x1deep_stride1+0x330> } 
  2ec1ec:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2ec1f0:	c3 49 18 8e	8e1849c3   	r3 += asl(r24,#0x9)
  2ec1f4:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2ec1f8:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2ec1fc:	b4 f8 df 5c	5cdff8b4   	if (p0.new) jump:t 0x2ec160 <hmx_v73_convbbb1x1deep_stride1+0x320> } 
  2ec200:	00 40 4d 75	754d4000 { 	p0 = cmp.gt(r13,#0x0)
  2ec204:	4a 58 20 5c	5c20584a   	if (!p0.new) jump:t 0x2ec294 <hmx_v73_convbbb1x1deep_stride1+0x454>
  2ec208:	00 40 00 7f	7f004000   	nop
  2ec20c:	00 c0 00 7f	7f00c000   	nop } 
  2ec210:	00 43 04 60	60044300 { 	loop0(0x2ec240,r4)
  2ec214:	09 41 09 8c	8c094109   	r9 = asr(r9,#0x1)
  2ec218:	11 40 6c 70	706c4011   	r17 = r12
  2ec21c:	00 c0 00 7f	7f00c000   	nop } 
  2ec220:	00 42 34 60	60344200 { 	loop1(0x2ec240,r20)
  2ec224:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2ec228:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2ec22c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2ec230:	00 40 00 7f	7f004000 { 	nop
  2ec234:	00 40 00 7f	7f004000   	nop
  2ec238:	00 40 00 7f	7f004000   	nop
  2ec23c:	00 c0 00 7f	7f00c000   	nop } 
  2ec240:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2ec244:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2ec248:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2ec24c:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2ec250:	00 5f 04 60	60045f00 { 	loop0(0x2ec240,r4)
  2ec254:	08 40 62 70	70624008   	r8 = r2
  2ec258:	15 40 78 70	70784015   	r21 = r24
  2ec25c:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2ec260:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2ec264:	f5 7f f5 bf	bff57ff5   	r21 = add(r21,#-0x1)
  2ec268:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2ec26c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ec270:	f8 40 bd 10	10bd40f8 { 	p0 = cmp.gt(r21,#0x0); if (p0.new) jump:nt 0x2ec260 <hmx_v73_convbbb1x1deep_stride1+0x420>
  2ec274:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ec278:	45 42 18 8e	8e184245 { 	r5 -= asl(r24,#0x2)
  2ec27c:	43 88 18 8e	8e188843   	r3 -= asl(r24,#0x8)
  2ec280:	80 40 00 b0	b0004080   	r0 = add(r0,#0x4)
  2ec284:	19 cb ea a6	a6eacb19   	mxmem(r10,r11):cm = cvt }  :endloop1
  2ec288:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ec28c:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ec290:	cc c0 b9 10	10b9c0cc   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2ec220 <hmx_v73_convbbb1x1deep_stride1+0x3e0> } 
  2ec294:	11 c0 e0 a6	a6e0c011 { 	mxclracc } 
  2ec298:	98 c0 dd 91	91ddc098 { 	r25:24 = memd(r29+#0x20) } 
  2ec29c:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2ec2a0:	1d 45 1d b0	b01d451d { 	r29 = add(r29,#0x28)
  2ec2a4:	00 40 9f 52	529f4000   	jumpr r31
  2ec2a8:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
