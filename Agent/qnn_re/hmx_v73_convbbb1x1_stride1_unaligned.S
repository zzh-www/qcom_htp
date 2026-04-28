
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

002eb780 <hmx_v73_convbbb1x1_stride1_unaligned>:
  2eb780:	1d 77 fd bf	bffd771d { 	r29 = add(r29,#-0x48)
  2eb784:	0c 7c 03 78	78037c0c   	r12 = #0x7e0
  2eb788:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2eb78c:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2eb790:	0c 47 0c f1	f10c470c { 	r12 = and(r12,r7)
  2eb794:	00 40 00 7f	7f004000   	nop
  2eb798:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2eb79c:	9c c1 84 91	9184c19c   	r28 = memw(r4+#0x30) } 
  2eb7a0:	8c 40 4c 8c	8c4c408c { 	r12 = ct0(r12)
  2eb7a4:	00 45 1c 85	851c4500   	p0 = tstbit(r28,#0x5)
  2eb7a8:	4f 40 81 91	9181404f   	r15 = memw(r1+#0x8)
  2eb7ac:	00 d0 dd a1	a1ddd000   	memd(r29+#0x0) = r17:16 } 
  2eb7b0:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2eb7b4:	6c 7f ec bf	bfec7f6c   	r12 = add(r12,#-0x5)
  2eb7b8:	16 0a 0d ea	ea0d0a16   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2eb7bc:	12 40 65 70	70654012 { 	r18 = r5
  2eb7c0:	65 40 81 91	91814065   	r5 = memw(r1+#0xc)
  2eb7c4:	8e c0 81 91	9181c08e   	r14 = memw(r1+#0x10) } 
  2eb7c8:	ed 4c 80 c6	c6804ced { 	r13 = lsl(#0x1,r12)
  2eb7cc:	f4 7f 03 78	78037ff4   	r20 = #0x7ff
  2eb7d0:	1f 0a 14 c1	c1140a1f   	r4 = memw(r1+#0x4); 	memd(r29+#0x18) = r23:22 } 
  2eb7d4:	e5 5f 0d e2	e20d5fe5 { 	r5 += add(r13,#-0x1)
  2eb7d8:	e4 7f e4 bf	bfe47fe4   	r4 = add(r4,#-0x1)
  2eb7dc:	01 40 81 91	91814001   	r1 = memw(r1+#0x0)
  2eb7e0:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2eb7e4:	45 4c 45 c6	c6454c45 { 	r5 = lsr(r5,r12)
  2eb7e8:	16 5c 00 78	78005c16   	r22 = #0xe0
  2eb7ec:	79 40 80 91	91804079   	r25 = memw(r0+#0xc)
  2eb7f0:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2eb7f4:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2eb7f8:	d9 6d 19 fb	fb196dd9   	if (!p2.new) r25 = add(r25,r13)
  2eb7fc:	09 01 08 02	02080109   	r16 = memw(r0+#0x8); 	r17 = memw(r0+#0x4) } 
  2eb800:	99 4c 85 cc	cc854c99 { 	r25 -= asl(r5,r12)
  2eb804:	cc 40 4c 76	764c40cc   	r12 = sub(#0x6,r12)
  2eb808:	bc 40 80 91	918040bc   	r28 = memw(r0+#0x14)
  2eb80c:	97 c0 80 91	9180c097   	r23 = memw(r0+#0x10) } 
  2eb810:	ec 4c 80 c6	c6804cec { 	r12 = lsl(#0x1,r12)
  2eb814:	f5 7f 03 78	78037ff5   	r21 = #0x7ff
  2eb818:	fc 43 1c b0	b01c43fc   	r28 = add(r28,#0x1f)
  2eb81c:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2eb820:	06 54 e6 d3	d3e65406 { 	r7:6 = and(r7:6,r21:20)
  2eb824:	17 4c 17 f3	f3174c17   	r23 = add(r23,r12)
  2eb828:	06 5e dd a1	a1dd5e06   	memd(r29+#0x30) = r31:30
  2eb82c:	0e c1 9d a1	a19dc10e   	memw(r29+#0x38) = r1 } 
  2eb830:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2eb834:	1c 45 1c 8c	8c1c451c   	r28 = asr(r28,#0x5)
  2eb838:	9a 40 00 b0	b000409a   	r26 = add(r0,#0x4)
  2eb83c:	0f d7 9d a1	a19dd70f   	memw(r29+#0x3c) = r23 } 
  2eb840:	18 40 6c 70	706c4018 { 	r24 = r12
  2eb844:	a0 40 00 5c	5c0040a0   	if (p0) jump:nt 0x2eb980 <hmx_v73_convbbb1x1_stride1_unaligned+0x200>
  2eb848:	29 40 09 74	74094029   	if (p0) r9 = add(r9,#0x1)
  2eb84c:	33 c0 92 9b	9b92c033   	r19 = memw(r18++#0x4) } 
  2eb850:	fc 7f fc bf	bffc7ffc { 	r28 = add(r28,#-0x1)
  2eb854:	1e 40 67 70	7067401e   	r30 = r7
  2eb858:	ff 7f 01 78	78017fff   	r31 = #0x3ff
  2eb85c:	00 c0 00 7f	7f00c000   	nop } 
  2eb860:	00 42 25 60	60254200 { 	loop1(0x2eb880,r5)
  2eb864:	17 40 00 78	78004017   	r23 = #0x0
  2eb868:	16 5c 00 78	78005c16   	r22 = #0xe0
  2eb86c:	00 c0 00 7f	7f00c000   	nop } 
  2eb870:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2eb874:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2eb878:	01 58 4e f2	f24e5801   	p1 = cmp.gt(r14,r24)
  2eb87c:	08 c0 dd a1	a1ddc008   	memd(r29+#0x40) = r1:0 } 
  2eb880:	00 42 04 60	60044200 { 	loop0(0x2eb8a0,r4)
  2eb884:	87 43 87 76	76874387   	r7 = or(r7,#0x1c)
  2eb888:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2eb88c:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2eb890:	37 57 36 fb	fb365737 { 	if (p1) r23 = sub(r23,r22)
  2eb894:	08 40 c2 bf	bfc24008   	r8 = add(r2,#-0x400)
  2eb898:	07 40 9e 74	749e4007   	if (!p0) r7 = add(r30,#0x0)
  2eb89c:	1a c0 20 5c	5c20c01a   	if (!p0) jump:nt 0x2eb8c4 <hmx_v73_convbbb1x1_stride1_unaligned+0x144> } 
  2eb8a0:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2eb8a4:	08 40 48 b0	b0484008   	r8 = add(r8,#0x400)
  2eb8a8:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2eb8ac:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2eb8b0:	06 94 e6 d3	d3e69406 { 	r7:6 = and(r7:6,r21:20)
  2eb8b4:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2eb8b8:	f1 47 06 92	920647f1   	activation.ub = mxmem(r6,r7):single:cm
  2eb8bc:	e6 ff 08 92	9208ffe6   	weight.b = mxmem(r8,r31):single }  :endloop0
  2eb8c0:	07 c0 7e 70	707ec007 { 	r7 = r30 } 
  2eb8c4:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2eb8c8:	08 40 48 b0	b0484008   	r8 = add(r8,#0x400)
  2eb8cc:	0a 54 0a f1	f10a540a   	r10 = and(r10,r20)
  2eb8d0:	0d c0 73 70	7073c00d   	r13 = r19 } 
  2eb8d4:	28 40 09 e2	e2094028 { 	r8 += add(r9,#0x1)
  2eb8d8:	06 54 e6 d3	d3e65406   	r7:6 = and(r7:6,r21:20)
  2eb8dc:	f1 47 06 92	920647f1   	activation.ub = mxmem(r6,r7):single:cm
  2eb8e0:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single } 
  2eb8e4:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2eb8e8:	ed 7f ed bf	bfed7fed   	r13 = add(r13,#-0x1)
  2eb8ec:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2eb8f0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2eb8f4:	03 40 4d 75	754d4003 { 	p3 = cmp.gt(r13,#0x0)
  2eb8f8:	f8 6b df 5c	5cdf6bf8   	if (p3.new) jump:nt 0x2eb8e4 <hmx_v73_convbbb1x1_stride1_unaligned+0x164>
  2eb8fc:	10 d7 f6 a6	a6f6d710   	cvt.ub = acc(r22) } 
  2eb900:	52 42 13 8e	8e134252 { 	r18 -= asl(r19,#0x2)
  2eb904:	43 48 13 8e	8e134843   	r3 -= asl(r19,#0x8)
  2eb908:	16 c0 80 91	9180c016   	r22 = memw(r0+#0x0) } 
  2eb90c:	0a 56 2a f1	f12a560a { 	r10 = or(r10,r22)
  2eb910:	06 c2 20 5c	5c20c206   	if (!p2) jump:nt 0x2eb918 <hmx_v73_convbbb1x1_stride1_unaligned+0x198> } 
  2eb914:	19 cb ea a6	a6eacb19 { 	mxmem(r10,r11):cm = cvt } 
  2eb918:	02 42 e2 6b	6be24202 { 	p2 = or(p2,!p2)
  2eb91c:	40 91 00 fb	fb009140   	if (p2) r0 = add(r0,r17)
  2eb920:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  2eb924:	03 40 59 75	75594003 { 	p3 = cmp.gt(r25,#0x0)
  2eb928:	10 5b 20 5c	5c205b10   	if (!p3.new) jump:t 0x2eb944 <hmx_v73_convbbb1x1_stride1_unaligned+0x1c4>
  2eb92c:	16 e0 60 7e	7e60e016   	if (p3.new) r22 = #0x0 } 
  2eb930:	0a 54 0a f1	f10a540a { 	r10 = and(r10,r20)
  2eb934:	16 40 80 91	91804016   	r22 = memw(r0+#0x0)
  2eb938:	10 d7 f6 a6	a6f6d710   	cvt.ub = acc(r22) } 
  2eb93c:	0a d6 2a f1	f12ad60a { 	r10 = or(r10,r22) } 
  2eb940:	19 cb ea a6	a6eacb19 { 	mxmem(r10,r11):cm = cvt } 
  2eb944:	18 4c 18 f3	f3184c18 { 	r24 = add(r24,r12)
  2eb948:	40 1e ff 3c	3cff1e40   	r23 = memw(r29+#0x3c); 	r1:0 = memd(r29+#0x40) } 
  2eb94c:	01 4f 01 f3	f3014f01 { 	r1 = add(r1,r15)
  2eb950:	00 50 00 f3	f3005000   	r0 = add(r0,r16)
  2eb954:	00 58 57 f2	f2575800   	p0 = cmp.gt(r23,r24)
  2eb958:	8a e8 df 5c	5cdfe88a   	if (p0.new) jump:nt 0x2eb860 <hmx_v73_convbbb1x1_stride1_unaligned+0xe0> } 
  2eb95c:	00 40 7a 70	707a4000 { 	r0 = r26
  2eb960:	fc 7f fc bf	bffc7ffc   	r28 = add(r28,#-0x1)
  2eb964:	02 40 68 70	70684002   	r2 = r8
  2eb968:	c1 c1 9d 91	919dc1c1   	r1 = memw(r29+#0x38) } 
  2eb96c:	7a 7f 7c 61	617c7f7a { 	if (r28>=#0) jump:t 0x2eb860
  2eb970:	c3 48 13 8e	8e1348c3   	r3 += asl(r19,#0x8)
  2eb974:	9a 40 1a b0	b01a409a   	r26 = add(r26,#0x4)
  2eb978:	18 c0 6c 70	706cc018   	r24 = r12 } 
  2eb97c:	9a c0 00 58	5800c09a { 	jump 0x2ebab0 <hmx_v73_convbbb1x1_stride1_unaligned+0x330> } 
  2eb980:	29 41 09 8c	8c094129 { 	r9 = lsr(r9,#0x1)
  2eb984:	00 40 00 7f	7f004000   	nop
  2eb988:	00 40 00 7f	7f004000   	nop
  2eb98c:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2eb990:	1b 5c 09 ed	ed095c1b { 	r27 = mpyi(r9,r28)
  2eb994:	e9 7f e9 bf	bfe97fe9   	r9 = add(r9,#-0x1)
  2eb998:	fc 7f fc bf	bffc7ffc   	r28 = add(r28,#-0x1)
  2eb99c:	00 c0 00 7f	7f00c000   	nop } 
  2eb9a0:	00 42 25 60	60254200 { 	loop1(0x2eb9c0,r5)
  2eb9a4:	17 40 00 78	78004017   	r23 = #0x0
  2eb9a8:	16 5c 00 78	78005c16   	r22 = #0xe0
  2eb9ac:	00 c0 00 7f	7f00c000   	nop } 
  2eb9b0:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2eb9b4:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2eb9b8:	01 58 4e f2	f24e5801   	p1 = cmp.gt(r14,r24)
  2eb9bc:	08 c0 dd a1	a1ddc008   	memd(r29+#0x40) = r1:0 } 
  2eb9c0:	00 42 04 60	60044200 { 	loop0(0x2eb9e0,r4)
  2eb9c4:	00 40 00 7f	7f004000   	nop
  2eb9c8:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2eb9cc:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2eb9d0:	06 54 e6 d3	d3e65406 { 	r7:6 = and(r7:6,r21:20)
  2eb9d4:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2eb9d8:	08 42 3b f3	f33b4208   	r8 = sub(r2,r27)
  2eb9dc:	18 c0 20 5c	5c20c018   	if (!p0) jump:nt 0x2eba00 <hmx_v73_convbbb1x1_stride1_unaligned+0x280> } 
  2eb9e0:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2eb9e4:	08 5b 08 f3	f3085b08   	r8 = add(r8,r27)
  2eb9e8:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2eb9ec:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2eb9f0:	06 94 e6 d3	d3e69406 { 	r7:6 = and(r7:6,r21:20)
  2eb9f4:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2eb9f8:	f1 47 06 92	920647f1   	activation.ub = mxmem(r6,r7):single:cm
  2eb9fc:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single }  :endloop0
  2eba00:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2eba04:	08 5b 08 f3	f3085b08   	r8 = add(r8,r27)
  2eba08:	0a d4 0a f1	f10ad40a   	r10 = and(r10,r20) } 
  2eba0c:	0d 40 73 70	7073400d { 	r13 = r19
  2eba10:	f1 47 06 92	920647f1   	activation.ub = mxmem(r6,r7):single:cm
  2eba14:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single } 
  2eba18:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2eba1c:	ed 7f ed bf	bfed7fed   	r13 = add(r13,#-0x1)
  2eba20:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2eba24:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2eba28:	03 40 4d 75	754d4003 { 	p3 = cmp.gt(r13,#0x0)
  2eba2c:	f8 6b df 5c	5cdf6bf8   	if (p3.new) jump:nt 0x2eba18 <hmx_v73_convbbb1x1_stride1_unaligned+0x298>
  2eba30:	10 d7 f6 a6	a6f6d710   	cvt.ub = acc(r22) } 
  2eba34:	52 42 13 8e	8e134252 { 	r18 -= asl(r19,#0x2)
  2eba38:	43 48 13 8e	8e134843   	r3 -= asl(r19,#0x8)
  2eba3c:	16 c0 80 91	9180c016   	r22 = memw(r0+#0x0) } 
  2eba40:	0a 56 2a f1	f12a560a { 	r10 = or(r10,r22)
  2eba44:	06 c2 20 5c	5c20c206   	if (!p2) jump:nt 0x2eba4c <hmx_v73_convbbb1x1_stride1_unaligned+0x2cc> } 
  2eba48:	19 cb ea a6	a6eacb19 { 	mxmem(r10,r11):cm = cvt } 
  2eba4c:	02 42 e2 6b	6be24202 { 	p2 = or(p2,!p2)
  2eba50:	40 91 00 fb	fb009140   	if (p2) r0 = add(r0,r17)
  2eba54:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  2eba58:	03 40 59 75	75594003 { 	p3 = cmp.gt(r25,#0x0)
  2eba5c:	10 5b 20 5c	5c205b10   	if (!p3.new) jump:t 0x2eba78 <hmx_v73_convbbb1x1_stride1_unaligned+0x2f8>
  2eba60:	16 e0 60 7e	7e60e016   	if (p3.new) r22 = #0x0 } 
  2eba64:	0a 54 0a f1	f10a540a { 	r10 = and(r10,r20)
  2eba68:	16 40 80 91	91804016   	r22 = memw(r0+#0x0)
  2eba6c:	10 d7 f6 a6	a6f6d710   	cvt.ub = acc(r22) } 
  2eba70:	0a d6 2a f1	f12ad60a { 	r10 = or(r10,r22) } 
  2eba74:	19 cb ea a6	a6eacb19 { 	mxmem(r10,r11):cm = cvt } 
  2eba78:	18 4c 18 f3	f3184c18 { 	r24 = add(r24,r12)
  2eba7c:	40 1e ff 3c	3cff1e40   	r23 = memw(r29+#0x3c); 	r1:0 = memd(r29+#0x40) } 
  2eba80:	01 4f 01 f3	f3014f01 { 	r1 = add(r1,r15)
  2eba84:	00 50 00 f3	f3005000   	r0 = add(r0,r16)
  2eba88:	00 58 57 f2	f2575800   	p0 = cmp.gt(r23,r24)
  2eba8c:	90 e8 df 5c	5cdfe890   	if (p0.new) jump:nt 0x2eb9a0 <hmx_v73_convbbb1x1_stride1_unaligned+0x220> } 
  2eba90:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2eba94:	00 40 7a 70	707a4000   	r0 = r26
  2eba98:	fc 7f fc bf	bffc7ffc   	r28 = add(r28,#-0x1)
  2eba9c:	c1 c1 9d 91	919dc1c1   	r1 = memw(r29+#0x38) } 
  2ebaa0:	80 7f 7c 61	617c7f80 { 	if (r28>=#0) jump:t 0x2eb9a0
  2ebaa4:	c3 48 13 8e	8e1348c3   	r3 += asl(r19,#0x8)
  2ebaa8:	9a 40 1a b0	b01a409a   	r26 = add(r26,#0x4)
  2ebaac:	18 c0 6c 70	706cc018   	r24 = r12 } 
  2ebab0:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2ebab4:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2ebab8:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2ebabc:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2ebac0:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2ebac4:	1d 49 1d b0	b01d491d { 	r29 = add(r29,#0x48)
  2ebac8:	00 40 9f 52	529f4000   	jumpr r31
  2ebacc:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...
