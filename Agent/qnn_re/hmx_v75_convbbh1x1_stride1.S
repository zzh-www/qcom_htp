
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

002ef680 <hmx_v75_convbbh1x1_stride1>:
  2ef680:	1d 7a fd bf	bffd7a1d { 	r29 = add(r29,#-0x30)
  2ef684:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2ef688:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2ef68c:	c9 40 84 91	918440c9 { 	r9 = memw(r4+#0x18)
  2ef690:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2ef694:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2ef698:	4f 42 0f 8c	8c0f424f   	r15 = asl(r15,#0x2)
  2ef69c:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2ef6a0:	36 40 09 b0	b0094036 { 	r22 = add(r9,#0x1)
  2ef6a4:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2ef6a8:	36 41 16 8c	8c164136 { 	r22 = lsr(r22,#0x1)
  2ef6ac:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2ef6b0:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2ef6b4:	20 40 44 75	75444020   	p0 = cmp.gt(r4,#0x1)
  2ef6b8:	04 58 dd a1	a1dd5804   	memd(r29+#0x20) = r25:24
  2ef6bc:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2ef6c0:	3c 41 04 8c	8c04413c { 	r28 = lsr(r4,#0x1)
  2ef6c4:	17 40 61 70	70614017   	r23 = r1
  2ef6c8:	38 40 85 9b	9b854038   	r24 = memw(r5++#0x4)
  2ef6cc:	8c c0 80 91	9180c08c   	r12 = memw(r0+#0x10) } 
  2ef6d0:	22 40 18 75	75184022 { 	p2 = cmp.eq(r24,#0x1)
  2ef6d4:	fa 40 0c b0	b00c40fa   	r26 = add(r12,#0x7)
  2ef6d8:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2ef6dc:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2ef6e0:	1a 43 1a 8c	8c1a431a   	r26 = asr(r26,#0x3)
  2ef6e4:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2ef6e8:	ad c0 80 91	9180c0ad   	r13 = memw(r0+#0x14) } 
  2ef6ec:	1a 50 1a ed	ed1a501a { 	r26 = mpyi(r26,r16)
  2ef6f0:	ed c3 0d b0	b00dc3ed   	r13 = add(r13,#0x1f) } 
  2ef6f4:	06 40 31 62	62314006 { 	m0 = r17
  2ef6f8:	2d 45 0d 8c	8c0d452d   	r13 = lsr(r13,#0x5)
  2ef6fc:	00 00 7c 40	407c0000   	r20 = add(r20,#0x7); 	r0 = memw(r0+#0x0) } 
  2ef700:	34 43 14 8c	8c144334 { 	r20 = lsr(r20,#0x3)
  2ef704:	12 40 01 f5	f5014012   	r19:18 = combine(r1,r0)
  2ef708:	19 d0 85 41	4185d019   	if (p2) r25 = memw(r5+#0x0) } 
  2ef70c:	1b 4c d9 8c	8cd94c1b { 	r27 = setbit(r25,#0xc)
  2ef710:	79 40 99 76	76994079   	r25 = or(r25,#0x3)
  2ef714:	8a c0 20 5c	5c20c08a   	if (!p0) jump:nt 0x2ef820 <hmx_v75_convbbh1x1_stride1+0x1a0> } 
  2ef718:	10 44 1c 60	601c4410 { 	loop0(0x2ef760,r28)
  2ef71c:	16 cd 16 ed	ed16cd16   	r22 = mpyi(r22,r13) } 
  2ef720:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ef724:	cd 7f ed bf	bfed7fcd   	r13 = add(r13,#-0x2)
  2ef728:	18 40 6c 70	706c4018   	r24 = r12
  2ef72c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef730:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ef734:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2ef738:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef73c:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ef740:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef744:	a3 5f 03 b0	b0035fa3 { 	r3 = add(r3,#0xfd)
  2ef748:	00 40 00 7f	7f004000   	nop
  2ef74c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef750:	00 41 34 60	60344100 { 	loop1(0x2ef760,r20)
  2ef754:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2ef758:	18 7f f8 bf	bff87f18   	r24 = add(r24,#-0x8)
  2ef75c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2ef760:	46 40 81 9b	9b814046 { 	r6 = memw(r1++#0x8)
  2ef764:	35 c0 81 91	9181c035   	r21 = memw(r1+#0x4) } 
  2ef768:	08 56 08 f3	f3085608 { 	r8 = add(r8,r22)
  2ef76c:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2ef770:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2ef774:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2ef778:	e1 47 15 92	921547e1   	activation.ub = mxmem(r21,r7):deep:cm
  2ef77c:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2ef780:	00 5e 1c 60	601c5e00 { 	loop0(0x2ef760,r28)
  2ef784:	00 5a 00 f3	f3005a00   	r0 = add(r0,r26)
  2ef788:	0e 41 20 5c	5c20410e   	if (!p1) jump:nt 0x2ef79c <hmx_v75_convbbh1x1_stride1+0x11c>
  2ef78c:	0a c0 80 91	9180c00a   	r10 = memw(r0+#0x0) } 
  2ef790:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2ef794:	e1 47 06 92	920647e1 { 	activation.ub = mxmem(r6,r7):deep:cm
  2ef798:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2ef79c:	19 4d d9 8c	8cd94d19 { 	r25 = setbit(r25,#0xd)
  2ef7a0:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ef7a4:	00 40 3a f3	f33a4000 { 	r0 = sub(r0,r26)
  2ef7a8:	0a 40 80 91	9180400a   	r10 = memw(r0+#0x0)
  2ef7ac:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2ef7b0:	1b 4d db 8c	8cdb4d1b { 	r27 = setbit(r27,#0xd)
  2ef7b4:	10 d7 fb a6	a6fbd710   	cvt.ub = acc(r27) } 
  2ef7b8:	00 5a 00 f3	f3005a00 { 	r0 = add(r0,r26)
  2ef7bc:	2a 40 80 91	9180402a   	r10 = memw(r0+#0x4)
  2ef7c0:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2ef7c4:	39 4d d9 8c	8cd94d39 { 	r25 = clrbit(r25,#0xd)
  2ef7c8:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ef7cc:	00 40 3a f3	f33a4000 { 	r0 = sub(r0,r26)
  2ef7d0:	2a 40 80 91	9180402a   	r10 = memw(r0+#0x4)
  2ef7d4:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2ef7d8:	3b 4d db 8c	8cdb4d3b { 	r27 = clrbit(r27,#0xd)
  2ef7dc:	10 d7 fb a6	a6fbd710   	cvt.ub = acc(r27) } 
  2ef7e0:	00 51 00 f3	f3005100 { 	r0 = add(r0,r17)
  2ef7e4:	08 80 62 70	70628008   	r8 = r2
  2ef7e8:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2ef7ec:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ef7f0:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ef7f4:	00 40 58 75	75584000   	p0 = cmp.gt(r24,#0x0)
  2ef7f8:	b2 e8 df 5c	5cdfe8b2   	if (p0.new) jump:nt 0x2ef750 <hmx_v75_convbbh1x1_stride1+0xd0> } 
  2ef7fc:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2ef800:	12 4e 17 f5	f5174e12   	r19:18 = combine(r23,r14)
  2ef804:	20 40 4d 75	754d4020   	p0 = cmp.gt(r13,#0x1)
  2ef808:	92 f8 df 5c	5cdff892   	if (p0.new) jump:t 0x2ef720 <hmx_v75_convbbh1x1_stride1+0xa0> } 
  2ef80c:	7e c0 00 58	5800c07e { 	jump 0x2ef908 <hmx_v75_convbbh1x1_stride1+0x288> } 
  2ef810:	00 40 00 7f	7f004000 { 	nop
  2ef814:	00 40 00 7f	7f004000   	nop
  2ef818:	00 40 00 7f	7f004000   	nop
  2ef81c:	00 c0 00 7f	7f00c000   	nop } 
  2ef820:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ef824:	cd 7f ed bf	bfed7fcd   	r13 = add(r13,#-0x2)
  2ef828:	18 40 6c 70	706c4018   	r24 = r12
  2ef82c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef830:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ef834:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2ef838:	00 40 00 7f	7f004000   	nop
  2ef83c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef840:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ef844:	00 40 00 7f	7f004000   	nop
  2ef848:	00 40 00 7f	7f004000   	nop
  2ef84c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef850:	a3 5f 03 b0	b0035fa3 { 	r3 = add(r3,#0xfd)
  2ef854:	00 40 00 7f	7f004000   	nop
  2ef858:	00 40 00 7f	7f004000   	nop
  2ef85c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef860:	00 42 14 60	60144200 { 	loop0(0x2ef880,r20)
  2ef864:	81 40 13 b0	b0134081   	r1 = add(r19,#0x4)
  2ef868:	18 7f f8 bf	bff87f18   	r24 = add(r24,#-0x8)
  2ef86c:	06 c0 93 91	9193c006   	r6 = memw(r19+#0x0) } 
  2ef870:	00 5a 12 f3	f3125a00 { 	r0 = add(r18,r26)
  2ef874:	15 40 74 70	70744015   	r21 = r20
  2ef878:	00 40 00 7f	7f004000   	nop
  2ef87c:	0a c0 92 91	9192c00a   	r10 = memw(r18+#0x0) } 
  2ef880:	f5 7f f5 bf	bff57ff5 { 	r21 = add(r21,#-0x1)
  2ef884:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2ef888:	e8 e9 02 92	9202e9e8   	weight.b = mxmem(r2,r9):deep } 
  2ef88c:	19 4d d9 8c	8cd94d19 { 	r25 = setbit(r25,#0xd)
  2ef890:	00 40 55 75	75554000   	p0 = cmp.gt(r21,#0x0)
  2ef894:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ef898:	00 40 3a f3	f33a4000 { 	r0 = sub(r0,r26)
  2ef89c:	0a 40 80 91	9180400a   	r10 = memw(r0+#0x0)
  2ef8a0:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2ef8a4:	1b 4d db 8c	8cdb4d1b { 	r27 = setbit(r27,#0xd)
  2ef8a8:	10 d7 fb a6	a6fbd710   	cvt.ub = acc(r27) } 
  2ef8ac:	00 5a 00 f3	f3005a00 { 	r0 = add(r0,r26)
  2ef8b0:	2a 40 80 91	9180402a   	r10 = memw(r0+#0x4)
  2ef8b4:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2ef8b8:	39 4d d9 8c	8cd94d39 { 	r25 = clrbit(r25,#0xd)
  2ef8bc:	26 60 81 9b	9b816026   	if (p0) r6 = memw(r1++#0x4)
  2ef8c0:	10 d7 f9 a6	a6f9d710   	cvt.ub = acc(r25) } 
  2ef8c4:	00 40 3a f3	f33a4000 { 	r0 = sub(r0,r26)
  2ef8c8:	2a 40 80 91	9180402a   	r10 = memw(r0+#0x4)
  2ef8cc:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2ef8d0:	3b 4d db 8c	8cdb4d3b { 	r27 = clrbit(r27,#0xd)
  2ef8d4:	00 51 00 f3	f3005100   	r0 = add(r0,r17)
  2ef8d8:	10 d7 fb a6	a6fbd710   	cvt.ub = acc(r27) } 
  2ef8dc:	00 9a 00 f3	f3009a00 { 	r0 = add(r0,r26)
  2ef8e0:	0a 40 80 41	4180400a   	if (p0) r10 = memw(r0+#0x0)
  2ef8e4:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop0
  2ef8e8:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ef8ec:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ef8f0:	00 40 58 75	75584000   	p0 = cmp.gt(r24,#0x0)
  2ef8f4:	bc e8 df 5c	5cdfe8bc   	if (p0.new) jump:nt 0x2ef860 <hmx_v75_convbbh1x1_stride1+0x1e0> } 
  2ef8f8:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2ef8fc:	12 4e 17 f5	f5174e12   	r19:18 = combine(r23,r14)
  2ef900:	20 40 4d 75	754d4020   	p0 = cmp.gt(r13,#0x1)
  2ef904:	94 f8 df 5c	5cdff894   	if (p0.new) jump:t 0x2ef820 <hmx_v75_convbbh1x1_stride1+0x1a0> } 
  2ef908:	00 40 4d 75	754d4000 { 	p0 = cmp.gt(r13,#0x0)
  2ef90c:	44 58 20 5c	5c205844   	if (!p0.new) jump:t 0x2ef990 <hmx_v75_convbbh1x1_stride1+0x310>
  2ef910:	00 c0 00 7f	7f00c000   	nop } 
  2ef914:	18 42 04 60	60044218 { 	loop0(0x2ef940,r4)
  2ef918:	09 41 09 8c	8c094109   	r9 = asr(r9,#0x1)
  2ef91c:	18 c0 6c 70	706cc018   	r24 = r12 } 
  2ef920:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2ef924:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef928:	e3 5f 03 b0	b0035fe3 { 	r3 = add(r3,#0xff)
  2ef92c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2ef930:	00 41 34 60	60344100 { 	loop1(0x2ef940,r20)
  2ef934:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2ef938:	18 7f f8 bf	bff87f18   	r24 = add(r24,#-0x8)
  2ef93c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2ef940:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2ef944:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2ef948:	e1 47 06 92	920647e1   	activation.ub = mxmem(r6,r7):deep:cm
  2ef94c:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2ef950:	00 5f 04 60	60045f00 { 	loop0(0x2ef940,r4)
  2ef954:	08 40 62 70	70624008   	r8 = r2
  2ef958:	00 5a 00 f3	f3005a00   	r0 = add(r0,r26)
  2ef95c:	0a c0 80 91	9180c00a   	r10 = memw(r0+#0x0) } 
  2ef960:	10 d7 f9 a6	a6f9d710 { 	cvt.ub = acc(r25) } 
  2ef964:	00 40 3a f3	f33a4000 { 	r0 = sub(r0,r26)
  2ef968:	0a 40 80 91	9180400a   	r10 = memw(r0+#0x0)
  2ef96c:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2ef970:	10 d7 fb a6	a6fbd710 { 	cvt.ub = acc(r27) } 
  2ef974:	00 51 00 f3	f3005100 { 	r0 = add(r0,r17)
  2ef978:	00 80 00 7f	7f008000   	nop
  2ef97c:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2ef980:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2ef984:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2ef988:	00 40 58 75	75584000   	p0 = cmp.gt(r24,#0x0)
  2ef98c:	d8 e8 df 5c	5cdfe8d8   	if (p0.new) jump:nt 0x2ef930 <hmx_v75_convbbh1x1_stride1+0x2b0> } 
  2ef990:	11 c0 e0 a6	a6e0c011 { 	mxclracc } 
  2ef994:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2ef998:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2ef99c:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2ef9a0:	1d 46 1d b0	b01d461d { 	r29 = add(r29,#0x30)
  2ef9a4:	00 40 9f 52	529f4000   	jumpr r31
  2ef9a8:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...
