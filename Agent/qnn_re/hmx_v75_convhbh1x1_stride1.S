
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

002f3580 <hmx_v75_convhbh1x1_stride1>:
  2f3580:	7c 70 03 78	7803707c { 	r28 = #0x783
  2f3584:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f3588:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f358c:	00 5c 86 c7	c7865c00 { 	p0 = bitsclr(r6,r28)
  2f3590:	00 5c 8a c7	c78a5c00   	p0 = bitsclr(r10,r28)
  2f3594:	88 c1 84 91	9184c188   	r8 = memw(r4+#0x30) } 
  2f3598:	2d 40 00 00	0000402d { 	immext(#0xb40)
  2f359c:	50 c0 20 5c	5c20c050   	if (!p0) jump:nt 0x2f4100 <hmx_v73_convhbh1x1_stride1_unaligned> } 
  2f35a0:	01 45 08 85	85084501 { 	p1 = tstbit(r8,#0x5)
  2f35a4:	30 59 07 5c	5c075930   	if (p1.new) jump:t 0x2f5200 <hmx_v75_convhbh1x1deep_stride1>
  2f35a8:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f35ac:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2f35b0:	1d 78 fd bf	bffd781d { 	r29 = add(r29,#-0x40)
  2f35b4:	cd 0b c4 eb	ebc40bcd   	memd(r29+#-0x40) = r17:16; 	memd(r29+#-0x38) = r19:18 } 
  2f35b8:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2f35bc:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f35c0:	02 d4 dd a1	a1ddd402   	memd(r29+#0x10) = r21:20 } 
  2f35c4:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2f35c8:	23 40 15 75	75154023   	p3 = cmp.eq(r21,#0x1)
  2f35cc:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f35d0:	20 40 44 75	75444020 { 	p0 = cmp.gt(r4,#0x1)
  2f35d4:	03 56 dd a1	a1dd5603   	memd(r29+#0x18) = r23:22
  2f35d8:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2f35dc:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2f35e0:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2f35e4:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2f35e8:	3c 41 04 8c	8c04413c   	r28 = lsr(r4,#0x1)
  2f35ec:	ad 40 80 91	918040ad   	r13 = memw(r0+#0x14)
  2f35f0:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f35f4:	06 40 31 62	62314006 { 	m0 = r17
  2f35f8:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2f35fc:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f3600:	74 40 14 b0	b0144074 { 	r20 = add(r20,#0x3)
  2f3604:	00 40 00 7f	7f004000   	nop
  2f3608:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2f360c:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f3610:	34 42 14 8c	8c144234 { 	r20 = lsr(r20,#0x2)
  2f3614:	8e 40 00 b0	b000408e   	r14 = add(r0,#0x4)
  2f3618:	12 40 01 f5	f5014012   	r19:18 = combine(r1,r0)
  2f361c:	0e c1 9d a1	a19dc10e   	memw(r29+#0x38) = r1 } 
  2f3620:	78 40 20 5c	5c204078 { 	if (!p0) jump:nt 0x2f3710 <hmx_v75_convhbh1x1_stride1+0x190>
  2f3624:	00 40 00 7f	7f004000   	nop
  2f3628:	00 40 00 7f	7f004000   	nop
  2f362c:	00 c0 00 7f	7f00c000   	nop } 
  2f3630:	1a 40 64 70	7064401a { 	r26 = r4
  2f3634:	18 4e 87 76	76874e18   	r24 = or(r7,#0x70)
  2f3638:	f9 7f 01 78	78017ff9   	r25 = #0x3ff
  2f363c:	c8 c3 20 5c	5c20c3c8   	if (!p3) jump:nt 0x2f37c0 <hmx_v75_convhbh1x1_stride1+0x240> } 
  2f3640:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f3644:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2f3648:	11 40 6c 70	706c4011   	r17 = r12
  2f364c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3650:	00 43 1c 60	601c4300 { 	loop0(0x2f3680,r28)
  2f3654:	e3 5f 03 b0	b0035fe3   	r3 = add(r3,#0xff)
  2f3658:	00 40 00 7f	7f004000   	nop
  2f365c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3660:	92 7f ee bf	bfee7f92 { 	r18 = add(r14,#-0x4)
  2f3664:	00 40 00 7f	7f004000   	nop
  2f3668:	00 40 00 7f	7f004000   	nop
  2f366c:	d3 c1 9d 91	919dc1d3   	r19 = memw(r29+#0x38) } 
  2f3670:	00 41 34 60	60344100 { 	loop1(0x2f3680,r20)
  2f3674:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f3678:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f367c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f3680:	40 40 1a 75	751a4040 { 	p0 = cmp.eq(r26,#0x2)
  2f3684:	da 7f fa bf	bffa7fda   	r26 = add(r26,#-0x2)
  2f3688:	46 40 81 9b	9b814046   	r6 = memw(r1++#0x8)
  2f368c:	37 c0 81 91	9181c037   	r23 = memw(r1+#0x4) } 
  2f3690:	08 40 48 b0	b0484008 { 	r8 = add(r8,#0x400)
  2f3694:	18 47 09 fd	fd094718   	if (p0) r25:24 = combine(r9,r7)
  2f3698:	ec 58 06 92	920658ec   	activation.ub = mxmem(r6,r24)
  2f369c:	eb f9 08 92	9208f9eb   	weight.b = mxmem(r8,r25):dilate } 
  2f36a0:	28 80 19 e2	e2198028 { 	r8 += add(r25,#0x1)
  2f36a4:	ec 58 17 92	921758ec   	activation.ub = mxmem(r23,r24)
  2f36a8:	eb f9 08 92	9208f9eb   	weight.b = mxmem(r8,r25):dilate }  :endloop0
  2f36ac:	08 5d 1c 60	601c5d08 { 	loop0(0x2f3680,r28)
  2f36b0:	18 4e 87 76	76874e18   	r24 = or(r7,#0x70)
  2f36b4:	10 41 20 5c	5c204110   	if (!p1) jump:nt 0x2f36cc <hmx_v75_convhbh1x1_stride1+0x14c>
  2f36b8:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f36bc:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f36c0:	28 40 09 e2	e2094028 { 	r8 += add(r9,#0x1)
  2f36c4:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f36c8:	eb e9 08 92	9208e9eb   	weight.b = mxmem(r8,r9):dilate } 
  2f36cc:	1a 44 08 f5	f508441a { 	r27:26 = combine(r8,r4)
  2f36d0:	08 40 62 70	70624008   	r8 = r2
  2f36d4:	36 c0 02 78	7802c036   	r22 = #0x401 } 
  2f36d8:	96 41 09 78	78094196 { 	r22 = #0x120c
  2f36dc:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f36e0:	10 d9 f6 a6	a6f6d910 { 	cvt.uh = acc(r22):2x2 } 
  2f36e4:	f9 7f 01 78	78017ff9 { 	r25 = #0x3ff
  2f36e8:	00 80 00 7f	7f008000   	nop
  2f36ec:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f36f0:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f36f4:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f36f8:	c0 c0 b9 10	10b9c0c0   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f3670 <hmx_v75_convhbh1x1_stride1+0xf0> } 
  2f36fc:	02 40 7b 70	707b4002 { 	r2 = r27
  2f3700:	8e 40 0e b0	b00e408e   	r14 = add(r14,#0x4)
  2f3704:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2f3708:	a2 f8 df 5c	5cdff8a2   	if (p0.new) jump:t 0x2f3640 <hmx_v75_convhbh1x1_stride1+0xc0> } 
  2f370c:	14 c1 00 58	5800c114 { 	jump 0x2f3934 <hmx_v75_convhbh1x1_stride1+0x3b4> } 
  2f3710:	c8 43 20 5c	5c2043c8 { 	if (!p3) jump:nt 0x2f38a0 <hmx_v75_convhbh1x1_stride1+0x320>
  2f3714:	00 40 00 7f	7f004000   	nop
  2f3718:	00 40 00 7f	7f004000   	nop
  2f371c:	00 c0 00 7f	7f00c000   	nop } 
  2f3720:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f3724:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2f3728:	11 40 6c 70	706c4011   	r17 = r12
  2f372c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3730:	e3 5f 03 b0	b0035fe3 { 	r3 = add(r3,#0xff)
  2f3734:	00 40 00 7f	7f004000   	nop
  2f3738:	00 40 00 7f	7f004000   	nop
  2f373c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3740:	92 7f ee bf	bfee7f92 { 	r18 = add(r14,#-0x4)
  2f3744:	00 40 00 7f	7f004000   	nop
  2f3748:	00 40 00 7f	7f004000   	nop
  2f374c:	d3 c1 9d 91	919dc1d3   	r19 = memw(r29+#0x38) } 
  2f3750:	00 41 14 60	60144100 { 	loop0(0x2f3760,r20)
  2f3754:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f3758:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f375c:	00 c0 00 7f	7f00c000   	nop } 
  2f3760:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2f3764:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f3768:	36 40 02 78	78024036 { 	r22 = #0x401
  2f376c:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f3770:	eb e9 02 92	9202e9eb   	weight.b = mxmem(r2,r9):dilate } 
  2f3774:	96 41 09 78	78094196 { 	r22 = #0x120c
  2f3778:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f377c:	10 d9 f6 a6	a6f6d910 { 	cvt.uh = acc(r22):2x2 } 
  2f3780:	00 80 00 7f	7f008000 { 	nop
  2f3784:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop0
  2f3788:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f378c:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f3790:	e4 c0 b9 10	10b9c0e4   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f3750 <hmx_v75_convhbh1x1_stride1+0x1d0> } 
  2f3794:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f3798:	8e 40 0e b0	b00e408e   	r14 = add(r14,#0x4)
  2f379c:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2f37a0:	c6 f8 df 5c	5cdff8c6   	if (p0.new) jump:t 0x2f3720 <hmx_v75_convhbh1x1_stride1+0x1a0> } 
  2f37a4:	c8 c0 00 58	5800c0c8 { 	jump 0x2f3934 <hmx_v75_convhbh1x1_stride1+0x3b4> } 
  2f37a8:	00 40 00 7f	7f004000 { 	nop
  2f37ac:	00 c0 00 7f	7f00c000   	nop } 
  2f37b0:	00 40 00 7f	7f004000 { 	nop
  2f37b4:	00 40 00 7f	7f004000   	nop
  2f37b8:	00 40 00 7f	7f004000   	nop
  2f37bc:	00 c0 00 7f	7f00c000   	nop } 
  2f37c0:	00 42 1c 60	601c4200 { 	loop0(0x2f37e0,r28)
  2f37c4:	00 40 00 7f	7f004000   	nop
  2f37c8:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2f37cc:	11 c0 6c 70	706cc011   	r17 = r12 } 
  2f37d0:	00 41 34 60	60344100 { 	loop1(0x2f37e0,r20)
  2f37d4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f37d8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f37dc:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f37e0:	40 40 1a 75	751a4040 { 	p0 = cmp.eq(r26,#0x2)
  2f37e4:	da 7f fa bf	bffa7fda   	r26 = add(r26,#-0x2)
  2f37e8:	46 40 81 9b	9b814046   	r6 = memw(r1++#0x8)
  2f37ec:	37 c0 81 91	9181c037   	r23 = memw(r1+#0x4) } 
  2f37f0:	08 40 48 b0	b0484008 { 	r8 = add(r8,#0x400)
  2f37f4:	18 47 09 fd	fd094718   	if (p0) r25:24 = combine(r9,r7)
  2f37f8:	ec 58 06 92	920658ec   	activation.ub = mxmem(r6,r24)
  2f37fc:	eb f9 08 92	9208f9eb   	weight.b = mxmem(r8,r25):dilate } 
  2f3800:	28 80 19 e2	e2198028 { 	r8 += add(r25,#0x1)
  2f3804:	ec 58 17 92	921758ec   	activation.ub = mxmem(r23,r24)
  2f3808:	eb f9 08 92	9208f9eb   	weight.b = mxmem(r8,r25):dilate }  :endloop0
  2f380c:	08 5d 1c 60	601c5d08 { 	loop0(0x2f37e0,r28)
  2f3810:	18 4e 87 76	76874e18   	r24 = or(r7,#0x70)
  2f3814:	10 41 20 5c	5c204110   	if (!p1) jump:nt 0x2f382c <hmx_v75_convhbh1x1_stride1+0x2ac>
  2f3818:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f381c:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f3820:	28 40 09 e2	e2094028 { 	r8 += add(r9,#0x1)
  2f3824:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f3828:	eb e9 08 92	9208e9eb   	weight.b = mxmem(r8,r9):dilate } 
  2f382c:	1a 44 08 f5	f508441a { 	r27:26 = combine(r8,r4)
  2f3830:	08 40 62 70	70624008   	r8 = r2
  2f3834:	19 c0 75 70	7075c019   	r25 = r21 } 
  2f3838:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f383c:	f9 7f f9 bf	bff97ff9   	r25 = add(r25,#-0x1)
  2f3840:	36 40 85 9b	9b854036   	r22 = memw(r5++#0x4)
  2f3844:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3848:	00 40 59 75	75594000 { 	p0 = cmp.gt(r25,#0x0)
  2f384c:	36 40 85 9b	9b854036   	r22 = memw(r5++#0x4)
  2f3850:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f3854:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f3858:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f385c:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f3838 <hmx_v75_convhbh1x1_stride1+0x2b8>
  2f3860:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f3864:	45 43 15 8e	8e154345 { 	r5 -= asl(r21,#0x3)
  2f3868:	43 89 15 8e	8e158943   	r3 -= asl(r21,#0x9)
  2f386c:	f9 7f 01 78	78017ff9   	r25 = #0x3ff
  2f3870:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f3874:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f3878:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f387c:	ae c0 b9 10	10b9c0ae   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f37d0 <hmx_v75_convhbh1x1_stride1+0x250> } 
  2f3880:	c3 49 15 8e	8e1549c3 { 	r3 += asl(r21,#0x9)
  2f3884:	12 40 6e 70	706e4012   	r18 = r14
  2f3888:	d3 c1 9d 91	919dc1d3   	r19 = memw(r29+#0x38) } 
  2f388c:	02 40 7b 70	707b4002 { 	r2 = r27
  2f3890:	8e 40 0e b0	b00e408e   	r14 = add(r14,#0x4)
  2f3894:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2f3898:	9a f8 df 5c	5cdff89a   	if (p0.new) jump:t 0x2f37c0 <hmx_v75_convhbh1x1_stride1+0x240> } 
  2f389c:	4c c0 00 58	5800c04c { 	jump 0x2f3934 <hmx_v75_convhbh1x1_stride1+0x3b4> } 
  2f38a0:	00 40 00 7f	7f004000 { 	nop
  2f38a4:	00 40 00 7f	7f004000   	nop
  2f38a8:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2f38ac:	11 c0 6c 70	706cc011   	r17 = r12 } 
  2f38b0:	00 41 14 60	60144100 { 	loop0(0x2f38c0,r20)
  2f38b4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f38b8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f38bc:	00 c0 00 7f	7f00c000   	nop } 
  2f38c0:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2f38c4:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f38c8:	19 40 75 70	70754019 { 	r25 = r21
  2f38cc:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f38d0:	eb e9 02 92	9202e9eb   	weight.b = mxmem(r2,r9):dilate } 
  2f38d4:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f38d8:	f9 7f f9 bf	bff97ff9   	r25 = add(r25,#-0x1)
  2f38dc:	36 40 85 9b	9b854036   	r22 = memw(r5++#0x4)
  2f38e0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f38e4:	00 40 59 75	75594000 { 	p0 = cmp.gt(r25,#0x0)
  2f38e8:	36 40 85 9b	9b854036   	r22 = memw(r5++#0x4)
  2f38ec:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f38f0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f38f4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f38f8:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f38d4 <hmx_v75_convhbh1x1_stride1+0x354>
  2f38fc:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f3900:	45 83 15 8e	8e158345 { 	r5 -= asl(r21,#0x3)
  2f3904:	43 49 15 8e	8e154943   	r3 -= asl(r21,#0x9)
  2f3908:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop0
  2f390c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f3910:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f3914:	d2 c0 b9 10	10b9c0d2   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f38b0 <hmx_v75_convhbh1x1_stride1+0x330> } 
  2f3918:	c3 49 15 8e	8e1549c3 { 	r3 += asl(r21,#0x9)
  2f391c:	12 40 6e 70	706e4012   	r18 = r14
  2f3920:	d3 c1 9d 91	919dc1d3   	r19 = memw(r29+#0x38) } 
  2f3924:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f3928:	8e 40 0e b0	b00e408e   	r14 = add(r14,#0x4)
  2f392c:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2f3930:	be f8 df 5c	5cdff8be   	if (p0.new) jump:t 0x2f38a0 <hmx_v75_convhbh1x1_stride1+0x320> } 
  2f3934:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2f3938:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f393c:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f3940:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f3944:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f3948:	1d 48 1d b0	b01d481d { 	r29 = add(r29,#0x40)
  2f394c:	00 40 9f 52	529f4000   	jumpr r31
  2f3950:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f3980 <hmx_v85_convhbh1x1_stride1>:
  2f3980:	c0 3f 00 5a	5a003fc0 { 	r0 = #-0x1; 	jumpr r31 } 
		...

002f39c0 <hmx_v73_convhbh1x1_lp_stride1>:
  2f39c0:	9c c1 84 91	9184c19c { 	r28 = memw(r4+#0x30) } 
  2f39c4:	00 45 1c 85	851c4500 { 	p0 = tstbit(r28,#0x5)
  2f39c8:	35 40 00 00	00004035   	immext(#0xd40)
  2f39cc:	78 d8 00 5c	5c00d878   	if (p0.new) jump:t 0x2f4740 <hmx_v73_convhbh1x1deep_lp_stride1> } 
  2f39d0:	1d 7c fd bf	bffd7c1d { 	r29 = add(r29,#-0x20)
  2f39d4:	ed 7f 03 78	78037fed   	r13 = #0x7ff
  2f39d8:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f39dc:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f39e0:	06 4d 06 f1	f1064d06 { 	r6 = and(r6,r13)
  2f39e4:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f39e8:	ac 40 80 91	918040ac   	r12 = memw(r0+#0x14)
  2f39ec:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f39f0:	ec 43 0c b0	b00c43ec { 	r12 = add(r12,#0x1f)
  2f39f4:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f39f8:	00 d0 dd a1	a1ddd000   	memd(r29+#0x0) = r17:16 } 
  2f39fc:	2c 45 0c 8c	8c0c452c { 	r12 = lsr(r12,#0x5)
  2f3a00:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f3a04:	21 40 44 75	75444021 { 	p1 = cmp.gt(r4,#0x1)
  2f3a08:	e4 7f e4 bf	bfe47fe4   	r4 = add(r4,#-0x1)
  2f3a0c:	16 0a 0d ea	ea0d0a16   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2f3a10:	16 4e 87 76	76874e16 { 	r22 = or(r7,#0x70)
  2f3a14:	f7 7f 01 78	78017ff7   	r23 = #0x3ff
  2f3a18:	1f 2a 88 dc	dc882a1f   	r16 = memw(r29+#0x20); 	memd(r29+#0x18) = r23:22 } 
  2f3a1c:	06 50 26 f1	f1265006 { 	r6 = or(r6,r16)
  2f3a20:	0f 40 61 70	7061400f   	r15 = r1
  2f3a24:	34 c0 85 9b	9b85c034   	r20 = memw(r5++#0x4) } 
  2f3a28:	00 40 00 7f	7f004000 { 	nop
  2f3a2c:	00 c0 00 7f	7f00c000   	nop } 
  2f3a30:	00 40 00 7f	7f004000 { 	nop
  2f3a34:	00 40 00 7f	7f004000   	nop
  2f3a38:	00 40 00 7f	7f004000   	nop
  2f3a3c:	00 c0 00 7f	7f00c000   	nop } 
  2f3a40:	00 42 04 60	60044200 { 	loop0(0x2f3a60,r4)
  2f3a44:	08 40 c2 bf	bfc24008   	r8 = add(r2,#-0x400)
  2f3a48:	ec 7f ec bf	bfec7fec   	r12 = add(r12,#-0x1)
  2f3a4c:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f3a50:	11 51 30 f3	f3305111 { 	r17 = sub(r17,r16)
  2f3a54:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f3a58:	18 41 20 5c	5c204118   	if (!p1) jump:nt 0x2f3a80 <hmx_v73_convhbh1x1_lp_stride1+0xc0>
  2f3a5c:	00 c0 00 7f	7f00c000   	nop } 
  2f3a60:	16 51 36 f1	f1365116 { 	r22 = or(r22,r17)
  2f3a64:	08 40 48 b0	b0484008   	r8 = add(r8,#0x400)
  2f3a68:	00 40 00 7f	7f004000   	nop
  2f3a6c:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f3a70:	11 91 30 f3	f3309111 { 	r17 = sub(r17,r16)
  2f3a74:	16 4d 16 f1	f1164d16   	r22 = and(r22,r13)
  2f3a78:	f0 56 06 92	920656f0   	activation.ub = mxmem(r6,r22):single
  2f3a7c:	e0 f7 08 92	9208f7e0   	weight.b = mxmem(r8,r23) }  :endloop0
  2f3a80:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f3a84:	08 40 48 b0	b0484008   	r8 = add(r8,#0x400)
  2f3a88:	0a 4d 0a f1	f10a4d0a   	r10 = and(r10,r13)
  2f3a8c:	32 c0 80 9b	9b80c032   	r18 = memw(r0++#0x4) } 
  2f3a90:	0a 52 2a f1	f12a520a { 	r10 = or(r10,r18)
  2f3a94:	13 40 74 70	70744013   	r19 = r20
  2f3a98:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f3a9c:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) } 
  2f3aa0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f3aa4:	f3 7f f3 bf	bff37ff3   	r19 = add(r19,#-0x1)
  2f3aa8:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f3aac:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3ab0:	00 40 53 75	75534000 { 	p0 = cmp.gt(r19,#0x0)
  2f3ab4:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f3ab8:	10 d9 f5 a6	a6f5d910   	cvt.uh = acc(r21):2x2 } 
  2f3abc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f3ac0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3ac4:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f3aa0 <hmx_v73_convhbh1x1_lp_stride1+0xe0>
  2f3ac8:	10 d9 f5 a6	a6f5d910   	cvt.uh = acc(r21):2x2 } 
  2f3acc:	45 43 14 8e	8e144345 { 	r5 -= asl(r20,#0x3)
  2f3ad0:	28 40 09 e2	e2094028   	r8 += add(r9,#0x1)
  2f3ad4:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f3ad8:	02 40 68 70	70684002 { 	r2 = r8
  2f3adc:	01 40 6f 70	706f4001   	r1 = r15
  2f3ae0:	00 40 4c 75	754c4000   	p0 = cmp.gt(r12,#0x0)
  2f3ae4:	b4 f8 df 5c	5cdff8b4   	if (p0.new) jump:t 0x2f3a40 <hmx_v73_convhbh1x1_lp_stride1+0x80> } 
  2f3ae8:	11 c0 e0 a6	a6e0c011 { 	mxclracc } 
  2f3aec:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f3af0:	1d 44 1d b0	b01d441d { 	r29 = add(r29,#0x20)
  2f3af4:	00 40 9f 52	529f4000   	jumpr r31
  2f3af8:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
  2f3afc:	00 00 00 00 <unknown>

002f3b00 <hmx_v85_convhbh1x1_lp_stride1>:
  2f3b00:	c0 3f 00 5a	5a003fc0 { 	r0 = #-0x1; 	jumpr r31 } 
		...

002f3b40 <hmx_convhbh1x1_stride1_unaligned>:
  2f3b40:	1d 78 fd bf	bffd781d { 	r29 = add(r29,#-0x40)
  2f3b44:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f3b48:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f3b4c:	00 40 00 7f	7f004000 { 	nop
  2f3b50:	00 40 00 7f	7f004000   	nop
  2f3b54:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f3b58:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2f3b5c:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2f3b60:	65 40 81 91	91814065   	r5 = memw(r1+#0xc)
  2f3b64:	8e c0 81 91	9181c08e   	r14 = memw(r1+#0x10) } 
  2f3b68:	65 40 05 b0	b0054065 { 	r5 = add(r5,#0x3)
  2f3b6c:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f3b70:	05 42 05 8c	8c054205 { 	r5 = asr(r5,#0x2)
  2f3b74:	ad 40 80 91	918040ad   	r13 = memw(r0+#0x14)
  2f3b78:	00 d0 dd a1	a1ddd000   	memd(r29+#0x0) = r17:16 } 
  2f3b7c:	e4 7f e4 bf	bfe47fe4 { 	r4 = add(r4,#-0x1)
  2f3b80:	00 40 00 7f	7f004000   	nop
  2f3b84:	0d 0a 08 c2	c2080a0d   	r16 = memw(r0+#0x8); 	memd(r29+#0x8) = r19:18 } 
  2f3b88:	f2 7f 03 78	78037ff2 { 	r18 = #0x7ff
  2f3b8c:	f3 7f 03 78	78037ff3   	r19 = #0x7ff
  2f3b90:	16 0a 09 c1	c1090a16   	r17 = memw(r0+#0x4); 	memd(r29+#0x10) = r21:20 } 
  2f3b94:	76 50 00 78	78005076 { 	r22 = #0x83
  2f3b98:	00 40 00 7f	7f004000   	nop
  2f3b9c:	1f 0a 0c c3	c30c0a1f   	r20 = memw(r0+#0xc); 	memd(r29+#0x18) = r23:22 } 
  2f3ba0:	51 42 11 8c	8c114251 { 	r17 = asl(r17,#0x2)
  2f3ba4:	94 40 14 b0	b0144094   	r20 = add(r20,#0x4)
  2f3ba8:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2f3bac:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f3bb0:	50 42 10 8c	8c104250 { 	r16 = asl(r16,#0x2)
  2f3bb4:	06 52 e6 d3	d3e65206   	r7:6 = and(r7:6,r19:18)
  2f3bb8:	04 58 dd a1	a1dd5804   	memd(r29+#0x20) = r25:24
  2f3bbc:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f3bc0:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2f3bc4:	1e 40 67 70	7067401e   	r30 = r7
  2f3bc8:	ff 7f 01 78	78017fff   	r31 = #0x3ff
  2f3bcc:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f3bd0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f3bd4:	1a 40 01 f5	f501401a   	r27:26 = combine(r1,r0)
  2f3bd8:	18 41 00 78	78004118   	r24 = #0x8
  2f3bdc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3be0:	00 42 25 60	60254200 { 	loop1(0x2f3c00,r5)
  2f3be4:	17 40 00 78	78004017   	r23 = #0x0
  2f3be8:	19 40 00 78	78004019   	r25 = #0x0
  2f3bec:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f3bf0:	01 58 4e f2	f24e5801 { 	p1 = cmp.gt(r14,r24)
  2f3bf4:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2f3bf8:	00 40 00 7f	7f004000   	nop
  2f3bfc:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f3c00:	00 42 04 60	60044200 { 	loop0(0x2f3c20,r4)
  2f3c04:	07 4e 87 76	76874e07   	r7 = or(r7,#0x70)
  2f3c08:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f3c0c:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f3c10:	37 57 36 fb	fb365737 { 	if (p1) r23 = sub(r23,r22)
  2f3c14:	08 40 c2 bf	bfc24008   	r8 = add(r2,#-0x400)
  2f3c18:	07 40 9e 74	749e4007   	if (!p0) r7 = add(r30,#0x0)
  2f3c1c:	1a c0 20 5c	5c20c01a   	if (!p0) jump:nt 0x2f3c44 <hmx_convhbh1x1_stride1_unaligned+0x104> } 
  2f3c20:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f3c24:	08 40 48 b0	b0484008   	r8 = add(r8,#0x400)
  2f3c28:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f3c2c:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f3c30:	06 92 e6 d3	d3e69206 { 	r7:6 = and(r7:6,r19:18)
  2f3c34:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f3c38:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f3c3c:	e6 ff 08 92	9208ffe6   	weight.b = mxmem(r8,r31):single }  :endloop0
  2f3c40:	07 c0 7e 70	707ec007 { 	r7 = r30 } 
  2f3c44:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f3c48:	08 40 48 b0	b0484008   	r8 = add(r8,#0x400)
  2f3c4c:	0a 52 0a f1	f10a520a   	r10 = and(r10,r18)
  2f3c50:	16 c0 80 91	9180c016   	r22 = memw(r0+#0x0) } 
  2f3c54:	28 40 09 e2	e2094028 { 	r8 += add(r9,#0x1)
  2f3c58:	0a 56 2a f1	f12a560a   	r10 = or(r10,r22)
  2f3c5c:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f3c60:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single } 
  2f3c64:	06 52 e6 d3	d3e65206 { 	r7:6 = and(r7:6,r19:18)
  2f3c68:	0c d2 20 5c	5c20d20c   	if (!p2) jump:t 0x2f3c7c <hmx_convhbh1x1_stride1_unaligned+0x13c> } 
  2f3c6c:	00 51 00 f3	f3005100 { 	r0 = add(r0,r17)
  2f3c70:	00 80 00 7f	7f008000   	nop
  2f3c74:	0a eb ea a6	a6eaeb0a   	mxmem(r10,r11):after:sat.uh = acc:2x1 }  :endloop1
  2f3c78:	18 c0 00 58	5800c018 { 	jump 0x2f3ca8 <hmx_convhbh1x1_stride1_unaligned+0x168> } 
  2f3c7c:	03 40 59 75	75594003 { 	p3 = cmp.gt(r25,#0x0)
  2f3c80:	08 4b 20 5c	5c204b08   	if (!p3.new) jump:nt 0x2f3c8c <hmx_convhbh1x1_stride1_unaligned+0x14c>
  2f3c84:	99 c0 19 b0	b019c099   	r25 = add(r25,#0x4) } 
  2f3c88:	0e eb f5 a6	a6f5eb0e { 	mxmem(r21,r11):after:retain:sat.uh = acc:2x1 } 
  2f3c8c:	03 59 54 f2	f2545903 { 	p3 = cmp.gt(r20,r25)
  2f3c90:	0e 4b 20 5c	5c204b0e   	if (!p3.new) jump:nt 0x2f3ca8 <hmx_convhbh1x1_stride1_unaligned+0x168>
  2f3c94:	9c 60 79 74	7479609c   	if (p3.new) r28 = add(r25,#0x4)
  2f3c98:	15 c0 6a 70	706ac015   	r21 = r10 } 
  2f3c9c:	03 5c 54 f2	f2545c03 { 	p3 = cmp.gt(r20,r28)
  2f3ca0:	60 b1 00 fb	fb00b160   	if (p3.new) r0 = add(r0,r17)
  2f3ca4:	02 eb ea a6	a6eaeb02   	mxmem(r10,r11):before:sat.uh = acc:2x1 }  :endloop1
  2f3ca8:	00 58 4c f2	f24c5800 { 	p0 = cmp.gt(r12,r24)
  2f3cac:	18 41 18 b0	b0184118   	r24 = add(r24,#0x8)
  2f3cb0:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f3cb4:	01 4f 01 f3	f3014f01 { 	r1 = add(r1,r15)
  2f3cb8:	00 50 00 f3	f3005000   	r0 = add(r0,r16)
  2f3cbc:	96 e0 df 5c	5cdfe096   	if (p0) jump:nt 0x2f3be0 <hmx_convhbh1x1_stride1_unaligned+0xa0> } 
  2f3cc0:	9a 40 1a b0	b01a409a { 	r26 = add(r26,#0x4)
  2f3cc4:	0d 7c ed bf	bfed7c0d   	r13 = add(r13,#-0x20)
  2f3cc8:	02 c0 68 70	7068c002   	r2 = r8 } 
  2f3ccc:	00 5a 1b f5	f51b5a00 { 	r1:0 = combine(r27,r26)
  2f3cd0:	00 40 4d 75	754d4000   	p0 = cmp.gt(r13,#0x0)
  2f3cd4:	82 f8 df 5c	5cdff882   	if (p0.new) jump:t 0x2f3bd0 <hmx_convhbh1x1_stride1_unaligned+0x90> } 
  2f3cd8:	de c0 dd 91	91ddc0de { 	r31:30 = memd(r29+#0x30) } 
  2f3cdc:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f3ce0:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f3ce4:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f3ce8:	1d 48 1d b0	b01d481d { 	r29 = add(r29,#0x40)
  2f3cec:	00 40 9f 52	529f4000   	jumpr r31
  2f3cf0:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f3d00 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity>:
  2f3d00:	1d 78 fd bf	bffd781d { 	r29 = add(r29,#-0x40)
  2f3d04:	00 40 00 7f	7f004000   	nop
  2f3d08:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f3d0c:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f3d10:	c9 40 84 91	918440c9 { 	r9 = memw(r4+#0x18)
  2f3d14:	9c c1 84 91	9184c19c   	r28 = memw(r4+#0x30) } 
  2f3d18:	00 45 1c 85	851c4500 { 	p0 = tstbit(r28,#0x5)
  2f3d1c:	4f 40 81 91	9181404f   	r15 = memw(r1+#0x8)
  2f3d20:	00 d0 dd a1	a1ddd000   	memd(r29+#0x0) = r17:16 } 
  2f3d24:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2f3d28:	f4 7f 03 78	78037ff4   	r20 = #0x7ff
  2f3d2c:	16 0a 0d ea	ea0d0a16   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2f3d30:	12 40 65 70	70654012 { 	r18 = r5
  2f3d34:	f5 7f 03 78	78037ff5   	r21 = #0x7ff
  2f3d38:	65 40 81 91	91814065   	r5 = memw(r1+#0xc)
  2f3d3c:	8e c0 81 91	9181c08e   	r14 = memw(r1+#0x10) } 
  2f3d40:	65 40 05 b0	b0054065 { 	r5 = add(r5,#0x3)
  2f3d44:	1f 0a 14 c1	c1140a1f   	r4 = memw(r1+#0x4); 	memd(r29+#0x18) = r23:22 } 
  2f3d48:	25 42 05 8c	8c054225 { 	r5 = lsr(r5,#0x2)
  2f3d4c:	18 41 00 78	78004118   	r24 = #0x8
  2f3d50:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2f3d54:	e4 7f e4 bf	bfe47fe4 { 	r4 = add(r4,#-0x1)
  2f3d58:	76 50 00 78	78005076   	r22 = #0x83
  2f3d5c:	01 40 81 91	91814001   	r1 = memw(r1+#0x0)
  2f3d60:	79 c0 80 91	9180c079   	r25 = memw(r0+#0xc) } 
  2f3d64:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2f3d68:	99 60 d9 74	74d96099   	if (!p2.new) r25 = add(r25,#0x4)
  2f3d6c:	09 01 08 02	02080109   	r16 = memw(r0+#0x8); 	r17 = memw(r0+#0x4) } 
  2f3d70:	06 54 e6 d3	d3e65406 { 	r7:6 = and(r7:6,r21:20)
  2f3d74:	0d 40 61 70	7061400d   	r13 = r1
  2f3d78:	bc 40 80 91	918040bc   	r28 = memw(r0+#0x14)
  2f3d7c:	8c c0 80 91	9180c08c   	r12 = memw(r0+#0x10) } 
  2f3d80:	59 42 05 8e	8e054259 { 	r25 -= asl(r5,#0x2)
  2f3d84:	fc 43 1c b0	b01c43fc   	r28 = add(r28,#0x1f)
  2f3d88:	00 40 80 91	91804000   	r0 = memw(r0+#0x0)
  2f3d8c:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f3d90:	1c 45 1c 8c	8c1c451c { 	r28 = asr(r28,#0x5)
  2f3d94:	0c 41 0c b0	b00c410c   	r12 = add(r12,#0x8)
  2f3d98:	33 40 92 9b	9b924033   	r19 = memw(r18++#0x4)
  2f3d9c:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f3da0:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2f3da4:	9a 40 00 b0	b000409a   	r26 = add(r0,#0x4)
  2f3da8:	e0 c0 00 5c	5c00c0e0   	if (p0) jump:nt 0x2f3f60 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x260> } 
  2f3dac:	fc 7f fc bf	bffc7ffc { 	r28 = add(r28,#-0x1)
  2f3db0:	1e 40 67 70	7067401e   	r30 = r7
  2f3db4:	1f 40 02 78	7802401f   	r31 = #0x400
  2f3db8:	f5 cf 09 76	7609cff5   	r21 = and(r9,#0x7f) } 
  2f3dbc:	03 40 15 75	75154003 { 	p3 = cmp.eq(r21,#0x0)
  2f3dc0:	10 cb 00 5c	5c00cb10   	if (p3.new) jump:nt 0x2f3ddc <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0xdc> } 
  2f3dc4:	35 c0 15 b0	b015c035 { 	r21 = add(r21,#0x1) } 
  2f3dc8:	15 c4 15 8c	8c15c415 { 	r21 = asr(r21,#0x4) } 
  2f3dcc:	35 c0 15 b0	b015c035 { 	r21 = add(r21,#0x1) } 
  2f3dd0:	1f 50 15 e0	e015501f { 	r31 = +mpyi(r21,#0x80)
  2f3dd4:	f5 cf 09 76	7609cff5   	r21 = and(r9,#0x7f) } 
  2f3dd8:	1f d5 3f f1	f13fd51f { 	r31 = or(r31,r21) } 
  2f3ddc:	1f f0 ff bf	bffff01f { 	r31 = add(r31,#-0x80) } 
  2f3de0:	f5 ff 03 78	7803fff5 { 	r21 = #0x7ff } 
  2f3de4:	00 40 00 7f	7f004000 { 	nop
  2f3de8:	00 40 00 7f	7f004000   	nop
  2f3dec:	00 c0 00 7f	7f00c000   	nop } 
  2f3df0:	00 40 00 7f	7f004000 { 	nop
  2f3df4:	00 40 00 7f	7f004000   	nop
  2f3df8:	00 40 00 7f	7f004000   	nop
  2f3dfc:	00 c0 00 7f	7f00c000   	nop } 
  2f3e00:	00 42 25 60	60254200 { 	loop1(0x2f3e20,r5)
  2f3e04:	17 40 00 78	78004017   	r23 = #0x0
  2f3e08:	76 50 00 78	78005076   	r22 = #0x83
  2f3e0c:	00 c0 00 7f	7f00c000   	nop } 
  2f3e10:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2f3e14:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2f3e18:	01 58 4e f2	f24e5801   	p1 = cmp.gt(r14,r24)
  2f3e1c:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f3e20:	18 42 04 60	60044218 { 	loop0(0x2f3e4c,r4)
  2f3e24:	07 4e 87 76	76874e07   	r7 = or(r7,#0x70)
  2f3e28:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f3e2c:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f3e30:	15 f0 3f 76	763ff015 { 	r21 = and(r31,#-0x80) } 
  2f3e34:	08 42 35 f3	f3354208 { 	r8 = sub(r2,r21)
  2f3e38:	f5 ff 03 78	7803fff5   	r21 = #0x7ff } 
  2f3e3c:	08 70 e8 bf	bfe87008 { 	r8 = add(r8,#-0x80)
  2f3e40:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f3e44:	07 40 9e 74	749e4007   	if (!p0) r7 = add(r30,#0x0)
  2f3e48:	20 c0 20 5c	5c20c020   	if (!p0) jump:nt 0x2f3e7c <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x17c> } 
  2f3e4c:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f3e50:	15 70 3f 76	763f7015   	r21 = and(r31,#-0x80)
  2f3e54:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f3e58:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f3e5c:	02 40 00 00	00004002 { 	immext(#0x80)
  2f3e60:	08 40 15 e2	e2154008   	r8 += add(r21,##0x80)
  2f3e64:	f5 ff 03 78	7803fff5   	r21 = #0x7ff } 
  2f3e68:	06 94 e6 d3	d3e69406 { 	r7:6 = and(r7:6,r21:20)
  2f3e6c:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f3e70:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f3e74:	e6 ff 08 92	9208ffe6   	weight.b = mxmem(r8,r31):single }  :endloop0
  2f3e78:	07 c0 7e 70	707ec007 { 	r7 = r30 } 
  2f3e7c:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f3e80:	15 70 3f 76	763f7015   	r21 = and(r31,#-0x80)
  2f3e84:	0a 54 0a f1	f10a540a   	r10 = and(r10,r20)
  2f3e88:	17 c0 73 70	7073c017   	r23 = r19 } 
  2f3e8c:	02 40 00 00	00004002 { 	immext(#0x80)
  2f3e90:	08 40 15 e2	e2154008   	r8 += add(r21,##0x80)
  2f3e94:	f5 ff 03 78	7803fff5   	r21 = #0x7ff } 
  2f3e98:	06 54 e6 d3	d3e65406 { 	r7:6 = and(r7:6,r21:20)
  2f3e9c:	15 70 29 76	76297015   	r21 = and(r9,#-0x80)
  2f3ea0:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f3ea4:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single } 
  2f3ea8:	02 40 00 00	00004002 { 	immext(#0x80)
  2f3eac:	08 c0 15 e2	e215c008   	r8 += add(r21,##0x80) } 
  2f3eb0:	f5 ff 03 78	7803fff5 { 	r21 = #0x7ff } 
  2f3eb4:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f3eb8:	f7 7f f7 bf	bff77ff7   	r23 = add(r23,#-0x1)
  2f3ebc:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2f3ec0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3ec4:	03 40 57 75	75574003 { 	p3 = cmp.gt(r23,#0x0)
  2f3ec8:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f3ecc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f3ed0:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2f3ed4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f3ed8:	ee 63 df 5c	5cdf63ee { 	if (p3) jump:nt 0x2f3eb4 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x1b4>
  2f3edc:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f3ee0:	52 43 13 8e	8e134352 { 	r18 -= asl(r19,#0x3)
  2f3ee4:	43 49 13 8e	8e134943   	r3 -= asl(r19,#0x9)
  2f3ee8:	16 c0 80 91	9180c016   	r22 = memw(r0+#0x0) } 
  2f3eec:	0a 56 2a f1	f12a560a { 	r10 = or(r10,r22)
  2f3ef0:	06 c2 20 5c	5c20c206   	if (!p2) jump:nt 0x2f3ef8 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x1f8> } 
  2f3ef4:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f3ef8:	02 42 e2 6b	6be24202 { 	p2 = or(p2,!p2)
  2f3efc:	40 91 00 fb	fb009140   	if (p2) r0 = add(r0,r17)
  2f3f00:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  2f3f04:	03 40 59 75	75594003 { 	p3 = cmp.gt(r25,#0x0)
  2f3f08:	10 5b 20 5c	5c205b10   	if (!p3.new) jump:t 0x2f3f24 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x224>
  2f3f0c:	16 e0 60 7e	7e60e016   	if (p3.new) r22 = #0x0 } 
  2f3f10:	0a 54 0a f1	f10a540a { 	r10 = and(r10,r20)
  2f3f14:	16 40 80 91	91804016   	r22 = memw(r0+#0x0)
  2f3f18:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f3f1c:	0a d6 2a f1	f12ad60a { 	r10 = or(r10,r22) } 
  2f3f20:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f3f24:	18 41 18 b0	b0184118 { 	r24 = add(r24,#0x8)
  2f3f28:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f3f2c:	01 4f 01 f3	f3014f01 { 	r1 = add(r1,r15)
  2f3f30:	00 50 00 f3	f3005000   	r0 = add(r0,r16)
  2f3f34:	00 58 4c f2	f24c5800   	p0 = cmp.gt(r12,r24)
  2f3f38:	6a e8 df 5c	5cdfe86a   	if (p0.new) jump:nt 0x2f3e00 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x100> } 
  2f3f3c:	01 40 6d 70	706d4001 { 	r1 = r13
  2f3f40:	00 40 7a 70	707a4000   	r0 = r26
  2f3f44:	fc 7f fc bf	bffc7ffc   	r28 = add(r28,#-0x1)
  2f3f48:	02 c0 68 70	7068c002   	r2 = r8 } 
  2f3f4c:	5a 7f 7c 61	617c7f5a { 	if (r28>=#0) jump:t 0x2f3e00
  2f3f50:	c3 49 13 8e	8e1349c3   	r3 += asl(r19,#0x9)
  2f3f54:	9a 40 1a b0	b01a409a   	r26 = add(r26,#0x4)
  2f3f58:	18 c1 00 78	7800c118   	r24 = #0x8 } 
  2f3f5c:	b8 c0 00 58	5800c0b8 { 	jump 0x2f40cc <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x3cc> } 
  2f3f60:	f5 4f 09 76	76094ff5 { 	r21 = and(r9,#0x7f)
  2f3f64:	09 f0 29 76	7629f009   	r9 = and(r9,#-0x80) } 
  2f3f68:	09 d0 09 b0	b009d009 { 	r9 = add(r9,#0x80) } 
  2f3f6c:	29 c1 09 8c	8c09c129 { 	r9 = lsr(r9,#0x1) } 
  2f3f70:	1b 5c 09 ed	ed095c1b { 	r27 = mpyi(r9,r28)
  2f3f74:	09 55 29 f1	f1295509   	r9 = or(r9,r21)
  2f3f78:	f5 7f 03 78	78037ff5   	r21 = #0x7ff
  2f3f7c:	fc ff fc bf	bffcfffc   	r28 = add(r28,#-0x1) } 
  2f3f80:	09 f0 e9 bf	bfe9f009 { 	r9 = add(r9,#-0x80) } 
  2f3f84:	00 40 00 7f	7f004000 { 	nop
  2f3f88:	00 40 00 7f	7f004000   	nop
  2f3f8c:	00 c0 00 7f	7f00c000   	nop } 
  2f3f90:	00 40 00 7f	7f004000 { 	nop
  2f3f94:	00 40 00 7f	7f004000   	nop
  2f3f98:	00 40 00 7f	7f004000   	nop
  2f3f9c:	00 c0 00 7f	7f00c000   	nop } 
  2f3fa0:	00 42 25 60	60254200 { 	loop1(0x2f3fc0,r5)
  2f3fa4:	17 40 00 78	78004017   	r23 = #0x0
  2f3fa8:	76 50 00 78	78005076   	r22 = #0x83
  2f3fac:	00 c0 00 7f	7f00c000   	nop } 
  2f3fb0:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2f3fb4:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2f3fb8:	01 58 4e f2	f24e5801   	p1 = cmp.gt(r14,r24)
  2f3fbc:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f3fc0:	00 42 04 60	60044200 { 	loop0(0x2f3fe0,r4)
  2f3fc4:	00 40 00 7f	7f004000   	nop
  2f3fc8:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f3fcc:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f3fd0:	06 54 e6 d3	d3e65406 { 	r7:6 = and(r7:6,r21:20)
  2f3fd4:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f3fd8:	08 42 3b f3	f33b4208   	r8 = sub(r2,r27)
  2f3fdc:	18 c0 20 5c	5c20c018   	if (!p0) jump:nt 0x2f4000 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x300> } 
  2f3fe0:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f3fe4:	08 5b 08 f3	f3085b08   	r8 = add(r8,r27)
  2f3fe8:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f3fec:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f3ff0:	06 94 e6 d3	d3e69406 { 	r7:6 = and(r7:6,r21:20)
  2f3ff4:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f3ff8:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f3ffc:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single }  :endloop0
  2f4000:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f4004:	08 5b 08 f3	f3085b08   	r8 = add(r8,r27)
  2f4008:	0a 54 0a f1	f10a540a   	r10 = and(r10,r20)
  2f400c:	17 c0 73 70	7073c017   	r23 = r19 } 
  2f4010:	f0 47 06 92	920647f0 { 	activation.ub = mxmem(r6,r7):single
  2f4014:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single } 
  2f4018:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f401c:	f7 7f f7 bf	bff77ff7   	r23 = add(r23,#-0x1)
  2f4020:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2f4024:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4028:	03 40 57 75	75574003 { 	p3 = cmp.gt(r23,#0x0)
  2f402c:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f4030:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4034:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2f4038:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f403c:	ee 63 df 5c	5cdf63ee { 	if (p3) jump:nt 0x2f4018 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x318>
  2f4040:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f4044:	52 43 13 8e	8e134352 { 	r18 -= asl(r19,#0x3)
  2f4048:	43 49 13 8e	8e134943   	r3 -= asl(r19,#0x9)
  2f404c:	16 c0 80 91	9180c016   	r22 = memw(r0+#0x0) } 
  2f4050:	0a 56 2a f1	f12a560a { 	r10 = or(r10,r22)
  2f4054:	06 c2 20 5c	5c20c206   	if (!p2) jump:nt 0x2f405c <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x35c> } 
  2f4058:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f405c:	02 42 e2 6b	6be24202 { 	p2 = or(p2,!p2)
  2f4060:	40 91 00 fb	fb009140   	if (p2) r0 = add(r0,r17)
  2f4064:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  2f4068:	03 40 59 75	75594003 { 	p3 = cmp.gt(r25,#0x0)
  2f406c:	10 5b 20 5c	5c205b10   	if (!p3.new) jump:t 0x2f4088 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x388>
  2f4070:	16 e0 60 7e	7e60e016   	if (p3.new) r22 = #0x0 } 
  2f4074:	0a 54 0a f1	f10a540a { 	r10 = and(r10,r20)
  2f4078:	16 40 80 91	91804016   	r22 = memw(r0+#0x0)
  2f407c:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f4080:	0a d6 2a f1	f12ad60a { 	r10 = or(r10,r22) } 
  2f4084:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f4088:	18 41 18 b0	b0184118 { 	r24 = add(r24,#0x8)
  2f408c:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f4090:	01 4f 01 f3	f3014f01 { 	r1 = add(r1,r15)
  2f4094:	00 50 00 f3	f3005000   	r0 = add(r0,r16)
  2f4098:	00 58 4c f2	f24c5800   	p0 = cmp.gt(r12,r24)
  2f409c:	88 e8 df 5c	5cdfe888   	if (p0.new) jump:nt 0x2f3fa0 <hmx_v73_convhbh1x1_stride1_unaligned_sparsity+0x2a0> } 
  2f40a0:	01 40 6d 70	706d4001 { 	r1 = r13
  2f40a4:	00 40 7a 70	707a4000   	r0 = r26
  2f40a8:	fc 7f fc bf	bffc7ffc   	r28 = add(r28,#-0x1)
  2f40ac:	15 f0 29 76	7629f015   	r21 = and(r9,#-0x80) } 
  2f40b0:	02 40 00 00	00004002 { 	immext(#0x80)
  2f40b4:	02 40 15 e2	e2154002   	r2 += add(r21,##0x80)
  2f40b8:	f5 ff 03 78	7803fff5   	r21 = #0x7ff } 
  2f40bc:	72 7f 7c 61	617c7f72 { 	if (r28>=#0) jump:t 0x2f3fa0
  2f40c0:	c3 49 13 8e	8e1349c3   	r3 += asl(r19,#0x9)
  2f40c4:	9a 40 1a b0	b01a409a   	r26 = add(r26,#0x4)
  2f40c8:	18 c1 00 78	7800c118   	r24 = #0x8 } 
  2f40cc:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2f40d0:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f40d4:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f40d8:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f40dc:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f40e0:	1d 48 1d b0	b01d481d { 	r29 = add(r29,#0x40)
  2f40e4:	00 40 9f 52	529f4000   	jumpr r31
  2f40e8:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f4100 <hmx_v73_convhbh1x1_stride1_unaligned>:
  2f4100:	1d 78 fd bf	bffd781d { 	r29 = add(r29,#-0x40)
  2f4104:	00 40 00 7f	7f004000   	nop
  2f4108:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f410c:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f4110:	c9 40 84 91	918440c9 { 	r9 = memw(r4+#0x18)
  2f4114:	9c c1 84 91	9184c19c   	r28 = memw(r4+#0x30) } 
  2f4118:	00 45 1c 85	851c4500 { 	p0 = tstbit(r28,#0x5)
  2f411c:	4f 40 81 91	9181404f   	r15 = memw(r1+#0x8)
  2f4120:	00 d0 dd a1	a1ddd000   	memd(r29+#0x0) = r17:16 } 
  2f4124:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2f4128:	f4 7f 03 78	78037ff4   	r20 = #0x7ff
  2f412c:	16 0a 0d ea	ea0d0a16   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2f4130:	12 40 65 70	70654012 { 	r18 = r5
  2f4134:	f5 7f 03 78	78037ff5   	r21 = #0x7ff
  2f4138:	65 40 81 91	91814065   	r5 = memw(r1+#0xc)
  2f413c:	8e c0 81 91	9181c08e   	r14 = memw(r1+#0x10) } 
  2f4140:	65 40 05 b0	b0054065 { 	r5 = add(r5,#0x3)
  2f4144:	1f 0a 14 c1	c1140a1f   	r4 = memw(r1+#0x4); 	memd(r29+#0x18) = r23:22 } 
  2f4148:	25 42 05 8c	8c054225 { 	r5 = lsr(r5,#0x2)
  2f414c:	18 41 00 78	78004118   	r24 = #0x8
  2f4150:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2f4154:	e4 7f e4 bf	bfe47fe4 { 	r4 = add(r4,#-0x1)
  2f4158:	76 50 00 78	78005076   	r22 = #0x83
  2f415c:	01 40 81 91	91814001   	r1 = memw(r1+#0x0)
  2f4160:	79 c0 80 91	9180c079   	r25 = memw(r0+#0xc) } 
  2f4164:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2f4168:	99 60 d9 74	74d96099   	if (!p2.new) r25 = add(r25,#0x4)
  2f416c:	09 01 08 02	02080109   	r16 = memw(r0+#0x8); 	r17 = memw(r0+#0x4) } 
  2f4170:	06 54 e6 d3	d3e65406 { 	r7:6 = and(r7:6,r21:20)
  2f4174:	0d 40 61 70	7061400d   	r13 = r1
  2f4178:	bc 40 80 91	918040bc   	r28 = memw(r0+#0x14)
  2f417c:	8c c0 80 91	9180c08c   	r12 = memw(r0+#0x10) } 
  2f4180:	59 42 05 8e	8e054259 { 	r25 -= asl(r5,#0x2)
  2f4184:	fc 43 1c b0	b01c43fc   	r28 = add(r28,#0x1f)
  2f4188:	00 40 80 91	91804000   	r0 = memw(r0+#0x0)
  2f418c:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f4190:	1c 45 1c 8c	8c1c451c { 	r28 = asr(r28,#0x5)
  2f4194:	0c 41 0c b0	b00c410c   	r12 = add(r12,#0x8)
  2f4198:	33 40 92 9b	9b924033   	r19 = memw(r18++#0x4)
  2f419c:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f41a0:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2f41a4:	9a 40 00 b0	b000409a   	r26 = add(r0,#0x4)
  2f41a8:	b0 40 00 5c	5c0040b0   	if (p0) jump:nt 0x2f4300 <hmx_v73_convhbh1x1_stride1_unaligned+0x200>
  2f41ac:	29 c0 09 74	7409c029   	if (p0) r9 = add(r9,#0x1) } 
  2f41b0:	fc 7f fc bf	bffc7ffc { 	r28 = add(r28,#-0x1)
  2f41b4:	1e 40 67 70	7067401e   	r30 = r7
  2f41b8:	ff ff 01 78	7801ffff   	r31 = #0x3ff } 
  2f41bc:	00 c0 00 7f	7f00c000 { 	nop } 
  2f41c0:	00 42 25 60	60254200 { 	loop1(0x2f41e0,r5)
  2f41c4:	17 40 00 78	78004017   	r23 = #0x0
  2f41c8:	76 50 00 78	78005076   	r22 = #0x83
  2f41cc:	00 c0 00 7f	7f00c000   	nop } 
  2f41d0:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2f41d4:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2f41d8:	01 58 4e f2	f24e5801   	p1 = cmp.gt(r14,r24)
  2f41dc:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f41e0:	00 42 04 60	60044200 { 	loop0(0x2f4200,r4)
  2f41e4:	07 4e 87 76	76874e07   	r7 = or(r7,#0x70)
  2f41e8:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f41ec:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f41f0:	37 57 36 fb	fb365737 { 	if (p1) r23 = sub(r23,r22)
  2f41f4:	08 40 c2 bf	bfc24008   	r8 = add(r2,#-0x400)
  2f41f8:	07 40 9e 74	749e4007   	if (!p0) r7 = add(r30,#0x0)
  2f41fc:	1a c0 20 5c	5c20c01a   	if (!p0) jump:nt 0x2f4224 <hmx_v73_convhbh1x1_stride1_unaligned+0x124> } 
  2f4200:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f4204:	08 40 48 b0	b0484008   	r8 = add(r8,#0x400)
  2f4208:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f420c:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f4210:	06 94 e6 d3	d3e69406 { 	r7:6 = and(r7:6,r21:20)
  2f4214:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f4218:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f421c:	e6 ff 08 92	9208ffe6   	weight.b = mxmem(r8,r31):single }  :endloop0
  2f4220:	07 c0 7e 70	707ec007 { 	r7 = r30 } 
  2f4224:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f4228:	08 40 48 b0	b0484008   	r8 = add(r8,#0x400)
  2f422c:	0a 54 0a f1	f10a540a   	r10 = and(r10,r20)
  2f4230:	17 c0 73 70	7073c017   	r23 = r19 } 
  2f4234:	28 40 09 e2	e2094028 { 	r8 += add(r9,#0x1)
  2f4238:	06 54 e6 d3	d3e65406   	r7:6 = and(r7:6,r21:20)
  2f423c:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f4240:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single } 
  2f4244:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4248:	f7 7f f7 bf	bff77ff7   	r23 = add(r23,#-0x1)
  2f424c:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2f4250:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4254:	03 40 57 75	75574003 { 	p3 = cmp.gt(r23,#0x0)
  2f4258:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f425c:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4260:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2f4264:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4268:	ee 63 df 5c	5cdf63ee { 	if (p3) jump:nt 0x2f4244 <hmx_v73_convhbh1x1_stride1_unaligned+0x144>
  2f426c:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f4270:	52 43 13 8e	8e134352 { 	r18 -= asl(r19,#0x3)
  2f4274:	43 49 13 8e	8e134943   	r3 -= asl(r19,#0x9)
  2f4278:	16 c0 80 91	9180c016   	r22 = memw(r0+#0x0) } 
  2f427c:	0a 56 2a f1	f12a560a { 	r10 = or(r10,r22)
  2f4280:	06 c2 20 5c	5c20c206   	if (!p2) jump:nt 0x2f4288 <hmx_v73_convhbh1x1_stride1_unaligned+0x188> } 
  2f4284:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f4288:	02 42 e2 6b	6be24202 { 	p2 = or(p2,!p2)
  2f428c:	40 91 00 fb	fb009140   	if (p2) r0 = add(r0,r17)
  2f4290:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  2f4294:	03 40 59 75	75594003 { 	p3 = cmp.gt(r25,#0x0)
  2f4298:	10 5b 20 5c	5c205b10   	if (!p3.new) jump:t 0x2f42b4 <hmx_v73_convhbh1x1_stride1_unaligned+0x1b4>
  2f429c:	16 e0 60 7e	7e60e016   	if (p3.new) r22 = #0x0 } 
  2f42a0:	0a 54 0a f1	f10a540a { 	r10 = and(r10,r20)
  2f42a4:	16 40 80 91	91804016   	r22 = memw(r0+#0x0)
  2f42a8:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f42ac:	0a d6 2a f1	f12ad60a { 	r10 = or(r10,r22) } 
  2f42b0:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f42b4:	18 41 18 b0	b0184118 { 	r24 = add(r24,#0x8)
  2f42b8:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f42bc:	01 4f 01 f3	f3014f01 { 	r1 = add(r1,r15)
  2f42c0:	00 50 00 f3	f3005000   	r0 = add(r0,r16)
  2f42c4:	00 58 4c f2	f24c5800   	p0 = cmp.gt(r12,r24)
  2f42c8:	82 e8 df 5c	5cdfe882   	if (p0.new) jump:nt 0x2f41c0 <hmx_v73_convhbh1x1_stride1_unaligned+0xc0> } 
  2f42cc:	01 40 6d 70	706d4001 { 	r1 = r13
  2f42d0:	00 40 7a 70	707a4000   	r0 = r26
  2f42d4:	fc 7f fc bf	bffc7ffc   	r28 = add(r28,#-0x1)
  2f42d8:	02 c0 68 70	7068c002   	r2 = r8 } 
  2f42dc:	72 7f 7c 61	617c7f72 { 	if (r28>=#0) jump:t 0x2f41c0
  2f42e0:	c3 49 13 8e	8e1349c3   	r3 += asl(r19,#0x9)
  2f42e4:	9a 40 1a b0	b01a409a   	r26 = add(r26,#0x4)
  2f42e8:	18 c1 00 78	7800c118   	r24 = #0x8 } 
  2f42ec:	aa c0 00 58	5800c0aa { 	jump 0x2f4440 <hmx_v73_convhbh1x1_stride1_unaligned+0x340> } 
  2f42f0:	00 40 00 7f	7f004000 { 	nop
  2f42f4:	00 40 00 7f	7f004000   	nop
  2f42f8:	00 40 00 7f	7f004000   	nop
  2f42fc:	00 c0 00 7f	7f00c000   	nop } 
  2f4300:	29 c1 09 8c	8c09c129 { 	r9 = lsr(r9,#0x1) } 
  2f4304:	1b 5c 09 ed	ed095c1b { 	r27 = mpyi(r9,r28)
  2f4308:	e9 7f e9 bf	bfe97fe9   	r9 = add(r9,#-0x1)
  2f430c:	fc ff fc bf	bffcfffc   	r28 = add(r28,#-0x1) } 
  2f4310:	00 40 00 7f	7f004000 { 	nop
  2f4314:	00 40 00 7f	7f004000   	nop
  2f4318:	00 40 00 7f	7f004000   	nop
  2f431c:	00 c0 00 7f	7f00c000   	nop } 
  2f4320:	00 42 25 60	60254200 { 	loop1(0x2f4340,r5)
  2f4324:	17 40 00 78	78004017   	r23 = #0x0
  2f4328:	76 50 00 78	78005076   	r22 = #0x83
  2f432c:	00 c0 00 7f	7f00c000   	nop } 
  2f4330:	02 56 8a c7	c78a5602 { 	p2 = bitsclr(r10,r22)
  2f4334:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2f4338:	01 58 4e f2	f24e5801   	p1 = cmp.gt(r14,r24)
  2f433c:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f4340:	00 42 04 60	60044200 { 	loop0(0x2f4360,r4)
  2f4344:	00 40 00 7f	7f004000   	nop
  2f4348:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f434c:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f4350:	06 54 e6 d3	d3e65406 { 	r7:6 = and(r7:6,r21:20)
  2f4354:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f4358:	08 42 3b f3	f33b4208   	r8 = sub(r2,r27)
  2f435c:	18 c0 20 5c	5c20c018   	if (!p0) jump:nt 0x2f4380 <hmx_v73_convhbh1x1_stride1_unaligned+0x280> } 
  2f4360:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f4364:	08 5b 08 f3	f3085b08   	r8 = add(r8,r27)
  2f4368:	36 40 81 9b	9b814036   	r22 = memw(r1++#0x4)
  2f436c:	37 cf 81 30	3081cf37   	if (p1) r23 = memw(r1+r15<<#0x0) } 
  2f4370:	06 94 e6 d3	d3e69406 { 	r7:6 = and(r7:6,r21:20)
  2f4374:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f4378:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f437c:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single }  :endloop0
  2f4380:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f4384:	08 5b 08 f3	f3085b08   	r8 = add(r8,r27)
  2f4388:	0a 54 0a f1	f10a540a   	r10 = and(r10,r20)
  2f438c:	17 c0 73 70	7073c017   	r23 = r19 } 
  2f4390:	f0 47 06 92	920647f0 { 	activation.ub = mxmem(r6,r7):single
  2f4394:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single } 
  2f4398:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f439c:	f7 7f f7 bf	bff77ff7   	r23 = add(r23,#-0x1)
  2f43a0:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2f43a4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f43a8:	03 40 57 75	75574003 { 	p3 = cmp.gt(r23,#0x0)
  2f43ac:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f43b0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f43b4:	36 40 92 9b	9b924036   	r22 = memw(r18++#0x4)
  2f43b8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f43bc:	ee 63 df 5c	5cdf63ee { 	if (p3) jump:nt 0x2f4398 <hmx_v73_convhbh1x1_stride1_unaligned+0x298>
  2f43c0:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f43c4:	52 43 13 8e	8e134352 { 	r18 -= asl(r19,#0x3)
  2f43c8:	43 49 13 8e	8e134943   	r3 -= asl(r19,#0x9)
  2f43cc:	16 c0 80 91	9180c016   	r22 = memw(r0+#0x0) } 
  2f43d0:	0a 56 2a f1	f12a560a { 	r10 = or(r10,r22)
  2f43d4:	06 c2 20 5c	5c20c206   	if (!p2) jump:nt 0x2f43dc <hmx_v73_convhbh1x1_stride1_unaligned+0x2dc> } 
  2f43d8:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f43dc:	02 42 e2 6b	6be24202 { 	p2 = or(p2,!p2)
  2f43e0:	40 91 00 fb	fb009140   	if (p2) r0 = add(r0,r17)
  2f43e4:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  2f43e8:	03 40 59 75	75594003 { 	p3 = cmp.gt(r25,#0x0)
  2f43ec:	10 5b 20 5c	5c205b10   	if (!p3.new) jump:t 0x2f4408 <hmx_v73_convhbh1x1_stride1_unaligned+0x308>
  2f43f0:	16 e0 60 7e	7e60e016   	if (p3.new) r22 = #0x0 } 
  2f43f4:	0a 54 0a f1	f10a540a { 	r10 = and(r10,r20)
  2f43f8:	16 40 80 91	91804016   	r22 = memw(r0+#0x0)
  2f43fc:	10 d9 f6 a6	a6f6d910   	cvt.uh = acc(r22):2x2 } 
  2f4400:	0a d6 2a f1	f12ad60a { 	r10 = or(r10,r22) } 
  2f4404:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f4408:	18 41 18 b0	b0184118 { 	r24 = add(r24,#0x8)
  2f440c:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f4410:	01 4f 01 f3	f3014f01 { 	r1 = add(r1,r15)
  2f4414:	00 50 00 f3	f3005000   	r0 = add(r0,r16)
  2f4418:	00 58 4c f2	f24c5800   	p0 = cmp.gt(r12,r24)
  2f441c:	88 e8 df 5c	5cdfe888   	if (p0.new) jump:nt 0x2f4320 <hmx_v73_convhbh1x1_stride1_unaligned+0x220> } 
  2f4420:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f4424:	01 40 6d 70	706d4001   	r1 = r13
  2f4428:	00 40 7a 70	707a4000   	r0 = r26
  2f442c:	fc ff fc bf	bffcfffc   	r28 = add(r28,#-0x1) } 
  2f4430:	78 7f 7c 61	617c7f78 { 	if (r28>=#0) jump:t 0x2f4320
  2f4434:	c3 49 13 8e	8e1349c3   	r3 += asl(r19,#0x9)
  2f4438:	9a 40 1a b0	b01a409a   	r26 = add(r26,#0x4)
  2f443c:	18 c1 00 78	7800c118   	r24 = #0x8 } 
  2f4440:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2f4444:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f4448:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f444c:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f4450:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f4454:	1d 48 1d b0	b01d481d { 	r29 = add(r29,#0x40)
  2f4458:	00 40 9f 52	529f4000   	jumpr r31
  2f445c:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f4480 <hmx_v85_convhbh1x1_stride1_unaligned>:
  2f4480:	c0 3f 00 5a	5a003fc0 { 	r0 = #-0x1; 	jumpr r31 } 
		...

002f44c0 <hmx_convhbh1x1deep_stride1>:
  2f44c0:	7c 70 03 78	7803707c { 	r28 = #0x783
  2f44c4:	00 40 00 7f	7f004000   	nop
  2f44c8:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f44cc:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f44d0:	00 5c 86 c7	c7865c00 { 	p0 = bitsclr(r6,r28)
  2f44d4:	00 5c 8a c7	c78a5c00   	p0 = bitsclr(r10,r28)
  2f44d8:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f44dc:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2f44e0:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2f44e4:	1d 7b fd bf	bffd7b1d   	r29 = add(r29,#-0x28)
  2f44e8:	e5 0b dc eb	ebdc0be5   	memd(r29+#-0x28) = r17:16; 	memd(r29+#-0x20) = r19:18 } 
  2f44ec:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2f44f0:	37 40 09 b0	b0094037   	r23 = add(r9,#0x1)
  2f44f4:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f44f8:	1c 40 61 70	7061401c { 	r28 = r1
  2f44fc:	20 40 44 75	75444020   	p0 = cmp.gt(r4,#0x1)
  2f4500:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f4504:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2f4508:	37 41 17 8c	8c174137   	r23 = lsr(r23,#0x1)
  2f450c:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2f4510:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2f4514:	25 41 04 8c	8c044125   	r5 = lsr(r4,#0x1)
  2f4518:	ad 40 80 91	918040ad   	r13 = memw(r0+#0x14)
  2f451c:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2f4520:	91 ff f1 bf	bff1ff91 { 	r17 = add(r17,#-0x4) } 
  2f4524:	06 40 31 62	62314006 { 	m0 = r17
  2f4528:	ed 43 0d b0	b00d43ed   	r13 = add(r13,#0x1f)
  2f452c:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2f4530:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f4534:	2d 45 0d 8c	8c0d452d { 	r13 = lsr(r13,#0x5)
  2f4538:	74 40 14 b0	b0144074   	r20 = add(r20,#0x3)
  2f453c:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2f4540:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f4544:	34 42 14 8c	8c144234 { 	r20 = lsr(r20,#0x2)
  2f4548:	12 40 01 f5	f5014012   	r19:18 = combine(r1,r0)
  2f454c:	6e c0 20 5c	5c20c06e   	if (!p0) jump:nt 0x2f4620 <hmx_convhbh1x1deep_stride1+0x160> } 
  2f4550:	00 43 05 60	60054300 { 	loop0(0x2f4580,r5)
  2f4554:	17 4d 17 ed	ed174d17   	r23 = mpyi(r23,r13)
  2f4558:	18 40 64 70	70644018   	r24 = r4
  2f455c:	16 ce 87 76	7687ce16   	r22 = or(r7,#0x70) } 
  2f4560:	0e 41 12 b0	b012410e { 	r14 = add(r18,#0x8)
  2f4564:	cd 7f ed bf	bfed7fcd   	r13 = add(r13,#-0x2)
  2f4568:	11 40 00 78	78004011   	r17 = #0x0
  2f456c:	00 c0 00 7f	7f00c000   	nop } 
  2f4570:	00 41 34 60	60344100 { 	loop1(0x2f4580,r20)
  2f4574:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f4578:	11 41 11 b0	b0114111   	r17 = add(r17,#0x8)
  2f457c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f4580:	40 40 18 75	75184040 { 	p0 = cmp.eq(r24,#0x2)
  2f4584:	d8 7f f8 bf	bff87fd8   	r24 = add(r24,#-0x2)
  2f4588:	46 40 81 9b	9b814046   	r6 = memw(r1++#0x8)
  2f458c:	35 c0 81 91	9181c035   	r21 = memw(r1+#0x4) } 
  2f4590:	08 57 08 f3	f3085708 { 	r8 = add(r8,r23)
  2f4594:	ec 56 06 92	920656ec   	activation.ub = mxmem(r6,r22)
  2f4598:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f459c:	08 97 08 f3	f3089708 { 	r8 = add(r8,r23)
  2f45a0:	ec 56 15 92	921556ec   	activation.ub = mxmem(r21,r22)
  2f45a4:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2f45a8:	10 5d 05 60	60055d10 { 	loop0(0x2f4580,r5)
  2f45ac:	16 4e 87 76	76874e16   	r22 = or(r7,#0x70)
  2f45b0:	0e 41 20 5c	5c20410e   	if (!p1) jump:nt 0x2f45c4 <hmx_convhbh1x1deep_stride1+0x104>
  2f45b4:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f45b8:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f45bc:	ec 47 06 92	920647ec { 	activation.ub = mxmem(r6,r7)
  2f45c0:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f45c4:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f45c8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f45cc:	0a eb ea a6	a6eaeb0a { 	mxmem(r10,r11):after:sat.uh = acc:2x1 } 
  2f45d0:	0a 40 80 9d	9d80400a { 	r10 = memw(r0++m0)
  2f45d4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f45d8:	08 40 62 70	70624008 { 	r8 = r2
  2f45dc:	03 a0 e3 bf	bfe3a003   	r3 = add(r3,#-0x100)
  2f45e0:	0a eb ea a6	a6eaeb0a   	mxmem(r10,r11):after:sat.uh = acc:2x1 }  :endloop1
  2f45e4:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f45e8:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f45ec:	00 51 4c f2	f24c5100   	p0 = cmp.gt(r12,r17)
  2f45f0:	c6 e8 df 5c	5cdfe8c6   	if (p0.new) jump:nt 0x2f4570 <hmx_convhbh1x1deep_stride1+0xb0> } 
  2f45f4:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f45f8:	12 4e 1c f5	f51c4e12   	r19:18 = combine(r28,r14)
  2f45fc:	03 c0 23 b0	b023c003   	r3 = add(r3,#0x200) } 
  2f4600:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f4604:	b0 f8 df 5c	5cdff8b0   	if (p0.new) jump:t 0x2f4560 <hmx_convhbh1x1deep_stride1+0xa0> } 
  2f4608:	44 c0 00 58	5800c044 { 	jump 0x2f4690 <hmx_convhbh1x1deep_stride1+0x1d0> } 
  2f460c:	00 c0 00 7f	7f00c000 { 	nop } 
  2f4610:	00 40 00 7f	7f004000 { 	nop
  2f4614:	00 40 00 7f	7f004000   	nop
  2f4618:	00 40 00 7f	7f004000   	nop
  2f461c:	00 c0 00 7f	7f00c000   	nop } 
  2f4620:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2f4624:	11 40 6c 70	706c4011   	r17 = r12
  2f4628:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2f462c:	00 c0 00 7f	7f00c000   	nop } 
  2f4630:	00 41 14 60	60144100 { 	loop0(0x2f4640,r20)
  2f4634:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f4638:	11 41 11 b0	b0114111   	r17 = add(r17,#0x8)
  2f463c:	00 c0 00 7f	7f00c000   	nop } 
  2f4640:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2f4644:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f4648:	ec 47 06 92	920647ec { 	activation.ub = mxmem(r6,r7)
  2f464c:	e8 e9 02 92	9202e9e8   	weight.b = mxmem(r2,r9):deep } 
  2f4650:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4654:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4658:	0a eb ea a6	a6eaeb0a { 	mxmem(r10,r11):after:sat.uh = acc:2x1 } 
  2f465c:	0a 40 80 9d	9d80400a { 	r10 = memw(r0++m0)
  2f4660:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4664:	03 a0 e3 bf	bfe3a003 { 	r3 = add(r3,#-0x100)
  2f4668:	0a eb ea a6	a6eaeb0a   	mxmem(r10,r11):after:sat.uh = acc:2x1 }  :endloop0
  2f466c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f4670:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f4674:	00 51 4c f2	f24c5100   	p0 = cmp.gt(r12,r17)
  2f4678:	e2 e8 df 5c	5cdfe8e2   	if (p0.new) jump:nt 0x2f4630 <hmx_convhbh1x1deep_stride1+0x170> } 
  2f467c:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f4680:	12 4e 1c f5	f51c4e12   	r19:18 = combine(r28,r14)
  2f4684:	03 c0 23 b0	b023c003   	r3 = add(r3,#0x200) } 
  2f4688:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f468c:	cc f8 df 5c	5cdff8cc   	if (p0.new) jump:t 0x2f4620 <hmx_convhbh1x1deep_stride1+0x160> } 
  2f4690:	00 40 4d 75	754d4000 { 	p0 = cmp.gt(r13,#0x0)
  2f4694:	36 58 20 5c	5c205836   	if (!p0.new) jump:t 0x2f46fc <hmx_convhbh1x1deep_stride1+0x23c>
  2f4698:	00 40 00 7f	7f004000   	nop
  2f469c:	00 c0 00 7f	7f00c000   	nop } 
  2f46a0:	00 42 04 60	60044200 { 	loop0(0x2f46c0,r4)
  2f46a4:	09 41 09 8c	8c094109   	r9 = asr(r9,#0x1)
  2f46a8:	11 40 6c 70	706c4011   	r17 = r12
  2f46ac:	00 c0 00 7f	7f00c000   	nop } 
  2f46b0:	00 41 34 60	60344100 { 	loop1(0x2f46c0,r20)
  2f46b4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f46b8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f46bc:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f46c0:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f46c4:	08 97 08 f3	f3089708 { 	r8 = add(r8,r23)
  2f46c8:	e0 47 06 92	920647e0   	activation.ub = mxmem(r6,r7):deep
  2f46cc:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2f46d0:	00 5f 04 60	60045f00 { 	loop0(0x2f46c0,r4)
  2f46d4:	08 40 62 70	70624008   	r8 = r2
  2f46d8:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f46dc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f46e0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f46e4:	03 60 e3 bf	bfe36003 { 	r3 = add(r3,#-0x100)
  2f46e8:	80 80 00 b0	b0008080   	r0 = add(r0,#0x4)
  2f46ec:	0a eb ea a6	a6eaeb0a   	mxmem(r10,r11):after:sat.uh = acc:2x1 }  :endloop1
  2f46f0:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f46f4:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f46f8:	e0 c0 b9 10	10b9c0e0   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f46b0 <hmx_convhbh1x1deep_stride1+0x1f0> } 
  2f46fc:	98 c0 dd 91	91ddc098 { 	r25:24 = memd(r29+#0x20) } 
  2f4700:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f4704:	1d 45 1d b0	b01d451d { 	r29 = add(r29,#0x28)
  2f4708:	00 40 9f 52	529f4000   	jumpr r31
  2f470c:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f4740 <hmx_v73_convhbh1x1deep_lp_stride1>:
  2f4740:	1d 7d fd bf	bffd7d1d { 	r29 = add(r29,#-0x18)
  2f4744:	ed 7f 03 78	78037fed   	r13 = #0x7ff
  2f4748:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f474c:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f4750:	06 4d 06 f1	f1064d06 { 	r6 = and(r6,r13)
  2f4754:	ac 40 80 91	918040ac   	r12 = memw(r0+#0x14)
  2f4758:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f475c:	ec 43 0c b0	b00c43ec { 	r12 = add(r12,#0x1f)
  2f4760:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f4764:	00 d0 dd a1	a1ddd000   	memd(r29+#0x0) = r17:16 } 
  2f4768:	3c 40 09 b0	b009403c { 	r28 = add(r9,#0x1)
  2f476c:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f4770:	21 40 44 75	75444021 { 	p1 = cmp.gt(r4,#0x1)
  2f4774:	e4 7f e4 bf	bfe47fe4   	r4 = add(r4,#-0x1)
  2f4778:	16 0a 0d ea	ea0d0a16   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2f477c:	2c 45 0c 8c	8c0c452c { 	r12 = lsr(r12,#0x5)
  2f4780:	3c 41 1c 8c	8c1c413c   	r28 = lsr(r28,#0x1)
  2f4784:	d0 c0 9d 91	919dc0d0   	r16 = memw(r29+#0x18) } 
  2f4788:	1c 4c 1c ed	ed1c4c1c { 	r28 = mpyi(r28,r12)
  2f478c:	06 50 26 f1	f1265006   	r6 = or(r6,r16)
  2f4790:	0f 40 61 70	7061400f   	r15 = r1
  2f4794:	34 c0 85 9b	9b85c034   	r20 = memw(r5++#0x4) } 
  2f4798:	00 40 00 7f	7f004000 { 	nop
  2f479c:	00 c0 00 7f	7f00c000   	nop } 
  2f47a0:	00 42 04 60	60044200 { 	loop0(0x2f47c0,r4)
  2f47a4:	0e 41 00 b0	b000410e   	r14 = add(r0,#0x8)
  2f47a8:	08 42 3c f3	f33c4208   	r8 = sub(r2,r28)
  2f47ac:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f47b0:	cc 7f ec bf	bfec7fcc { 	r12 = add(r12,#-0x2)
  2f47b4:	11 51 30 f3	f3305111   	r17 = sub(r17,r16)
  2f47b8:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f47bc:	16 c1 20 5c	5c20c116   	if (!p1) jump:nt 0x2f47dc <hmx_v73_convhbh1x1deep_lp_stride1+0x9c> } 
  2f47c0:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f47c4:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f47c8:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f47cc:	11 91 30 f3	f3309111 { 	r17 = sub(r17,r16)
  2f47d0:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f47d4:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f47d8:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2f47dc:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f47e0:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f47e4:	0a 4d 0a f1	f10a4d0a   	r10 = and(r10,r13)
  2f47e8:	32 c0 80 9b	9b80c032   	r18 = memw(r0++#0x4) } 
  2f47ec:	0a 52 2a f1	f12a520a { 	r10 = or(r10,r18)
  2f47f0:	13 40 74 70	70744013   	r19 = r20
  2f47f4:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f47f8:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f47fc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4800:	f3 7f f3 bf	bff37ff3   	r19 = add(r19,#-0x1)
  2f4804:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4808:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f480c:	00 40 53 75	75534000 { 	p0 = cmp.gt(r19,#0x0)
  2f4810:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4814:	10 d9 f5 a6	a6f5d910   	cvt.uh = acc(r21):2x2 } 
  2f4818:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f481c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4820:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f47fc <hmx_v73_convhbh1x1deep_lp_stride1+0xbc>
  2f4824:	10 d9 f5 a6	a6f5d910   	cvt.uh = acc(r21):2x2 } 
  2f4828:	45 43 14 8e	8e144345 { 	r5 -= asl(r20,#0x3)
  2f482c:	0a 4d 0a f1	f10a4d0a   	r10 = and(r10,r13)
  2f4830:	32 40 80 9b	9b804032   	r18 = memw(r0++#0x4)
  2f4834:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f4838:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f483c:	0a 52 2a f1	f12a520a   	r10 = or(r10,r18)
  2f4840:	13 40 74 70	70744013   	r19 = r20
  2f4844:	00 ce 0f f5	f50fce00   	r1:0 = combine(r15,r14) } 
  2f4848:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f484c:	f3 7f f3 bf	bff37ff3   	r19 = add(r19,#-0x1)
  2f4850:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4854:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4858:	00 40 53 75	75534000 { 	p0 = cmp.gt(r19,#0x0)
  2f485c:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4860:	10 d9 f5 a6	a6f5d910   	cvt.uh = acc(r21):2x2 } 
  2f4864:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4868:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f486c:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f4848 <hmx_v73_convhbh1x1deep_lp_stride1+0x108>
  2f4870:	10 d9 f5 a6	a6f5d910   	cvt.uh = acc(r21):2x2 } 
  2f4874:	45 43 14 8e	8e144345 { 	r5 -= asl(r20,#0x3)
  2f4878:	20 40 4c 75	754c4020   	p0 = cmp.gt(r12,#0x1)
  2f487c:	96 78 df 5c	5cdf7896   	if (p0.new) jump:t 0x2f47a0 <hmx_v73_convhbh1x1deep_lp_stride1+0x60>
  2f4880:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f4884:	00 40 4c 75	754c4000 { 	p0 = cmp.gt(r12,#0x0)
  2f4888:	54 58 20 5c	5c205854   	if (!p0.new) jump:t 0x2f492c <hmx_v73_convhbh1x1deep_lp_stride1+0x1ec>
  2f488c:	00 40 00 7f	7f004000   	nop
  2f4890:	00 c0 00 7f	7f00c000   	nop } 
  2f4894:	18 42 04 60	60044218 { 	loop0(0x2f48c0,r4)
  2f4898:	09 41 09 8c	8c094109   	r9 = asr(r9,#0x1)
  2f489c:	08 42 3c f3	f33c4208   	r8 = sub(r2,r28)
  2f48a0:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f48a4:	cc 7f ec bf	bfec7fcc { 	r12 = add(r12,#-0x2)
  2f48a8:	11 51 30 f3	f3305111   	r17 = sub(r17,r16)
  2f48ac:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f48b0:	1c c1 20 5c	5c20c11c   	if (!p1) jump:nt 0x2f48dc <hmx_v73_convhbh1x1deep_lp_stride1+0x19c> } 
  2f48b4:	00 40 00 7f	7f004000 { 	nop
  2f48b8:	00 40 00 7f	7f004000   	nop
  2f48bc:	00 c0 00 7f	7f00c000   	nop } 
  2f48c0:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f48c4:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f48c8:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f48cc:	11 91 30 f3	f3309111 { 	r17 = sub(r17,r16)
  2f48d0:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f48d4:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f48d8:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2f48dc:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f48e0:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f48e4:	0a 4d 0a f1	f10a4d0a   	r10 = and(r10,r13)
  2f48e8:	32 c0 80 9b	9b80c032   	r18 = memw(r0++#0x4) } 
  2f48ec:	0a 52 2a f1	f12a520a { 	r10 = or(r10,r18)
  2f48f0:	13 40 74 70	70744013   	r19 = r20
  2f48f4:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f48f8:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) } 
  2f48fc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4900:	f3 7f f3 bf	bff37ff3   	r19 = add(r19,#-0x1)
  2f4904:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4908:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f490c:	00 40 53 75	75534000 { 	p0 = cmp.gt(r19,#0x0)
  2f4910:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4914:	10 d9 f5 a6	a6f5d910   	cvt.uh = acc(r21):2x2 } 
  2f4918:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f491c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4920:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f48fc <hmx_v73_convhbh1x1deep_lp_stride1+0x1bc>
  2f4924:	10 d9 f5 a6	a6f5d910   	cvt.uh = acc(r21):2x2 } 
  2f4928:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f492c:	54 40 dd 91	91dd4054 { 	r21:20 = memd(r29+#0x10)
  2f4930:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f4934:	1d 43 1d b0	b01d431d { 	r29 = add(r29,#0x18)
  2f4938:	00 40 9f 52	529f4000   	jumpr r31
  2f493c:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 

002f4940 <hmx_v85_convhbh1x1deep_lp_stride1>:
  2f4940:	1d 7d fd bf	bffd7d1d { 	r29 = add(r29,#-0x18)
  2f4944:	ed 7f 03 78	78037fed   	r13 = #0x7ff
  2f4948:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f494c:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f4950:	06 4d 06 f1	f1064d06 { 	r6 = and(r6,r13)
  2f4954:	ac 40 80 91	918040ac   	r12 = memw(r0+#0x14)
  2f4958:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f495c:	ec 43 0c b0	b00c43ec { 	r12 = add(r12,#0x1f)
  2f4960:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f4964:	00 d0 dd a1	a1ddd000   	memd(r29+#0x0) = r17:16 } 
  2f4968:	3c 40 09 b0	b009403c { 	r28 = add(r9,#0x1)
  2f496c:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f4970:	21 40 44 75	75444021 { 	p1 = cmp.gt(r4,#0x1)
  2f4974:	e4 7f e4 bf	bfe47fe4   	r4 = add(r4,#-0x1)
  2f4978:	16 0a 0d ea	ea0d0a16   	memd(r29+#0x8) = r19:18; 	memd(r29+#0x10) = r21:20 } 
  2f497c:	2c 45 0c 8c	8c0c452c { 	r12 = lsr(r12,#0x5)
  2f4980:	3c 41 1c 8c	8c1c413c   	r28 = lsr(r28,#0x1)
  2f4984:	d0 c0 9d 91	919dc0d0   	r16 = memw(r29+#0x18) } 
  2f4988:	1c 4c 1c ed	ed1c4c1c { 	r28 = mpyi(r28,r12)
  2f498c:	06 50 26 f1	f1265006   	r6 = or(r6,r16)
  2f4990:	0f 40 61 70	7061400f   	r15 = r1
  2f4994:	34 c0 85 9b	9b85c034   	r20 = memw(r5++#0x4) } 
  2f4998:	00 40 00 7f	7f004000 { 	nop
  2f499c:	00 c0 00 7f	7f00c000   	nop } 
  2f49a0:	00 42 04 60	60044200 { 	loop0(0x2f49c0,r4)
  2f49a4:	0e 41 00 b0	b000410e   	r14 = add(r0,#0x8)
  2f49a8:	08 42 3c f3	f33c4208   	r8 = sub(r2,r28)
  2f49ac:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f49b0:	cc 7f ec bf	bfec7fcc { 	r12 = add(r12,#-0x2)
  2f49b4:	11 51 30 f3	f3305111   	r17 = sub(r17,r16)
  2f49b8:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f49bc:	16 c1 20 5c	5c20c116   	if (!p1) jump:nt 0x2f49dc <hmx_v85_convhbh1x1deep_lp_stride1+0x9c> } 
  2f49c0:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f49c4:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f49c8:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f49cc:	11 91 30 f3	f3309111 { 	r17 = sub(r17,r16)
  2f49d0:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f49d4:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f49d8:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2f49dc:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f49e0:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f49e4:	0a 4d 0a f1	f10a4d0a   	r10 = and(r10,r13)
  2f49e8:	32 c0 80 9b	9b80c032   	r18 = memw(r0++#0x4) } 
  2f49ec:	0a 52 2a f1	f12a520a { 	r10 = or(r10,r18)
  2f49f0:	13 40 74 70	70744013   	r19 = r20
  2f49f4:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f49f8:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f49fc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4a00:	f3 7f f3 bf	bff37ff3   	r19 = add(r19,#-0x1)
  2f4a04:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4a08:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4a0c:	f8 40 bb 10	10bb40f8 { 	p0 = cmp.gt(r19,#0x0); if (p0.new) jump:nt 0x2f49fc <hmx_v85_convhbh1x1deep_lp_stride1+0xbc>
  2f4a10:	10 d8 f5 a6	a6f5d810   	cvt.uh = acc(r21):2x1 } 
  2f4a14:	45 42 14 8e	8e144245 { 	r5 -= asl(r20,#0x2)
  2f4a18:	0a 4d 0a f1	f10a4d0a   	r10 = and(r10,r13)
  2f4a1c:	32 40 80 9b	9b804032   	r18 = memw(r0++#0x4)
  2f4a20:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f4a24:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f4a28:	0a 52 2a f1	f12a520a   	r10 = or(r10,r18)
  2f4a2c:	13 40 74 70	70744013   	r19 = r20
  2f4a30:	00 ce 0f f5	f50fce00   	r1:0 = combine(r15,r14) } 
  2f4a34:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4a38:	f3 7f f3 bf	bff37ff3   	r19 = add(r19,#-0x1)
  2f4a3c:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4a40:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4a44:	f8 40 bb 10	10bb40f8 { 	p0 = cmp.gt(r19,#0x0); if (p0.new) jump:nt 0x2f4a34 <hmx_v85_convhbh1x1deep_lp_stride1+0xf4>
  2f4a48:	10 d8 f5 a6	a6f5d810   	cvt.uh = acc(r21):2x1 } 
  2f4a4c:	45 42 14 8e	8e144245 { 	r5 -= asl(r20,#0x2)
  2f4a50:	20 40 4c 75	754c4020   	p0 = cmp.gt(r12,#0x1)
  2f4a54:	aa 78 df 5c	5cdf78aa   	if (p0.new) jump:t 0x2f49a0 <hmx_v85_convhbh1x1deep_lp_stride1+0x60>
  2f4a58:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f4a5c:	00 40 4c 75	754c4000 { 	p0 = cmp.gt(r12,#0x0)
  2f4a60:	4e 58 20 5c	5c20584e   	if (!p0.new) jump:t 0x2f4af8 <hmx_v85_convhbh1x1deep_lp_stride1+0x1b8>
  2f4a64:	00 40 00 7f	7f004000   	nop
  2f4a68:	00 c0 00 7f	7f00c000   	nop } 
  2f4a6c:	08 43 04 60	60044308 { 	loop0(0x2f4aa0,r4)
  2f4a70:	09 41 09 8c	8c094109   	r9 = asr(r9,#0x1)
  2f4a74:	08 42 3c f3	f33c4208   	r8 = sub(r2,r28)
  2f4a78:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f4a7c:	cc 7f ec bf	bfec7fcc { 	r12 = add(r12,#-0x2)
  2f4a80:	11 51 30 f3	f3305111   	r17 = sub(r17,r16)
  2f4a84:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f4a88:	20 c1 20 5c	5c20c120   	if (!p1) jump:nt 0x2f4abc <hmx_v85_convhbh1x1deep_lp_stride1+0x17c> } 
  2f4a8c:	00 c0 00 7f	7f00c000 { 	nop } 
  2f4a90:	00 40 00 7f	7f004000 { 	nop
  2f4a94:	00 40 00 7f	7f004000   	nop
  2f4a98:	00 40 00 7f	7f004000   	nop
  2f4a9c:	00 c0 00 7f	7f00c000   	nop } 
  2f4aa0:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f4aa4:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f4aa8:	31 c0 81 9b	9b81c031   	r17 = memw(r1++#0x4) } 
  2f4aac:	11 91 30 f3	f3309111 { 	r17 = sub(r17,r16)
  2f4ab0:	07 4d 07 f1	f1074d07   	r7 = and(r7,r13)
  2f4ab4:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f4ab8:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2f4abc:	07 51 27 f1	f1275107 { 	r7 = or(r7,r17)
  2f4ac0:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f4ac4:	0a 4d 0a f1	f10a4d0a   	r10 = and(r10,r13)
  2f4ac8:	32 c0 80 9b	9b80c032   	r18 = memw(r0++#0x4) } 
  2f4acc:	0a 52 2a f1	f12a520a { 	r10 = or(r10,r18)
  2f4ad0:	13 40 74 70	70744013   	r19 = r20
  2f4ad4:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f4ad8:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) } 
  2f4adc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4ae0:	f3 7f f3 bf	bff37ff3   	r19 = add(r19,#-0x1)
  2f4ae4:	35 40 85 9b	9b854035   	r21 = memw(r5++#0x4)
  2f4ae8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4aec:	f8 40 bb 10	10bb40f8 { 	p0 = cmp.gt(r19,#0x0); if (p0.new) jump:nt 0x2f4adc <hmx_v85_convhbh1x1deep_lp_stride1+0x19c>
  2f4af0:	10 d8 f5 a6	a6f5d810   	cvt.uh = acc(r21):2x1 } 
  2f4af4:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f4af8:	54 40 dd 91	91dd4054 { 	r21:20 = memd(r29+#0x10)
  2f4afc:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f4b00:	1d 43 1d b0	b01d431d { 	r29 = add(r29,#0x18)
  2f4b04:	00 40 9f 52	529f4000   	jumpr r31
  2f4b08:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f4b40 <hmx_v73_convhbh1x1deep_stride1_sparsity>:
  2f4b40:	1d 79 fd bf	bffd791d { 	r29 = add(r29,#-0x38)
  2f4b44:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f4b48:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f4b4c:	00 40 00 7f	7f004000 { 	nop
  2f4b50:	00 40 00 7f	7f004000   	nop
  2f4b54:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f4b58:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2f4b5c:	0c de 9d a1	a19dde0c { 	memw(r29+#0x30) = r30 } 
  2f4b60:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2f4b64:	00 40 00 7f	7f004000   	nop
  2f4b68:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f4b6c:	1e 70 29 76	7629701e { 	r30 = and(r9,#-0x80)
  2f4b70:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f4b74:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2f4b78:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f4b7c:	20 40 44 75	75444020 { 	p0 = cmp.gt(r4,#0x1)
  2f4b80:	04 58 dd a1	a1dd5804   	memd(r29+#0x20) = r25:24
  2f4b84:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f4b88:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2f4b8c:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2f4b90:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2f4b94:	3c 41 04 8c	8c04413c   	r28 = lsr(r4,#0x1)
  2f4b98:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2f4b9c:	ad c0 80 91	9180c0ad   	r13 = memw(r0+#0x14) } 
  2f4ba0:	16 70 29 76	76297016 { 	r22 = and(r9,#-0x80)
  2f4ba4:	1f 30 c9 27	27c9301f   	r17 = add(r17,#-0x4); 	r23 = r1 } 
  2f4ba8:	06 40 31 62	62314006 { 	m0 = r17
  2f4bac:	74 40 14 b0	b0144074   	r20 = add(r20,#0x3)
  2f4bb0:	ed 43 0d b0	b00d43ed   	r13 = add(r13,#0x1f)
  2f4bb4:	16 d0 16 b0	b016d016   	r22 = add(r22,#0x80) } 
  2f4bb8:	2d 45 0d 8c	8c0d452d { 	r13 = lsr(r13,#0x5)
  2f4bbc:	36 41 16 8c	8c164136   	r22 = lsr(r22,#0x1)
  2f4bc0:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2f4bc4:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f4bc8:	34 42 14 8c	8c144234 { 	r20 = lsr(r20,#0x2)
  2f4bcc:	12 40 01 f5	f5014012   	r19:18 = combine(r1,r0)
  2f4bd0:	9c 40 20 5c	5c20409c   	if (!p0) jump:nt 0x2f4d00 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x1c0>
  2f4bd4:	38 c0 85 9b	9b85c038   	r24 = memw(r5++#0x4) } 
  2f4bd8:	10 44 1c 60	601c4410 { 	loop0(0x2f4c20,r28)
  2f4bdc:	16 4d 16 ed	ed164d16   	r22 = mpyi(r22,r13)
  2f4be0:	00 40 00 7f	7f004000   	nop
  2f4be4:	00 c0 00 7f	7f00c000   	nop } 
  2f4be8:	00 40 00 7f	7f004000 { 	nop
  2f4bec:	00 c0 00 7f	7f00c000   	nop } 
  2f4bf0:	00 40 00 7f	7f004000 { 	nop
  2f4bf4:	00 40 00 7f	7f004000   	nop
  2f4bf8:	00 40 00 7f	7f004000   	nop
  2f4bfc:	00 c0 00 7f	7f00c000   	nop } 
  2f4c00:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2f4c04:	11 40 6c 70	706c4011   	r17 = r12
  2f4c08:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2f4c0c:	00 c0 00 7f	7f00c000   	nop } 
  2f4c10:	00 41 34 60	60344100 { 	loop1(0x2f4c20,r20)
  2f4c14:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f4c18:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f4c1c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f4c20:	46 40 81 9b	9b814046 { 	r6 = memw(r1++#0x8)
  2f4c24:	35 c0 81 91	9181c035   	r21 = memw(r1+#0x4) } 
  2f4c28:	08 56 08 f3	f3085608 { 	r8 = add(r8,r22)
  2f4c2c:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f4c30:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f4c34:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2f4c38:	ec 47 15 92	921547ec   	activation.ub = mxmem(r21,r7)
  2f4c3c:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2f4c40:	00 5e 1c 60	601c5e00 { 	loop0(0x2f4c20,r28)
  2f4c44:	1a 40 78 70	7078401a   	r26 = r24
  2f4c48:	0e 41 20 5c	5c20410e   	if (!p1) jump:nt 0x2f4c5c <hmx_v73_convhbh1x1deep_stride1_sparsity+0x11c>
  2f4c4c:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f4c50:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f4c54:	ec 47 06 92	920647ec { 	activation.ub = mxmem(r6,r7)
  2f4c58:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f4c5c:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4c60:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f4c64:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4c68:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4c6c:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f4c70:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4c74:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4c78:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4c7c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4c80:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f4c5c <hmx_v73_convhbh1x1deep_stride1_sparsity+0x11c>
  2f4c84:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4c88:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f4c8c:	1a 40 78 70	7078401a   	r26 = r24
  2f4c90:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2f4c94:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f4c98:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4c9c:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f4ca0:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4ca4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4ca8:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f4cac:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4cb0:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4cb4:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4cb8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4cbc:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f4c98 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x158>
  2f4cc0:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4cc4:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f4cc8:	43 8a 18 8e	8e188a43   	r3 -= asl(r24,#0xa)
  2f4ccc:	08 40 62 70	70624008   	r8 = r2
  2f4cd0:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f4cd4:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f4cd8:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f4cdc:	9e c0 b9 10	10b9c09e   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f4c10 <hmx_v73_convhbh1x1deep_stride1_sparsity+0xd0> } 
  2f4ce0:	c3 4a 18 8e	8e184ac3 { 	r3 += asl(r24,#0xa)
  2f4ce4:	02 40 00 00	00004002   	immext(#0x80)
  2f4ce8:	02 40 1e e2	e21e4002   	r2 += add(r30,##0x80)
  2f4cec:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2f4cf0:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f4cf4:	88 f8 df 5c	5cdff888   	if (p0.new) jump:t 0x2f4c00 <hmx_v73_convhbh1x1deep_stride1_sparsity+0xc0> } 
  2f4cf8:	6a c0 00 58	5800c06a { 	jump 0x2f4dcc <hmx_v73_convhbh1x1deep_stride1_sparsity+0x28c> } 
  2f4cfc:	00 c0 00 7f	7f00c000 { 	nop } 
  2f4d00:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2f4d04:	11 40 6c 70	706c4011   	r17 = r12
  2f4d08:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2f4d0c:	00 c0 00 7f	7f00c000   	nop } 
  2f4d10:	00 41 14 60	60144100 { 	loop0(0x2f4d20,r20)
  2f4d14:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f4d18:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f4d1c:	00 c0 00 7f	7f00c000   	nop } 
  2f4d20:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2f4d24:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f4d28:	1a 40 78 70	7078401a { 	r26 = r24
  2f4d2c:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f4d30:	e8 e9 02 92	9202e9e8   	weight.b = mxmem(r2,r9):deep } 
  2f4d34:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4d38:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f4d3c:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4d40:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4d44:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f4d48:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4d4c:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4d50:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4d54:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4d58:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f4d34 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x1f4>
  2f4d5c:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4d60:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f4d64:	1a 40 78 70	7078401a   	r26 = r24
  2f4d68:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2f4d6c:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f4d70:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4d74:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f4d78:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4d7c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4d80:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f4d84:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4d88:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4d8c:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4d90:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4d94:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f4d70 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x230>
  2f4d98:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4d9c:	45 83 18 8e	8e188345 { 	r5 -= asl(r24,#0x3)
  2f4da0:	43 4a 18 8e	8e184a43   	r3 -= asl(r24,#0xa)
  2f4da4:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop0
  2f4da8:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f4dac:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f4db0:	b4 c0 b9 10	10b9c0b4   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f4d10 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x1d0> } 
  2f4db4:	02 40 00 00	00004002 { 	immext(#0x80)
  2f4db8:	02 40 1e e2	e21e4002   	r2 += add(r30,##0x80)
  2f4dbc:	c3 4a 18 8e	8e184ac3   	r3 += asl(r24,#0xa)
  2f4dc0:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2f4dc4:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f4dc8:	9e f8 df 5c	5cdff89e   	if (p0.new) jump:t 0x2f4d00 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x1c0> } 
  2f4dcc:	00 40 4d 75	754d4000 { 	p0 = cmp.gt(r13,#0x0)
  2f4dd0:	5e 58 20 5c	5c20585e   	if (!p0.new) jump:t 0x2f4e88 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x348>
  2f4dd4:	fe 4f 09 76	76094ffe   	r30 = and(r9,#0x7f)
  2f4dd8:	09 f0 29 76	7629f009   	r9 = and(r9,#-0x80) } 
  2f4ddc:	09 d0 09 b0	b009d009 { 	r9 = add(r9,#0x80) } 
  2f4de0:	09 c1 09 8c	8c09c109 { 	r9 = asr(r9,#0x1) } 
  2f4de4:	09 f0 e9 bf	bfe9f009 { 	r9 = add(r9,#-0x80) } 
  2f4de8:	10 43 04 60	60044310 { 	loop0(0x2f4e20,r4)
  2f4dec:	09 5e 29 f1	f1295e09   	r9 = or(r9,r30)
  2f4df0:	1e 70 29 76	7629701e   	r30 = and(r9,#-0x80)
  2f4df4:	11 c0 6c 70	706cc011   	r17 = r12 } 
  2f4df8:	10 42 34 60	60344210 { 	loop1(0x2f4e20,r20)
  2f4dfc:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f4e00:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f4e04:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f4e08:	00 40 00 7f	7f004000 { 	nop
  2f4e0c:	00 c0 00 7f	7f00c000   	nop } 
  2f4e10:	00 40 00 7f	7f004000 { 	nop
  2f4e14:	00 40 00 7f	7f004000   	nop
  2f4e18:	00 40 00 7f	7f004000   	nop
  2f4e1c:	00 c0 00 7f	7f00c000   	nop } 
  2f4e20:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f4e24:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2f4e28:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f4e2c:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2f4e30:	00 5f 04 60	60045f00 { 	loop0(0x2f4e20,r4)
  2f4e34:	08 40 62 70	70624008   	r8 = r2
  2f4e38:	1a 40 78 70	7078401a   	r26 = r24
  2f4e3c:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f4e40:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4e44:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f4e48:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4e4c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4e50:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f4e54:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4e58:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4e5c:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4e60:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4e64:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f4e40 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x300>
  2f4e68:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4e6c:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f4e70:	43 89 18 8e	8e188943   	r3 -= asl(r24,#0x9)
  2f4e74:	80 40 00 b0	b0004080   	r0 = add(r0,#0x4)
  2f4e78:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f4e7c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f4e80:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f4e84:	be c0 b9 10	10b9c0be   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f4df8 <hmx_v73_convhbh1x1deep_stride1_sparsity+0x2b8> } 
  2f4e88:	9e 41 9d 91	919d419e { 	r30 = memw(r29+#0x30)
  2f4e8c:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f4e90:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f4e94:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f4e98:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f4e9c:	1d 47 1d b0	b01d471d { 	r29 = add(r29,#0x38)
  2f4ea0:	00 40 9f 52	529f4000   	jumpr r31
  2f4ea4:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f4ec0 <hmx_v73_convhbh1x1deep_stride1>:
  2f4ec0:	1d 7a fd bf	bffd7a1d { 	r29 = add(r29,#-0x30)
  2f4ec4:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f4ec8:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f4ecc:	00 40 00 7f	7f004000 { 	nop
  2f4ed0:	00 40 00 7f	7f004000   	nop
  2f4ed4:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f4ed8:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2f4edc:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2f4ee0:	00 40 00 7f	7f004000   	nop
  2f4ee4:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f4ee8:	00 40 00 7f	7f004000 { 	nop
  2f4eec:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f4ef0:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2f4ef4:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f4ef8:	20 40 44 75	75444020 { 	p0 = cmp.gt(r4,#0x1)
  2f4efc:	04 58 dd a1	a1dd5804   	memd(r29+#0x20) = r25:24
  2f4f00:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f4f04:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2f4f08:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2f4f0c:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2f4f10:	3c 41 04 8c	8c04413c   	r28 = lsr(r4,#0x1)
  2f4f14:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2f4f18:	ad c0 80 91	9180c0ad   	r13 = memw(r0+#0x14) } 
  2f4f1c:	1f 30 c9 27	27c9301f { 	r17 = add(r17,#-0x4); 	r23 = r1 } 
  2f4f20:	06 40 31 62	62314006 { 	m0 = r17
  2f4f24:	74 40 14 b0	b0144074   	r20 = add(r20,#0x3)
  2f4f28:	ed 43 0d b0	b00d43ed   	r13 = add(r13,#0x1f)
  2f4f2c:	36 c0 09 b0	b009c036   	r22 = add(r9,#0x1) } 
  2f4f30:	2d 45 0d 8c	8c0d452d { 	r13 = lsr(r13,#0x5)
  2f4f34:	36 41 16 8c	8c164136   	r22 = lsr(r22,#0x1)
  2f4f38:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2f4f3c:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f4f40:	34 42 14 8c	8c144234 { 	r20 = lsr(r20,#0x2)
  2f4f44:	12 40 01 f5	f5014012   	r19:18 = combine(r1,r0)
  2f4f48:	90 40 20 5c	5c204090   	if (!p0) jump:nt 0x2f5060 <hmx_v73_convhbh1x1deep_stride1+0x1a0>
  2f4f4c:	38 c0 85 9b	9b85c038   	r24 = memw(r5++#0x4) } 
  2f4f50:	00 43 1c 60	601c4300 { 	loop0(0x2f4f80,r28)
  2f4f54:	16 4d 16 ed	ed164d16   	r22 = mpyi(r22,r13)
  2f4f58:	00 40 00 7f	7f004000   	nop
  2f4f5c:	00 c0 00 7f	7f00c000   	nop } 
  2f4f60:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2f4f64:	11 40 6c 70	706c4011   	r17 = r12
  2f4f68:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2f4f6c:	00 c0 00 7f	7f00c000   	nop } 
  2f4f70:	00 41 34 60	60344100 { 	loop1(0x2f4f80,r20)
  2f4f74:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f4f78:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f4f7c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f4f80:	46 40 81 9b	9b814046 { 	r6 = memw(r1++#0x8)
  2f4f84:	35 c0 81 91	9181c035   	r21 = memw(r1+#0x4) } 
  2f4f88:	08 56 08 f3	f3085608 { 	r8 = add(r8,r22)
  2f4f8c:	e0 47 06 92	920647e0   	activation.ub = mxmem(r6,r7):deep
  2f4f90:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f4f94:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2f4f98:	e0 47 15 92	921547e0   	activation.ub = mxmem(r21,r7):deep
  2f4f9c:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2f4fa0:	00 5e 1c 60	601c5e00 { 	loop0(0x2f4f80,r28)
  2f4fa4:	1a 40 78 70	7078401a   	r26 = r24
  2f4fa8:	0e 41 20 5c	5c20410e   	if (!p1) jump:nt 0x2f4fbc <hmx_v73_convhbh1x1deep_stride1+0xfc>
  2f4fac:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f4fb0:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f4fb4:	e0 47 06 92	920647e0 { 	activation.ub = mxmem(r6,r7):deep
  2f4fb8:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f4fbc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4fc0:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f4fc4:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4fc8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4fcc:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f4fd0:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f4fd4:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4fd8:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4fdc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f4fe0:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f4fbc <hmx_v73_convhbh1x1deep_stride1+0xfc>
  2f4fe4:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f4fe8:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f4fec:	1a 40 78 70	7078401a   	r26 = r24
  2f4ff0:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2f4ff4:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f4ff8:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f4ffc:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f5000:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5004:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5008:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f500c:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5010:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5014:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5018:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f501c:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f4ff8 <hmx_v73_convhbh1x1deep_stride1+0x138>
  2f5020:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5024:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f5028:	43 8a 18 8e	8e188a43   	r3 -= asl(r24,#0xa)
  2f502c:	08 40 62 70	70624008   	r8 = r2
  2f5030:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f5034:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f5038:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f503c:	9e c0 b9 10	10b9c09e   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f4f70 <hmx_v73_convhbh1x1deep_stride1+0xb0> } 
  2f5040:	c3 4a 18 8e	8e184ac3 { 	r3 += asl(r24,#0xa)
  2f5044:	22 40 09 e2	e2094022   	r2 += add(r9,#0x1)
  2f5048:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2f504c:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f5050:	8a f8 df 5c	5cdff88a   	if (p0.new) jump:t 0x2f4f60 <hmx_v73_convhbh1x1deep_stride1+0xa0> } 
  2f5054:	6a c0 00 58	5800c06a { 	jump 0x2f5128 <hmx_v73_convhbh1x1deep_stride1+0x268> } 
  2f5058:	00 40 00 7f	7f004000 { 	nop
  2f505c:	00 c0 00 7f	7f00c000   	nop } 
  2f5060:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2f5064:	11 40 6c 70	706c4011   	r17 = r12
  2f5068:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2f506c:	00 c0 00 7f	7f00c000   	nop } 
  2f5070:	00 41 14 60	60144100 { 	loop0(0x2f5080,r20)
  2f5074:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f5078:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f507c:	00 c0 00 7f	7f00c000   	nop } 
  2f5080:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2f5084:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f5088:	1a 40 78 70	7078401a { 	r26 = r24
  2f508c:	e0 47 06 92	920647e0   	activation.ub = mxmem(r6,r7):deep
  2f5090:	e8 e9 02 92	9202e9e8   	weight.b = mxmem(r2,r9):deep } 
  2f5094:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5098:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f509c:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f50a0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f50a4:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f50a8:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f50ac:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f50b0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f50b4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f50b8:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f5094 <hmx_v73_convhbh1x1deep_stride1+0x1d4>
  2f50bc:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f50c0:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f50c4:	1a 40 78 70	7078401a   	r26 = r24
  2f50c8:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2f50cc:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f50d0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f50d4:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f50d8:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f50dc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f50e0:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f50e4:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f50e8:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f50ec:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f50f0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f50f4:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f50d0 <hmx_v73_convhbh1x1deep_stride1+0x210>
  2f50f8:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f50fc:	45 83 18 8e	8e188345 { 	r5 -= asl(r24,#0x3)
  2f5100:	43 4a 18 8e	8e184a43   	r3 -= asl(r24,#0xa)
  2f5104:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop0
  2f5108:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f510c:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f5110:	b4 c0 b9 10	10b9c0b4   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f5070 <hmx_v73_convhbh1x1deep_stride1+0x1b0> } 
  2f5114:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f5118:	c3 4a 18 8e	8e184ac3   	r3 += asl(r24,#0xa)
  2f511c:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2f5120:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f5124:	a0 f8 df 5c	5cdff8a0   	if (p0.new) jump:t 0x2f5060 <hmx_v73_convhbh1x1deep_stride1+0x1a0> } 
  2f5128:	00 40 4d 75	754d4000 { 	p0 = cmp.gt(r13,#0x0)
  2f512c:	50 58 20 5c	5c205850   	if (!p0.new) jump:t 0x2f51c8 <hmx_v73_convhbh1x1deep_stride1+0x308>
  2f5130:	00 40 00 7f	7f004000   	nop
  2f5134:	00 c0 00 7f	7f00c000   	nop } 
  2f5138:	10 42 04 60	60044210 { 	loop0(0x2f5160,r4)
  2f513c:	09 41 09 8c	8c094109   	r9 = asr(r9,#0x1)
  2f5140:	11 40 6c 70	706c4011   	r17 = r12
  2f5144:	00 c0 00 7f	7f00c000   	nop } 
  2f5148:	10 41 34 60	60344110 { 	loop1(0x2f5160,r20)
  2f514c:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f5150:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f5154:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f5158:	00 40 00 7f	7f004000 { 	nop
  2f515c:	00 c0 00 7f	7f00c000   	nop } 
  2f5160:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f5164:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2f5168:	e0 47 06 92	920647e0   	activation.ub = mxmem(r6,r7):deep
  2f516c:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2f5170:	00 5f 04 60	60045f00 { 	loop0(0x2f5160,r4)
  2f5174:	08 40 62 70	70624008   	r8 = r2
  2f5178:	1a 40 78 70	7078401a   	r26 = r24
  2f517c:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f5180:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5184:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f5188:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f518c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5190:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f5194:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5198:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f519c:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f51a0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f51a4:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f5180 <hmx_v73_convhbh1x1deep_stride1+0x2c0>
  2f51a8:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f51ac:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f51b0:	43 89 18 8e	8e188943   	r3 -= asl(r24,#0x9)
  2f51b4:	80 40 00 b0	b0004080   	r0 = add(r0,#0x4)
  2f51b8:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f51bc:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f51c0:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f51c4:	c6 c0 b9 10	10b9c0c6   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f5148 <hmx_v73_convhbh1x1deep_stride1+0x288> } 
  2f51c8:	11 c0 e0 a6	a6e0c011 { 	mxclracc } 
  2f51cc:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f51d0:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f51d4:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f51d8:	1d 46 1d b0	b01d461d { 	r29 = add(r29,#0x30)
  2f51dc:	00 40 9f 52	529f4000   	jumpr r31
  2f51e0:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...
