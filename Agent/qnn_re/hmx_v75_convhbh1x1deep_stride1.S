
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

002f5200 <hmx_v75_convhbh1x1deep_stride1>:
  2f5200:	1d 7a fd bf	bffd7a1d { 	r29 = add(r29,#-0x30)
  2f5204:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f5208:	0a c0 c4 91	91c4c00a   	r11:10 = memd(r4+#0x0) } 
  2f520c:	00 40 00 7f	7f004000 { 	nop
  2f5210:	c9 40 84 91	918440c9   	r9 = memw(r4+#0x18)
  2f5214:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2f5218:	4f 42 0f 8c	8c0f424f { 	r15 = asl(r15,#0x2)
  2f521c:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f5220:	1f 0a 16 ea	ea160a1f { 	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f5224:	07 4b 07 8d	8d074b07 { 	r7 = extractu(r7,#0xb,#0x0)
  2f5228:	11 00 14 01	01140011   	r4 = memw(r1+#0x4); 	r1 = memw(r1+#0x0) } 
  2f522c:	20 40 44 75	75444020 { 	p0 = cmp.gt(r4,#0x1)
  2f5230:	04 58 dd a1	a1dd5804   	memd(r29+#0x20) = r25:24
  2f5234:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f5238:	01 40 04 85	85044001 { 	p1 = tstbit(r4,#0x0)
  2f523c:	08 02 09 01	01090208   	r17 = memw(r0+#0x4); 	r16 = memw(r0+#0x8) } 
  2f5240:	50 42 50 80	80504250 { 	r17:16 = vaslw(r17:16,#0x2)
  2f5244:	3c 41 04 8c	8c04413c   	r28 = lsr(r4,#0x1)
  2f5248:	74 40 80 91	91804074   	r20 = memw(r0+#0xc)
  2f524c:	ad c0 80 91	9180c0ad   	r13 = memw(r0+#0x14) } 
  2f5250:	91 7f f1 bf	bff17f91 { 	r17 = add(r17,#-0x4)
  2f5254:	17 40 61 70	70614017   	r23 = r1
  2f5258:	38 c0 85 9b	9b85c038   	r24 = memw(r5++#0x4) } 
  2f525c:	06 40 31 62	62314006 { 	m0 = r17
  2f5260:	74 40 14 b0	b0144074   	r20 = add(r20,#0x3)
  2f5264:	ed 43 0d b0	b00d43ed   	r13 = add(r13,#0x1f)
  2f5268:	36 c0 09 b0	b009c036   	r22 = add(r9,#0x1) } 
  2f526c:	2d 45 0d 8c	8c0d452d { 	r13 = lsr(r13,#0x5)
  2f5270:	36 41 16 8c	8c164136   	r22 = lsr(r22,#0x1)
  2f5274:	8c 40 80 91	9180408c   	r12 = memw(r0+#0x10)
  2f5278:	00 c0 80 91	9180c000   	r0 = memw(r0+#0x0) } 
  2f527c:	34 42 14 8c	8c144234 { 	r20 = lsr(r20,#0x2)
  2f5280:	12 40 01 f5	f5014012   	r19:18 = combine(r1,r0)
  2f5284:	23 40 18 75	75184023   	p3 = cmp.eq(r24,#0x1)
  2f5288:	82 c0 20 5c	5c20c082   	if (!p0) jump:nt 0x2f5380 <hmx_v75_convhbh1x1deep_stride1+0x180> } 
  2f528c:	08 45 1c 60	601c4508 { 	loop0(0x2f52e0,r28)
  2f5290:	16 4d 16 ed	ed164d16   	r22 = mpyi(r22,r13)
  2f5294:	1a e3 20 5c	5c20e31a   	if (!p3) jump:nt 0x2f54c0 <hmx_v75_convhbh1x1deep_stride1+0x2c0> } 
  2f5298:	11 40 6c 70	706c4011 { 	r17 = r12
  2f529c:	0e c1 12 b0	b012c10e   	r14 = add(r18,#0x8) } 
  2f52a0:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f52a4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f52a8:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f52ac:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f52b0:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f52b4:	00 40 00 7f	7f004000   	nop
  2f52b8:	00 40 00 7f	7f004000   	nop
  2f52bc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f52c0:	a3 5f 03 b0	b0035fa3 { 	r3 = add(r3,#0xfd)
  2f52c4:	00 40 00 7f	7f004000   	nop
  2f52c8:	00 40 00 7f	7f004000   	nop
  2f52cc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f52d0:	00 41 34 60	60344100 { 	loop1(0x2f52e0,r20)
  2f52d4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f52d8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f52dc:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f52e0:	46 40 81 9b	9b814046 { 	r6 = memw(r1++#0x8)
  2f52e4:	35 c0 81 91	9181c035   	r21 = memw(r1+#0x4) } 
  2f52e8:	08 56 08 f3	f3085608 { 	r8 = add(r8,r22)
  2f52ec:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f52f0:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f52f4:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2f52f8:	ec 47 15 92	921547ec   	activation.ub = mxmem(r21,r7)
  2f52fc:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2f5300:	00 5e 1c 60	601c5e00 { 	loop0(0x2f52e0,r28)
  2f5304:	39 40 02 78	78024039   	r25 = #0x401
  2f5308:	0e 41 20 5c	5c20410e   	if (!p1) jump:nt 0x2f531c <hmx_v75_convhbh1x1deep_stride1+0x11c>
  2f530c:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f5310:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f5314:	ec 47 06 92	920647ec { 	activation.ub = mxmem(r6,r7)
  2f5318:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f531c:	99 41 09 78	78094199 { 	r25 = #0x120c
  2f5320:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5324:	39 40 12 78	78124039 { 	r25 = #0x2401
  2f5328:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f532c:	0a 40 80 9d	9d80400a { 	r10 = memw(r0++m0)
  2f5330:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f5334:	99 41 19 78	78194199 { 	r25 = #0x320c
  2f5338:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f533c:	10 d9 f9 a6	a6f9d910 { 	cvt.uh = acc(r25):2x2 } 
  2f5340:	08 40 62 70	70624008 { 	r8 = r2
  2f5344:	00 80 00 7f	7f008000   	nop
  2f5348:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f534c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f5350:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f5354:	c2 c0 b9 10	10b9c0c2   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f52d0 <hmx_v75_convhbh1x1deep_stride1+0xd0> } 
  2f5358:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f535c:	12 4e 17 f5	f5174e12   	r19:18 = combine(r23,r14)
  2f5360:	cd 7f ed bf	bfed7fcd   	r13 = add(r13,#-0x2)
  2f5364:	11 c0 6c 70	706cc011   	r17 = r12 } 
  2f5368:	0e 41 12 b0	b012410e { 	r14 = add(r18,#0x8)
  2f536c:	20 40 4d 75	754d4020   	p0 = cmp.gt(r13,#0x1)
  2f5370:	9c f8 df 5c	5cdff89c   	if (p0.new) jump:t 0x2f52a0 <hmx_v75_convhbh1x1deep_stride1+0xa0> } 
  2f5374:	54 c0 00 58	5800c054 { 	jump 0x2f541c <hmx_v75_convhbh1x1deep_stride1+0x21c> } 
  2f5378:	00 40 00 7f	7f004000 { 	nop
  2f537c:	00 c0 00 7f	7f00c000   	nop } 
  2f5380:	20 63 20 5c	5c206320 { 	if (!p3) jump:nt 0x2f55c0 <hmx_v75_convhbh1x1deep_stride1+0x3c0>
  2f5384:	11 c0 6c 70	706cc011   	r17 = r12 } 
  2f5388:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f538c:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2f5390:	cd 7f ed bf	bfed7fcd   	r13 = add(r13,#-0x2)
  2f5394:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5398:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f539c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f53a0:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f53a4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f53a8:	a3 5f 03 b0	b0035fa3 { 	r3 = add(r3,#0xfd)
  2f53ac:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f53b0:	00 41 14 60	60144100 { 	loop0(0x2f53c0,r20)
  2f53b4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f53b8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f53bc:	00 c0 00 7f	7f00c000   	nop } 
  2f53c0:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2f53c4:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f53c8:	39 40 02 78	78024039 { 	r25 = #0x401
  2f53cc:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f53d0:	e8 e9 02 92	9202e9e8   	weight.b = mxmem(r2,r9):deep } 
  2f53d4:	99 41 09 78	78094199 { 	r25 = #0x120c
  2f53d8:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f53dc:	39 40 12 78	78124039 { 	r25 = #0x2401
  2f53e0:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f53e4:	0a 40 80 9d	9d80400a { 	r10 = memw(r0++m0)
  2f53e8:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f53ec:	99 41 19 78	78194199 { 	r25 = #0x320c
  2f53f0:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f53f4:	10 d9 f9 a6	a6f9d910 { 	cvt.uh = acc(r25):2x2 } 
  2f53f8:	00 80 00 7f	7f008000 { 	nop
  2f53fc:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop0
  2f5400:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f5404:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f5408:	d8 c0 b9 10	10b9c0d8   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f53b0 <hmx_v75_convhbh1x1deep_stride1+0x1b0> } 
  2f540c:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f5410:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2f5414:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f5418:	b6 f8 df 5c	5cdff8b6   	if (p0.new) jump:t 0x2f5380 <hmx_v75_convhbh1x1deep_stride1+0x180> } 
  2f541c:	00 40 4d 75	754d4000 { 	p0 = cmp.gt(r13,#0x0)
  2f5420:	86 78 20 5c	5c207886   	if (!p0.new) jump:t 0x2f5728 <hmx_v75_convhbh1x1deep_stride1+0x528>
  2f5424:	00 40 00 7f	7f004000   	nop
  2f5428:	00 c0 00 7f	7f00c000   	nop } 
  2f542c:	08 43 04 60	60044308 { 	loop0(0x2f5460,r4)
  2f5430:	09 41 09 8c	8c094109   	r9 = asr(r9,#0x1)
  2f5434:	11 c0 6c 70	706cc011   	r17 = r12 } 
  2f5438:	23 60 03 b0	b0036023 { 	r3 = add(r3,#0x101)
  2f543c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5440:	e3 5f 03 b0	b0035fe3 { 	r3 = add(r3,#0xff)
  2f5444:	00 40 00 7f	7f004000   	nop
  2f5448:	00 40 00 7f	7f004000   	nop
  2f544c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5450:	00 41 34 60	60344100 { 	loop1(0x2f5460,r20)
  2f5454:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f5458:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f545c:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f5460:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f5464:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2f5468:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f546c:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2f5470:	00 5f 04 60	60045f00 { 	loop0(0x2f5460,r4)
  2f5474:	08 40 62 70	70624008   	r8 = r2
  2f5478:	39 40 02 78	78024039   	r25 = #0x401
  2f547c:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f5480:	99 41 09 78	78094199 { 	r25 = #0x120c
  2f5484:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5488:	10 d9 f9 a6	a6f9d910 { 	cvt.uh = acc(r25):2x2 } 
  2f548c:	80 40 00 b0	b0004080 { 	r0 = add(r0,#0x4)
  2f5490:	00 80 00 7f	7f008000   	nop
  2f5494:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f5498:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f549c:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f54a0:	dc c0 b9 10	10b9c0dc   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f5450 <hmx_v75_convhbh1x1deep_stride1+0x250> } 
  2f54a4:	42 c1 00 58	5800c142 { 	jump 0x2f5728 <hmx_v75_convhbh1x1deep_stride1+0x528> } 
  2f54a8:	00 40 00 7f	7f004000 { 	nop
  2f54ac:	00 c0 00 7f	7f00c000   	nop } 
  2f54b0:	00 40 00 7f	7f004000 { 	nop
  2f54b4:	00 40 00 7f	7f004000   	nop
  2f54b8:	00 40 00 7f	7f004000   	nop
  2f54bc:	00 c0 00 7f	7f00c000   	nop } 
  2f54c0:	00 42 1c 60	601c4200 { 	loop0(0x2f54e0,r28)
  2f54c4:	cd 7f ed bf	bfed7fcd   	r13 = add(r13,#-0x2)
  2f54c8:	11 40 6c 70	706c4011   	r17 = r12
  2f54cc:	0e c1 12 b0	b012c10e   	r14 = add(r18,#0x8) } 
  2f54d0:	00 41 34 60	60344100 { 	loop1(0x2f54e0,r20)
  2f54d4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f54d8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f54dc:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f54e0:	46 40 81 9b	9b814046 { 	r6 = memw(r1++#0x8)
  2f54e4:	35 c0 81 91	9181c035   	r21 = memw(r1+#0x4) } 
  2f54e8:	08 56 08 f3	f3085608 { 	r8 = add(r8,r22)
  2f54ec:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f54f0:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f54f4:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2f54f8:	ec 47 15 92	921547ec   	activation.ub = mxmem(r21,r7)
  2f54fc:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep }  :endloop0
  2f5500:	00 5e 1c 60	601c5e00 { 	loop0(0x2f54e0,r28)
  2f5504:	1a 40 78 70	7078401a   	r26 = r24
  2f5508:	0e 41 20 5c	5c20410e   	if (!p1) jump:nt 0x2f551c <hmx_v75_convhbh1x1deep_stride1+0x31c>
  2f550c:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f5510:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f5514:	ec 47 06 92	920647ec { 	activation.ub = mxmem(r6,r7)
  2f5518:	e8 e9 08 92	9208e9e8   	weight.b = mxmem(r8,r9):deep } 
  2f551c:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5520:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f5524:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5528:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f552c:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f5530:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5534:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5538:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f553c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5540:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f551c <hmx_v75_convhbh1x1deep_stride1+0x31c>
  2f5544:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5548:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f554c:	1a 40 78 70	7078401a   	r26 = r24
  2f5550:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2f5554:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f5558:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f555c:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f5560:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5564:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5568:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f556c:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5570:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5574:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5578:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f557c:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f5558 <hmx_v75_convhbh1x1deep_stride1+0x358>
  2f5580:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5584:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f5588:	43 8a 18 8e	8e188a43   	r3 -= asl(r24,#0xa)
  2f558c:	08 40 62 70	70624008   	r8 = r2
  2f5590:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f5594:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f5598:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f559c:	9e c0 b9 10	10b9c09e   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f54d0 <hmx_v75_convhbh1x1deep_stride1+0x2d0> } 
  2f55a0:	c3 4a 18 8e	8e184ac3 { 	r3 += asl(r24,#0xa)
  2f55a4:	22 40 09 e2	e2094022   	r2 += add(r9,#0x1)
  2f55a8:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2f55ac:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f55b0:	8a f8 df 5c	5cdff88a   	if (p0.new) jump:t 0x2f54c0 <hmx_v75_convhbh1x1deep_stride1+0x2c0> } 
  2f55b4:	6e c0 00 58	5800c06e { 	jump 0x2f5690 <hmx_v75_convhbh1x1deep_stride1+0x490> } 
  2f55b8:	00 40 00 7f	7f004000 { 	nop
  2f55bc:	00 c0 00 7f	7f00c000   	nop } 
  2f55c0:	cd 7f ed bf	bfed7fcd { 	r13 = add(r13,#-0x2)
  2f55c4:	11 40 6c 70	706c4011   	r17 = r12
  2f55c8:	0e 41 12 b0	b012410e   	r14 = add(r18,#0x8)
  2f55cc:	00 c0 00 7f	7f00c000   	nop } 
  2f55d0:	00 41 14 60	60144100 { 	loop0(0x2f55e0,r20)
  2f55d4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f55d8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f55dc:	00 c0 00 7f	7f00c000   	nop } 
  2f55e0:	26 40 81 9b	9b814026 { 	r6 = memw(r1++#0x4)
  2f55e4:	2a c0 80 9b	9b80c02a   	r10 = memw(r0++#0x4) } 
  2f55e8:	1a 40 78 70	7078401a { 	r26 = r24
  2f55ec:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f55f0:	e8 e9 02 92	9202e9e8   	weight.b = mxmem(r2,r9):deep } 
  2f55f4:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f55f8:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f55fc:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5600:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5604:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f5608:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f560c:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5610:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5614:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5618:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f55f4 <hmx_v75_convhbh1x1deep_stride1+0x3f4>
  2f561c:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f5620:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f5624:	1a 40 78 70	7078401a   	r26 = r24
  2f5628:	0a 40 80 9d	9d80400a   	r10 = memw(r0++m0)
  2f562c:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt } 
  2f5630:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5634:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f5638:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f563c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5640:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f5644:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f5648:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f564c:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5650:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5654:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f5630 <hmx_v75_convhbh1x1deep_stride1+0x430>
  2f5658:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f565c:	45 83 18 8e	8e188345 { 	r5 -= asl(r24,#0x3)
  2f5660:	43 4a 18 8e	8e184a43   	r3 -= asl(r24,#0xa)
  2f5664:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop0
  2f5668:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f566c:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f5670:	b4 c0 b9 10	10b9c0b4   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f55d0 <hmx_v75_convhbh1x1deep_stride1+0x3d0> } 
  2f5674:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f5678:	c3 4a 18 8e	8e184ac3   	r3 += asl(r24,#0xa)
  2f567c:	12 ce 17 f5	f517ce12   	r19:18 = combine(r23,r14) } 
  2f5680:	20 40 4d 75	754d4020 { 	p0 = cmp.gt(r13,#0x1)
  2f5684:	a0 78 df 5c	5cdf78a0   	if (p0.new) jump:t 0x2f55c0 <hmx_v75_convhbh1x1deep_stride1+0x3c0>
  2f5688:	00 40 00 7f	7f004000   	nop
  2f568c:	00 c0 00 7f	7f00c000   	nop } 
  2f5690:	00 40 4d 75	754d4000 { 	p0 = cmp.gt(r13,#0x0)
  2f5694:	4c 58 20 5c	5c20584c   	if (!p0.new) jump:t 0x2f5728 <hmx_v75_convhbh1x1deep_stride1+0x528>
  2f5698:	00 40 00 7f	7f004000   	nop
  2f569c:	00 c0 00 7f	7f00c000   	nop } 
  2f56a0:	00 42 04 60	60044200 { 	loop0(0x2f56c0,r4)
  2f56a4:	09 41 09 8c	8c094109   	r9 = asr(r9,#0x1)
  2f56a8:	11 40 6c 70	706c4011   	r17 = r12
  2f56ac:	00 c0 00 7f	7f00c000   	nop } 
  2f56b0:	00 41 34 60	60344100 { 	loop1(0x2f56c0,r20)
  2f56b4:	00 52 13 f5	f5135200   	r1:0 = combine(r19,r18)
  2f56b8:	11 7f f1 bf	bff17f11   	r17 = add(r17,#-0x8)
  2f56bc:	08 c0 62 70	7062c008   	r8 = r2 } 
  2f56c0:	26 c0 81 9b	9b81c026 { 	r6 = memw(r1++#0x4) } 
  2f56c4:	08 96 08 f3	f3089608 { 	r8 = add(r8,r22)
  2f56c8:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f56cc:	e0 e9 08 92	9208e9e0   	weight.b = mxmem(r8,r9) }  :endloop0
  2f56d0:	00 5f 04 60	60045f00 { 	loop0(0x2f56c0,r4)
  2f56d4:	08 40 62 70	70624008   	r8 = r2
  2f56d8:	1a 40 78 70	7078401a   	r26 = r24
  2f56dc:	0a c0 80 9d	9d80c00a   	r10 = memw(r0++m0) } 
  2f56e0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f56e4:	fa 7f fa bf	bffa7ffa   	r26 = add(r26,#-0x1)
  2f56e8:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f56ec:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f56f0:	00 40 5a 75	755a4000 { 	p0 = cmp.gt(r26,#0x0)
  2f56f4:	39 40 85 9b	9b854039   	r25 = memw(r5++#0x4)
  2f56f8:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f56fc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5700:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5704:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f56e0 <hmx_v75_convhbh1x1deep_stride1+0x4e0>
  2f5708:	10 d9 f9 a6	a6f9d910   	cvt.uh = acc(r25):2x2 } 
  2f570c:	45 43 18 8e	8e184345 { 	r5 -= asl(r24,#0x3)
  2f5710:	43 89 18 8e	8e188943   	r3 -= asl(r24,#0x9)
  2f5714:	80 40 00 b0	b0004080   	r0 = add(r0,#0x4)
  2f5718:	18 cb ea a6	a6eacb18   	mxmem(r10,r11) = cvt }  :endloop1
  2f571c:	13 4f 13 f3	f3134f13 { 	r19 = add(r19,r15)
  2f5720:	12 50 12 f3	f3125012   	r18 = add(r18,r16)
  2f5724:	ca c0 b9 10	10b9c0ca   	p0 = cmp.gt(r17,#0x0); if (p0.new) jump:nt 0x2f56b0 <hmx_v75_convhbh1x1deep_stride1+0x4b0> } 
  2f5728:	11 c0 e0 a6	a6e0c011 { 	mxclracc } 
  2f572c:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f5730:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f5734:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f5738:	1d 46 1d b0	b01d461d { 	r29 = add(r29,#0x30)
  2f573c:	00 40 9f 52	529f4000   	jumpr r31
  2f5740:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f5780 <hmx_v85_convhbh1x1deep_stride1>:
  2f5780:	c0 3f 00 5a	5a003fc0 { 	r0 = #-0x1; 	jumpr r31 } 
		...

002f57c0 <hmx_convhbh1xN_stride2>:
  2f57c0:	1d 77 fd bf	bffd771d { 	r29 = add(r29,#-0x48)
  2f57c4:	ed 7f 03 78	78037fed   	r13 = #0x7ff
  2f57c8:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f57cc:	48 c0 c4 91	91c4c048   	r9:8 = memd(r4+#0x10) } 
  2f57d0:	0a 40 c4 91	91c4400a { 	r11:10 = memd(r4+#0x0)
  2f57d4:	6e c0 c4 91	91c4c06e   	r15:14 = memd(r4+#0x18) } 
  2f57d8:	10 4d 06 f1	f1064d10 { 	r16 = and(r6,r13)
  2f57dc:	11 4d 08 f1	f1084d11   	r17 = and(r8,r13)
  2f57e0:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f57e4:	13 4d 0a f1	f10a4d13 { 	r19 = and(r10,r13)
  2f57e8:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f57ec:	6d 50 00 78	7800506d { 	r13 = #0x83
  2f57f0:	15 03 14 01	01140315   	r4 = memw(r1+#0x4); 	r5 = memw(r1+#0xc) } 
  2f57f4:	24 41 04 8c	8c044124 { 	r4 = lsr(r4,#0x1)
  2f57f8:	02 4d 8a c7	c78a4d02   	p2 = bitsclr(r10,r13)
  2f57fc:	04 58 dd a1	a1dd5804   	memd(r29+#0x20) = r25:24
  2f5800:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f5804:	e4 7f e4 bf	bfe47fe4 { 	r4 = add(r4,#-0x1)
  2f5808:	11 00 1d 02	021d0011   	r21 = memw(r1+#0x8); 	r1 = memw(r1+#0x0) } 
  2f580c:	01 40 44 75	75444001 { 	p1 = cmp.gt(r4,#0x0)
  2f5810:	39 40 80 91	91804039   	r25 = memw(r0+#0x4)
  2f5814:	77 c0 80 91	9180c077   	r23 = memw(r0+#0xc) } 
  2f5818:	59 42 19 8c	8c194259 { 	r25 = asl(r25,#0x2)
  2f581c:	45 45 17 f4	f4174545   	r5 = mux(p2,r23,r5)
  2f5820:	b8 40 80 91	918040b8   	r24 = memw(r0+#0x14)
  2f5824:	54 c0 80 91	9180c054   	r20 = memw(r0+#0x8) } 
  2f5828:	54 42 54 80	80544254 { 	r21:20 = vaslw(r21:20,#0x2)
  2f582c:	65 40 05 b0	b0054065   	r5 = add(r5,#0x3)
  2f5830:	00 00 0e 04	040e0000   	r22 = memw(r0+#0x10); 	r0 = memw(r0+#0x0) } 
  2f5834:	25 42 05 8c	8c054225 { 	r5 = lsr(r5,#0x2)
  2f5838:	0c 5e 9d a1	a19d5e0c   	memw(r29+#0x30) = r30
  2f583c:	11 c1 9d a1	a19dc111   	memw(r29+#0x44) = r1 } 
  2f5840:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5844:	18 7c f8 bf	bff87c18   	r24 = add(r24,#-0x20)
  2f5848:	9a 40 00 b0	b000409a   	r26 = add(r0,#0x4)
  2f584c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5850:	1c 40 00 78	7800401c { 	r28 = #0x0
  2f5854:	00 40 00 7f	7f004000   	nop
  2f5858:	00 40 00 7f	7f004000   	nop
  2f585c:	10 da 9d a1	a19dda10   	memw(r29+#0x40) = r26 } 
  2f5860:	00 42 25 60	60254200 { 	loop1(0x2f5880,r5)
  2f5864:	1c 41 1c b0	b01c411c   	r28 = add(r28,#0x8)
  2f5868:	1e 40 00 78	7800401e   	r30 = #0x0
  2f586c:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f5870:	03 43 63 6b	6b634303 { 	p3 = and(p3,!p3)
  2f5874:	00 40 00 7f	7f004000   	nop
  2f5878:	00 40 00 7f	7f004000   	nop
  2f587c:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f5880:	00 42 04 60	60044200 { 	loop0(0x2f58a0,r4)
  2f5884:	0c 40 62 70	7062400c   	r12 = r2
  2f5888:	5a 40 81 9b	9b81405a   	r26 = memw(r1++#0x8)
  2f588c:	3b c0 81 91	9181c03b   	r27 = memw(r1+#0x4) } 
  2f5890:	06 5a 30 f1	f1305a06 { 	r6 = or(r16,r26)
  2f5894:	08 5b 31 f1	f1315b08   	r8 = or(r17,r27)
  2f5898:	1c 41 20 5c	5c20411c   	if (!p1) jump:nt 0x2f58c8 <hmx_convhbh1xN_stride2+0x108>
  2f589c:	0d c0 80 91	9180c00d   	r13 = memw(r0+#0x0) } 
  2f58a0:	5a 40 81 9b	9b81405a { 	r26 = memw(r1++#0x8)
  2f58a4:	3b c0 81 91	9181c03b   	r27 = memw(r1+#0x4) } 
  2f58a8:	2c 40 0e e2	e20e402c { 	r12 += add(r14,#0x1)
  2f58ac:	06 5a 30 f1	f1305a06   	r6 = or(r16,r26)
  2f58b0:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f58b4:	eb ee 0c 92	920ceeeb   	weight.b = mxmem(r12,r14):dilate } 
  2f58b8:	2c 80 0f e2	e20f802c { 	r12 += add(r15,#0x1)
  2f58bc:	08 5b 31 f1	f1315b08   	r8 = or(r17,r27)
  2f58c0:	ec 49 08 92	920849ec   	activation.ub = mxmem(r8,r9)
  2f58c4:	eb ef 0c 92	920cefeb   	weight.b = mxmem(r12,r15):dilate }  :endloop0
  2f58c8:	2c 40 0e e2	e20e402c { 	r12 += add(r14,#0x1)
  2f58cc:	0a 4d 33 f1	f1334d0a   	r10 = or(r19,r13)
  2f58d0:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f58d4:	eb ee 0c 92	920ceeeb   	weight.b = mxmem(r12,r14):dilate } 
  2f58d8:	2c 40 0f e2	e20f402c { 	r12 += add(r15,#0x1)
  2f58dc:	1c 42 00 5c	5c00421c   	if (p2) jump:nt 0x2f5910 <hmx_convhbh1xN_stride2+0x150>
  2f58e0:	ec 49 08 92	920849ec   	activation.ub = mxmem(r8,r9)
  2f58e4:	eb ef 0c 92	920cefeb   	weight.b = mxmem(r12,r15):dilate } 
  2f58e8:	04 c3 20 5c	5c20c304 { 	if (!p3) jump:nt 0x2f58f0 <hmx_convhbh1xN_stride2+0x130> } 
  2f58ec:	0e eb f2 a6	a6f2eb0e { 	mxmem(r18,r11):after:retain:sat.uh = acc:2x1 } 
  2f58f0:	03 5e 57 f2	f2575e03 { 	p3 = cmp.gt(r23,r30)
  2f58f4:	16 4b 20 5c	5c204b16   	if (!p3.new) jump:nt 0x2f591c <hmx_convhbh1xN_stride2+0x15c>
  2f58f8:	9e c0 1e b0	b01ec09e   	r30 = add(r30,#0x4) } 
  2f58fc:	12 40 6a 70	706a4012 { 	r18 = r10
  2f5900:	00 9e 57 f2	f2579e00   	p0 = cmp.gt(r23,r30)
  2f5904:	00 79 00 fb	fb007900   	if (p0.new) r0 = add(r0,r25)
  2f5908:	02 eb ea a6	a6eaeb02   	mxmem(r10,r11):before:sat.uh = acc:2x1 }  :endloop1
  2f590c:	08 c0 00 58	5800c008 { 	jump 0x2f591c <hmx_convhbh1xN_stride2+0x15c> } 
  2f5910:	00 59 00 f3	f3005900 { 	r0 = add(r0,r25)
  2f5914:	00 80 00 7f	7f008000   	nop
  2f5918:	0a eb ea a6	a6eaeb0a   	mxmem(r10,r11):after:sat.uh = acc:2x1 }  :endloop1
  2f591c:	00 5c 56 f2	f2565c00 { 	p0 = cmp.gt(r22,r28)
  2f5920:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f5924:	a0 54 00 d3	d30054a0 { 	r1:0 = vaddw(r1:0,r21:20)
  2f5928:	9e e0 df 5c	5cdfe09e   	if (p0) jump:nt 0x2f5860 <hmx_convhbh1xN_stride2+0xa0> } 
  2f592c:	02 40 6c 70	706c4002 { 	r2 = r12
  2f5930:	00 40 58 75	75584000   	p0 = cmp.gt(r24,#0x0)
  2f5934:	8a 78 df 5c	5cdf788a   	if (p0.new) jump:t 0x2f5840 <hmx_convhbh1xN_stride2+0x80>
  2f5938:	00 c1 dd 91	91ddc100   	r1:0 = memd(r29+#0x40) } 
  2f593c:	9e c1 9d 91	919dc19e { 	r30 = memw(r29+#0x30) } 
  2f5940:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f5944:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f5948:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f594c:	1d 49 1d b0	b01d491d { 	r29 = add(r29,#0x48)
  2f5950:	00 40 9f 52	529f4000   	jumpr r31
  2f5954:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f5980 <hmx_v73_convhbh1xN_stride2_sparsity>:
  2f5980:	1d 77 fd bf	bffd771d { 	r29 = add(r29,#-0x48)
  2f5984:	ed 7f 03 78	78037fed   	r13 = #0x7ff
  2f5988:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f598c:	48 c0 c4 91	91c4c048   	r9:8 = memd(r4+#0x10) } 
  2f5990:	00 40 00 7f	7f004000 { 	nop
  2f5994:	00 40 00 7f	7f004000   	nop
  2f5998:	0a 40 c4 91	91c4400a   	r11:10 = memd(r4+#0x0)
  2f599c:	6e c0 c4 91	91c4c06e   	r15:14 = memd(r4+#0x18) } 
  2f59a0:	10 4d 06 f1	f1064d10 { 	r16 = and(r6,r13)
  2f59a4:	11 4d 08 f1	f1084d11   	r17 = and(r8,r13)
  2f59a8:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f59ac:	12 4d 0a f1	f10a4d12 { 	r18 = and(r10,r13)
  2f59b0:	13 40 65 70	70654013   	r19 = r5
  2f59b4:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f59b8:	6d 50 00 78	7800506d { 	r13 = #0x83
  2f59bc:	15 03 14 01	01140315   	r4 = memw(r1+#0x4); 	r5 = memw(r1+#0xc) } 
  2f59c0:	24 41 04 8c	8c044124 { 	r4 = lsr(r4,#0x1)
  2f59c4:	02 4d 8a c7	c78a4d02   	p2 = bitsclr(r10,r13)
  2f59c8:	04 58 dd a1	a1dd5804   	memd(r29+#0x20) = r25:24
  2f59cc:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f59d0:	e4 7f e4 bf	bfe47fe4 { 	r4 = add(r4,#-0x1)
  2f59d4:	11 00 1d 02	021d0011   	r21 = memw(r1+#0x8); 	r1 = memw(r1+#0x0) } 
  2f59d8:	01 40 44 75	75444001 { 	p1 = cmp.gt(r4,#0x0)
  2f59dc:	77 40 80 91	91804077   	r23 = memw(r0+#0xc)
  2f59e0:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f59e4:	45 45 17 f4	f4174545 { 	r5 = mux(p2,r23,r5)
  2f59e8:	97 40 d7 74	74d74097   	if (!p2) r23 = add(r23,#0x4)
  2f59ec:	39 40 80 91	91804039   	r25 = memw(r0+#0x4)
  2f59f0:	54 c0 80 91	9180c054   	r20 = memw(r0+#0x8) } 
  2f59f4:	59 42 19 8c	8c194259 { 	r25 = asl(r25,#0x2)
  2f59f8:	65 40 05 b0	b0054065   	r5 = add(r5,#0x3)
  2f59fc:	b8 40 80 91	918040b8   	r24 = memw(r0+#0x14)
  2f5a00:	96 c0 80 91	9180c096   	r22 = memw(r0+#0x10) } 
  2f5a04:	54 42 54 80	80544254 { 	r21:20 = vaslw(r21:20,#0x2)
  2f5a08:	85 7f 25 76	76257f85   	r5 = and(r5,#-0x4)
  2f5a0c:	11 09 00 c0	c0000911   	r0 = memw(r0+#0x0); 	memw(r29+#0x44) = r1 } 
  2f5a10:	25 42 05 8c	8c054225 { 	r5 = lsr(r5,#0x2)
  2f5a14:	03 45 57 f2	f2574503   	p3 = cmp.gt(r23,r5)
  2f5a18:	00 40 00 7f	7f004000   	nop
  2f5a1c:	3e c0 93 9b	9b93c03e   	r30 = memw(r19++#0x4) } 
  2f5a20:	1c 40 00 78	7800401c { 	r28 = #0x0
  2f5a24:	10 c0 9d a1	a19dc010   	memw(r29+#0x40) = r0 } 
  2f5a28:	10 41 25 60	60254110 { 	loop1(0x2f5a40,r5)
  2f5a2c:	1c 41 1c b0	b01c411c   	r28 = add(r28,#0x8)
  2f5a30:	6d 50 00 78	7800506d   	r13 = #0x83
  2f5a34:	00 c0 00 7f	7f00c000   	nop } 
  2f5a38:	02 4d 8a c7	c78a4d02 { 	p2 = bitsclr(r10,r13)
  2f5a3c:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f5a40:	00 42 04 60	60044200 { 	loop0(0x2f5a60,r4)
  2f5a44:	0c 40 62 70	7062400c   	r12 = r2
  2f5a48:	5a 40 81 9b	9b81405a   	r26 = memw(r1++#0x8)
  2f5a4c:	3b c0 81 91	9181c03b   	r27 = memw(r1+#0x4) } 
  2f5a50:	06 5a 30 f1	f1305a06 { 	r6 = or(r16,r26)
  2f5a54:	08 5b 31 f1	f1315b08   	r8 = or(r17,r27)
  2f5a58:	24 41 20 5c	5c204124   	if (!p1) jump:nt 0x2f5a98 <hmx_v73_convhbh1xN_stride2_sparsity+0x118>
  2f5a5c:	0d c0 80 91	9180c00d   	r13 = memw(r0+#0x0) } 
  2f5a60:	5a 40 81 9b	9b81405a { 	r26 = memw(r1++#0x8)
  2f5a64:	3b c0 81 91	9181c03b   	r27 = memw(r1+#0x4) } 
  2f5a68:	06 5a 30 f1	f1305a06 { 	r6 = or(r16,r26)
  2f5a6c:	0d 70 2e 76	762e700d   	r13 = and(r14,#-0x80)
  2f5a70:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f5a74:	eb ee 0c 92	920ceeeb   	weight.b = mxmem(r12,r14):dilate } 
  2f5a78:	02 40 00 00	00004002 { 	immext(#0x80)
  2f5a7c:	0c 40 0d e2	e20d400c   	r12 += add(r13,##0x80)
  2f5a80:	0d f0 2f 76	762ff00d   	r13 = and(r15,#-0x80) } 
  2f5a84:	08 5b 31 f1	f1315b08 { 	r8 = or(r17,r27)
  2f5a88:	ec 49 08 92	920849ec   	activation.ub = mxmem(r8,r9)
  2f5a8c:	eb ef 0c 92	920cefeb   	weight.b = mxmem(r12,r15):dilate } 
  2f5a90:	02 80 00 00	00008002 { 	immext(#0x80)
  2f5a94:	0c c0 0d e2	e20dc00c   	r12 += add(r13,##0x80) }  :endloop0
  2f5a98:	0d 70 2e 76	762e700d { 	r13 = and(r14,#-0x80)
  2f5a9c:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f5aa0:	eb ee 0c 92	920ceeeb   	weight.b = mxmem(r12,r14):dilate } 
  2f5aa4:	02 40 00 00	00004002 { 	immext(#0x80)
  2f5aa8:	0c 40 0d e2	e20d400c   	r12 += add(r13,##0x80)
  2f5aac:	0d f0 2f 76	762ff00d   	r13 = and(r15,#-0x80) } 
  2f5ab0:	1f 40 7e 70	707e401f { 	r31 = r30
  2f5ab4:	ec 49 08 92	920849ec   	activation.ub = mxmem(r8,r9)
  2f5ab8:	eb ef 0c 92	920cefeb   	weight.b = mxmem(r12,r15):dilate } 
  2f5abc:	02 40 00 00	00004002 { 	immext(#0x80)
  2f5ac0:	0c c0 0d e2	e20dc00c   	r12 += add(r13,##0x80) } 
  2f5ac4:	0d c0 80 91	9180c00d { 	r13 = memw(r0+#0x0) } 
  2f5ac8:	0a cd 32 f1	f132cd0a { 	r10 = or(r18,r13) } 
  2f5acc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5ad0:	ff 7f ff bf	bfff7fff   	r31 = add(r31,#-0x1)
  2f5ad4:	3a 40 93 9b	9b93403a   	r26 = memw(r19++#0x4)
  2f5ad8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5adc:	00 40 5f 75	755f4000 { 	p0 = cmp.gt(r31,#0x0)
  2f5ae0:	3a 40 93 9b	9b93403a   	r26 = memw(r19++#0x4)
  2f5ae4:	10 d9 fa a6	a6fad910   	cvt.uh = acc(r26):2x2 } 
  2f5ae8:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5aec:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5af0:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f5acc <hmx_v73_convhbh1xN_stride2_sparsity+0x14c>
  2f5af4:	10 d9 fa a6	a6fad910   	cvt.uh = acc(r26):2x2 } 
  2f5af8:	53 43 1e 8e	8e1e4353 { 	r19 -= asl(r30,#0x3)
  2f5afc:	43 c9 1e 8e	8e1ec943   	r3 -= asl(r30,#0x9) } 
  2f5b00:	04 c2 20 5c	5c20c204 { 	if (!p2) jump:nt 0x2f5b08 <hmx_v73_convhbh1xN_stride2_sparsity+0x188> } 
  2f5b04:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f5b08:	02 42 e2 6b	6be24202 { 	p2 = or(p2,!p2)
  2f5b0c:	40 99 00 fb	fb009940   	if (p2) r0 = add(r0,r25)
  2f5b10:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  2f5b14:	0c 53 20 5c	5c20530c { 	if (!p3) jump:t 0x2f5b2c <hmx_v73_convhbh1xN_stride2_sparsity+0x1ac>
  2f5b18:	1a c0 60 7e	7e60c01a   	if (p3) r26 = #0x0 } 
  2f5b1c:	0d 40 80 91	9180400d { 	r13 = memw(r0+#0x0)
  2f5b20:	10 d9 fa a6	a6fad910   	cvt.uh = acc(r26):2x2 } 
  2f5b24:	0a cd 32 f1	f132cd0a { 	r10 = or(r18,r13) } 
  2f5b28:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f5b2c:	00 5c 56 f2	f2565c00 { 	p0 = cmp.gt(r22,r28)
  2f5b30:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f5b34:	a0 54 00 d3	d30054a0 { 	r1:0 = vaddw(r1:0,r21:20)
  2f5b38:	7a e0 df 5c	5cdfe07a   	if (p0) jump:nt 0x2f5a28 <hmx_v73_convhbh1xN_stride2_sparsity+0xa8> } 
  2f5b3c:	c3 49 1e 8e	8e1e49c3 { 	r3 += asl(r30,#0x9)
  2f5b40:	02 40 6c 70	706c4002   	r2 = r12
  2f5b44:	18 7c f8 bf	bff87c18   	r24 = add(r24,#-0x20)
  2f5b48:	00 c1 dd 91	91ddc100   	r1:0 = memd(r29+#0x40) } 
  2f5b4c:	80 40 00 b0	b0004080 { 	r0 = add(r0,#0x4)
  2f5b50:	00 40 58 75	75584000   	p0 = cmp.gt(r24,#0x0)
  2f5b54:	6a f8 df 5c	5cdff86a   	if (p0.new) jump:t 0x2f5a20 <hmx_v73_convhbh1xN_stride2_sparsity+0xa0> } 
  2f5b58:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2f5b5c:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f5b60:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f5b64:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f5b68:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f5b6c:	1d 49 1d b0	b01d491d { 	r29 = add(r29,#0x48)
  2f5b70:	00 40 9f 52	529f4000   	jumpr r31
  2f5b74:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f5b80 <hmx_v73_convhbh1xN_stride2>:
  2f5b80:	1d 77 fd bf	bffd771d { 	r29 = add(r29,#-0x48)
  2f5b84:	ed 7f 03 78	78037fed   	r13 = #0x7ff
  2f5b88:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f5b8c:	48 c0 c4 91	91c4c048   	r9:8 = memd(r4+#0x10) } 
  2f5b90:	00 40 00 7f	7f004000 { 	nop
  2f5b94:	00 40 00 7f	7f004000   	nop
  2f5b98:	0a 40 c4 91	91c4400a   	r11:10 = memd(r4+#0x0)
  2f5b9c:	6e c0 c4 91	91c4c06e   	r15:14 = memd(r4+#0x18) } 
  2f5ba0:	10 4d 06 f1	f1064d10 { 	r16 = and(r6,r13)
  2f5ba4:	11 4d 08 f1	f1084d11   	r17 = and(r8,r13)
  2f5ba8:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f5bac:	12 4d 0a f1	f10a4d12 { 	r18 = and(r10,r13)
  2f5bb0:	13 40 65 70	70654013   	r19 = r5
  2f5bb4:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f5bb8:	6d 50 00 78	7800506d { 	r13 = #0x83
  2f5bbc:	00 40 00 7f	7f004000   	nop
  2f5bc0:	24 40 81 91	91814024   	r4 = memw(r1+#0x4)
  2f5bc4:	8c c1 84 91	9184c18c   	r12 = memw(r4+#0x30) } 
  2f5bc8:	24 41 04 8c	8c044124 { 	r4 = lsr(r4,#0x1)
  2f5bcc:	02 4d 8a c7	c78a4d02   	p2 = bitsclr(r10,r13)
  2f5bd0:	65 40 81 91	91814065   	r5 = memw(r1+#0xc)
  2f5bd4:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2f5bd8:	e4 7f e4 bf	bfe47fe4 { 	r4 = add(r4,#-0x1)
  2f5bdc:	00 40 00 7f	7f004000   	nop
  2f5be0:	11 00 1d 02	021d0011   	r21 = memw(r1+#0x8); 	r1 = memw(r1+#0x0) } 
  2f5be4:	01 40 44 75	75444001 { 	p1 = cmp.gt(r4,#0x0)
  2f5be8:	00 40 00 7f	7f004000   	nop
  2f5bec:	77 40 80 91	91804077   	r23 = memw(r0+#0xc)
  2f5bf0:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f5bf4:	45 45 17 f4	f4174545 { 	r5 = mux(p2,r23,r5)
  2f5bf8:	97 40 d7 74	74d74097   	if (!p2) r23 = add(r23,#0x4)
  2f5bfc:	39 40 80 91	91804039   	r25 = memw(r0+#0x4)
  2f5c00:	54 c0 80 91	9180c054   	r20 = memw(r0+#0x8) } 
  2f5c04:	59 42 19 8c	8c194259 { 	r25 = asl(r25,#0x2)
  2f5c08:	65 40 05 b0	b0054065   	r5 = add(r5,#0x3)
  2f5c0c:	b8 40 80 91	918040b8   	r24 = memw(r0+#0x14)
  2f5c10:	96 c0 80 91	9180c096   	r22 = memw(r0+#0x10) } 
  2f5c14:	54 42 54 80	80544254 { 	r21:20 = vaslw(r21:20,#0x2)
  2f5c18:	85 7f 25 76	76257f85   	r5 = and(r5,#-0x4)
  2f5c1c:	11 09 00 c0	c0000911   	r0 = memw(r0+#0x0); 	memw(r29+#0x44) = r1 } 
  2f5c20:	25 42 05 8c	8c054225 { 	r5 = lsr(r5,#0x2)
  2f5c24:	03 45 57 f2	f2574503   	p3 = cmp.gt(r23,r5)
  2f5c28:	3e 40 93 9b	9b93403e   	r30 = memw(r19++#0x4)
  2f5c2c:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f5c30:	00 40 00 7f	7f004000 { 	nop
  2f5c34:	00 40 00 7f	7f004000   	nop
  2f5c38:	00 40 00 7f	7f004000   	nop
  2f5c3c:	00 c0 00 7f	7f00c000   	nop } 
  2f5c40:	1c 40 00 78	7800401c { 	r28 = #0x0
  2f5c44:	10 c0 9d a1	a19dc010   	memw(r29+#0x40) = r0 } 
  2f5c48:	10 41 25 60	60254110 { 	loop1(0x2f5c60,r5)
  2f5c4c:	1c 41 1c b0	b01c411c   	r28 = add(r28,#0x8)
  2f5c50:	6d 50 00 78	7800506d   	r13 = #0x83
  2f5c54:	00 c0 00 7f	7f00c000   	nop } 
  2f5c58:	02 4d 8a c7	c78a4d02 { 	p2 = bitsclr(r10,r13)
  2f5c5c:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f5c60:	00 42 04 60	60044200 { 	loop0(0x2f5c80,r4)
  2f5c64:	0c 40 62 70	7062400c   	r12 = r2
  2f5c68:	5a 40 81 9b	9b81405a   	r26 = memw(r1++#0x8)
  2f5c6c:	3b c0 81 91	9181c03b   	r27 = memw(r1+#0x4) } 
  2f5c70:	06 5a 30 f1	f1305a06 { 	r6 = or(r16,r26)
  2f5c74:	08 5b 31 f1	f1315b08   	r8 = or(r17,r27)
  2f5c78:	1c 41 20 5c	5c20411c   	if (!p1) jump:nt 0x2f5ca8 <hmx_v73_convhbh1xN_stride2+0x128>
  2f5c7c:	0d c0 80 91	9180c00d   	r13 = memw(r0+#0x0) } 
  2f5c80:	5a 40 81 9b	9b81405a { 	r26 = memw(r1++#0x8)
  2f5c84:	3b c0 81 91	9181c03b   	r27 = memw(r1+#0x4) } 
  2f5c88:	2c 40 0e e2	e20e402c { 	r12 += add(r14,#0x1)
  2f5c8c:	06 5a 30 f1	f1305a06   	r6 = or(r16,r26)
  2f5c90:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f5c94:	eb ee 0c 92	920ceeeb   	weight.b = mxmem(r12,r14):dilate } 
  2f5c98:	2c 80 0f e2	e20f802c { 	r12 += add(r15,#0x1)
  2f5c9c:	08 5b 31 f1	f1315b08   	r8 = or(r17,r27)
  2f5ca0:	ec 49 08 92	920849ec   	activation.ub = mxmem(r8,r9)
  2f5ca4:	eb ef 0c 92	920cefeb   	weight.b = mxmem(r12,r15):dilate }  :endloop0
  2f5ca8:	2c 40 0e e2	e20e402c { 	r12 += add(r14,#0x1)
  2f5cac:	0a 4d 32 f1	f1324d0a   	r10 = or(r18,r13)
  2f5cb0:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f5cb4:	eb ee 0c 92	920ceeeb   	weight.b = mxmem(r12,r14):dilate } 
  2f5cb8:	2c 40 0f e2	e20f402c { 	r12 += add(r15,#0x1)
  2f5cbc:	1f 40 7e 70	707e401f   	r31 = r30
  2f5cc0:	ec 49 08 92	920849ec   	activation.ub = mxmem(r8,r9)
  2f5cc4:	eb ef 0c 92	920cefeb   	weight.b = mxmem(r12,r15):dilate } 
  2f5cc8:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5ccc:	ff 7f ff bf	bfff7fff   	r31 = add(r31,#-0x1)
  2f5cd0:	3a 40 93 9b	9b93403a   	r26 = memw(r19++#0x4)
  2f5cd4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5cd8:	00 40 5f 75	755f4000 { 	p0 = cmp.gt(r31,#0x0)
  2f5cdc:	3a 40 93 9b	9b93403a   	r26 = memw(r19++#0x4)
  2f5ce0:	10 d9 fa a6	a6fad910   	cvt.uh = acc(r26):2x2 } 
  2f5ce4:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5ce8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5cec:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f5cc8 <hmx_v73_convhbh1xN_stride2+0x148>
  2f5cf0:	10 d9 fa a6	a6fad910   	cvt.uh = acc(r26):2x2 } 
  2f5cf4:	53 43 1e 8e	8e1e4353 { 	r19 -= asl(r30,#0x3)
  2f5cf8:	43 c9 1e 8e	8e1ec943   	r3 -= asl(r30,#0x9) } 
  2f5cfc:	04 c2 20 5c	5c20c204 { 	if (!p2) jump:nt 0x2f5d04 <hmx_v73_convhbh1xN_stride2+0x184> } 
  2f5d00:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f5d04:	02 42 e2 6b	6be24202 { 	p2 = or(p2,!p2)
  2f5d08:	40 99 00 fb	fb009940   	if (p2) r0 = add(r0,r25)
  2f5d0c:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  2f5d10:	0c 53 20 5c	5c20530c { 	if (!p3) jump:t 0x2f5d28 <hmx_v73_convhbh1xN_stride2+0x1a8>
  2f5d14:	1a c0 60 7e	7e60c01a   	if (p3) r26 = #0x0 } 
  2f5d18:	0d 40 80 91	9180400d { 	r13 = memw(r0+#0x0)
  2f5d1c:	10 d9 fa a6	a6fad910   	cvt.uh = acc(r26):2x2 } 
  2f5d20:	0a cd 32 f1	f132cd0a { 	r10 = or(r18,r13) } 
  2f5d24:	18 cb ea a6	a6eacb18 { 	mxmem(r10,r11) = cvt } 
  2f5d28:	00 5c 56 f2	f2565c00 { 	p0 = cmp.gt(r22,r28)
  2f5d2c:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f5d30:	a0 54 00 d3	d30054a0 { 	r1:0 = vaddw(r1:0,r21:20)
  2f5d34:	8c e0 df 5c	5cdfe08c   	if (p0) jump:nt 0x2f5c48 <hmx_v73_convhbh1xN_stride2+0xc8> } 
  2f5d38:	c3 49 1e 8e	8e1e49c3 { 	r3 += asl(r30,#0x9)
  2f5d3c:	02 40 6c 70	706c4002   	r2 = r12
  2f5d40:	18 7c f8 bf	bff87c18   	r24 = add(r24,#-0x20)
  2f5d44:	00 c1 dd 91	91ddc100   	r1:0 = memd(r29+#0x40) } 
  2f5d48:	80 40 00 b0	b0004080 { 	r0 = add(r0,#0x4)
  2f5d4c:	00 40 58 75	75584000   	p0 = cmp.gt(r24,#0x0)
  2f5d50:	7c f8 df 5c	5cdff87c   	if (p0.new) jump:t 0x2f5c40 <hmx_v73_convhbh1xN_stride2+0xc0> } 
  2f5d54:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2f5d58:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f5d5c:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f5d60:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f5d64:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f5d68:	1d 49 1d b0	b01d491d { 	r29 = add(r29,#0x48)
  2f5d6c:	00 40 9f 52	529f4000   	jumpr r31
  2f5d70:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f5d80 <hmx_convhbhNx1_stride2>:
  2f5d80:	1d 78 fd bf	bffd781d { 	r29 = add(r29,#-0x40)
  2f5d84:	ef 7f 03 78	78037fef   	r15 = #0x7ff
  2f5d88:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f5d8c:	48 c1 84 91	9184c148   	r8 = memw(r4+#0x28) } 
  2f5d90:	10 4f 06 f1	f1064f10 { 	r16 = and(r6,r15)
  2f5d94:	11 4f 08 f1	f1084f11   	r17 = and(r8,r15)
  2f5d98:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f5d9c:	0e 4f 07 f1	f1074f0e { 	r14 = and(r7,r15)
  2f5da0:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f5da4:	ca 40 84 91	918440ca { 	r10 = memw(r4+#0x18)
  2f5da8:	0b c1 84 91	9184c10b   	r11 = memw(r4+#0x20) } 
  2f5dac:	0c 40 c4 91	91c4400c { 	r13:12 = memd(r4+#0x0)
  2f5db0:	04 d8 dd a1	a1ddd804   	memd(r29+#0x20) = r25:24 } 
  2f5db4:	00 40 00 7f	7f004000 { 	nop
  2f5db8:	00 40 00 7f	7f004000   	nop
  2f5dbc:	1b 02 14 01	0114021b   	r4 = memw(r1+#0x4); 	r19 = memw(r1+#0x8) } 
  2f5dc0:	24 41 04 8c	8c044124 { 	r4 = lsr(r4,#0x1)
  2f5dc4:	11 00 15 04	04150011   	r5 = memw(r1+#0x10); 	r1 = memw(r1+#0x0) } 
  2f5dc8:	0f 4f 0c f1	f10c4f0f { 	r15 = and(r12,r15)
  2f5dcc:	e4 7f e4 bf	bfe47fe4   	r4 = add(r4,#-0x1)
  2f5dd0:	bc 40 80 91	918040bc   	r28 = memw(r0+#0x14)
  2f5dd4:	36 c0 80 91	9180c036   	r22 = memw(r0+#0x4) } 
  2f5dd8:	56 42 16 8c	8c164256 { 	r22 = asl(r22,#0x2)
  2f5ddc:	00 40 00 7f	7f004000   	nop
  2f5de0:	0d 03 0a 02	020a030d   	r18 = memw(r0+#0x8); 	r21 = memw(r0+#0xc) } 
  2f5de4:	52 42 52 80	80524252 { 	r19:18 = vaslw(r19:18,#0x2)
  2f5de8:	75 40 15 b0	b0154075   	r21 = add(r21,#0x3)
  2f5dec:	00 00 0c 04	040c0000   	r20 = memw(r0+#0x10); 	r0 = memw(r0+#0x0) } 
  2f5df0:	07 40 36 62	62364007 { 	m1 = r22
  2f5df4:	15 42 15 8c	8c154215   	r21 = asr(r21,#0x2)
  2f5df8:	05 5a dd a1	a1dd5a05   	memd(r29+#0x28) = r27:26
  2f5dfc:	0f c1 9d a1	a19dc10f   	memw(r29+#0x3c) = r1 } 
  2f5e00:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f5e04:	1c 7c fc bf	bffc7c1c   	r28 = add(r28,#-0x20)
  2f5e08:	9b 40 00 b0	b000409b   	r27 = add(r0,#0x4)
  2f5e0c:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f5e10:	1a 41 00 78	7800411a { 	r26 = #0x8
  2f5e14:	00 40 00 7f	7f004000   	nop
  2f5e18:	00 40 00 7f	7f004000   	nop
  2f5e1c:	0e db 9d a1	a19ddb0e   	memw(r29+#0x38) = r27 } 
  2f5e20:	00 43 35 60	60354300 { 	loop1(0x2f5e50,r21)
  2f5e24:	1b 40 00 78	7800401b   	r27 = #0x0
  2f5e28:	17 60 03 78	78036017   	r23 = #0x700
  2f5e2c:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f5e30:	01 57 a6 c7	c7a65701 { 	p1 = !bitsclr(r6,r23)
  2f5e34:	02 57 a8 c7	c7a85702   	p2 = !bitsclr(r8,r23)
  2f5e38:	01 5a 45 f2	f2455a01   	p1 = cmp.gt(r5,r26)
  2f5e3c:	02 da 45 f2	f245da02   	p2 = cmp.gt(r5,r26) } 
  2f5e40:	18 40 00 7c	7c004018 { 	r25:24 = combine(#0x0,#0x0)
  2f5e44:	00 40 00 7f	7f004000   	nop
  2f5e48:	00 40 00 7f	7f004000   	nop
  2f5e4c:	06 c0 dd a1	a1ddc006   	memd(r29+#0x30) = r1:0 } 
  2f5e50:	00 43 04 60	60044300 { 	loop0(0x2f5e80,r4)
  2f5e54:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2f5e58:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f5e5c:	38 d3 81 30	3081d338   	if (p1) r24 = memw(r1+r19<<#0x0) } 
  2f5e60:	38 58 37 fb	fb375838 { 	if (p1) r24 = sub(r24,r23)
  2f5e64:	06 57 30 f1	f1305706   	r6 = or(r16,r23)
  2f5e68:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f5e6c:	59 d3 81 30	3081d359   	if (p2) r25 = memw(r1+r19<<#0x0) } 
  2f5e70:	07 58 2e f1	f12e5807 { 	r7 = or(r14,r24)
  2f5e74:	09 40 62 70	70624009   	r9 = r2
  2f5e78:	28 40 20 5c	5c204028   	if (!p0) jump:nt 0x2f5ec0 <hmx_convhbhNx1_stride2+0x140>
  2f5e7c:	00 c0 00 7f	7f00c000   	nop } 
  2f5e80:	59 59 37 fb	fb375959 { 	if (p2) r25 = sub(r25,r23)
  2f5e84:	08 57 31 f1	f1315708   	r8 = or(r17,r23)
  2f5e88:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f5e8c:	38 d3 81 30	3081d338   	if (p1) r24 = memw(r1+r19<<#0x0) } 
  2f5e90:	29 40 0a e2	e20a4029 { 	r9 += add(r10,#0x1)
  2f5e94:	07 59 2e f1	f12e5907   	r7 = or(r14,r25)
  2f5e98:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f5e9c:	eb ea 09 92	9209eaeb   	weight.b = mxmem(r9,r10):dilate } 
  2f5ea0:	38 58 37 fb	fb375838 { 	if (p1) r24 = sub(r24,r23)
  2f5ea4:	06 57 30 f1	f1305706   	r6 = or(r16,r23)
  2f5ea8:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f5eac:	59 d3 81 30	3081d359   	if (p2) r25 = memw(r1+r19<<#0x0) } 
  2f5eb0:	29 80 0b e2	e20b8029 { 	r9 += add(r11,#0x1)
  2f5eb4:	07 58 2e f1	f12e5807   	r7 = or(r14,r24)
  2f5eb8:	ec 47 08 92	920847ec   	activation.ub = mxmem(r8,r7)
  2f5ebc:	eb eb 09 92	9209ebeb   	weight.b = mxmem(r9,r11):dilate }  :endloop0
  2f5ec0:	59 59 37 fb	fb375959 { 	if (p2) r25 = sub(r25,r23)
  2f5ec4:	08 57 31 f1	f1315708   	r8 = or(r17,r23)
  2f5ec8:	17 e0 80 9d	9d80e017   	r23 = memw(r0++m1) } 
  2f5ecc:	29 40 0a e2	e20a4029 { 	r9 += add(r10,#0x1)
  2f5ed0:	07 59 2e f1	f12e5907   	r7 = or(r14,r25)
  2f5ed4:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f5ed8:	eb ea 09 92	9209eaeb   	weight.b = mxmem(r9,r10):dilate } 
  2f5edc:	0c 57 2f f1	f12f570c { 	r12 = or(r15,r23)
  2f5ee0:	ec 47 08 92	920847ec   	activation.ub = mxmem(r8,r7)
  2f5ee4:	eb eb 09 92	9209ebeb   	weight.b = mxmem(r9,r11):dilate } 
  2f5ee8:	29 40 0b e2	e20b4029 { 	r9 += add(r11,#0x1)
  2f5eec:	00 80 00 7f	7f008000   	nop
  2f5ef0:	0a ed ec a6	a6eced0a   	mxmem(r12,r13):after:sat.uh = acc:2x1 }  :endloop1
  2f5ef4:	00 5a 54 f2	f2545a00 { 	p0 = cmp.gt(r20,r26)
  2f5ef8:	1a 41 1a b0	b01a411a   	r26 = add(r26,#0x8)
  2f5efc:	c0 c0 dd 91	91ddc0c0   	r1:0 = memd(r29+#0x30) } 
  2f5f00:	a0 52 00 d3	d30052a0 { 	r1:0 = vaddw(r1:0,r19:18)
  2f5f04:	90 e0 df 5c	5cdfe090   	if (p0) jump:nt 0x2f5e20 <hmx_convhbhNx1_stride2+0xa0> } 
  2f5f08:	02 40 69 70	70694002 { 	r2 = r9
  2f5f0c:	00 40 5c 75	755c4000   	p0 = cmp.gt(r28,#0x0)
  2f5f10:	7c 78 df 5c	5cdf787c   	if (p0.new) jump:t 0x2f5e00 <hmx_convhbhNx1_stride2+0x80>
  2f5f14:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f5f18:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f5f1c:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f5f20:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f5f24:	1d 48 1d b0	b01d481d { 	r29 = add(r29,#0x40)
  2f5f28:	00 40 9f 52	529f4000   	jumpr r31
  2f5f2c:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f5f40 <hmx_v73_convhbhNx1_stride2_sparsity>:
  2f5f40:	1d 77 fd bf	bffd771d { 	r29 = add(r29,#-0x48)
  2f5f44:	ef 7f 03 78	78037fef   	r15 = #0x7ff
  2f5f48:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f5f4c:	48 c1 84 91	9184c148   	r8 = memw(r4+#0x28) } 
  2f5f50:	10 4f 06 f1	f1064f10 { 	r16 = and(r6,r15)
  2f5f54:	11 4f 08 f1	f1084f11   	r17 = and(r8,r15)
  2f5f58:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f5f5c:	0e 4f 07 f1	f1074f0e { 	r14 = and(r7,r15)
  2f5f60:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f5f64:	04 58 dd a1	a1dd5804 { 	memd(r29+#0x20) = r25:24
  2f5f68:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f5f6c:	ca 40 84 91	918440ca { 	r10 = memw(r4+#0x18)
  2f5f70:	0b c1 84 91	9184c10b   	r11 = memw(r4+#0x20) } 
  2f5f74:	1e 40 65 70	7065401e { 	r30 = r5
  2f5f78:	0c 40 c4 91	91c4400c   	r13:12 = memd(r4+#0x0)
  2f5f7c:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f5f80:	1b 02 14 01	0114021b { 	r4 = memw(r1+#0x4); 	r19 = memw(r1+#0x8) } 
  2f5f84:	24 41 04 8c	8c044124 { 	r4 = lsr(r4,#0x1)
  2f5f88:	11 00 15 04	04150011   	r5 = memw(r1+#0x10); 	r1 = memw(r1+#0x0) } 
  2f5f8c:	0f 4f 0c f1	f10c4f0f { 	r15 = and(r12,r15)
  2f5f90:	e4 7f e4 bf	bfe47fe4   	r4 = add(r4,#-0x1)
  2f5f94:	bc 40 80 91	918040bc   	r28 = memw(r0+#0x14)
  2f5f98:	36 c0 80 91	9180c036   	r22 = memw(r0+#0x4) } 
  2f5f9c:	56 42 16 8c	8c164256 { 	r22 = asl(r22,#0x2)
  2f5fa0:	0d 03 0a 02	020a030d   	r18 = memw(r0+#0x8); 	r21 = memw(r0+#0xc) } 
  2f5fa4:	52 42 52 80	80524252 { 	r19:18 = vaslw(r19:18,#0x2)
  2f5fa8:	75 40 15 b0	b0154075   	r21 = add(r21,#0x3)
  2f5fac:	00 00 0c 04	040c0000   	r20 = memw(r0+#0x10); 	r0 = memw(r0+#0x0) } 
  2f5fb0:	07 40 36 62	62364007 { 	m1 = r22
  2f5fb4:	15 42 15 8c	8c154215   	r21 = asr(r21,#0x2)
  2f5fb8:	3f 40 9e 9b	9b9e403f   	r31 = memw(r30++#0x4)
  2f5fbc:	11 c1 9d a1	a19dc111   	memw(r29+#0x44) = r1 } 
  2f5fc0:	1c 7c fc bf	bffc7c1c { 	r28 = add(r28,#-0x20)
  2f5fc4:	1a 41 00 78	7800411a   	r26 = #0x8
  2f5fc8:	10 c0 9d a1	a19dc010   	memw(r29+#0x40) = r0 } 
  2f5fcc:	08 42 35 60	60354208 { 	loop1(0x2f5ff0,r21)
  2f5fd0:	17 60 03 78	78036017   	r23 = #0x700
  2f5fd4:	00 c0 00 7f	7f00c000   	nop } 
  2f5fd8:	01 57 a6 c7	c7a65701 { 	p1 = !bitsclr(r6,r23)
  2f5fdc:	02 57 a8 c7	c7a85702   	p2 = !bitsclr(r8,r23)
  2f5fe0:	01 5a 45 f2	f2455a01   	p1 = cmp.gt(r5,r26)
  2f5fe4:	02 da 45 f2	f245da02   	p2 = cmp.gt(r5,r26) } 
  2f5fe8:	18 40 00 7c	7c004018 { 	r25:24 = combine(#0x0,#0x0)
  2f5fec:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f5ff0:	00 43 04 60	60044300 { 	loop0(0x2f6020,r4)
  2f5ff4:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2f5ff8:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f5ffc:	38 d3 81 30	3081d338   	if (p1) r24 = memw(r1+r19<<#0x0) } 
  2f6000:	38 58 37 fb	fb375838 { 	if (p1) r24 = sub(r24,r23)
  2f6004:	06 57 30 f1	f1305706   	r6 = or(r16,r23)
  2f6008:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f600c:	59 d3 81 30	3081d359   	if (p2) r25 = memw(r1+r19<<#0x0) } 
  2f6010:	07 58 2e f1	f12e5807 { 	r7 = or(r14,r24)
  2f6014:	09 40 62 70	70624009   	r9 = r2
  2f6018:	30 40 20 5c	5c204030   	if (!p0) jump:nt 0x2f6070 <hmx_v73_convhbhNx1_stride2_sparsity+0x130>
  2f601c:	00 c0 00 7f	7f00c000   	nop } 
  2f6020:	59 59 37 fb	fb375959 { 	if (p2) r25 = sub(r25,r23)
  2f6024:	08 57 31 f1	f1315708   	r8 = or(r17,r23)
  2f6028:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f602c:	38 d3 81 30	3081d338   	if (p1) r24 = memw(r1+r19<<#0x0) } 
  2f6030:	0f 70 2a 76	762a700f { 	r15 = and(r10,#-0x80)
  2f6034:	07 59 2e f1	f12e5907   	r7 = or(r14,r25)
  2f6038:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f603c:	eb ea 09 92	9209eaeb   	weight.b = mxmem(r9,r10):dilate } 
  2f6040:	02 40 00 00	00004002 { 	immext(#0x80)
  2f6044:	09 40 0f e2	e20f4009   	r9 += add(r15,##0x80)
  2f6048:	0f f0 2b 76	762bf00f   	r15 = and(r11,#-0x80) } 
  2f604c:	38 58 37 fb	fb375838 { 	if (p1) r24 = sub(r24,r23)
  2f6050:	06 57 30 f1	f1305706   	r6 = or(r16,r23)
  2f6054:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f6058:	59 d3 81 30	3081d359   	if (p2) r25 = memw(r1+r19<<#0x0) } 
  2f605c:	07 58 2e f1	f12e5807 { 	r7 = or(r14,r24)
  2f6060:	ec 47 08 92	920847ec   	activation.ub = mxmem(r8,r7)
  2f6064:	eb eb 09 92	9209ebeb   	weight.b = mxmem(r9,r11):dilate } 
  2f6068:	02 80 00 00	00008002 { 	immext(#0x80)
  2f606c:	09 c0 0f e2	e20fc009   	r9 += add(r15,##0x80) }  :endloop0
  2f6070:	59 59 37 fb	fb375959 { 	if (p2) r25 = sub(r25,r23)
  2f6074:	08 57 31 f1	f1315708   	r8 = or(r17,r23)
  2f6078:	1b 40 7f 70	707f401b   	r27 = r31
  2f607c:	17 e0 80 9d	9d80e017   	r23 = memw(r0++m1) } 
  2f6080:	0f 70 2a 76	762a700f { 	r15 = and(r10,#-0x80)
  2f6084:	07 59 2e f1	f12e5907   	r7 = or(r14,r25)
  2f6088:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f608c:	eb ea 09 92	9209eaeb   	weight.b = mxmem(r9,r10):dilate } 
  2f6090:	02 40 00 00	00004002 { 	immext(#0x80)
  2f6094:	09 40 0f e2	e20f4009   	r9 += add(r15,##0x80)
  2f6098:	0f f0 2b 76	762bf00f   	r15 = and(r11,#-0x80) } 
  2f609c:	ec 47 08 92	920847ec { 	activation.ub = mxmem(r8,r7)
  2f60a0:	eb eb 09 92	9209ebeb   	weight.b = mxmem(r9,r11):dilate } 
  2f60a4:	02 40 00 00	00004002 { 	immext(#0x80)
  2f60a8:	09 40 0f e2	e20f4009   	r9 += add(r15,##0x80)
  2f60ac:	1f 40 00 00	0000401f   	immext(#0x7c0)
  2f60b0:	ef c7 0c 76	760cc7ef   	r15 = and(r12,##0x7ff) } 
  2f60b4:	0c d7 2f f1	f12fd70c { 	r12 = or(r15,r23) } 
  2f60b8:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f60bc:	fb 7f fb bf	bffb7ffb   	r27 = add(r27,#-0x1)
  2f60c0:	37 40 9e 9b	9b9e4037   	r23 = memw(r30++#0x4)
  2f60c4:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f60c8:	00 40 5b 75	755b4000 { 	p0 = cmp.gt(r27,#0x0)
  2f60cc:	37 40 9e 9b	9b9e4037   	r23 = memw(r30++#0x4)
  2f60d0:	10 d9 f7 a6	a6f7d910   	cvt.uh = acc(r23):2x2 } 
  2f60d4:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f60d8:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f60dc:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f60b8 <hmx_v73_convhbhNx1_stride2_sparsity+0x178>
  2f60e0:	10 d9 f7 a6	a6f7d910   	cvt.uh = acc(r23):2x2 } 
  2f60e4:	5e 43 1f 8e	8e1f435e { 	r30 -= asl(r31,#0x3)
  2f60e8:	43 89 1f 8e	8e1f8943   	r3 -= asl(r31,#0x9)
  2f60ec:	18 cd ec a6	a6eccd18   	mxmem(r12,r13) = cvt }  :endloop1
  2f60f0:	00 5a 54 f2	f2545a00 { 	p0 = cmp.gt(r20,r26)
  2f60f4:	1a 41 1a b0	b01a411a   	r26 = add(r26,#0x8)
  2f60f8:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f60fc:	a0 52 00 d3	d30052a0 { 	r1:0 = vaddw(r1:0,r19:18)
  2f6100:	68 e0 df 5c	5cdfe068   	if (p0) jump:nt 0x2f5fcc <hmx_v73_convhbhNx1_stride2_sparsity+0x8c> } 
  2f6104:	00 40 5c 75	755c4000 { 	p0 = cmp.gt(r28,#0x0)
  2f6108:	02 40 69 70	70694002   	r2 = r9
  2f610c:	00 c1 dd 91	91ddc100   	r1:0 = memd(r29+#0x40) } 
  2f6110:	c3 49 1f 8e	8e1f49c3 { 	r3 += asl(r31,#0x9)
  2f6114:	80 40 00 b0	b0004080   	r0 = add(r0,#0x4)
  2f6118:	58 f0 df 5c	5cdff058   	if (p0) jump:t 0x2f5fc0 <hmx_v73_convhbhNx1_stride2_sparsity+0x80> } 
  2f611c:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2f6120:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f6124:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f6128:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f612c:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f6130:	1d 49 1d b0	b01d491d { 	r29 = add(r29,#0x48)
  2f6134:	00 40 9f 52	529f4000   	jumpr r31
  2f6138:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
  2f613c:	00 00 00 00 <unknown>

002f6140 <hmx_v73_convhbhNx1_stride2>:
  2f6140:	1d 77 fd bf	bffd771d { 	r29 = add(r29,#-0x48)
  2f6144:	ef 7f 03 78	78037fef   	r15 = #0x7ff
  2f6148:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f614c:	48 c1 84 91	9184c148   	r8 = memw(r4+#0x28) } 
  2f6150:	10 4f 06 f1	f1064f10 { 	r16 = and(r6,r15)
  2f6154:	11 4f 08 f1	f1084f11   	r17 = and(r8,r15)
  2f6158:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f615c:	0e 4f 07 f1	f1074f0e { 	r14 = and(r7,r15)
  2f6160:	1f 0a 16 ea	ea160a1f   	memd(r29+#0x10) = r21:20; 	memd(r29+#0x18) = r23:22 } 
  2f6164:	04 58 dd a1	a1dd5804 { 	memd(r29+#0x20) = r25:24
  2f6168:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f616c:	ca 40 84 91	918440ca { 	r10 = memw(r4+#0x18)
  2f6170:	0b c1 84 91	9184c10b   	r11 = memw(r4+#0x20) } 
  2f6174:	1e 40 65 70	7065401e { 	r30 = r5
  2f6178:	00 40 00 7f	7f004000   	nop
  2f617c:	0c 40 c4 91	91c4400c   	r13:12 = memd(r4+#0x0)
  2f6180:	89 c1 84 91	9184c189   	r9 = memw(r4+#0x30) } 
  2f6184:	00 40 00 7f	7f004000 { 	nop
  2f6188:	1b 02 14 01	0114021b   	r4 = memw(r1+#0x4); 	r19 = memw(r1+#0x8) } 
  2f618c:	24 41 04 8c	8c044124 { 	r4 = lsr(r4,#0x1)
  2f6190:	00 40 00 7f	7f004000   	nop
  2f6194:	11 00 15 04	04150011   	r5 = memw(r1+#0x10); 	r1 = memw(r1+#0x0) } 
  2f6198:	0f 4f 0c f1	f10c4f0f { 	r15 = and(r12,r15)
  2f619c:	e4 7f e4 bf	bfe47fe4   	r4 = add(r4,#-0x1)
  2f61a0:	bc 40 80 91	918040bc   	r28 = memw(r0+#0x14)
  2f61a4:	36 c0 80 91	9180c036   	r22 = memw(r0+#0x4) } 
  2f61a8:	56 42 16 8c	8c164256 { 	r22 = asl(r22,#0x2)
  2f61ac:	00 40 00 7f	7f004000   	nop
  2f61b0:	0d 03 0a 02	020a030d   	r18 = memw(r0+#0x8); 	r21 = memw(r0+#0xc) } 
  2f61b4:	52 42 52 80	80524252 { 	r19:18 = vaslw(r19:18,#0x2)
  2f61b8:	75 40 15 b0	b0154075   	r21 = add(r21,#0x3)
  2f61bc:	00 00 0c 04	040c0000   	r20 = memw(r0+#0x10); 	r0 = memw(r0+#0x0) } 
  2f61c0:	07 40 36 62	62364007 { 	m1 = r22
  2f61c4:	15 42 15 8c	8c154215   	r21 = asr(r21,#0x2)
  2f61c8:	3f 40 9e 9b	9b9e403f   	r31 = memw(r30++#0x4)
  2f61cc:	11 c1 9d a1	a19dc111   	memw(r29+#0x44) = r1 } 
  2f61d0:	00 40 00 7f	7f004000 { 	nop
  2f61d4:	00 40 00 7f	7f004000   	nop
  2f61d8:	00 40 00 7f	7f004000   	nop
  2f61dc:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f61e0:	1c 7c fc bf	bffc7c1c { 	r28 = add(r28,#-0x20)
  2f61e4:	1a 41 00 78	7800411a   	r26 = #0x8
  2f61e8:	10 c0 9d a1	a19dc010   	memw(r29+#0x40) = r0 } 
  2f61ec:	08 42 35 60	60354208 { 	loop1(0x2f6210,r21)
  2f61f0:	17 60 03 78	78036017   	r23 = #0x700
  2f61f4:	00 c0 00 7f	7f00c000   	nop } 
  2f61f8:	01 57 a6 c7	c7a65701 { 	p1 = !bitsclr(r6,r23)
  2f61fc:	02 57 a8 c7	c7a85702   	p2 = !bitsclr(r8,r23)
  2f6200:	01 5a 45 f2	f2455a01   	p1 = cmp.gt(r5,r26)
  2f6204:	02 da 45 f2	f245da02   	p2 = cmp.gt(r5,r26) } 
  2f6208:	18 40 00 7c	7c004018 { 	r25:24 = combine(#0x0,#0x0)
  2f620c:	07 c0 dd a1	a1ddc007   	memd(r29+#0x38) = r1:0 } 
  2f6210:	00 43 04 60	60044300 { 	loop0(0x2f6240,r4)
  2f6214:	00 40 44 75	75444000   	p0 = cmp.gt(r4,#0x0)
  2f6218:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f621c:	38 d3 81 30	3081d338   	if (p1) r24 = memw(r1+r19<<#0x0) } 
  2f6220:	38 58 37 fb	fb375838 { 	if (p1) r24 = sub(r24,r23)
  2f6224:	06 57 30 f1	f1305706   	r6 = or(r16,r23)
  2f6228:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f622c:	59 d3 81 30	3081d359   	if (p2) r25 = memw(r1+r19<<#0x0) } 
  2f6230:	07 58 2e f1	f12e5807 { 	r7 = or(r14,r24)
  2f6234:	09 40 62 70	70624009   	r9 = r2
  2f6238:	28 40 20 5c	5c204028   	if (!p0) jump:nt 0x2f6280 <hmx_v73_convhbhNx1_stride2+0x140>
  2f623c:	00 c0 00 7f	7f00c000   	nop } 
  2f6240:	59 59 37 fb	fb375959 { 	if (p2) r25 = sub(r25,r23)
  2f6244:	08 57 31 f1	f1315708   	r8 = or(r17,r23)
  2f6248:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f624c:	38 d3 81 30	3081d338   	if (p1) r24 = memw(r1+r19<<#0x0) } 
  2f6250:	29 40 0a e2	e20a4029 { 	r9 += add(r10,#0x1)
  2f6254:	07 59 2e f1	f12e5907   	r7 = or(r14,r25)
  2f6258:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f625c:	eb ea 09 92	9209eaeb   	weight.b = mxmem(r9,r10):dilate } 
  2f6260:	38 58 37 fb	fb375838 { 	if (p1) r24 = sub(r24,r23)
  2f6264:	06 57 30 f1	f1305706   	r6 = or(r16,r23)
  2f6268:	37 40 81 9b	9b814037   	r23 = memw(r1++#0x4)
  2f626c:	59 d3 81 30	3081d359   	if (p2) r25 = memw(r1+r19<<#0x0) } 
  2f6270:	29 80 0b e2	e20b8029 { 	r9 += add(r11,#0x1)
  2f6274:	07 58 2e f1	f12e5807   	r7 = or(r14,r24)
  2f6278:	ec 47 08 92	920847ec   	activation.ub = mxmem(r8,r7)
  2f627c:	eb eb 09 92	9209ebeb   	weight.b = mxmem(r9,r11):dilate }  :endloop0
  2f6280:	59 59 37 fb	fb375959 { 	if (p2) r25 = sub(r25,r23)
  2f6284:	08 57 31 f1	f1315708   	r8 = or(r17,r23)
  2f6288:	1b 40 7f 70	707f401b   	r27 = r31
  2f628c:	17 e0 80 9d	9d80e017   	r23 = memw(r0++m1) } 
  2f6290:	29 40 0a e2	e20a4029 { 	r9 += add(r10,#0x1)
  2f6294:	07 59 2e f1	f12e5907   	r7 = or(r14,r25)
  2f6298:	ec 47 06 92	920647ec   	activation.ub = mxmem(r6,r7)
  2f629c:	eb ea 09 92	9209eaeb   	weight.b = mxmem(r9,r10):dilate } 
  2f62a0:	29 40 0b e2	e20b4029 { 	r9 += add(r11,#0x1)
  2f62a4:	0c 57 2f f1	f12f570c   	r12 = or(r15,r23)
  2f62a8:	ec 47 08 92	920847ec   	activation.ub = mxmem(r8,r7)
  2f62ac:	eb eb 09 92	9209ebeb   	weight.b = mxmem(r9,r11):dilate } 
  2f62b0:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f62b4:	fb 7f fb bf	bffb7ffb   	r27 = add(r27,#-0x1)
  2f62b8:	37 40 9e 9b	9b9e4037   	r23 = memw(r30++#0x4)
  2f62bc:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f62c0:	00 40 5b 75	755b4000 { 	p0 = cmp.gt(r27,#0x0)
  2f62c4:	37 40 9e 9b	9b9e4037   	r23 = memw(r30++#0x4)
  2f62c8:	10 d9 f7 a6	a6f7d910   	cvt.uh = acc(r23):2x2 } 
  2f62cc:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f62d0:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f62d4:	ee 60 df 5c	5cdf60ee { 	if (p0) jump:nt 0x2f62b0 <hmx_v73_convhbhNx1_stride2+0x170>
  2f62d8:	10 d9 f7 a6	a6f7d910   	cvt.uh = acc(r23):2x2 } 
  2f62dc:	5e 43 1f 8e	8e1f435e { 	r30 -= asl(r31,#0x3)
  2f62e0:	43 89 1f 8e	8e1f8943   	r3 -= asl(r31,#0x9)
  2f62e4:	18 cd ec a6	a6eccd18   	mxmem(r12,r13) = cvt }  :endloop1
  2f62e8:	00 5a 54 f2	f2545a00 { 	p0 = cmp.gt(r20,r26)
  2f62ec:	1a 41 1a b0	b01a411a   	r26 = add(r26,#0x8)
  2f62f0:	e0 c0 dd 91	91ddc0e0   	r1:0 = memd(r29+#0x38) } 
  2f62f4:	a0 52 00 d3	d30052a0 { 	r1:0 = vaddw(r1:0,r19:18)
  2f62f8:	7c e0 df 5c	5cdfe07c   	if (p0) jump:nt 0x2f61ec <hmx_v73_convhbhNx1_stride2+0xac> } 
  2f62fc:	00 40 5c 75	755c4000 { 	p0 = cmp.gt(r28,#0x0)
  2f6300:	02 40 69 70	70694002   	r2 = r9
  2f6304:	00 c1 dd 91	91ddc100   	r1:0 = memd(r29+#0x40) } 
  2f6308:	c3 49 1f 8e	8e1f49c3 { 	r3 += asl(r31,#0x9)
  2f630c:	80 40 00 b0	b0004080   	r0 = add(r0,#0x4)
  2f6310:	6c f0 df 5c	5cdff06c   	if (p0) jump:t 0x2f61e0 <hmx_v73_convhbhNx1_stride2+0xa0> } 
  2f6314:	de 40 dd 91	91dd40de { 	r31:30 = memd(r29+#0x30)
  2f6318:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f631c:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f6320:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f6324:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f6328:	1d 49 1d b0	b01d491d { 	r29 = add(r29,#0x48)
  2f632c:	00 40 9f 52	529f4000   	jumpr r31
  2f6330:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f6340 <hmx_convhbh_dilate_stride1>:
  2f6340:	1d 71 fd bf	bffd711d { 	r29 = add(r29,#-0x78)
  2f6344:	0c 7c 03 78	78037c0c   	r12 = #0x7e0
  2f6348:	26 40 c4 91	91c44026   	r7:6 = memd(r4+#0x8)
  2f634c:	11 c0 e0 a6	a6e0c011   	mxclracc } 
  2f6350:	a7 42 07 8f	8f0742a7 { 	r7 = insert(r7,#0x2,#0x5)
  2f6354:	0b 40 67 70	7067400b   	r11 = r7
  2f6358:	dc 40 84 91	918440dc   	r28 = memw(r4+#0x18)
  2f635c:	e9 c0 84 91	9184c0e9   	r9 = memw(r4+#0x1c) } 
  2f6360:	3c 40 1c b0	b01c403c { 	r28 = add(r28,#0x1)
  2f6364:	0c 4c 07 f1	f1074c0c   	r12 = and(r7,r12)
  2f6368:	0d 0a 04 ea	ea040a0d   	memd(r29+#0x0) = r17:16; 	memd(r29+#0x8) = r19:18 } 
  2f636c:	8c 40 4c 8c	8c4c408c { 	r12 = ct0(r12)
  2f6370:	07 40 6b 70	706b4007   	r7 = r11
  2f6374:	8e 41 84 91	9184418e   	r14 = memw(r4+#0x30)
  2f6378:	02 d4 dd a1	a1ddd402   	memd(r29+#0x10) = r21:20 } 
  2f637c:	4c 7f ec bf	bfec7f4c { 	r12 = add(r12,#-0x6)
  2f6380:	e1 7f 6e 75	756e7fe1   	p1 = cmp.gt(r14,#-0x1)
  2f6384:	0a 40 c4 91	91c4400a   	r11:10 = memd(r4+#0x0)
  2f6388:	03 d6 dd a1	a1ddd603   	memd(r29+#0x18) = r23:22 } 
  2f638c:	ec 4c 80 c6	c6804cec { 	r12 = lsl(#0x1,r12)
  2f6390:	ad 40 4c 76	764c40ad   	r13 = sub(#0x5,r12)
  2f6394:	04 58 dd a1	a1dd5804   	memd(r29+#0x20) = r25:24
  2f6398:	05 da dd a1	a1ddda05   	memd(r29+#0x28) = r27:26 } 
  2f639c:	13 40 82 7a	7a824013 { 	r19 = mux(p1,#0x0,#0x4)
  2f63a0:	2e 40 81 91	9181402e   	r14 = memw(r1+#0x4)
  2f63a4:	4f c0 81 91	9181c04f   	r15 = memw(r1+#0x8) } 
  2f63a8:	24 40 2e 73	732e4024 { 	r4 = mux(p1,r14,#0x1)
  2f63ac:	36 40 c5 9b	9bc54036   	r23:22 = memd(r5++#0x8)
  2f63b0:	1d d3 9d a1	a19dd31d   	memw(r29+#0x74) = r19 } 
  2f63b4:	7b 45 16 c4	c416457b { 	r27 = addasl(r5,r22,#0x3)
  2f63b8:	1a 40 65 70	7065401a   	r26 = r5
  2f63bc:	11 00 1a 04	041a0011   	r18 = memw(r1+#0x10); 	r1 = memw(r1+#0x0) } 
  2f63c0:	65 40 80 91	91804065 { 	r5 = memw(r0+#0xc)
  2f63c4:	06 de dd a1	a1ddde06   	memd(r29+#0x30) = r31:30 } 
  2f63c8:	0a 44 1c ed	ed1c440a { 	r10 = mpyi(r28,r4)
  2f63cc:	09 02 08 01	01080209   	r16 = memw(r0+#0x4); 	r17 = memw(r0+#0x8) } 
  2f63d0:	12 4f 12 ed	ed124f12 { 	r18 = mpyi(r18,r15)
  2f63d4:	50 42 50 80	80504250   	r17:16 = vaslw(r17:16,#0x2)
  2f63d8:	99 40 80 91	91804099   	r25 = memw(r0+#0x10)
  2f63dc:	b5 c0 80 91	9180c0b5   	r21 = memw(r0+#0x14) } 
  2f63e0:	e4 7f e4 bf	bfe47fe4 { 	r4 = add(r4,#-0x1)
  2f63e4:	3f 0a 00 c0	c0000a3f   	r0 = memw(r0+#0x0); 	memd(r29+#0x38) = r23:22 } 
  2f63e8:	07 40 30 62	62304007 { 	m1 = r16
  2f63ec:	19 4f 19 ed	ed194f19   	r25 = mpyi(r25,r15)
  2f63f0:	11 4a 9d a1	a19d4a11   	memw(r29+#0x44) = r10
  2f63f4:	15 d2 9d a1	a19dd215   	memw(r29+#0x54) = r18 } 
  2f63f8:	4e 42 4e 80	804e424e { 	r15:14 = vaslw(r15:14,#0x2)
  2f63fc:	0b 5a dd a1	a1dd5a0b   	memd(r29+#0x58) = r27:26
  2f6400:	10 c2 9d a1	a19dc210   	memw(r29+#0x40) = r2 } 
  2f6404:	00 40 00 7f	7f004000 { 	nop
  2f6408:	14 d9 9d a1	a19dd914   	memw(r29+#0x50) = r25 } 
  2f640c:	00 c0 00 7f	7f00c000 { 	nop } 
  2f6410:	00 40 00 7f	7f004000 { 	nop
  2f6414:	00 40 00 7f	7f004000   	nop
  2f6418:	00 40 00 7f	7f004000   	nop
  2f641c:	00 c0 00 7f	7f00c000   	nop } 
  2f6420:	03 60 03 b0	b0036003 { 	r3 = add(r3,#0x100)
  2f6424:	fe c3 03 92	9203c3fe   	bias = mxmem2(r3) } 
  2f6428:	18 40 00 78	78004018 { 	r24 = #0x0
  2f642c:	00 40 00 7f	7f004000   	nop
  2f6430:	09 c0 dd a1	a1ddc009   	memd(r29+#0x48) = r1:0 } 
  2f6434:	01 41 2e f3	f32e4101 { 	r1 = sub(r1,r14)
  2f6438:	d9 7f ed bf	bfed7fd9   	r25 = add(r13,#-0x2)
  2f643c:	fb 42 9d 91	919d42fb   	r27 = memw(r29+#0x5c)
  2f6440:	0c c0 dd a1	a1ddc00c   	memd(r29+#0x60) = r1:0 } 
  2f6444:	98 59 cf cc	cccf5998 { 	r24 += asl(r15,r25)
  2f6448:	1f 4f 01 f3	f3014f1f   	r31 = add(r1,r15)
  2f644c:	fc 1c 02 3d	3d021cfc   	r2 = memw(r29+#0x40); 	r20 = memw(r29+#0x3c) } 
  2f6450:	00 40 00 7f	7f004000 { 	nop
  2f6454:	00 40 00 7f	7f004000   	nop
  2f6458:	1a 41 9d a1	a19d411a   	memw(r29+#0x68) = r1
  2f645c:	1b df 9d a1	a19ddf1b   	memw(r29+#0x6c) = r31 } 
  2f6460:	f4 7f f4 bf	bff47ff4 { 	r20 = add(r20,#-0x1)
  2f6464:	13 40 78 70	70784013   	r19 = r24
  2f6468:	36 40 db 9b	9bdb4036   	r23:22 = memd(r27++#0x8)
  2f646c:	b2 c2 9d 91	919dc2b2   	r18 = memw(r29+#0x54) } 
  2f6470:	41 41 16 c4	c4164141 { 	r1 = addasl(r1,r22,#0x2)
  2f6474:	93 4d d6 cc	ccd64d93   	r19 += asl(r22,r13)
  2f6478:	da 42 9d 91	919d42da   	r26 = memw(r29+#0x58)
  2f647c:	ca c1 9d 91	919dc1ca   	r10 = memw(r29+#0x38) } 
  2f6480:	00 42 2a 60	602a4200 { 	loop1(0x2f64a0,r10)
  2f6484:	01 40 57 75	75574001   	p1 = cmp.gt(r23,#0x0)
  2f6488:	01 53 52 f2	f2525301   	p1 = cmp.gt(r18,r19)
  2f648c:	7f c3 9d 91	919dc37f   	r31 = memw(r29+#0x6c) } 
  2f6490:	06 43 37 8f	8f374306 { 	r6 = insert(r23,#0x3,#0x8)
  2f6494:	df 42 16 8e	8e1642df   	r31 += asl(r22,#0x2)
  2f6498:	73 50 00 78	78005073   	r19 = #0x83
  2f649c:	36 c0 da 9b	9bdac036   	r23:22 = memd(r26++#0x8) } 
  2f64a0:	00 44 04 60	60044400 { 	loop0(0x2f64e0,r4)
  2f64a4:	5e 41 16 c4	c416415e   	r30 = addasl(r1,r22,#0x2)
  2f64a8:	06 57 66 f1	f1665706   	r6 = xor(r6,r23)
  2f64ac:	17 c0 00 78	7800c017   	r23 = #0x0 } 
  2f64b0:	9e 5e bf d5	d5bf5e9e { 	r30 = minu(r30,r31)
  2f64b4:	00 53 86 c7	c7865300   	p0 = bitsclr(r6,r19)
  2f64b8:	30 48 00 5c	5c004830   	if (p0.new) jump:nt 0x2f6510 <hmx_convhbh_dilate_stride1+0x1d0>
  2f64bc:	f2 ff 03 78	7803fff2   	r18 = #0x7ff } 
  2f64c0:	00 40 44 75	75444000 { 	p0 = cmp.gt(r4,#0x0)
  2f64c4:	f3 7f 03 78	78037ff3   	r19 = #0x7ff
  2f64c8:	36 40 9e 9b	9b9e4036   	r22 = memw(r30++#0x4)
  2f64cc:	37 cf 9e 30	309ecf37   	if (p1) r23 = memw(r30+r15<<#0x0) } 
  2f64d0:	06 52 e6 d3	d3e65206 { 	r7:6 = and(r7:6,r19:18)
  2f64d4:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f64d8:	08 42 3c f3	f33c4208   	r8 = sub(r2,r28)
  2f64dc:	18 c0 20 5c	5c20c018   	if (!p0) jump:nt 0x2f6500 <hmx_convhbh_dilate_stride1+0x1c0> } 
  2f64e0:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f64e4:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f64e8:	36 40 9e 9b	9b9e4036   	r22 = memw(r30++#0x4)
  2f64ec:	37 cf 9e 30	309ecf37   	if (p1) r23 = memw(r30+r15<<#0x0) } 
  2f64f0:	06 92 e6 d3	d3e69206 { 	r7:6 = and(r7:6,r19:18)
  2f64f4:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f64f8:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f64fc:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single }  :endloop0
  2f6500:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f6504:	08 dc 08 f3	f308dc08   	r8 = add(r8,r28) } 
  2f6508:	f0 47 06 92	920647f0 { 	activation.ub = mxmem(r6,r7):single
  2f650c:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single } 
  2f6510:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f6514:	73 90 00 78	78009073   	r19 = #0x83
  2f6518:	36 c0 da 9b	9bdac036   	r23:22 = memd(r26++#0x8) }  :endloop1
  2f651c:	a2 7f 34 61	61347fa2 { 	if (r20!=#0) jump:t 0x2f6460
  2f6520:	06 46 93 f1	f1934606   	r6 = and(r6,~r19)
  2f6524:	41 c3 9d 91	919dc341   	r1 = memw(r29+#0x68) } 
  2f6528:	0a c0 80 91	9180c00a { 	r10 = memw(r0+#0x0) } 
  2f652c:	01 4e 01 f3	f3014e01 { 	r1 = add(r1,r14)
  2f6530:	19 40 00 78	78004019   	r25 = #0x0
  2f6534:	0a eb ea a6	a6eaeb0a   	mxmem(r10,r11):after:sat.uh = acc:2x1 } 
  2f6538:	fb 42 9d 91	919d42fb { 	r27 = memw(r29+#0x5c)
  2f653c:	1a c1 9d a1	a19dc11a   	memw(r29+#0x68) = r1 } 
  2f6540:	fc 1c 02 3d	3d021cfc { 	r2 = memw(r29+#0x40); 	r20 = memw(r29+#0x3c) } 
  2f6544:	00 40 00 7f	7f004000 { 	nop
  2f6548:	00 40 00 7f	7f004000   	nop
  2f654c:	00 c0 00 7f	7f00c000   	nop } 
  2f6550:	00 40 00 7f	7f004000 { 	nop
  2f6554:	00 40 00 7f	7f004000   	nop
  2f6558:	00 40 00 7f	7f004000   	nop
  2f655c:	00 c0 00 7f	7f00c000   	nop } 
  2f6560:	f4 7f f4 bf	bff47ff4 { 	r20 = add(r20,#-0x1)
  2f6564:	13 40 78 70	70784013   	r19 = r24
  2f6568:	36 40 db 9b	9bdb4036   	r23:22 = memd(r27++#0x8)
  2f656c:	b2 c2 9d 91	919dc2b2   	r18 = memw(r29+#0x54) } 
  2f6570:	41 41 16 c4	c4164141 { 	r1 = addasl(r1,r22,#0x2)
  2f6574:	93 4d d6 cc	ccd64d93   	r19 += asl(r22,r13)
  2f6578:	da 42 9d 91	919d42da   	r26 = memw(r29+#0x58)
  2f657c:	ca c1 9d 91	919dc1ca   	r10 = memw(r29+#0x38) } 
  2f6580:	00 42 2a 60	602a4200 { 	loop1(0x2f65a0,r10)
  2f6584:	01 40 57 75	75574001   	p1 = cmp.gt(r23,#0x0)
  2f6588:	01 53 52 f2	f2525301   	p1 = cmp.gt(r18,r19)
  2f658c:	7f c3 9d 91	919dc37f   	r31 = memw(r29+#0x6c) } 
  2f6590:	06 43 37 8f	8f374306 { 	r6 = insert(r23,#0x3,#0x8)
  2f6594:	df 42 16 8e	8e1642df   	r31 += asl(r22,#0x2)
  2f6598:	00 40 00 7f	7f004000   	nop
  2f659c:	36 c0 da 9b	9bdac036   	r23:22 = memd(r26++#0x8) } 
  2f65a0:	00 44 04 60	60044400 { 	loop0(0x2f65e0,r4)
  2f65a4:	5e 41 16 c4	c416415e   	r30 = addasl(r1,r22,#0x2)
  2f65a8:	06 57 66 f1	f1665706   	r6 = xor(r6,r23)
  2f65ac:	17 c0 00 78	7800c017   	r23 = #0x0 } 
  2f65b0:	9e 5e bf d5	d5bf5e9e { 	r30 = minu(r30,r31)
  2f65b4:	f2 7f 03 78	78037ff2   	r18 = #0x7ff
  2f65b8:	f3 7f 03 78	78037ff3   	r19 = #0x7ff
  2f65bc:	00 c0 44 75	7544c000   	p0 = cmp.gt(r4,#0x0) } 
  2f65c0:	06 52 e6 d3	d3e65206 { 	r7:6 = and(r7:6,r19:18)
  2f65c4:	00 40 00 7f	7f004000   	nop
  2f65c8:	36 40 9e 9b	9b9e4036   	r22 = memw(r30++#0x4)
  2f65cc:	37 cf 9e 30	309ecf37   	if (p1) r23 = memw(r30+r15<<#0x0) } 
  2f65d0:	37 57 36 fb	fb365737 { 	if (p1) r23 = sub(r23,r22)
  2f65d4:	08 42 3c f3	f33c4208   	r8 = sub(r2,r28)
  2f65d8:	18 40 20 5c	5c204018   	if (!p0) jump:nt 0x2f6600 <hmx_convhbh_dilate_stride1+0x2c0>
  2f65dc:	00 c0 00 7f	7f00c000   	nop } 
  2f65e0:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f65e4:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f65e8:	36 40 9e 9b	9b9e4036   	r22 = memw(r30++#0x4)
  2f65ec:	37 cf 9e 30	309ecf37   	if (p1) r23 = memw(r30+r15<<#0x0) } 
  2f65f0:	06 92 e6 d3	d3e69206 { 	r7:6 = and(r7:6,r19:18)
  2f65f4:	37 57 36 fb	fb365737   	if (p1) r23 = sub(r23,r22)
  2f65f8:	f0 47 06 92	920647f0   	activation.ub = mxmem(r6,r7):single
  2f65fc:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single }  :endloop0
  2f6600:	46 56 e6 d3	d3e65646 { 	r7:6 = or(r7:6,r23:22)
  2f6604:	08 5c 08 f3	f3085c08   	r8 = add(r8,r28)
  2f6608:	36 c0 da 9b	9bdac036   	r23:22 = memd(r26++#0x8) } 
  2f660c:	22 40 09 e2	e2094022 { 	r2 += add(r9,#0x1)
  2f6610:	f0 87 06 92	920687f0   	activation.ub = mxmem(r6,r7):single
  2f6614:	e6 e9 08 92	9208e9e6   	weight.b = mxmem(r8,r9):single }  :endloop1
  2f6618:	a4 7f 34 61	61347fa4 { 	if (r20!=#0) jump:t 0x2f6560
  2f661c:	fd 7f ff 0f	0fff7ffd   	immext(#0xffffff40)
  2f6620:	86 47 06 76	76064786   	r6 = and(r6,##-0x84)
  2f6624:	41 c3 9d 91	919dc341   	r1 = memw(r29+#0x68) } 
  2f6628:	19 4c 19 f3	f3194c19 { 	r25 = add(r25,r12)
  2f662c:	0a e0 80 9d	9d80e00a   	r10 = memw(r0++m1) } 
  2f6630:	01 4e 01 f3	f3014e01 { 	r1 = add(r1,r14)
  2f6634:	00 59 45 f2	f2455900   	p0 = cmp.gt(r5,r25)
  2f6638:	84 78 df 5c	5cdf7884   	if (p0.new) jump:t 0x2f6538 <hmx_convhbh_dilate_stride1+0x1f8>
  2f663c:	0a eb ea a6	a6eaeb0a   	mxmem(r10,r11):after:sat.uh = acc:2x1 } 
  2f6640:	80 41 dd 91	91dd4180 { 	r1:0 = memd(r29+#0x60)
  2f6644:	99 c2 9d 91	919dc299   	r25 = memw(r29+#0x50) } 
  2f6648:	01 4f 01 f3	f3014f01 { 	r1 = add(r1,r15)
  2f664c:	00 51 00 f3	f3005100   	r0 = add(r0,r17)
  2f6650:	00 58 59 f2	f2595800   	p0 = cmp.gt(r25,r24)
  2f6654:	f6 c8 df 5c	5cdfc8f6   	if (p0.new) jump:nt 0x2f6434 <hmx_convhbh_dilate_stride1+0xf4> } 
  2f6658:	15 7c f5 bf	bff57c15 { 	r21 = add(r21,#-0x20)
  2f665c:	02 42 9d 91	919d4202   	r2 = memw(r29+#0x40)
  2f6660:	2a c2 9d 91	919dc22a   	r10 = memw(r29+#0x44) } 
  2f6664:	02 4a 02 f3	f3024a02 { 	r2 = add(r2,r10)
  2f6668:	00 40 55 75	75554000   	p0 = cmp.gt(r21,#0x0)
  2f666c:	20 41 dd 91	91dd4120   	r1:0 = memd(r29+#0x48)
  2f6670:	bb c3 9d 91	919dc3bb   	r27 = memw(r29+#0x74) } 
  2f6674:	01 5b 01 f3	f3015b01 { 	r1 = add(r1,r27)
  2f6678:	d6 50 df 5c	5cdf50d6   	if (p0) jump:t 0x2f6420 <hmx_convhbh_dilate_stride1+0xe0>
  2f667c:	02 29 40 60	60402902   	r0 = add(r0,#0x4); 	memw(r29+#0x40) = r2 } 
  2f6680:	de c0 dd 91	91ddc0de { 	r31:30 = memd(r29+#0x30) } 
  2f6684:	ba 40 dd 91	91dd40ba { 	r27:26 = memd(r29+#0x28)
  2f6688:	98 c0 dd 91	91ddc098   	r25:24 = memd(r29+#0x20) } 
  2f668c:	16 1e 1f 3e	3e1f1e16 { 	r23:22 = memd(r29+#0x18); 	r21:20 = memd(r29+#0x10) } 
  2f6690:	1d 4f 1d b0	b01d4f1d { 	r29 = add(r29,#0x78)
  2f6694:	00 40 9f 52	529f4000   	jumpr r31
  2f6698:	04 1e 0d 3e	3e0d1e04   	r19:18 = memd(r29+#0x8); 	r17:16 = memd(r29+#0x0) } 
		...

002f66c0 <hmx_v73_convhbh_dilate_stride1>:
  2f66c0:	c0 3f 00 5a	5a003fc0 { 	r0 = #-0x1; 	jumpr r31 } 
		...
