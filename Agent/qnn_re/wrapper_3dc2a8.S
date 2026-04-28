
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

003d7600 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv>:
  3dc2a8:	b0 1d f4 eb	ebf41db0 { 	memd(r29+#-0x10) = r17:16; 	allocframe(#0xd8) } 
  3dc2ac:	16 44 01 f5	f5014416 { 	r23:22 = combine(r1,r4)
  3dc2b0:	04 40 63 70	70634004   	r4 = r3
  3dc2b4:	bf 0a 13 c2	c2130abf   	r3 = memw(r1+#0x8); 	memd(r29+#0xb8) = r23:22 } 
  3dc2b8:	1b 0a 05 c2	c2050a1b { 	r5 = memw(r0+#0x8); 	memd(r29+#0x18) = r7:6 } 
  3dc2bc:	06 41 83 91	91834106 { 	r6 = memw(r3+#0x20)
  3dc2c0:	15 da dd a1	a1ddda15   	memd(r29+#0xa8) = r27:26 } 
  3dc2c4:	26 45 06 8c	8c064526 { 	r6 = lsr(r6,#0x5)
  3dc2c8:	15 43 1d b0	b01d4315   	r21 = add(r29,#0x18)
  3dc2cc:	c6 0a 32 c7	c7320ac6   	r2 = memw(r3+#0x1c); 	memd(r29+#0xc0) = r21:20 } 
  3dc2d0:	27 43 02 8c	8c024327 { 	r7 = lsr(r2,#0x3)
  3dc2d4:	18 40 60 70	70604018   	r24 = r0
  3dc2d8:	c3 40 83 91	918340c3   	r3 = memw(r3+#0x18)
  3dc2dc:	16 d8 dd a1	a1ddd816   	memd(r29+#0xb0) = r25:24 } 
  3dc2e0:	06 46 07 ed	ed074606 { 	r6 = mpyi(r7,r6)
  3dc2e4:	23 43 03 8c	8c034323   	r3 = lsr(r3,#0x3)
  3dc2e8:	00 55 18 f5	f5185500   	r1:0 = combine(r24,r21)
  3dc2ec:	19 d2 dd a1	a1ddd219   	memd(r29+#0xc8) = r19:18 } 
  3dc2f0:	1b 43 06 ed	ed06431b { 	r27 = mpyi(r6,r3)
  3dc2f4:	56 07 53 08	08530756   	r3 = memw(r5+#0x20); 	r6 = memw(r5+#0x1c) } 
  3dc2f8:	23 45 03 8c	8c034523 { 	r3 = lsr(r3,#0x5)
  3dc2fc:	26 43 06 8c	8c064326   	r6 = lsr(r6,#0x3)
  3dc300:	08 08 55 c6	c6550808   	r5 = memw(r5+#0x18); 	memw(r29+#0x0) = r16 } 
  3dc304:	06 43 06 ed	ed064306 { 	r6 = mpyi(r6,r3)
  3dc308:	27 43 05 8c	8c054327   	r7 = lsr(r5,#0x3)
  3dc30c:	02 57 11 f5	f5115702   	r3:2 = combine(r17,r23)
  3dc310:	05 c0 76 70	7076c005   	r5 = r22 } 
  3dc314:	1a 47 06 ed	ed06471a { 	r26 = mpyi(r6,r7)
  3dc318:	06 db ff 5b	5bffdb06   	call 0x3d7920 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x320> } 
  3dc31c:	13 42 15 b0	b0154213 { 	r19 = add(r21,#0x10)
  3dc320:	15 49 15 b0	b0154915   	r21 = add(r21,#0x48)
  3dc324:	89 3c 92 02	02923c89   	r2 = memw(r17+#0x8); 	r17 = memw(r29+#0x20) } 
  3dc328:	14 43 13 b0	b0134314 { 	r20 = add(r19,#0x18)
  3dc32c:	32 c1 9d 91	919dc132   	r18 = memw(r29+#0x24) } 
  3dc330:	22 02 23 01	01230222 { 	r3 = memw(r2+#0x4); 	r2 = memw(r2+#0x8) } 
  3dc334:	20 40 02 75	75024020 { 	p0 = cmp.eq(r2,#0x1)
  3dc338:	21 c0 03 75	7503c021   	p1 = cmp.eq(r3,#0x1) } 
  3dc33c:	00 c0 01 6b	6b01c000 { 	p0 = and(p0,p1) } 
  3dc340:	02 40 40 89	89404002 { 	r2 = p0
  3dc344:	00 50 91 85	85915000   	p0 = bitsclr(r17,#0x10)
  3dc348:	ba 58 20 5c	5c2058ba   	if (!p0.new) jump:t 0x3dc4b4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4eb4>
  3dc34c:	05 d6 bd a1	a1bdd605   	memw(r29+#0x14) = r2.new } 
  3dc350:	42 c0 97 91	9197c042 { 	r2 = memw(r23+#0x8) } 
  3dc354:	37 40 82 91	91824037 { 	r23 = memw(r2+#0x4)
  3dc358:	0a 40 00 00	0000400a   	immext(#0x280)
  3dc35c:	40 c0 02 24	2402c040   	if (cmp.eq(r23.new,#0x0)) jump:nt 0x3dc5f4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4ff4> } 
  3dc360:	58 42 5a 80	805a4258 { 	r25:24 = vaslw(r27:26,#0x2)
  3dc364:	03 42 16 76	76164203   	r3 = and(r22,#0x10)
  3dc368:	b2 00 0e 48	480e00b2   	r22 = #0x0; 	r2 = memw(r19+#0x0) } 
  3dc36c:	00 40 03 75	75034000 { 	p0 = cmp.eq(r3,#0x0)
  3dc370:	1a c0 00 78	7800c01a   	r26 = #0x0 } 
  3dc374:	03 40 40 89	89404003 { 	r3 = p0
  3dc378:	00 40 59 75	75594000   	p0 = cmp.gt(r25,#0x0)
  3dc37c:	04 d4 bd a1	a1bdd404   	memw(r29+#0x10) = r3.new } 
  3dc380:	03 40 40 89	89404003 { 	r3 = p0
  3dc384:	00 40 58 75	75584000   	p0 = cmp.gt(r24,#0x0)
  3dc388:	03 d4 bd a1	a1bdd403   	memw(r29+#0xc) = r3.new } 
  3dc38c:	03 40 40 89	89404003 { 	r3 = p0
  3dc390:	02 d2 bd a1	a1bdd202   	memw(r29+#0x8) = r3.new } 
  3dc394:	63 c0 9d 91	919dc063 { 	r3 = memw(r29+#0xc) } 
  3dc398:	00 40 43 85	85434000 { 	p0 = r3
  3dc39c:	04 68 16 74	74166804   	if (p0.new) r4 = add(r22,#0x40)
  3dc3a0:	e3 67 19 74	741967e3   	if (p0.new) r3 = add(r25,#0x3f)
  3dc3a4:	1c c8 20 5c	5c20c81c   	if (!p0.new) jump:nt 0x3dc3d0 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4dd0> } 
  3dc3a8:	23 c6 03 8c	8c03c623 { 	r3 = lsr(r3,#0x6) } 
  3dc3ac:	e5 7f e3 bf	bfe37fe5 { 	r5 = add(r3,#-0x1)
  3dc3b0:	20 40 83 75	75834020   	p0 = cmp.gtu(r3,#0x1)
  3dc3b4:	03 d6 02 f3	f302d603   	r3 = add(r2,r22) } 
  3dc3b8:	10 40 05 60	60054010 { 	loop0(0x3dc3c0,r5)
  3dc3bc:	0a c0 20 5c	5c20c00a   	if (!p0) jump:nt 0x3dc3cc <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4dcc> } 
  3dc3c0:	03 84 02 f3	f3028403 { 	r3 = add(r2,r4)
  3dc3c4:	04 48 04 b0	b0044804   	r4 = add(r4,#0x40)
  3dc3c8:	00 c0 03 94	9403c000   	dcfetch(r3+#0x0) }  :endloop0
  3dc3cc:	00 c0 03 94	9403c000 { 	dcfetch(r3+#0x0) } 
  3dc3d0:	42 c0 9d 91	919dc042 { 	r2 = memw(r29+#0x8) } 
  3dc3d4:	00 40 42 85	85424000 { 	p0 = r2
  3dc3d8:	e3 67 18 74	741867e3   	if (p0.new) r3 = add(r24,#0x3f)
  3dc3dc:	1e c8 20 5c	5c20c81e   	if (!p0.new) jump:nt 0x3dc410 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e10> } 
  3dc3e0:	c2 00 04 48	480400c2 { 	r4 = #0x0; 	r2 = memw(r20+#0x0) } 
  3dc3e4:	23 c6 03 8c	8c03c623 { 	r3 = lsr(r3,#0x6) } 
  3dc3e8:	e5 7f e3 bf	bfe37fe5 { 	r5 = add(r3,#-0x1)
  3dc3ec:	20 40 83 75	75834020   	p0 = cmp.gtu(r3,#0x1)
  3dc3f0:	03 44 02 f3	f3024403   	r3 = add(r2,r4)
  3dc3f4:	04 c8 04 b0	b004c804   	r4 = add(r4,#0x40) } 
  3dc3f8:	10 40 05 60	60054010 { 	loop0(0x3dc400,r5)
  3dc3fc:	0a c0 20 5c	5c20c00a   	if (!p0) jump:nt 0x3dc40c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e0c> } 
  3dc400:	03 84 02 f3	f3028403 { 	r3 = add(r2,r4)
  3dc404:	04 48 04 b0	b0044804   	r4 = add(r4,#0x40)
  3dc408:	00 c0 03 94	9403c000   	dcfetch(r3+#0x0) }  :endloop0
  3dc40c:	00 c0 03 94	9403c000 { 	dcfetch(r3+#0x0) } 
  3dc410:	82 c0 9d 91	919dc082 { 	r2 = memw(r29+#0x10) } 
  3dc414:	00 40 42 85	85424000 { 	p0 = r2
  3dc418:	80 74 13 fd	fd137480   	if (!p0.new) r1:0 = combine(r19,r20)
  3dc41c:	1e 58 20 5c	5c20581e   	if (!p0.new) jump:t 0x3dc450 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e50>
  3dc420:	a2 c0 9d 47	479dc0a2   	if (!p0.new) r2 = memw(r29+#0x14) } 
  3dc424:	a2 c0 9d 91	919dc0a2 { 	r2 = memw(r29+#0x14) } 
  3dc428:	00 40 42 85	85424000 { 	p0 = r2
  3dc42c:	00 74 13 fd	fd137400   	if (p0.new) r1:0 = combine(r19,r20)
  3dc430:	82 71 12 fd	fd127182   	if (!p0.new) r3:2 = combine(r18,r17)
  3dc434:	24 d8 20 5c	5c20d824   	if (!p0.new) jump:t 0x3dc470 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e70> } 
  3dc438:	82 44 9d 91	919d4482 { 	r2 = memw(r29+#0x90)
  3dc43c:	28 c0 c2 26	26c2c028   	if (!cmp.gt(r2.new,#-0x1)) jump:nt 0x3dc488 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e88> } 
  3dc440:	c0 74 e1 5b	5be174c0 { 	call 0x2eadc0 <hmx_v73_convbbb1x1_stride1>
  3dc444:	02 51 12 f5	f5125102   	r3:2 = combine(r18,r17)
  3dc448:	04 d5 10 f5	f510d504   	r5:4 = combine(r16,r21) } 
  3dc44c:	24 c0 00 58	5800c024 { 	jump 0x3dc494 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e94> } 
  3dc450:	00 40 42 85	85424000 { 	p0 = r2
  3dc454:	82 71 12 fd	fd127182   	if (!p0.new) r3:2 = combine(r18,r17)
  3dc458:	84 75 10 fd	fd107584   	if (!p0.new) r5:4 = combine(r16,r21)
  3dc45c:	18 d8 20 5c	5c20d818   	if (!p0.new) jump:t 0x3dc480 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e80> } 
  3dc460:	90 61 e2 5b	5be26190 { 	call 0x2f0780 <hmx_v73_convbnb1x1_stride1>
  3dc464:	02 51 12 f5	f5125102   	r3:2 = combine(r18,r17)
  3dc468:	04 d5 10 f5	f510d504   	r5:4 = combine(r16,r21) } 
  3dc46c:	14 c0 00 58	5800c014 { 	jump 0x3dc494 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e94> } 
  3dc470:	e8 4f e2 5b	5be24fe8 { 	call 0x2ee440 <hmx_v73_convbbb_stride1>
  3dc474:	00 54 13 f5	f5135400   	r1:0 = combine(r19,r20)
  3dc478:	04 d5 10 f5	f510d504   	r5:4 = combine(r16,r21) } 
  3dc47c:	0c c0 00 58	5800c00c { 	jump 0x3dc494 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e94> } 
  3dc480:	80 ec e2 5b	5be2ec80 { 	call 0x2f1d80 <hmx_v73_convbnb_stride1> } 
  3dc484:	08 c0 00 58	5800c008 { 	jump 0x3dc494 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4e94> } 
  3dc488:	fc 64 e4 5b	5be464fc { 	call 0x300e80 <hmx_v73_dwconvbbb1x1_stride1>
  3dc48c:	02 51 12 f5	f5125102   	r3:2 = combine(r18,r17)
  3dc490:	04 d5 10 f5	f510d504   	r5:4 = combine(r16,r21) } 
  3dc494:	3a 40 1a b0	b01a403a { 	r26 = add(r26,#0x1)
  3dc498:	42 41 9d 91	919d4142   	r2 = memw(r29+#0x28)
  3dc49c:	18 c8 5d 3e	3e5dc818   	memw(r29+#0x40) += r24 } 
  3dc4a0:	42 42 1b c4	c41b4242 { 	r2 = addasl(r2,r27,#0x2)
  3dc4a4:	00 57 1a f2	f21a5700   	p0 = cmp.eq(r26,r23)
  3dc4a8:	7a 78 ff 5c	5cff787a   	if (!p0.new) jump:t 0x3dc394 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x4d94>
  3dc4ac:	0a d6 bd a1	a1bdd60a   	memw(r29+#0x28) = r2.new } 
