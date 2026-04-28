
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

003d7600 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv>:
  3d7920:	10 40 60 70	70604010 { 	r16 = r0
  3d7924:	e0 1c f4 eb	ebf41ce0   	memd(r29+#-0x10) = r17:16; 	allocframe(#0x70) } 
  3d7928:	ff 7f 0f 00	000f7fff { 	immext(#0xffffc0)
  3d792c:	e0 47 85 75	758547e0   	p0 = cmp.gtu(r5,##0xffffff)
  3d7930:	5c 40 83 91	9183405c   	r28 = memw(r3+#0x8)
  3d7934:	0b d4 dd a1	a1ddd40b   	memd(r29+#0x58) = r21:20 } 
  3d7938:	14 40 65 70	70654014 { 	r20 = r5
  3d793c:	65 0a 36 c3	c3360a65   	r6 = memw(r3+#0xc); 	memd(r29+#0x60) = r19:18 } 
  3d7940:	39 58 05 8c	8c055839 { 	r25 = lsr(r5,#0x18)
  3d7944:	18 40 61 70	70614018   	r24 = r1
  3d7948:	68 40 9c 91	919c4068   	r8 = memw(r28+#0xc)
  3d794c:	09 d8 dd a1	a1ddd809   	memd(r29+#0x48) = r25:24 } 
  3d7950:	28 45 08 8c	8c084528 { 	r8 = lsr(r8,#0x5)
  3d7954:	21 42 19 74	74194221   	if (p0) r1 = add(r25,#0x11)
  3d7958:	e7 44 3c 91	913c44e7   	r7 = memub(r28+#0x27)
  3d795c:	4f c0 9c 91	919cc04f   	r15 = memw(r28+#0x8) } 
  3d7960:	2c 45 07 8c	8c07452c { 	r12 = lsr(r7,#0x5)
  3d7964:	47 42 07 8c	8c074247   	r7 = asl(r7,#0x2)
  3d7968:	c9 44 3c 91	913c44c9   	r9 = memub(r28+#0x26)
  3d796c:	0a d6 dd a1	a1ddd60a   	memd(r29+#0x50) = r23:22 } 
  3d7970:	08 48 0c ed	ed0c4808 { 	r8 = mpyi(r12,r8)
  3d7974:	0e 40 69 70	7069400e   	r14 = r9
  3d7978:	2c 40 9c 91	919c402c   	r12 = memw(r28+#0x4)
  3d797c:	08 da dd a1	a1ddda08   	memd(r29+#0x40) = r27:26 } 
  3d7980:	a8 45 09 8e	8e0945a8 { 	r8 += lsr(r9,#0x5)
  3d7984:	0e 40 00 00	0000400e   	immext(#0x380)
  3d7988:	00 45 0e de	de0e4500   	r14 = and(##0x380,asl(r14,#0x5))
  3d798c:	8d c4 3c 91	913cc48d   	r13 = memub(r28+#0x24) } 
  3d7990:	6e 40 49 da	da49406e { 	r9 = or(r14,and(r9,#0x3))
  3d7994:	03 40 40 89	89404003   	r3 = p0
  3d7998:	16 42 04 f5	f5044216   	r23:22 = combine(r4,r2)
  3d799c:	08 c3 9d a1	a19dc308   	memw(r29+#0x20) = r3 } 
  3d79a0:	0d 4c 08 ef	ef084c0d { 	r13 += mpyi(r8,r12)
  3d79a4:	89 4f 47 da	da474f89   	r7 = or(r9,and(r7,#0x7c))
  3d79a8:	ac 44 3c 91	913c44ac   	r12 = memub(r28+#0x25)
  3d79ac:	0a c3 9d a1	a19dc30a   	memw(r29+#0x28) = r3 } 
  3d79b0:	0c cf 0d ef	ef0dcf0c { 	r12 += mpyi(r13,r15) } 
  3d79b4:	c7 ca 4c 8e	8e4ccac7 { 	r7 |= asl(r12,#0xa) } 
  3d79b8:	06 47 06 f3	f3064706 { 	r6 = add(r6,r7)
  3d79bc:	02 d2 a0 a1	a1a0d202   	memw(r0+#0x8) = r6.new } 
  3d79c0:	66 40 84 91	91844066 { 	r6 = memw(r4+#0xc)
  3d79c4:	03 d2 a0 a1	a1a0d203   	memw(r0+#0xc) = r6.new } 
  3d79c8:	51 40 82 91	91824051 { 	r17 = memw(r2+#0x8)
  3d79cc:	55 c0 9c 91	919cc055   	r21 = memw(r28+#0x8) } 
  3d79d0:	22 40 9c 91	919c4022 { 	r2 = memw(r28+#0x4)
  3d79d4:	6a c0 9c 91	919cc06a   	r10 = memw(r28+#0xc) } 
  3d79d8:	1a 40 62 70	7062401a { 	r26 = r2
  3d79dc:	9b 04 9a 02	029a049b   	r18 = memw(r17+#0x8); 	r19 = memw(r17+#0x10) } 
  3d79e0:	0c 40 20 5c	5c20400c { 	if (!p0) jump:nt 0x3d79f8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x3f8>
  3d79e4:	7b c0 91 91	9191c07b   	r27 = memw(r17+#0xc) } 
  3d79e8:	e4 70 b9 5b	5bb970e4 { 	call 0x1a5bb0 <__hexagon_udivsi3@plt>
  3d79ec:	40 c7 0a 8c	8c0ac740   	r0 = asl(r10,#0x7) } 
  3d79f0:	02 40 7a 70	707a4002 { 	r2 = r26
  3d79f4:	0a c0 60 70	7060c00a   	r10 = r0 } 
  3d79f8:	e4 43 13 b0	b01343e4 { 	r4 = add(r19,#0x1f)
  3d79fc:	20 40 15 75	75154020   	p0 = cmp.eq(r21,#0x1)
  3d7a00:	a7 44 31 91	913144a7   	r7 = memub(r17+#0x25)
  3d7a04:	c1 c0 91 91	9191c0c1   	r1 = memw(r17+#0x18) } 
  3d7a08:	46 48 07 8c	8c074846 { 	r6 = asl(r7,#0x8)
  3d7a0c:	e7 40 07 76	760740e7   	r7 = and(r7,#0x7)
  3d7a10:	c5 44 31 91	913144c5   	r5 = memub(r17+#0x26)
  3d7a14:	ed c0 91 91	9191c0ed   	r13 = memw(r17+#0x1c) } 
  3d7a18:	00 45 e5 de	dee54500 { 	r5 = and(#0xe0,asl(r5,#0x5))
  3d7a1c:	e9 40 05 76	760540e9   	r9 = and(r5,#0x7)
  3d7a20:	07 52 07 f3	f3075207   	r7 = add(r7,r18)
  3d7a24:	f3 c4 31 91	9131c4f3   	r19 = memub(r17+#0x27) } 
  3d7a28:	1c 40 00 00	0000401c { 	immext(#0x700)
  3d7a2c:	05 40 46 da	da464005   	r6 = or(r5,and(r6,##0x700))
  3d7a30:	09 5b 09 f3	f3095b09   	r9 = add(r9,r27)
  3d7a34:	43 c0 98 91	9198c043   	r3 = memw(r24+#0x8) } 
  3d7a38:	e6 43 53 da	da5343e6 { 	r19 = or(r6,and(r19,#0x1f))
  3d7a3c:	26 43 0d 8c	8c0d4326   	r6 = lsr(r13,#0x3)
  3d7a40:	05 41 91 91	91914105   	r5 = memw(r17+#0x20)
  3d7a44:	2d c0 96 91	9196c02d   	r13 = memw(r22+#0x4) } 
  3d7a48:	25 45 05 8c	8c054525 { 	r5 = lsr(r5,#0x5)
  3d7a4c:	04 7c 24 76	76247c04   	r4 = and(r4,#-0x20)
  3d7a50:	48 40 83 91	91834048   	r8 = memw(r3+#0x8)
  3d7a54:	0f c1 83 91	9183c10f   	r15 = memw(r3+#0x20) } 
  3d7a58:	1a 46 05 ed	ed05461a { 	r26 = mpyi(r5,r6)
  3d7a5c:	20 45 13 8c	8c134520   	r0 = lsr(r19,#0x5)
  3d7a60:	2d 40 8d 91	918d402d   	r13 = memw(r13+#0x4)
  3d7a64:	fc c0 83 91	9183c0fc   	r28 = memw(r3+#0x1c) } 
  3d7a68:	16 48 13 de	de134816 { 	r19 = sub(#0x0,lsr(r19,#0x8))
  3d7a6c:	26 45 0f 8c	8c0f4526   	r6 = lsr(r15,#0x5)
  3d7a70:	6b 40 83 91	9183406b   	r11 = memw(r3+#0xc)
  3d7a74:	06 cd 9d a1	a19dcd06   	memw(r29+#0x18) = r13 } 
  3d7a78:	2d 43 1c 8c	8c1c432d { 	r13 = lsr(r28,#0x3)
  3d7a7c:	02 47 93 85	85934702   	p2 = bitsclr(r19,#0x7)
  3d7a80:	8c 40 83 91	9183408c   	r12 = memw(r3+#0x10)
  3d7a84:	07 db 9d a1	a19ddb07   	memw(r29+#0x1c) = r27 } 
  3d7a88:	1b 46 0d ed	ed0d461b { 	r27 = mpyi(r13,r6)
  3d7a8c:	11 40 40 76	76404011   	r17 = sub(#0x0,r0)
  3d7a90:	00 40 15 f3	f3154000   	r0 = add(r21,r0)
  3d7a94:	c3 c0 83 91	9183c0c3   	r3 = memw(r3+#0x18) } 
  3d7a98:	01 47 b1 85	85b14701 { 	p1 = !bitsclr(r17,#0x7)
  3d7a9c:	23 40 02 75	75024023   	p3 = cmp.eq(r2,#0x1)
  3d7aa0:	6e 40 96 91	9196406e   	r14 = memw(r22+#0xc)
  3d7aa4:	09 c4 90 a1	a190c409   	memw(r16+#0x24) = r4 } 
  3d7aa8:	06 5a 90 a1	a1905a06 { 	memw(r16+#0x18) = r26
  3d7aac:	04 ce 90 a1	a190ce04   	memw(r16+#0x10) = r14 } 
  3d7ab0:	02 40 03 6b	6b034002 { 	p2 = and(p0,p3)
  3d7ab4:	04 40 04 7b	7b044004   	r4 = mux(p2,#0x0,#0x8)
  3d7ab8:	05 45 90 a1	a1904505   	memw(r16+#0x14) = r5
  3d7abc:	07 c9 90 a1	a190c907   	memw(r16+#0x1c) = r9 } 
  3d7ac0:	32 43 03 8c	8c034332 { 	r18 = lsr(r3,#0x3)
  3d7ac4:	08 47 90 a1	a1904708   	memw(r16+#0x20) = r7
  3d7ac8:	65 c0 98 91	9198c065   	r5 = memw(r24+#0xc) } 
  3d7acc:	38 40 40 76	76404038 { 	r24 = sub(#0x1,r0)
  3d7ad0:	86 0b 85 aa	aa850b86   	memw(r16+#0x28) = r5; 	memw(r16+#0x2c) = r6 } 
  3d7ad4:	0c 5b 90 a1	a1905b0c { 	memw(r16+#0x30) = r27
  3d7ad8:	0d cb 90 a1	a190cb0d   	memw(r16+#0x34) = r11 } 
  3d7adc:	0e 48 90 a1	a190480e { 	memw(r16+#0x38) = r8
  3d7ae0:	0f cc 90 a1	a190cc0f   	memw(r16+#0x3c) = r12 } 
  3d7ae4:	35 43 01 8c	8c014335 { 	r21 = lsr(r1,#0x3)
  3d7ae8:	17 42 c4 8c	8cc44217   	r23 = setbit(r4,#0x2)
  3d7aec:	9d 08 f5 c2	c2f5089d   	r5 = memw(r23+#0x8); 	memw(r29+#0x24) = r21 } 
  3d7af0:	17 c0 a4 74	74a4c017 { 	if (!p1) r23 = add(r4,#0x0) } 
  3d7af4:	85 c0 85 91	9185c085 { 	r5 = memw(r5+#0x10) } 
  3d7af8:	00 50 05 75	75055000 { 	p0 = cmp.eq(r5,#0x80)
  3d7afc:	48 d8 20 5c	5c20d848   	if (!p0.new) jump:t 0x3d7b88 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x588> } 
  3d7b00:	80 45 1d b0	b01d4580 { 	r0 = add(r29,#0x2c)
  3d7b04:	01 40 76 70	70764001   	r1 = r22
  3d7b08:	42 08 e2 c0	c0e20842   	r2 = memw(r22+#0x0); 	memw(r29+#0x10) = r2 } 
  3d7b0c:	03 40 42 89	89424003 { 	r3 = p2
  3d7b10:	02 d2 bd a1	a1bdd202   	memw(r29+#0x8) = r3.new } 
  3d7b14:	22 41 82 91	91824122 { 	r2 = memw(r2+#0x24)
  3d7b18:	03 cb 9d a1	a19dcb03   	memw(r29+#0xc) = r11 } 
  3d7b1c:	00 40 a2 50	50a24000 { 	callr r2
  3d7b20:	05 ca 9d a1	a19dca05   	memw(r29+#0x14) = r10 } 
  3d7b24:	80 45 1d b0	b01d4580 { 	r0 = add(r29,#0x2c)
  3d7b28:	b3 1c 81 3c	3c811cb3   	r1 = memw(r29+#0x20); 	r3 = memw(r29+#0x2c) } 
  3d7b2c:	40 c0 03 75	7503c040 { 	p0 = cmp.eq(r3,#0x2) } 
  3d7b30:	03 40 40 89	89404003 { 	r3 = p0
  3d7b34:	02 40 81 91	91814002   	r2 = memw(r1+#0x0)
  3d7b38:	08 d4 bd a1	a1bdd408   	memw(r29+#0x20) = r3.new } 
  3d7b3c:	22 c1 82 91	9182c122 { 	r2 = memw(r2+#0x24) } 
  3d7b40:	00 c0 a2 50	50a2c000 { 	callr r2 } 
  3d7b44:	02 c1 9d 91	919dc102 { 	r2 = memw(r29+#0x20) } 
  3d7b48:	00 40 42 85	85424000 { 	p0 = r2
  3d7b4c:	1c 58 20 5c	5c20581c   	if (!p0.new) jump:t 0x3d7b80 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x580>
  3d7b50:	aa 40 9d 47	479d40aa   	if (!p0.new) r10 = memw(r29+#0x14)
  3d7b54:	6b c0 9d 47	479dc06b   	if (!p0.new) r11 = memw(r29+#0xc) } 
  3d7b58:	03 47 d7 8c	8cd74703 { 	r3 = setbit(r23,#0x7)
  3d7b5c:	62 41 9d 91	919d4162   	r2 = memw(r29+#0x2c)
  3d7b60:	aa c0 9d 91	919dc0aa   	r10 = memw(r29+#0x14) } 
  3d7b64:	e0 40 02 75	750240e0 { 	p0 = cmp.eq(r2,#0x7)
  3d7b68:	17 60 03 74	74036017   	if (p0.new) r23 = add(r3,#0x0)
  3d7b6c:	6b 40 9d 91	919d406b   	r11 = memw(r29+#0xc)
  3d7b70:	82 c0 9d 91	919dc082   	r2 = memw(r29+#0x10) } 
  3d7b74:	43 c0 9d 91	919dc043 { 	r3 = memw(r29+#0x8) } 
  3d7b78:	02 40 43 85	85434002 { 	p2 = r3
  3d7b7c:	08 c0 00 58	5800c008   	jump 0x3d7b88 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x588> } 
  3d7b80:	42 1c 23 3c	3c231c42 { 	r3 = memw(r29+#0x8); 	r2 = memw(r29+#0x10) } 
  3d7b84:	02 c0 43 85	8543c002 { 	p2 = r3 } 
  3d7b88:	47 42 14 8c	8c144247 { 	r7 = asl(r20,#0x2)
  3d7b8c:	00 58 00 00	00005800   	immext(#0x60000)
  3d7b90:	c6 30 03 28	280330c6   	r3 = ##0x60000; 	r6 = r20 } 
  3d7b94:	1a 55 1a ed	ed1a551a { 	r26 = mpyi(r26,r21)
  3d7b98:	00 43 54 c7	c7544300   	p0 = bitsset(r20,r3)
  3d7b9c:	00 50 00 00	00005000   	immext(#0x40000)
  3d7ba0:	04 c4 00 78	7800c404   	r4 = ##0x40020 } 
  3d7ba4:	00 40 00 08	08004000 { 	immext(#0x80000000)
  3d7ba8:	00 5e 06 de	de065e00   	r6 = and(##0x80000000,asl(r6,#0x1e))
  3d7bac:	03 40 74 70	70744003   	r3 = r20
  3d7bb0:	f5 c0 13 76	7613c0f5   	r21 = and(r19,#0x7) } 
  3d7bb4:	7c 40 00 00	0000407c { 	immext(#0x1f00)
  3d7bb8:	06 40 43 da	da434006   	r3 = or(r6,and(r3,##0x1f00))
  3d7bbc:	01 44 94 c7	c7944401   	p1 = bitsclr(r20,r4)
  3d7bc0:	e6 c0 11 76	7611c0e6   	r6 = and(r17,#0x7) } 
  3d7bc4:	16 5b 12 ed	ed125b16 { 	r22 = mpyi(r18,r27)
  3d7bc8:	01 48 00 00	00004801   	immext(#0x20040)
  3d7bcc:	05 40 20 7a	7a204005   	r5 = mux(p0,##0x20040,#0x40)
  3d7bd0:	db c3 9d 91	919dc3db   	r27 = memw(r29+#0x78) } 
  3d7bd4:	00 49 10 b0	b0104900 { 	r0 = add(r16,#0x48)
  3d7bd8:	a3 43 25 f9	f92543a3   	if (!p1) r3 = or(r5,r3)
  3d7bdc:	e5 c0 18 76	7618c0e5   	r5 = and(r24,#0x7) } 
  3d7be0:	08 57 23 f1	f1235708 { 	r8 = or(r3,r23)
  3d7be4:	80 40 00 00	00004080   	immext(#0x2000)
  3d7be8:	83 40 c0 7e	7ec04083   	if (!p2) r3 = ##0x2004
  3d7bec:	26 c2 20 5c	5c20c226   	if (!p2) jump:nt 0x3d7c2c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x62c> } 
  3d7bf0:	08 44 47 da	da474408 { 	r7 = or(r8,and(r7,#0x20))
  3d7bf4:	00 50 94 85	85945000   	p0 = bitsclr(r20,#0x10)
  3d7bf8:	02 60 2a 73	732a6002   	r3:2 = combine(#0x0,r10)
  3d7bfc:	84 e6 15 fd	fd15e684   	if (!p0.new) r5:4 = combine(r21,r6) } 
  3d7c00:	01 40 07 7e	7e074001 { 	if (p0) r1 = #0x700
  3d7c04:	4c c0 20 5c	5c20c04c   	if (!p0) jump:nt 0x3d7c98 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x698> } 
  3d7c08:	00 47 00 00	00004700 { 	immext(#0x1c000)
  3d7c0c:	02 c0 14 76	7614c002   	r2 = and(r20,##0x1c000) } 
  3d7c10:	00 42 00 00	00004200 { 	immext(#0x8000)
  3d7c14:	00 40 02 75	75024000   	p0 = cmp.eq(r2,##0x8000)
  3d7c18:	02 6a 06 fd	fd066a02   	if (p0.new) r3:2 = combine(r6,r10)
  3d7c1c:	66 d8 20 5c	5c20d866   	if (!p0.new) jump:t 0x3d7cdc <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x6dc> } 
  3d7c20:	60 4c d6 5b	5bd64c60 { 	call 0x2894e0 <_Z25set_hmx_params_convw2b1x1P10hmx_paramsmmmmm>
  3d7c24:	04 d5 07 f5	f507d504   	r5:4 = combine(r7,r21) } 
  3d7c28:	6c c0 00 58	5800c06c { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7c2c:	00 43 b4 c7	c7b44300 { 	p0 = !bitsclr(r20,r3)
  3d7c30:	e3 60 8b 74	748b60e3   	if (!p0.new) r3 = add(r11,#0x7)
  3d7c34:	24 d8 00 5c	5c00d824   	if (p0.new) jump:t 0x3d7c74 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x674> } 
  3d7c38:	06 7f 63 da	da637f06 { 	r3 = or(r6,and(r3,#-0x8))
  3d7c3c:	64 3c 60 59	59603c64   	p0 = cmp.eq(r6,#0x0); 	r4 = memw(r29+#0x18) } 
  3d7c40:	01 40 04 75	75044001 { 	p1 = cmp.eq(r4,#0x0)
  3d7c44:	e4 c0 9d 91	919dc0e4   	r4 = memw(r29+#0x1c) } 
  3d7c48:	00 41 60 6b	6b604100 { 	p0 = and(p1,!p0)
  3d7c4c:	03 44 23 f3	f3234403   	r3 = sub(r4,r3)
  3d7c50:	24 c1 9d 91	919dc124   	r4 = memw(r29+#0x24) } 
  3d7c54:	01 40 43 75	75434001 { 	p1 = cmp.gt(r3,#0x0)
  3d7c58:	02 c3 44 f2	f244c302   	p2 = cmp.gt(r4,r3) } 
  3d7c5c:	41 c0 12 6b	6b12c041 { 	p1 = and(p2,and(p0,p1)) } 
  3d7c60:	63 c0 81 7a	7a81c063 { 	r3 = mux(p1,#0x3,#0x2) } 
  3d7c64:	04 40 03 73	73034004 { 	r4 = mux(p0,r3,#0x0)
  3d7c68:	23 c0 80 7e	7e80c023   	if (!p0) r3 = #0x1 } 
  3d7c6c:	03 c0 a4 74	74a4c003 { 	if (!p1) r3 = add(r4,#0x0) } 
  3d7c70:	08 c8 23 f1	f123c808 { 	r8 = or(r3,r8) } 
  3d7c74:	03 46 14 76	76144603 { 	r3 = and(r20,#0x30)
  3d7c78:	1a f0 42 24	2442f01a   	if (!cmp.eq(r3.new,#0x10)) jump:t 0x3d7ca8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x6a8> } 
  3d7c7c:	01 60 03 78	78036001 { 	r1 = #0x700
  3d7c80:	04 4a 06 f5	f5064a04   	r5:4 = combine(r6,r10)
  3d7c84:	23 41 9d 91	919d4123   	r3 = memw(r29+#0x24)
  3d7c88:	01 c8 9d a1	a19dc801   	memw(r29+#0x4) = r8 } 
  3d7c8c:	ea 43 d6 5b	5bd643ea { 	call 0x288460 <_Z30set_hmx_params_convw4b_stride1P10hmx_paramsmmmmmmm>
  3d7c90:	00 d5 9d a1	a19dd500   	memw(r29+#0x0) = r21 } 
  3d7c94:	36 c0 00 58	5800c036 { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7c98:	74 4b d6 5b	5bd64b74 { 	call 0x289380 <_Z25set_hmx_params_convw4b1x1P10hmx_paramsmmmmmm>
  3d7c9c:	01 60 03 78	78036001   	r1 = #0x700
  3d7ca0:	00 c7 9d a1	a19dc700   	memw(r29+#0x0) = r7 } 
  3d7ca4:	2e c0 00 58	5800c02e { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7ca8:	00 47 00 00	00004700 { 	immext(#0x1c000)
  3d7cac:	03 c2 14 76	7614c203   	r3 = and(r20,##0x1c010) } 
  3d7cb0:	00 42 00 00	00004200 { 	immext(#0x8000)
  3d7cb4:	00 40 03 75	75034000   	p0 = cmp.eq(r3,##0x8000)
  3d7cb8:	04 60 8a 74	748a6004   	if (!p0.new) r4 = add(r10,#0x0)
  3d7cbc:	1e d8 20 5c	5c20d81e   	if (!p0.new) jump:t 0x3d7cec <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x6ec> } 
  3d7cc0:	01 60 03 78	78036001 { 	r1 = #0x700
  3d7cc4:	04 40 6a 70	706a4004   	r4 = r10
  3d7cc8:	23 41 9d 91	919d4123   	r3 = memw(r29+#0x24)
  3d7ccc:	01 c8 9d a1	a19dc801   	memw(r29+#0x4) = r8 } 
  3d7cd0:	58 46 d6 5b	5bd64658 { 	call 0x288980 <_Z30set_hmx_params_convw2b_stride1P10hmx_paramsmmmmmmm>
  3d7cd4:	00 d5 9d a1	a19dd500   	memw(r29+#0x0) = r21 } 
  3d7cd8:	14 c0 00 58	5800c014 { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7cdc:	72 40 d6 5b	5bd64072 { 	call 0x287dc0 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm>
  3d7ce0:	02 4a 05 f5	f5054a02   	r3:2 = combine(r5,r10)
  3d7ce4:	04 d5 07 f5	f507d504   	r5:4 = combine(r7,r21) } 
  3d7ce8:	0c c0 00 58	5800c00c { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7cec:	01 60 03 78	78036001 { 	r1 = #0x700
  3d7cf0:	23 41 9d 91	919d4123   	r3 = memw(r29+#0x24)
  3d7cf4:	01 c8 9d a1	a19dc801   	memw(r29+#0x4) = r8 } 
  3d7cf8:	a4 7d d5 5b	5bd57da4 { 	call 0x287840 <_Z22set_hmx_params_stride1P10hmx_paramsmmmmmmm>
  3d7cfc:	00 d5 9d a1	a19dd500   	memw(r29+#0x0) = r21 } 
  3d7d00:	eb 5a 06 00	00065aeb { 	immext(#0x66bac0)
  3d7d04:	02 5c 49 6a	6a495c02   	r2 = add(pc,##0x66baf8)
  3d7d08:	20 5b 90 a1	a1905b20   	memw(r16+#0x80) = r27
  3d7d0c:	10 da 90 a1	a190da10   	memw(r16+#0x40) = r26 } 
  3d7d10:	11 56 90 a1	a1905611 { 	memw(r16+#0x44) = r22
  3d7d14:	21 d5 90 a1	a190d521   	memw(r16+#0x84) = r21 } 
  3d7d18:	e8 7f ff 0f	0fff7fe8 { 	immext(#0xfffffa00)
  3d7d1c:	83 43 82 91	91824383   	r3 = memw(r2+##-0x5e4)
  3d7d20:	fc 7f ff 0f	0fff7ffc   	immext(#0xffffff00)
  3d7d24:	04 c6 82 91	9182c604   	r4 = memw(r2+##-0xd0) } 
  3d7d28:	03 43 e3 6b	6be34303 { 	p3 = or(p3,!p3)
  3d7d2c:	21 40 83 91	91834021   	r1 = memw(r3+#0x4)
  3d7d30:	18 c4 02 20	2002c418   	if (cmp.eq(r1.new,r4)) jump:nt 0x3d7d58 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x758> } 
  3d7d34:	c6 70 b9 5b	5bb970c6 { 	call 0x1a5ec0 <strcmp@plt>
  3d7d38:	fc 7f ff 0f	0fff7ffc   	immext(#0xffffff00)
  3d7d3c:	00 c6 82 91	9182c600   	r0 = memw(r2+##-0xd0) } 
  3d7d40:	03 40 00 75	75004003 { 	p3 = cmp.eq(r0,#0x0)
  3d7d44:	12 4b 00 5c	5c004b12   	if (p3.new) jump:nt 0x3d7d64 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x764>
  3d7d48:	42 d9 9d 47	479dd942   	if (!p3.new) r2 = memw(r29+#0x28) } 
  3d7d4c:	02 40 42 85	85424002 { 	p2 = r2
  3d7d50:	a0 da 20 5c	5c20daa0   	if (!p2.new) jump:t 0x3d7e8c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x88c> } 
  3d7d54:	0c c0 00 58	5800c00c { 	jump 0x3d7d6c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x76c> } 
  3d7d58:	42 c1 9d 91	919dc142 { 	r2 = memw(r29+#0x28) } 
  3d7d5c:	02 40 42 85	85424002 { 	p2 = r2
  3d7d60:	08 c0 00 58	5800c008   	jump 0x3d7d6c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x76c> } 
  3d7d64:	42 c1 9d 91	919dc142 { 	r2 = memw(r29+#0x28) } 
  3d7d68:	02 c0 42 85	8542c002 { 	p2 = r2 } 
  3d7d6c:	03 43 90 91	91904303 { 	r3 = memw(r16+#0x60)
  3d7d70:	22 c3 90 91	9190c322   	r2 = memw(r16+#0x64) } 
  3d7d74:	00 40 03 75	75034000 { 	p0 = cmp.eq(r3,#0x0)
  3d7d78:	23 40 03 b0	b0034023   	r3 = add(r3,#0x1)
  3d7d7c:	01 40 02 75	75024001   	p1 = cmp.eq(r2,#0x0)
  3d7d80:	c0 d4 b0 46	46b0d4c0   	if (!p0.new) memw(r16+#0x60) = r3.new } 
  3d7d84:	22 40 a2 74	74a24022 { 	if (!p1) r2 = add(r2,#0x1)
  3d7d88:	63 43 90 91	91904363   	r3 = memw(r16+#0x6c)
  3d7d8c:	c9 d4 b0 44	44b0d4c9   	if (!p1) memw(r16+#0x64) = r2.new } 
  3d7d90:	01 40 03 75	75034001 { 	p1 = cmp.eq(r3,#0x0)
  3d7d94:	42 c3 90 91	9190c342   	r2 = memw(r16+#0x68) } 
  3d7d98:	00 40 02 75	75024000 { 	p0 = cmp.eq(r2,#0x0)
  3d7d9c:	22 40 02 b0	b0024022   	r2 = add(r2,#0x1)
  3d7da0:	d0 d2 b0 46	46b0d2d0   	if (!p0.new) memw(r16+#0x68) = r2.new } 
  3d7da4:	22 40 03 b0	b0034022 { 	r2 = add(r3,#0x1)
  3d7da8:	03 cc 50 74	7450cc03   	if (p2) r3 = add(r16,#0x60) } 
  3d7dac:	22 40 59 74	74594022 { 	if (p2) r2 = add(r25,#0x1)
  3d7db0:	84 40 43 74	74434084   	if (p2) r4 = add(r3,#0x4)
  3d7db4:	50 42 20 5c	5c204250   	if (!p2) jump:nt 0x3d7e4c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x84c>
  3d7db8:	d9 c2 90 44	4490c2d9   	if (!p1) memw(r16+#0x6c) = r2 } 
  3d7dbc:	0b 41 00 69	6900410b { 	loop0(0x3d7dd0,#0x3)
  3d7dc0:	05 c0 83 91	9183c005   	r5 = memw(r3+#0x0) } 
  3d7dc4:	1c 44 02 de	de02441c { 	r2 = add(#0x1,lsr(r2,#0x4))
  3d7dc8:	26 43 05 8c	8c054326   	r6 = lsr(r5,#0x3)
  3d7dcc:	00 c0 05 75	7505c000   	p0 = cmp.eq(r5,#0x0) } 
  3d7dd0:	06 42 06 ed	ed064206 { 	r6 = mpyi(r6,r2)
  3d7dd4:	84 40 04 b0	b0044084   	r4 = add(r4,#0x4)
  3d7dd8:	47 00 45 50	50450047   	r5 = r4; 	r7 = memw(r4+#0x0) } 
  3d7ddc:	19 70 66 da	da667019 { 	r6 = or(r25,and(r6,#-0x80))
  3d7de0:	00 40 07 75	75074000   	p0 = cmp.eq(r7,#0x0)
  3d7de4:	03 40 65 70	70654003   	r3 = r5
  3d7de8:	00 d6 a3 44	44a3d600   	if (!p0) memw(r3+#0x0) = r6.new } 
  3d7dec:	26 83 07 8c	8c078326 { 	r6 = lsr(r7,#0x3)
  3d7df0:	00 c0 00 7f	7f00c000   	nop }  :endloop0
  3d7df4:	02 42 06 ed	ed064202 { 	r2 = mpyi(r6,r2)
  3d7df8:	03 40 43 89	89434003   	r3 = p3
  3d7dfc:	0a d2 bd a1	a1bdd20a   	memw(r29+#0x28) = r3.new } 
