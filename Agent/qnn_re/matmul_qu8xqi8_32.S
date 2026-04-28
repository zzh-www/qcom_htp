
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

0021be60 <matmul_qu8xqi8_32>:
  21be60:	e8 41 05 c4	c40541e8 { 	r8 = addasl(r1,r5,#0x7)
  21be64:	7a 40 04 10	1004407a   	p0 = cmp.eq(r4,#0x0); if (p0.new) jump:nt 0x21bf54 <matmul_qu8xqi8_32+0xf4>
  21be68:	06 c0 9d 47	479dc006   	if (!p0.new) r6 = memw(r29+#0x0) } 
  21be6c:	28 5f 05 00	00055f28 { 	immext(#0x57ca00)
  21be70:	0c 4a 49 6a	6a494a0c   	r12 = add(pc,##0x57ca14)
  21be74:	00 40 05 75	75054000   	p0 = cmp.eq(r5,#0x0)
  21be78:	01 c0 06 75	7506c001   	p1 = cmp.eq(r6,#0x0) } 
  21be7c:	47 45 05 8c	8c054547 { 	r7 = asl(r5,#0x5)
  21be80:	09 c0 00 78	7800c009   	r9 = #0x0 } 
  21be84:	00 c0 0c 28	280cc000 { 	v0 = vmem(r12+#0x0) } 
  21be88:	8c 41 02 fd	fd02418c { 	if (!p0) r13:12 = combine(r2,r1)
  21be8c:	32 40 00 5c	5c004032   	if (p0) jump:nt 0x21beec <matmul_qu8xqi8_32+0x8c>
  21be90:	01 c0 03 28	2803c001   	v1 = vmem(r3+#0x0) } 
  21be94:	18 40 25 60	60254018 { 	loop1(0x21bea0,r5)
  21be98:	00 40 00 7f	7f004000   	nop
  21be9c:	00 c0 00 7f	7f00c000   	nop } 
  21bea0:	e3 42 00 69	690042e3 { 	loop0(0x21bec0,#0x1f)
  21bea4:	0e 40 6d 70	706d400e   	r14 = r13
  21bea8:	e7 c0 0c 28	280cc0e7   	v7 = vmemu(r12+#0x0) } 
  21beac:	63 c7 20 1f	1f20c763 { 	v3 = vrdelta(v7,v0) } 
  21beb0:	85 e7 27 1e	1e27e785 { 	v5 = valign(v7,v7,#0x4) } 
  21beb4:	00 40 00 7f	7f004000 { 	nop
  21beb8:	00 40 00 7f	7f004000   	nop
  21bebc:	00 c0 00 7f	7f00c000   	nop } 
  21bec0:	41 63 08 1c	1c086341 { 	v1.w += vrmpy(v3.ub,v8.b)
  21bec4:	63 45 20 1f	1f204563   	v3 = vrdelta(v5,v0)
  21bec8:	28 c1 0e 29	290ec128   	v8.cur = vmem(r14++#0x1) } 
  21becc:	00 80 00 7f	7f008000 { 	nop
  21bed0:	85 e5 25 1e	1e25e585   	v5 = valign(v5,v5,#0x4) }  :endloop0
  21bed4:	41 63 09 1c	1c096341 { 	v1.w += vrmpy(v3.ub,v9.b)
  21bed8:	29 c0 0e 28	280ec029   	v9.cur = vmem(r14+#0x0) } 
  21bedc:	0c 50 0c b0	b00c500c { 	r12 = add(r12,#0x80)
  21bee0:	0d 80 0d b1	b10d800d   	r13 = add(r13,#0x1000)
  21bee4:	00 c0 00 7f	7f00c000   	nop }  :endloop1
  21bee8:	e2 c2 07 c4	c407c2e2 { 	r2 = addasl(r2,r7,#0x7) } 
  21beec:	ec 5f a6 74	74a65fec { 	if (!p1) r12 = add(r6,#-0x1)
  21bef0:	2a c1 00 5c	5c00c12a   	if (p1) jump:nt 0x21bf40 <matmul_qu8xqi8_32+0xe0> } 
  21bef4:	22 40 86 75	75864022 { 	p2 = cmp.gtu(r6,#0x1)
  21bef8:	e6 c0 08 28	2808c0e6   	v6 = vmemu(r8+#0x0) } 
  21befc:	08 42 0c 60	600c4208 { 	loop0(0x21bf20,r12)
  21bf00:	0c c0 62 70	7062c00c   	r12 = r2 } 
  21bf04:	62 c6 20 1f	1f20c662 { 	v2 = vrdelta(v6,v0) } 
  21bf08:	16 42 20 5c	5c204216 { 	if (!p2) jump:nt 0x21bf34 <matmul_qu8xqi8_32+0xd4>
  21bf0c:	84 e6 26 1e	1e26e684   	v4 = valign(v6,v6,#0x4) } 
  21bf10:	00 40 00 7f	7f004000 { 	nop
  21bf14:	00 40 00 7f	7f004000   	nop
  21bf18:	00 40 00 7f	7f004000   	nop
  21bf1c:	00 c0 00 7f	7f00c000   	nop } 
  21bf20:	41 62 0a 1c	1c0a6241 { 	v1.w += vrmpy(v2.ub,v10.b)
  21bf24:	62 44 20 1f	1f204462   	v2 = vrdelta(v4,v0)
  21bf28:	2a c1 0c 29	290cc12a   	v10.cur = vmem(r12++#0x1) } 
  21bf2c:	00 80 00 7f	7f008000 { 	nop
  21bf30:	84 e4 24 1e	1e24e484   	v4 = valign(v4,v4,#0x4) }  :endloop0
  21bf34:	e2 42 06 c4	c40642e2 { 	r2 = addasl(r2,r6,#0x7)
  21bf38:	41 62 0b 1c	1c0b6241   	v1.w += vrmpy(v2.ub,v11.b)
  21bf3c:	2b c0 0c 28	280cc02b   	v11.cur = vmem(r12+#0x0) } 
  21bf40:	29 40 09 b0	b0094029 { 	r9 = add(r9,#0x1)
  21bf44:	03 50 03 b0	b0035003   	r3 = add(r3,#0x80)
  21bf48:	e1 c1 20 29	2920c1e1   	vmemu(r0++#0x1) = v1 } 
  21bf4c:	02 44 09 f2	f2094402 { 	p2 = cmp.eq(r9,r4)
  21bf50:	9e fa ff 5c	5cfffa9e   	if (!p2.new) jump:t 0x21be88 <matmul_qu8xqi8_32+0x28> } 
  21bf54:	00 c0 9f 52	529fc000 { 	jumpr r31 } 
		...
