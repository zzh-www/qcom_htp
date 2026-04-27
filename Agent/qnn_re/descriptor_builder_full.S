
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

003d7600 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv>:
  3d7920: { 	r16 = r0
  3d7924:   	memd(r29+#-0x10) = r17:16; 	allocframe(#0x70) } 
  3d7928: { 	immext(#0xffffc0)
  3d792c:   	p0 = cmp.gtu(r5,##0xffffff)
  3d7930:   	r28 = memw(r3+#0x8)
  3d7934:   	memd(r29+#0x58) = r21:20 } 
  3d7938: { 	r20 = r5
  3d793c:   	r6 = memw(r3+#0xc); 	memd(r29+#0x60) = r19:18 } 
  3d7940: { 	r25 = lsr(r5,#0x18)
  3d7944:   	r24 = r1
  3d7948:   	r8 = memw(r28+#0xc)
  3d794c:   	memd(r29+#0x48) = r25:24 } 
  3d7950: { 	r8 = lsr(r8,#0x5)
  3d7954:   	if (p0) r1 = add(r25,#0x11)
  3d7958:   	r7 = memub(r28+#0x27)
  3d795c:   	r15 = memw(r28+#0x8) } 
  3d7960: { 	r12 = lsr(r7,#0x5)
  3d7964:   	r7 = asl(r7,#0x2)
  3d7968:   	r9 = memub(r28+#0x26)
  3d796c:   	memd(r29+#0x50) = r23:22 } 
  3d7970: { 	r8 = mpyi(r12,r8)
  3d7974:   	r14 = r9
  3d7978:   	r12 = memw(r28+#0x4)
  3d797c:   	memd(r29+#0x40) = r27:26 } 
  3d7980: { 	r8 += lsr(r9,#0x5)
  3d7984:   	immext(#0x380)
  3d7988:   	r14 = and(##0x380,asl(r14,#0x5))
  3d798c:   	r13 = memub(r28+#0x24) } 
  3d7990: { 	r9 = or(r14,and(r9,#0x3))
  3d7994:   	r3 = p0
  3d7998:   	r23:22 = combine(r4,r2)
  3d799c:   	memw(r29+#0x20) = r3 } 
  3d79a0: { 	r13 += mpyi(r8,r12)
  3d79a4:   	r7 = or(r9,and(r7,#0x7c))
  3d79a8:   	r12 = memub(r28+#0x25)
  3d79ac:   	memw(r29+#0x28) = r3 } 
  3d79b0: { 	r12 += mpyi(r13,r15) } 
  3d79b4: { 	r7 |= asl(r12,#0xa) } 
  3d79b8: { 	r6 = add(r6,r7)
  3d79bc:   	memw(r0+#0x8) = r6.new } 
  3d79c0: { 	r6 = memw(r4+#0xc)
  3d79c4:   	memw(r0+#0xc) = r6.new } 
  3d79c8: { 	r17 = memw(r2+#0x8)
  3d79cc:   	r21 = memw(r28+#0x8) } 
  3d79d0: { 	r2 = memw(r28+#0x4)
  3d79d4:   	r10 = memw(r28+#0xc) } 
  3d79d8: { 	r26 = r2
  3d79dc:   	r18 = memw(r17+#0x8); 	r19 = memw(r17+#0x10) } 
  3d79e0: { 	if (!p0) jump:nt 0x3d79f8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x3f8>
  3d79e4:   	r27 = memw(r17+#0xc) } 
  3d79e8: { 	call 0x1a5bb0 <__hexagon_udivsi3@plt>
  3d79ec:   	r0 = asl(r10,#0x7) } 
  3d79f0: { 	r2 = r26
  3d79f4:   	r10 = r0 } 
  3d79f8: { 	r4 = add(r19,#0x1f)
  3d79fc:   	p0 = cmp.eq(r21,#0x1)
  3d7a00:   	r7 = memub(r17+#0x25)
  3d7a04:   	r1 = memw(r17+#0x18) } 
  3d7a08: { 	r6 = asl(r7,#0x8)
  3d7a0c:   	r7 = and(r7,#0x7)
  3d7a10:   	r5 = memub(r17+#0x26)
  3d7a14:   	r13 = memw(r17+#0x1c) } 
  3d7a18: { 	r5 = and(#0xe0,asl(r5,#0x5))
  3d7a1c:   	r9 = and(r5,#0x7)
  3d7a20:   	r7 = add(r7,r18)
  3d7a24:   	r19 = memub(r17+#0x27) } 
  3d7a28: { 	immext(#0x700)
  3d7a2c:   	r6 = or(r5,and(r6,##0x700))
  3d7a30:   	r9 = add(r9,r27)
  3d7a34:   	r3 = memw(r24+#0x8) } 
  3d7a38: { 	r19 = or(r6,and(r19,#0x1f))
  3d7a3c:   	r6 = lsr(r13,#0x3)
  3d7a40:   	r5 = memw(r17+#0x20)
  3d7a44:   	r13 = memw(r22+#0x4) } 
  3d7a48: { 	r5 = lsr(r5,#0x5)
  3d7a4c:   	r4 = and(r4,#-0x20)
  3d7a50:   	r8 = memw(r3+#0x8)
  3d7a54:   	r15 = memw(r3+#0x20) } 
  3d7a58: { 	r26 = mpyi(r5,r6)
  3d7a5c:   	r0 = lsr(r19,#0x5)
  3d7a60:   	r13 = memw(r13+#0x4)
  3d7a64:   	r28 = memw(r3+#0x1c) } 
  3d7a68: { 	r19 = sub(#0x0,lsr(r19,#0x8))
  3d7a6c:   	r6 = lsr(r15,#0x5)
  3d7a70:   	r11 = memw(r3+#0xc)
  3d7a74:   	memw(r29+#0x18) = r13 } 
  3d7a78: { 	r13 = lsr(r28,#0x3)
  3d7a7c:   	p2 = bitsclr(r19,#0x7)
  3d7a80:   	r12 = memw(r3+#0x10)
  3d7a84:   	memw(r29+#0x1c) = r27 } 
  3d7a88: { 	r27 = mpyi(r13,r6)
  3d7a8c:   	r17 = sub(#0x0,r0)
  3d7a90:   	r0 = add(r21,r0)
  3d7a94:   	r3 = memw(r3+#0x18) } 
  3d7a98: { 	p1 = !bitsclr(r17,#0x7)
  3d7a9c:   	p3 = cmp.eq(r2,#0x1)
  3d7aa0:   	r14 = memw(r22+#0xc)
  3d7aa4:   	memw(r16+#0x24) = r4 } 
  3d7aa8: { 	memw(r16+#0x18) = r26
  3d7aac:   	memw(r16+#0x10) = r14 } 
  3d7ab0: { 	p2 = and(p0,p3)
  3d7ab4:   	r4 = mux(p2,#0x0,#0x8)
  3d7ab8:   	memw(r16+#0x14) = r5
  3d7abc:   	memw(r16+#0x1c) = r9 } 
  3d7ac0: { 	r18 = lsr(r3,#0x3)
  3d7ac4:   	memw(r16+#0x20) = r7
  3d7ac8:   	r5 = memw(r24+#0xc) } 
  3d7acc: { 	r24 = sub(#0x1,r0)
  3d7ad0:   	memw(r16+#0x28) = r5; 	memw(r16+#0x2c) = r6 } 
  3d7ad4: { 	memw(r16+#0x30) = r27
  3d7ad8:   	memw(r16+#0x34) = r11 } 
  3d7adc: { 	memw(r16+#0x38) = r8
  3d7ae0:   	memw(r16+#0x3c) = r12 } 
  3d7ae4: { 	r21 = lsr(r1,#0x3)
  3d7ae8:   	r23 = setbit(r4,#0x2)
  3d7aec:   	r5 = memw(r23+#0x8); 	memw(r29+#0x24) = r21 } 
  3d7af0: { 	if (!p1) r23 = add(r4,#0x0) } 
  3d7af4: { 	r5 = memw(r5+#0x10) } 
  3d7af8: { 	p0 = cmp.eq(r5,#0x80)
  3d7afc:   	if (!p0.new) jump:t 0x3d7b88 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x588> } 
  3d7b00: { 	r0 = add(r29,#0x2c)
  3d7b04:   	r1 = r22
  3d7b08:   	r2 = memw(r22+#0x0); 	memw(r29+#0x10) = r2 } 
  3d7b0c: { 	r3 = p2
  3d7b10:   	memw(r29+#0x8) = r3.new } 
  3d7b14: { 	r2 = memw(r2+#0x24)
  3d7b18:   	memw(r29+#0xc) = r11 } 
  3d7b1c: { 	callr r2
  3d7b20:   	memw(r29+#0x14) = r10 } 
  3d7b24: { 	r0 = add(r29,#0x2c)
  3d7b28:   	r1 = memw(r29+#0x20); 	r3 = memw(r29+#0x2c) } 
  3d7b2c: { 	p0 = cmp.eq(r3,#0x2) } 
  3d7b30: { 	r3 = p0
  3d7b34:   	r2 = memw(r1+#0x0)
  3d7b38:   	memw(r29+#0x20) = r3.new } 
  3d7b3c: { 	r2 = memw(r2+#0x24) } 
  3d7b40: { 	callr r2 } 
  3d7b44: { 	r2 = memw(r29+#0x20) } 
  3d7b48: { 	p0 = r2
  3d7b4c:   	if (!p0.new) jump:t 0x3d7b80 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x580>
  3d7b50:   	if (!p0.new) r10 = memw(r29+#0x14)
  3d7b54:   	if (!p0.new) r11 = memw(r29+#0xc) } 
  3d7b58: { 	r3 = setbit(r23,#0x7)
  3d7b5c:   	r2 = memw(r29+#0x2c)
  3d7b60:   	r10 = memw(r29+#0x14) } 
  3d7b64: { 	p0 = cmp.eq(r2,#0x7)
  3d7b68:   	if (p0.new) r23 = add(r3,#0x0)
  3d7b6c:   	r11 = memw(r29+#0xc)
  3d7b70:   	r2 = memw(r29+#0x10) } 
  3d7b74: { 	r3 = memw(r29+#0x8) } 
  3d7b78: { 	p2 = r3
  3d7b7c:   	jump 0x3d7b88 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x588> } 
  3d7b80: { 	r3 = memw(r29+#0x8); 	r2 = memw(r29+#0x10) } 
  3d7b84: { 	p2 = r3 } 
  3d7b88: { 	r7 = asl(r20,#0x2)
  3d7b8c:   	immext(#0x60000)
  3d7b90:   	r3 = ##0x60000; 	r6 = r20 } 
  3d7b94: { 	r26 = mpyi(r26,r21)
  3d7b98:   	p0 = bitsset(r20,r3)
  3d7b9c:   	immext(#0x40000)
  3d7ba0:   	r4 = ##0x40020 } 
  3d7ba4: { 	immext(#0x80000000)
  3d7ba8:   	r6 = and(##0x80000000,asl(r6,#0x1e))
  3d7bac:   	r3 = r20
  3d7bb0:   	r21 = and(r19,#0x7) } 
  3d7bb4: { 	immext(#0x1f00)
  3d7bb8:   	r3 = or(r6,and(r3,##0x1f00))
  3d7bbc:   	p1 = bitsclr(r20,r4)
  3d7bc0:   	r6 = and(r17,#0x7) } 
  3d7bc4: { 	r22 = mpyi(r18,r27)
  3d7bc8:   	immext(#0x20040)
  3d7bcc:   	r5 = mux(p0,##0x20040,#0x40)
  3d7bd0:   	r27 = memw(r29+#0x78) } 
  3d7bd4: { 	r0 = add(r16,#0x48)
  3d7bd8:   	if (!p1) r3 = or(r5,r3)
  3d7bdc:   	r5 = and(r24,#0x7) } 
  3d7be0: { 	r8 = or(r3,r23)
  3d7be4:   	immext(#0x2000)
  3d7be8:   	if (!p2) r3 = ##0x2004
  3d7bec:   	if (!p2) jump:nt 0x3d7c2c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x62c> } 
  3d7bf0: { 	r7 = or(r8,and(r7,#0x20))
  3d7bf4:   	p0 = bitsclr(r20,#0x10)
  3d7bf8:   	r3:2 = combine(#0x0,r10)
  3d7bfc:   	if (!p0.new) r5:4 = combine(r21,r6) } 
  3d7c00: { 	if (p0) r1 = #0x700
  3d7c04:   	if (!p0) jump:nt 0x3d7c98 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x698> } 
  3d7c08: { 	immext(#0x1c000)
  3d7c0c:   	r2 = and(r20,##0x1c000) } 
  3d7c10: { 	immext(#0x8000)
  3d7c14:   	p0 = cmp.eq(r2,##0x8000)
  3d7c18:   	if (p0.new) r3:2 = combine(r6,r10)
  3d7c1c:   	if (!p0.new) jump:t 0x3d7cdc <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x6dc> } 
  3d7c20: { 	call 0x2894e0 <_Z25set_hmx_params_convw2b1x1P10hmx_paramsmmmmm>
  3d7c24:   	r5:4 = combine(r7,r21) } 
  3d7c28: { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7c2c: { 	p0 = !bitsclr(r20,r3)
  3d7c30:   	if (!p0.new) r3 = add(r11,#0x7)
  3d7c34:   	if (p0.new) jump:t 0x3d7c74 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x674> } 
  3d7c38: { 	r3 = or(r6,and(r3,#-0x8))
  3d7c3c:   	p0 = cmp.eq(r6,#0x0); 	r4 = memw(r29+#0x18) } 
