
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

00287dc0 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm>:
  287dc0: { 	r13 = extractu(r5,#0x5,#0x8)
  287dc4:   	r6 = extractu(r1,#0x1,#0x2)
  287dc8:   	p3 = cmp.gtu(r2,#0x20)
  287dcc:   	r2 = add(r2,#0x1f) } 
  287dd0: { 	r7 = extractu(r2,#0x3,#0x2)
  287dd4:   	p1 = cmp.eq(r13,#0x2)
  287dd8:   	p0 = cmp.gtu(r13,#0x2)
  287ddc:   	allocframe(#0x8) } 
  287de0: { 	r2 = extractu(r1,#0x6,#0x5)
  287de4:   	p2 = bitsclr(r5,#0x20)
  287de8:   	r6 = add(r6,#0x1)
  287dec:   	r12 = mux(p3,#0x7,r7) } 
  287df0: { 	r8 = p1
  287df4:   	if (p0) jump:nt 0x287e34 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x74>
  287df8:   	memw(r29+#0x4) = r8.new } 
  287dfc: { 	r8 = r7
  287e00:   	r14 = and(r5,#0x20)
  287e04:   	r11 = memw(r29+#0x4) } 
  287e08: { 	r8 = add(#0x80,asl(r8,#0x7))
  287e0c:   	p1 = cmp.eq(r14,#0x0) } 
  287e10: { 	r9 = mpyi(r8,r6)
  287e14:   	r8 = #0x1f } 
  287e18: { 	r15 = asl(r9,#0x1) } 
  287e1c: { 	immext(#0x800)
  287e20:   	if (p3) r15 = ##0x800 } 
  287e24: { 	p1 = r11
  287e28:   	if (!p1) r9 = add(r15,#0x0)
  287e2c:   	if (!p1.new) jump:nt 0x287e40 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x80> } 
  287e30: { 	p1 = cmp.eq(r6,#0x2); if (!p1.new) jump:nt 0x287e40 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x80> } 
  287e34: { 	r9 = mpyi(r6,r13)
  287e38:   	r8 = add(r13,#-0x1) } 
  287e3c: { 	r9 = asl(r9,#0x5) } 
  287e40: { 	r13 = #0x2
  287e44:   	r4 = sub(#0x0,r4)
  287e48:   	if (!p2) r7 = add(r12,#0x0)
  287e4c:   	p1 = cmp.gt(r2,#0x3b) } 
  287e50: { 	if (!p1) r13 = #0x4
  287e54:   	r12 = and(r4,#0x7)
  287e58:   	if (p1) jump:nt 0x287e74 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0xb4> } 
  287e5c: { 	p1 = cmp.eq(r2,#0x20)
  287e60:   	if (p1.new) r13 = #0x5
  287e64:   	if (p1.new) jump:nt 0x287e90 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0xd0> } 
  287e68: { 	if (!p1.new) jump:nt 0x287e8c <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0xcc>
  287e6c:   	jump 0x287e90 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0xd0>
  287e70:   	p1 = cmp.eq(r2,#0x30) } 
  287e74: { 	p1 = cmp.eq(r2,#0x3c)
  287e78:   	if (!p1.new) r13 = #0x1
  287e7c:   	if (p1.new) jump:t 0x287e90 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0xd0> } 
  287e80: { 	if (!p1.new) jump:nt 0x287e8c <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0xcc>
  287e84:   	jump 0x287e90 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0xd0>
  287e88:   	p1 = cmp.eq(r2,#0x3e) } 
  287e8c: { 	r13 = #0x3 } 
  287e90: { 	r4 = asl(r2,#0x5)
  287e94:   	r9 = add(r9,#-0x1)
  287e98:   	memw(r0+#0x18) = r9.new } 
  287e9c: { 	r15:14 = combine(r4,r2) } 
  287ea0: { 	r15 |= asl(r7,#0x2)
  287ea4:   	r14 &= asl(r12,r13) } 
  287ea8: { 	r7 = asl(r14,#0x5)
  287eac:   	r9 = or(r15,#0x3)
  287eb0:   	memw(r0+#0x8) = r7.new } 
  287eb4: { 	if (p0) jump:nt 0x287ecc <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x10c>
  287eb8:   	memw(r0+#0xc) = r9 } 
  287ebc: { 	r12 = memw(r29+#0x4) } 
  287ec0: { 	p0 = r12
  287ec4:   	if (!p0.new) jump:nt 0x287ee8 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x128> } 
  287ec8: { 	p0 = cmp.eq(r6,#0x2); if (!p0.new) jump:nt 0x287ee8 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x128> } 
  287ecc: { 	r6 = #0x1f
  287ed0:   	r9 = or(r8,r4)
  287ed4:   	immext(#0xffffe0c0)
  287ed8:   	r5 = and(r5,##-0x1f01) } 
  287edc: { 	r6 ^= or(r7,r8)
  287ee0:   	memw(r0+#0xc) = r9 } 
  287ee4: { 	r7 = r6; 	memw(r0+#0x8) = r6 } 
  287ee8: { 	r6 = asl(r3,#0x5)
  287eec:   	p0 = bitsclr(r1,#0x3)
  287ef0:   	memw(r0+#0x30) = r5; 	memw(r0+#0x4) = r4 } 
  287ef4: { 	r5 = extractu(r6,#0xb,#0x0)
  287ef8:   	memw(r0+#0x0) = r5.new } 
  287efc: { 	if (p0) jump:nt 0x287f5c <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x19c>
  287f00:   	deallocframe } 
  287f04: { 	r5 = extractu(r7,#0x2,#0x5)
  287f08:   	r8 = and(r7,#-0x80)
  287f0c:   	p0 = tstbit(r1,#0); if (!p0.new) jump:t 0x287f30 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x170> } 
  287f10: { 	r7 = and(#0x7c,asl(r7,#0x2))
  287f14:   	r13 = and(r9,#-0x80) } 
  287f18: { 	r12 = extractu(r9,#0x2,#0x5)
  287f1c:   	r9 = and(#0x7c,asl(r9,#0x2)) } 
  287f20: { 	r5 |= or(r7,r8)
  287f24:   	r12 |= or(r9,r13)
  287f28:   	memw(r0+#0x8) = r5.new } 
  287f2c: { 	memw(r0+#0xc) = r12 } 
  287f30: { 	p0 = bitsclr(r1,#0x2)
  287f34:   	r2 = and(r2,#0x3)
  287f38:   	if (p0.new) jump:t 0x287f5c <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm+0x19c> } 
  287f3c: { 	immext(#0x780)
  287f40:   	r5 = and(r6,##0x780) } 
  287f44: { 	r3 = or(r5,and(r3,#0x3))
  287f48:   	immext(#0x780)
  287f4c:   	r4 = or(r2,and(r4,##0x780))
  287f50:   	memw(r0+#0x0) = r3.new } 
  287f54: { 	jumpr r31
  287f58:   	memw(r0+#0x4) = r4 } 
  287f5c: { 	jumpr r31 } 
