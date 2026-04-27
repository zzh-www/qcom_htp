
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

003d7600 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv>:
  3d7c40: { 	p1 = cmp.eq(r4,#0x0)
  3d7c44:   	r4 = memw(r29+#0x1c) } 
  3d7c48: { 	p0 = and(p1,!p0)
  3d7c4c:   	r3 = sub(r4,r3)
  3d7c50:   	r4 = memw(r29+#0x24) } 
  3d7c54: { 	p1 = cmp.gt(r3,#0x0)
  3d7c58:   	p2 = cmp.gt(r4,r3) } 
  3d7c5c: { 	p1 = and(p2,and(p0,p1)) } 
  3d7c60: { 	r3 = mux(p1,#0x3,#0x2) } 
  3d7c64: { 	r4 = mux(p0,r3,#0x0)
  3d7c68:   	if (!p0) r3 = #0x1 } 
  3d7c6c: { 	if (!p1) r3 = add(r4,#0x0) } 
  3d7c70: { 	r8 = or(r3,r8) } 
  3d7c74: { 	r3 = and(r20,#0x30)
  3d7c78:   	if (!cmp.eq(r3.new,#0x10)) jump:t 0x3d7ca8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x6a8> } 
  3d7c7c: { 	r1 = #0x700
  3d7c80:   	r5:4 = combine(r6,r10)
  3d7c84:   	r3 = memw(r29+#0x24)
  3d7c88:   	memw(r29+#0x4) = r8 } 
  3d7c8c: { 	call 0x288460 <_Z30set_hmx_params_convw4b_stride1P10hmx_paramsmmmmmmm>
  3d7c90:   	memw(r29+#0x0) = r21 } 
  3d7c94: { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7c98: { 	call 0x289380 <_Z25set_hmx_params_convw4b1x1P10hmx_paramsmmmmmm>
  3d7c9c:   	r1 = #0x700
  3d7ca0:   	memw(r29+#0x0) = r7 } 
  3d7ca4: { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7ca8: { 	immext(#0x1c000)
  3d7cac:   	r3 = and(r20,##0x1c010) } 
  3d7cb0: { 	immext(#0x8000)
  3d7cb4:   	p0 = cmp.eq(r3,##0x8000)
  3d7cb8:   	if (!p0.new) r4 = add(r10,#0x0)
  3d7cbc:   	if (!p0.new) jump:t 0x3d7cec <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x6ec> } 
  3d7cc0: { 	r1 = #0x700
  3d7cc4:   	r4 = r10
  3d7cc8:   	r3 = memw(r29+#0x24)
  3d7ccc:   	memw(r29+#0x4) = r8 } 
  3d7cd0: { 	call 0x288980 <_Z30set_hmx_params_convw2b_stride1P10hmx_paramsmmmmmmm>
  3d7cd4:   	memw(r29+#0x0) = r21 } 
  3d7cd8: { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7cdc: { 	call 0x287dc0 <_Z22set_hmx_params_conv1x1P10hmx_paramsmmmmm>
  3d7ce0:   	r3:2 = combine(r5,r10)
  3d7ce4:   	r5:4 = combine(r7,r21) } 
  3d7ce8: { 	jump 0x3d7d00 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x700> } 
  3d7cec: { 	r1 = #0x700
  3d7cf0:   	r3 = memw(r29+#0x24)
  3d7cf4:   	memw(r29+#0x4) = r8 } 
  3d7cf8: { 	call 0x287840 <_Z22set_hmx_params_stride1P10hmx_paramsmmmmmmm>
  3d7cfc:   	memw(r29+#0x0) = r21 } 
  3d7d00: { 	immext(#0x66bac0)
  3d7d04:   	r2 = add(pc,##0x66baf8)
  3d7d08:   	memw(r16+#0x80) = r27
  3d7d0c:   	memw(r16+#0x40) = r26 } 
  3d7d10: { 	memw(r16+#0x44) = r22
  3d7d14:   	memw(r16+#0x84) = r21 } 
  3d7d18: { 	immext(#0xfffffa00)
  3d7d1c:   	r3 = memw(r2+##-0x5e4)
  3d7d20:   	immext(#0xffffff00)
  3d7d24:   	r4 = memw(r2+##-0xd0) } 
  3d7d28: { 	p3 = or(p3,!p3)
  3d7d2c:   	r1 = memw(r3+#0x4)
  3d7d30:   	if (cmp.eq(r1.new,r4)) jump:nt 0x3d7d58 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x758> } 
  3d7d34: { 	call 0x1a5ec0 <strcmp@plt>
  3d7d38:   	immext(#0xffffff00)
  3d7d3c:   	r0 = memw(r2+##-0xd0) } 
  3d7d40: { 	p3 = cmp.eq(r0,#0x0)
  3d7d44:   	if (p3.new) jump:nt 0x3d7d64 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x764>
  3d7d48:   	if (!p3.new) r2 = memw(r29+#0x28) } 
  3d7d4c: { 	p2 = r2
  3d7d50:   	if (!p2.new) jump:t 0x3d7e8c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x88c> } 
  3d7d54: { 	jump 0x3d7d6c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x76c> } 
  3d7d58: { 	r2 = memw(r29+#0x28) } 
  3d7d5c: { 	p2 = r2
  3d7d60:   	jump 0x3d7d6c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x76c> } 
  3d7d64: { 	r2 = memw(r29+#0x28) } 
  3d7d68: { 	p2 = r2 } 
  3d7d6c: { 	r3 = memw(r16+#0x60)
  3d7d70:   	r2 = memw(r16+#0x64) } 
  3d7d74: { 	p0 = cmp.eq(r3,#0x0)
  3d7d78:   	r3 = add(r3,#0x1)
  3d7d7c:   	p1 = cmp.eq(r2,#0x0)
  3d7d80:   	if (!p0.new) memw(r16+#0x60) = r3.new } 
  3d7d84: { 	if (!p1) r2 = add(r2,#0x1)
  3d7d88:   	r3 = memw(r16+#0x6c)
  3d7d8c:   	if (!p1) memw(r16+#0x64) = r2.new } 
  3d7d90: { 	p1 = cmp.eq(r3,#0x0)
  3d7d94:   	r2 = memw(r16+#0x68) } 
  3d7d98: { 	p0 = cmp.eq(r2,#0x0)
  3d7d9c:   	r2 = add(r2,#0x1)
  3d7da0:   	if (!p0.new) memw(r16+#0x68) = r2.new } 
  3d7da4: { 	r2 = add(r3,#0x1)
  3d7da8:   	if (p2) r3 = add(r16,#0x60) } 
  3d7dac: { 	if (p2) r2 = add(r25,#0x1)
  3d7db0:   	if (p2) r4 = add(r3,#0x4)
  3d7db4:   	if (!p2) jump:nt 0x3d7e4c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x84c>
  3d7db8:   	if (!p1) memw(r16+#0x6c) = r2 } 
  3d7dbc: { 	loop0(0x3d7dd0,#0x3)
  3d7dc0:   	r5 = memw(r3+#0x0) } 
  3d7dc4: { 	r2 = add(#0x1,lsr(r2,#0x4))
  3d7dc8:   	r6 = lsr(r5,#0x3)
  3d7dcc:   	p0 = cmp.eq(r5,#0x0) } 
  3d7dd0: { 	r6 = mpyi(r6,r2)
  3d7dd4:   	r4 = add(r4,#0x4)
  3d7dd8:   	r5 = r4; 	r7 = memw(r4+#0x0) } 
  3d7ddc: { 	r6 = or(r25,and(r6,#-0x80))
  3d7de0:   	p0 = cmp.eq(r7,#0x0)
  3d7de4:   	r3 = r5
  3d7de8:   	if (!p0) memw(r3+#0x0) = r6.new } 
  3d7dec: { 	r6 = lsr(r7,#0x3)
  3d7df0:   	nop }  :endloop0
  3d7df4: { 	r2 = mpyi(r6,r2)
  3d7df8:   	r3 = p3
  3d7dfc:   	memw(r29+#0x28) = r3.new } 
  3d7e00: { 	r2 = or(r25,and(r2,#-0x80))
  3d7e04:   	if (!p0) memw(r5+#0x0) = r2.new } 
  3d7e08: { 	call 0x289e40 <GetLogPriorityLevel>
  3d7e0c:   	memw(r16+#0x8) = setbit(#0x4) } 
  3d7e10: { 	p0 = cmp.gt(r0,#0x3)
  3d7e14:   	r0 = #0x4; 	r2 = memw(r29+#0x28) } 
  3d7e18: { 	p3 = r2
  3d7e1c:   	if (!p0) jump:nt 0x3d7e4c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x84c> } 
  3d7e20: { 	immext(#0x3aaa80)
  3d7e24:   	r2 = add(pc,##0x3aaab9)
  3d7e28:   	memw(r29+#0x4) = r2.new } 
  3d7e2c: { 	immext(#0x40b500)
  3d7e30:   	r1 = add(pc,##0x40b51c) } 
  3d7e34: { 	immext(#0x40b500)
  3d7e38:   	r2 = add(pc,##0x40b508)
  3d7e3c:   	call 0x289cc0 <qnndsp_log>
  3d7e40:   	memw(r29+#0x0) = r2.new } 
  3d7e44: { 	r2 = memw(r29+#0x28) } 
  3d7e48: { 	p3 = r2 } 
  3d7e4c: { 	loop0(0x3d7e60,#0x4)
  3d7e50:   	r4 = #0x1; 	r3:2 = combine(#0,#0x0) } 
  3d7e54: { 	if (!p3) r3 = memw(r16+#0x78) } 
  3d7e58: { 	r3 = add(r16,#0x60)
  3d7e5c:   	memw(r16+#0x78) = r3 } 
  3d7e60: { 	r5 = memw(r3+#0x0)
  3d7e64:   	if (cmp.eq(r5.new,#0x0)) jump:nt 0x3d7e84 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x884> } 
  3d7e68: { 	r5 = add(r5,#-0x80)
  3d7e6c:   	memw(r3+#0x0) = r5.new } 
  3d7e70: { 	r5 = memw(r16+#0x78) } 
  3d7e74: { 	r6 = r5 } 
  3d7e78: { 	r6 |= asl(r4,r2) } 
  3d7e7c: { 	if (!p3) r6 = add(r5,#0x0) } 
  3d7e80: { 	memw(r16+#0x78) = r6 } 
  3d7e84: { 	nop
  3d7e88:   	r3 = add(r3,#0x4); 	r2 = add(r2,#1) }  :endloop0
  3d7e8c: { 	r17:16 = memd(r29+#0x68); 	r19:18 = memd(r29+#0x60) } 
  3d7e90: { 	r21:20 = memd(r29+#0x58); 	r23:22 = memd(r29+#0x50) } 
  3d7e94: { 	r25:24 = memd(r29+#0x48)
  3d7e98:   	r27:26 = memd(r29+#0x40) } 
  3d7e9c: { 	dealloc_return } 
  3d7ea0: { 	r1 = add(r0,#0x10)
  3d7ea4:   	allocframe(#0x0) } 
  3d7ea8: { 	dcfetch(r0+#0x0) } 
  3d7eac: { 	dcfetch(r0+#0x40) } 
  3d7eb0: { 	dcfetch(r0+#0x80) } 
  3d7eb4: { 	r2 = memw(r0+#0x40) } 
  3d7eb8: { 	r2 = asl(r2,#0x2) } 
  3d7ebc: { 	r3 = add(r2,#0x3f)
  3d7ec0:   	p0 = cmp.gt(r2,#0x0); if (!p0.new) jump:nt 0x3d7ef4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x8f4> } 
  3d7ec4: { 	r4 = #0x0; 	r2 = memw(r1+#0x0) } 
  3d7ec8: { 	r3 = lsr(r3,#0x6) } 
  3d7ecc: { 	r5 = add(r3,#-0x1)
  3d7ed0:   	p0 = cmp.gtu(r3,#0x1)
  3d7ed4:   	r3 = add(r2,r4)
  3d7ed8:   	r4 = add(r4,#0x40) } 
  3d7edc: { 	loop0(0x3d7ee4,r5)
  3d7ee0:   	if (!p0) jump:nt 0x3d7ef0 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x8f0> } 
  3d7ee4: { 	r3 = add(r2,r4)
  3d7ee8:   	r4 = add(r4,#0x40)
  3d7eec:   	dcfetch(r3+#0x0) }  :endloop0
  3d7ef0: { 	dcfetch(r3+#0x0) } 
  3d7ef4: { 	r5 = add(r0,#0x28)
  3d7ef8:   	r2 = memw(r0+#0x44) } 
  3d7efc: { 	r2 = asl(r2,#0x2) } 
  3d7f00: { 	r3 = add(r2,#0x3f)
  3d7f04:   	p0 = cmp.gt(r2,#0x0); if (!p0.new) jump:nt 0x3d7f38 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x938> } 
  3d7f08: { 	r4 = #0x0; 	r2 = memw(r5+#0x0) } 
  3d7f0c: { 	r3 = lsr(r3,#0x6) } 
  3d7f10: { 	r6 = add(r3,#-0x1)
  3d7f14:   	p0 = cmp.gtu(r3,#0x1)
  3d7f18:   	r3 = add(r2,r4)
  3d7f1c:   	r4 = add(r4,#0x40) } 
  3d7f20: { 	loop0(0x3d7f28,r6)
  3d7f24:   	if (!p0) jump:nt 0x3d7f34 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x934> } 
  3d7f28: { 	r3 = add(r2,r4)
  3d7f2c:   	r4 = add(r4,#0x40)
  3d7f30:   	dcfetch(r3+#0x0) }  :endloop0
  3d7f34: { 	dcfetch(r3+#0x0) } 
  3d7f38: { 	r4 = add(r0,#0x48)
  3d7f3c:   	r0 = r5
  3d7f40:   	r2 = memw(r0+#0x8); 	r3 = memw(r0+#0xc) } 
  3d7f44: { 	call 0x2edd80 <hmx_convbbb_stride1> } 
  3d7f48: { 	r0 = #0x0; 	dealloc_return } 
		...
  3d7f60: { 	r1 = add(r0,#0x10)
  3d7f64:   	allocframe(#0x0) } 
  3d7f68: { 	dcfetch(r0+#0x0) } 
  3d7f6c: { 	dcfetch(r0+#0x40) } 
  3d7f70: { 	dcfetch(r0+#0x80) } 
  3d7f74: { 	r2 = memw(r0+#0x40) } 
  3d7f78: { 	r2 = asl(r2,#0x2) } 
  3d7f7c: { 	r3 = add(r2,#0x3f)
  3d7f80:   	p0 = cmp.gt(r2,#0x0); if (!p0.new) jump:nt 0x3d7fb4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x9b4> } 
  3d7f84: { 	r4 = #0x0; 	r2 = memw(r1+#0x0) } 
  3d7f88: { 	r3 = lsr(r3,#0x6) } 
  3d7f8c: { 	r5 = add(r3,#-0x1)
  3d7f90:   	p0 = cmp.gtu(r3,#0x1)
  3d7f94:   	r3 = add(r2,r4)
  3d7f98:   	r4 = add(r4,#0x40) } 
  3d7f9c: { 	loop0(0x3d7fa4,r5)
  3d7fa0:   	if (!p0) jump:nt 0x3d7fb0 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x9b0> } 
  3d7fa4: { 	r3 = add(r2,r4)
  3d7fa8:   	r4 = add(r4,#0x40)
  3d7fac:   	dcfetch(r3+#0x0) }  :endloop0
  3d7fb0: { 	dcfetch(r3+#0x0) } 
  3d7fb4: { 	r5 = add(r0,#0x28)
  3d7fb8:   	r2 = memw(r0+#0x44) } 
  3d7fbc: { 	r2 = asl(r2,#0x2) } 
  3d7fc0: { 	r3 = add(r2,#0x3f)
  3d7fc4:   	p0 = cmp.gt(r2,#0x0); if (!p0.new) jump:nt 0x3d7ff8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x9f8> } 
  3d7fc8: { 	r4 = #0x0; 	r2 = memw(r5+#0x0) } 
  3d7fcc: { 	r3 = lsr(r3,#0x6) } 
  3d7fd0: { 	r6 = add(r3,#-0x1)
  3d7fd4:   	p0 = cmp.gtu(r3,#0x1)
  3d7fd8:   	r3 = add(r2,r4)
  3d7fdc:   	r4 = add(r4,#0x40) } 
  3d7fe0: { 	loop0(0x3d7fe8,r6)
  3d7fe4:   	if (!p0) jump:nt 0x3d7ff4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x9f4> } 
  3d7fe8: { 	r3 = add(r2,r4)
  3d7fec:   	r4 = add(r4,#0x40)
  3d7ff0:   	dcfetch(r3+#0x0) }  :endloop0
  3d7ff4: { 	dcfetch(r3+#0x0) } 
  3d7ff8: { 	r4 = add(r0,#0x48)
  3d7ffc:   	r0 = r5
  3d8000:   	r2 = memw(r0+#0x8); 	r3 = memw(r0+#0xc) } 
