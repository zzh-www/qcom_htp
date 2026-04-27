
/home/zzh/work/qcom_htp/tools/qnn-sdk/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so:	file format elf32-hexagon

Disassembly of section .text:

003d7600 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv>:
  3d7620: { 	r7:6 = combine(#0x0,#0x0)
  3d7624:   	memd(r29+#-0x10) = r17:16; 	allocframe(#0xd8) } 
  3d7628: { 	r22 = r4
  3d762c:   	r4 = r3
  3d7630:   	r3 = memw(r1+#0x8); 	memd(r29+#0xb8) = r23:22 } 
  3d7634: { 	r21 = r5
  3d7638:   	r20 = add(r29,#0x18)
  3d763c:   	memd(r29+#0x18) = r7:6; 	memd(r29+#0xc0) = r21:20 } 
  3d7640: { 	r6 = memw(r3+#0x20)
  3d7644:   	memd(r29+#0xb0) = r25:24 } 
  3d7648: { 	r6 = lsr(r6,#0x5)
  3d764c:   	r7 = memw(r3+#0x1c); 	r5 = memw(r0+#0x8) } 
  3d7650: { 	r7 = lsr(r7,#0x3)
  3d7654:   	r19:18 = combine(r0,r2)
  3d7658:   	r3 = memw(r3+#0x18); 	memd(r29+#0xc8) = r19:18 } 
  3d765c: { 	r6 = mpyi(r7,r6)
  3d7660:   	r3 = lsr(r3,#0x3)
  3d7664:   	r2 = r1
  3d7668:   	r1:0 = combine(r19,r20) } 
  3d766c: { 	memd(r29+#0xa8) = r27:26 } 
  3d7670: { 	r25 = mpyi(r6,r3)
  3d7674:   	r3 = memw(r5+#0x20); 	r6 = memw(r5+#0x1c) } 
  3d7678: { 	r3 = lsr(r3,#0x5)
  3d767c:   	r6 = lsr(r6,#0x3)
  3d7680:   	r5 = memw(r5+#0x18); 	memw(r29+#0x0) = r21 } 
  3d7684: { 	r6 = mpyi(r6,r3)
  3d7688:   	r7 = lsr(r5,#0x3)
  3d768c:   	r3 = r18; 	r5 = r22 } 
  3d7690: { 	r24 = mpyi(r6,r7)
  3d7694:   	call 0x3d7920 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x320> } 
  3d7698: { 	r18 = add(r20,#0x10)
  3d769c:   	r20 = add(r20,#0x48)
  3d76a0:   	r2 = memw(r18+#0x8); 	r4 = memw(r19+#0x8) } 
  3d76a4: { 	r19 = add(r18,#0x18)
  3d76a8:   	r16 = memw(r29+#0x20) } 
  3d76ac: { 	r3 = memw(r2+#0x4); 	r2 = memw(r2+#0x8) } 
  3d76b0: { 	p0 = cmp.eq(r2,#0x1)
  3d76b4:   	p1 = cmp.eq(r3,#0x1)
  3d76b8:   	r23 = memw(r4+#0x4); 	r17 = memw(r29+#0x24) } 
  3d76bc: { 	p0 = and(p0,p1) } 
  3d76c0: { 	r2 = p0
  3d76c4:   	p0 = bitsclr(r16,#0x10)
  3d76c8:   	if (!p0.new) jump:t 0x3d77bc <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x1bc>
  3d76cc:   	memw(r29+#0x14) = r2.new } 
  3d76d0: { 	r27:26 = vaslw(r25:24,#0x2)
  3d76d4:   	p0 = cmp.eq(r23,#0x0); if (p0.new) jump:nt 0x3d78f4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2f4> } 
  3d76d8: { 	r22 = #0x0
  3d76dc:   	p0 = cmp.gt(r27,#0x0)
  3d76e0:   	r21 = #0x0; 	r2 = memw(r18+#0x0) } 
  3d76e4: { 	r3 = p0
  3d76e8:   	p0 = cmp.gt(r26,#0x0)
  3d76ec:   	memw(r29+#0x10) = r3.new } 
  3d76f0: { 	r3 = p0
  3d76f4:   	memw(r29+#0xc) = r3.new } 
  3d76f8: { 	r3 = memw(r29+#0x10) } 
  3d76fc: { 	p0 = r3
  3d7700:   	if (p0.new) r4 = add(r21,#0x40)
  3d7704:   	if (p0.new) r3 = add(r27,#0x3f)
  3d7708:   	if (!p0.new) jump:nt 0x3d7734 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x134> } 
  3d770c: { 	r3 = lsr(r3,#0x6) } 
  3d7710: { 	r5 = add(r3,#-0x1)
  3d7714:   	p0 = cmp.gtu(r3,#0x1)
  3d7718:   	r3 = add(r2,r21) } 
  3d771c: { 	loop0(0x3d7724,r5)
  3d7720:   	if (!p0) jump:nt 0x3d7730 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x130> } 
  3d7724: { 	r3 = add(r2,r4)
  3d7728:   	r4 = add(r4,#0x40)
  3d772c:   	dcfetch(r3+#0x0) }  :endloop0
  3d7730: { 	dcfetch(r3+#0x0) } 
  3d7734: { 	r2 = memw(r29+#0xc) } 
  3d7738: { 	p0 = r2
  3d773c:   	if (p0.new) r3 = add(r26,#0x3f)
  3d7740:   	if (!p0.new) jump:nt 0x3d7774 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x174> } 
  3d7744: { 	r4 = #0x0; 	r2 = memw(r19+#0x0) } 
  3d7748: { 	r3 = lsr(r3,#0x6) } 
  3d774c: { 	r5 = add(r3,#-0x1)
  3d7750:   	p0 = cmp.gtu(r3,#0x1)
  3d7754:   	r3 = add(r2,r4)
  3d7758:   	r4 = add(r4,#0x40) } 
  3d775c: { 	loop0(0x3d7764,r5)
  3d7760:   	if (!p0) jump:nt 0x3d7770 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x170> } 
  3d7764: { 	r3 = add(r2,r4)
  3d7768:   	r4 = add(r4,#0x40)
  3d776c:   	dcfetch(r3+#0x0) }  :endloop0
  3d7770: { 	dcfetch(r3+#0x0) } 
  3d7774: { 	r1:0 = combine(r18,r19)
  3d7778:   	r2 = memw(r29+#0x14) } 
  3d777c: { 	p0 = r2
  3d7780:   	if (!p0.new) r3:2 = combine(r17,r16)
  3d7784:   	if (!p0.new) r4 = add(r20,#0x0)
  3d7788:   	if (!p0.new) jump:t 0x3d779c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x19c> } 
  3d778c: { 	call 0x2ea740 <hmx_convbbb1x1_stride1>
  3d7790:   	r3:2 = combine(r17,r16)
  3d7794:   	r4 = r20 } 
  3d7798: { 	jump 0x3d77a0 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x1a0> } 
  3d779c: { 	call 0x2edd80 <hmx_convbbb_stride1> } 
  3d77a0: { 	r22 = add(r22,#0x1)
  3d77a4:   	r2 = memw(r29+#0x28)
  3d77a8:   	memw(r29+#0x40) += r26 } 
  3d77ac: { 	r2 = addasl(r2,r25,#0x2)
  3d77b0:   	p0 = cmp.eq(r22,r23); if (!p0.new) jump:t 0x3d76f8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0xf8>
  3d77b4:   	memw(r29+#0x28) = r2.new } 
  3d77b8: { 	jump 0x3d78f4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2f4> } 
  3d77bc: { 	r27:26 = vaslw(r25:24,#0x2)
  3d77c0:   	r3 = and(r22,#0x10)
  3d77c4:   	p0 = cmp.eq(r23,#0x0); if (p0.new) jump:nt 0x3d78f4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2f4> } 
  3d77c8: { 	p0 = cmp.eq(r3,#0x0)
  3d77cc:   	r24 = #0x0
  3d77d0:   	r22 = #0x0; 	r2 = memw(r18+#0x0) } 
  3d77d4: { 	r3 = p0
  3d77d8:   	p0 = cmp.gt(r27,#0x0)
  3d77dc:   	memw(r29+#0x10) = r3.new } 
  3d77e0: { 	r3 = p0
  3d77e4:   	p0 = cmp.gt(r26,#0x0)
  3d77e8:   	memw(r29+#0xc) = r3.new } 
  3d77ec: { 	r3 = p0
  3d77f0:   	memw(r29+#0x8) = r3.new } 
  3d77f4: { 	r3 = memw(r29+#0xc) } 
  3d77f8: { 	p0 = r3
  3d77fc:   	if (p0.new) r4 = add(r22,#0x40)
  3d7800:   	if (p0.new) r3 = add(r27,#0x3f)
  3d7804:   	if (!p0.new) jump:nt 0x3d7830 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x230> } 
  3d7808: { 	r3 = lsr(r3,#0x6) } 
  3d780c: { 	r5 = add(r3,#-0x1)
  3d7810:   	p0 = cmp.gtu(r3,#0x1)
  3d7814:   	r3 = add(r2,r22) } 
  3d7818: { 	loop0(0x3d7820,r5)
  3d781c:   	if (!p0) jump:nt 0x3d782c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x22c> } 
  3d7820: { 	r3 = add(r2,r4)
  3d7824:   	r4 = add(r4,#0x40)
  3d7828:   	dcfetch(r3+#0x0) }  :endloop0
  3d782c: { 	dcfetch(r3+#0x0) } 
  3d7830: { 	r2 = memw(r29+#0x8) } 
  3d7834: { 	p0 = r2
  3d7838:   	if (p0.new) r3 = add(r26,#0x3f)
  3d783c:   	if (!p0.new) jump:nt 0x3d7870 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x270> } 
  3d7840: { 	r4 = #0x0; 	r2 = memw(r19+#0x0) } 
  3d7844: { 	r3 = lsr(r3,#0x6) } 
  3d7848: { 	r5 = add(r3,#-0x1)
  3d784c:   	p0 = cmp.gtu(r3,#0x1)
  3d7850:   	r3 = add(r2,r4)
  3d7854:   	r4 = add(r4,#0x40) } 
  3d7858: { 	loop0(0x3d7860,r5)
  3d785c:   	if (!p0) jump:nt 0x3d786c <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x26c> } 
  3d7860: { 	r3 = add(r2,r4)
  3d7864:   	r4 = add(r4,#0x40)
  3d7868:   	dcfetch(r3+#0x0) }  :endloop0
  3d786c: { 	dcfetch(r3+#0x0) } 
  3d7870: { 	r2 = memw(r29+#0x10) } 
  3d7874: { 	p0 = r2
  3d7878:   	if (!p0.new) r1:0 = combine(r18,r19)
  3d787c:   	if (!p0.new) jump:t 0x3d78a4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2a4>
  3d7880:   	if (!p0.new) r2 = memw(r29+#0x14) } 
  3d7884: { 	r1:0 = combine(r18,r19)
  3d7888:   	r2 = memw(r29+#0x14) } 
  3d788c: { 	p0 = r2
  3d7890:   	if (p0.new) r3:2 = combine(r17,r16)
  3d7894:   	if (p0.new) r5:4 = combine(r21,r20)
  3d7898:   	if (!p0.new) jump:t 0x3d78c4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2c4> } 
  3d789c: { 	call 0x2ea940 <hmx_v73_convbbb1x1_stride1_sparsity> } 
  3d78a0: { 	jump 0x3d78d8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2d8> } 
  3d78a4: { 	p0 = r2
  3d78a8:   	if (!p0.new) r3:2 = combine(r17,r16)
  3d78ac:   	if (!p0.new) r5:4 = combine(r21,r20)
  3d78b0:   	if (!p0.new) jump:t 0x3d78d4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2d4> } 
  3d78b4: { 	call 0x2ef9c0 <hmx_v73_convbnb1x1_stride1_sparsity>
  3d78b8:   	r3:2 = combine(r17,r16)
  3d78bc:   	r5:4 = combine(r21,r20) } 
  3d78c0: { 	jump 0x3d78d8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2d8> } 
  3d78c4: { 	call 0x2ee000 <hmx_v73_convbbb_stride1_sparsity>
  3d78c8:   	r3:2 = combine(r17,r16)
  3d78cc:   	r5:4 = combine(r21,r20) } 
  3d78d0: { 	jump 0x3d78d8 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x2d8> } 
  3d78d4: { 	call 0x2f15c0 <hmx_v73_convbnb_stride1_sparsity> } 
  3d78d8: { 	r24 = add(r24,#0x1)
  3d78dc:   	r2 = memw(r29+#0x28)
  3d78e0:   	memw(r29+#0x40) += r26 } 
  3d78e4: { 	r2 = addasl(r2,r25,#0x2)
  3d78e8:   	p0 = cmp.eq(r24,r23)
  3d78ec:   	if (!p0.new) jump:t 0x3d77f4 <_Z17code_to_type_nameI14ConcreteTensorIN5Tdefs16pkWeightsF16_TCMEEEPKcv+0x1f4>
  3d78f0:   	memw(r29+#0x28) = r2.new } 
  3d78f4: { 	r0 = #0x0
  3d78f8:   	r17:16 = memd(r29+#0xd0); 	r19:18 = memd(r29+#0xc8) } 
  3d78fc: { 	r21:20 = memd(r29+#0xc0); 	r23:22 = memd(r29+#0xb8) } 
  3d7900: { 	r25:24 = memd(r29+#0xb0)
  3d7904:   	r27:26 = memd(r29+#0xa8) } 
  3d7908: { 	dealloc_return } 
		...
