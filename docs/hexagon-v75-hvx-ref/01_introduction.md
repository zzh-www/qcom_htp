<!-- chapter: 1 Introduction -->
<!-- source: 80-N2040-58_AB_Hexagon_V75_HVX_Programmers_Reference_Manual.pdf -->
<!-- pages: 8-12 (1-based as printed in PDF) -->

# 1 Introduction

This document describes the Qualcomm® Hexagon™ Vector eXtensions (HVX) instruction set architecture. These extensions are implemented in an optional coprocessor. This document assumes that the reader is familiar with the Hexagon architecture. For a full description of the architecture, refer to the _Qualcomm Hexagon Programmer's Reference Manual_.

## 1.1 SIMD coprocessor

HVX instructions are primarily implemented in a single instruction multiple data (SIMD) coprocessor block that includes vector registers, vector compute elements, and dedicated memory. This extends the baseline Hexagon architecture to enable high-performance computer vision, image processing, or other workloads that can map to SIMD parallel processing.

![Hexagon core with attached SIMD coprocessor](images/p008_0.png)

**----- Start of picture text -----**<br>
Hexagon core<br>I$<br>R0-R31<br>P0-P3<br>S3 S2 S1 S0<br>SIMD coprocessor<br>V0-V31<br>D$<br>Q0-Q3<br>mem VTCM<br>mpy mpy shift xlane scat<br>ALU ALU ALU ALU gath<br>L2/TCM<br>**----- End of picture text -----**<br>

**Figure 1-1 Hexagon core with attached SIMD coprocessor**

The Hexagon instruction set architecture (ISA) is extended with HVX instructions. These instructions use HVX compute resources and can freely mix with normal Hexagon instructions in a very long instruction word (VLIW) packet. HVX instructions can also use scalar source operands from the core.

## 1.2 HVX features

HVX adds very wide SIMD capability to the Hexagon ISA. SIMD operations execute on vector registers (currently up to 1024 bits each), and multiple SIMD instructions can execute in parallel.

### 1.2.1 Vector length

HVX supports 1024-bit vectors (128 byte). To minimize porting effort, software should strive to treat vector length as an arbitrary constant power of two.

![Registers using 128 byte with a vector length of 1024 bits](images/p009_0.png)

**----- Start of picture text -----**<br>
Vector context<br>V0 1024-bit Q0 128-bit<br>128sss ...<br>... Q3<br>sbit<br>V31 1024-bit<br>**----- End of picture text -----**<br>

**Figure 1-2 Registers using 128 byte with a vector length of 1024 bits**

### 1.2.2 Vector contexts

A vector context consists of a vector register file, vector predicate file, and the ability to execute instructions using this state.

HVX hardware threads dynamically attach to a vector context; this enables the thread to execute HVX instructions. Multiple hardware threads can execute in parallel, each with a different vector context. The number of supported vector contexts is implementation-defined.

The HVX scalar core can contain any number of hardware threads greater or equal to the number of vector contexts. The scalar hardware thread is assignable to a vector context through perthread SSR:XA programming, as follows:

- SSR:XA = 4: HVX instructions use vector context 0.

- SSR:XA = 5: HVX instructions use vector context 1, if it is available.

- SSR:XA = 6: HVX instructions use vector context 2, if it is available.

- SSR:XA = 7: HVX instructions use vector context 3, if it is available.

Figure 1-3 shows a vector context configuration with four hardware threads, but with two of the threads configured to use 128 byte vectors. In this configuration, two of the threads can execute 128 byte vector instructions, while the other two threads can execute scalar-only instructions.

![Four hardware threads (two HVX-enabled threads and two scalar-only threads)](images/p010_0.png)

**----- Start of picture text -----**<br>
Hexagon<br>D$<br>GRF  GRF<br>thread0 thread2<br>I$<br>GRF  GRF<br>thread1 thread3<br>L2$/<br>Coprocessor<br>L2-TCM<br>instruction port<br>SIMD coprocessor<br>Vector context #0<br>Vector Context #0<br>V0 1024-bit Q0 128bit<br>...<br>... Q3 128bit<br>V31 1024-bit<br>Vector context #1<br>V0 1024- bit Q0 128bit<br>...<br>... Q3 128bit<br>V31 1024-bit<br>**----- End of picture text -----**<br>

**Figure 1-3 Four hardware threads (two HVX-enabled threads and two scalar-only threads)**

### 1.2.3 Memory access

The HVX memory instructions (referred to as VMEM instructions) use the Hexagon general registers (R0 through R31) to form addresses that access memory. The memory access size of these instructions is the vector length or the size of a vector register.

VMEM loads and stores share a 32-bit virtual address space as normal scalar load/stores. VMEM load/stores are coherent with scalar load/stores and hardware maintains coherency.

### 1.2.4 Vector registers

HVX has two sets of registers:

- Data registers consist of 32 vector length registers. Certain operations can access a pair of registers to effectively double the vector length for the operand.

- Predicate registers consist of four registers, each with one bit per byte of vector length. These registers provide operands to compare, mux, and other special instructions.

The vector registers are partitioned into lanes that operate in SIMD fashion. For example, with 1024-bit (128 byte) vector length, each vector register can contain any of the following items:

- 32 words (32-bit elements)

- 64 halfwords (16-bit elements)

- 128 bytes (8-bit elements)

Element ordering is little-endian with the lowest byte in the least-significant position, as shown in Figure 1-4.

|-4.||||||||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|127|126|125|124||7|6|5|4|3|2|1|0|Word<br>Halfword<br>Byte|
|63||62|||3||2||1||0|||
|31|||||1||||0|||||

**Figure 1-4 1024-bit SIMD register**

### 1.2.5 Vector compute instructions

Vector instructions process vector register data in SIMD fashion. The operation is performed on each vector lane in parallel. For example, the following instruction performs a signed ADD operation over each halfword:

V2.h = VADD(V3.h,V4.h)

In this instruction, the halfwords in V3 are summed with the corresponding halfwords in V4, and the results stored in V2.

When vectors are specified in instructions, the element type is also usually specified:

- .b for signed byte

- .ub for unsigned byte

- .h for signed halfword

- .uh for unsigned halfword

- .w for signed word

- .uw for unsigned word

- .qf16 for qfloat16

- .qf32 for qfloat32

- .hf for half precision

- .sf for single precision

For example:

v0.b = vadd(v1.b,v2.b) // Add vectors of bytes v1:0.b = vadd(v3:2.b, v5:4.b) // Add vector pairs of bytes v1:0.h = vadd(v3:2.h, v5:4.h) // Add vector pairs of halfwords v5:4.w = vmpy(v0.h,v1.h) // Widening vector 16 x 16 to 32 // multiplies: halfword inputs, // word outputs

For operations with mixed element sizes, each operand with the smaller element size uses a single vector register and each operand with the larger element size (double the smaller) uses a vector register pair. One vector in a pair contains even elements and the other odd elements.

## 1.3 Technical assistance

For assistance or clarification on information in this document, open a technical support case at https://support.qualcomm.com/.

You will need to register for a Qualcomm ID account and your company must have support enabled to access our Case system.

Other systems and support resources are listed on https://qualcomm.com/support.

If you need further assistance, you can send an email to qualcomm.support@qti.qualcomm.com.
