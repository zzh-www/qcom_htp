<!-- chapter: 1 Introduction -->
<!-- source: 80-N2040-57_AB_Hexagon_V75_Programmers_Reference_Manual.pdf -->
<!-- pages: 17-21 (1-based as printed in PDF) -->

# 1 Introduction

The Qualcomm Hexagon™ processor is a general-purpose digital signal processor designed for high performance and low power across a wide variety of multimedia and modem applications. V75 is a member of the sixth generation of the Hexagon processor architecture.

## 1.1 Hexagon V75 processor architecture

### 1.1.1 Memory

The Hexagon processor features a unified byte-addressable memory. This memory has a single 32-bit virtual address space, which holds both instructions and data. It operates in little endian mode.

The load/store architecture supports a complete set of addressing modes for both compiler code generation and DSP application programming.

#### 1.1.1.1 Cache memory

Memory accesses are cached or uncached. Separate L1 instruction and data caches exist for program code and data. A unified L2 cache is partly or wholly configured as tightly coupled memory (TCM).

#### 1.1.1.2 Virtual memory

Memory is addressed virtually, with virtual-to-physical memory mapping handled by a resident OS. Virtual memory supports the implementation of memory management and memory protection in a hardware-independent manner.

### 1.1.2 Registers

The Hexagon processor has two sets of registers: General registers and Control registers.

The general registers include thirty-two 32-bit registers (named R0 through R31), which are accessed either as single registers or as aligned 64-bit register pairs. The general registers contain all data, including pointer, scalar, vector, and accumulator data.

The control registers include special-purpose registers such as program counter, status register, loop registers, and so on.

### 1.1.3 Instruction sequencer

The instruction sequencer processes packets of one to four instructions in each cycle. If a packet contains more than one instruction, the instructions execute in parallel.

The instruction combinations allowed in a packet are limited to the instruction types that can execute in parallel in the four execution units (shown in Figure 1-1).

**==> picture [417 x 436] intentionally omitted <==**

**----- Start of picture text -----**<br>
Memory<br>( unified address space )<br>Load/store 64 Load/store<br>4 ´ 32 bit 64<br>128<br>instructions S3: X unit<br>XTYPE instructions<br>ALU32 instructions<br>J instructions<br>CR instructions<br>Sequencer<br>Packets of<br>1 to 4 instructions<br>S2: X unit<br>XTYPE instructions<br>ALU32 instructions<br>J instructions<br>JR instructions<br>Control registers<br>  Hardware loop  General registers<br>registers<br>S1: Load/store unit<br>  Modifier registers  R0 to R31<br>LD instructions<br>  Status register<br>ST instructions<br>  Program counter<br>ALU32 instructions<br>  Predicate registers<br>User general pointer<br>Global pointer<br>  Circular start registers<br>S0: Load/store unit<br>LD instructions<br>ST instructions<br>ALU32 instructions<br>MEMOP<br>instructions<br>NV instructions<br>SYSTEM<br>instructions<br>**----- End of picture text -----**<br>

**Figure 1-1 Hexagon V75 processor architecture**

### 1.1.4 Execution units

The dual execution units are identical: each includes a 64-bit shifter and a vector

multiply/accumulate unit with four 16 × 16 multipliers to support both scalar and vector instructions.

These units also perform 32-bit and 64-bit ALU instructions, and jump and loop instructions.

**NOTE:** Each execution unit supports floating-point instructions.

### 1.1.5 Load/store units

The two load/store units can operate on signed or unsigned bytes, halfwords (16-bit), words (32bit), or double words (64-bit).

To increase the number of instruction combinations allowed in packets, the load units also support 32-bit ALU instructions.

## 1.2 Instruction set

For the Hexagon processor to achieve large amounts of work per cycle, the instruction set is designed with the following properties:

- Static grouping (VLIW) architecture

- Static fusing of simple dependent instructions

- Extensive compound instructions

- A large set of SIMD and application-specific instructions

To support efficient compilation, the instruction set is orthogonal with respect to registers, addressing modes, and load/store access size.

### 1.2.1 Addressing modes

The Hexagon processor supports the following memory addressing modes:

- 32-bit absolute

- 32-bit absolute-set

- Absolute with register offset

- Global pointer relative

- Indirect

- Indirect with offset

- Indirect with register offset

- Indirect with auto-increment (immediate or register)

- Circular with auto-increment (immediate or register)

- Bit-reversed with auto-increment register

For example: R2 = memw(##myvariable) R2 = memw(R3=##myvariable) R2 = memw(R4<<#3+##myvariable) R2 = memw(GP+#200) R2 = memw(R1) R2 = memw(R3+#100) R2 = memw(R3+R4<<#2) R2 = memw(R3++#4) R2 = memw(R0++M1) R0 = memw(R2++#8:circ(M0)) R0 = memw(R2++I:circ(M0)) R2 = memw(R0++M1:brev)

Auto-increment with register addressing uses one of the two dedicated address-modify registers M0 and M1 (which are part of the control registers).

**NOTE:** Atomic memory operations (load locked/store conditional) are supported to implement multithread synchronization.

### 1.2.2 Program flow

The Hexagon processor supports zero-overhead hardware loops. For example:

loop0(start,#3)      // loop 3 times start: { R0 = mpyi(R0,R0) } :endloop0

The loop instructions support nestable loops, with few restrictions on their use.

Software branches use a predicated branch mechanism. Explicit compare instructions generate a predicate bit, which conditional branch instructions then test. For example:

P1 = cmp.eq(R2, R3) if (P1) jump end

Jumps and subroutine calls are conditional or unconditional, and support both PC-relative and register indirect addressing modes. For example:

jump end jumpr R1 call function callr R2

The subroutine call instructions store the return address in register R31. Subroutine returns are performed using a jump indirect instruction through this register. For example:

jumpr R31     // Subroutine return

Two program flow instructions can be grouped into one packet.

### 1.2.3 Instruction pipeline

Hardware resolves pipeline hazards: pipeline restrictions do not constrain instruction scheduling.

## 1.3 Technical assistance

For assistance or clarification on information in this document, open a technical support case at https://support.qualcomm.com/.

You will need to register for a Qualcomm ID account and your company must have support enabled to access our Case system.

Other systems and support resources are listed on https://qualcomm.com/support.

If you need further assistance, you can send an email to qualcomm.support@qti.qualcomm.com.
