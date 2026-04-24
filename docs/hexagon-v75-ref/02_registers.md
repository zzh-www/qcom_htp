<!-- chapter: 2 Registers -->
<!-- source: 80-N2040-57_AB_Hexagon_V75_Programmers_Reference_Manual.pdf -->
<!-- pages: 22-34 (1-based as printed in PDF) -->

# 2 Registers

General registers are used for general-purpose computation, including address generation, and scalar and vector arithmetic.

Control registers support special-purpose processor features such as hardware loops and predicates.

## 2.1 General registers

The Hexagon processor has thirty-two 32-bit general-purpose registers (named R0 through R31). These registers store operands in the instructions:

-  Memory addresses for load/store instructions

- Data operands for arithmetic/logic instructions

- Vector operands for vector instructions

For example:

R1 = memh(R0)              // Load from address R0 R4 = add(R2,R3)            // Add R28 = vaddh(R11,R10)       // Vector add halfword

**==> picture [152 x 160] intentionally omitted <==**

**----- Start of picture text -----**<br>
R3 R2 R1 R0<br>R3:2 R1:0<br>.<br>.<br>.<br>R31 R30 R29 R28<br>R31:30 R29:28<br>**----- End of picture text -----**<br>

**Figure 2-1 General registers**

## Aliased registers

Three of the general registers - R29 through R31 - support subroutines (Section 8.3.2) and the Software stack. The subroutine and stack instructions implicitly modify the registers. They have symbol aliases that indicate when these registers are accessed as subroutine and stack registers.

For example:

SP = add(SP, #-8)   // SP is an alias of R29 allocframe          // Modifies SP (R29) and FP (R30) call init           // Modifies LR (R31)

**Table 2-1 General register aliases**

|**Register**|**Alias**|**Name**|**Description**|
|---|---|---|---|
|R29|SP|Stack pointer|Points to the topmost element of the stack in memory.|
|R30|FP|Frame pointer|Points to the current procedure frame on the stack.<br>Used by external debuggers to examine the stack and determine<br>call sequence, parameters, local variables, and so on.|
|R31|LR|Link register|Stores the return address of a subroutine call.|

## Register pairs

The general registers can be specified as register pairs that represent a single 64-bit register. For example:

R1:0 = memd(R3)                   // Load doubleword

R7:6 = valignb(R9:8,R7:6, #2)     // Vector align

**NOTE:** The first register in a register pair must always be odd-numbered, and the second must be the next lower register.

**Table 2-2 General register pairs**

|**Register**|**Register pair**|
|---|---|
|R0|R1:0|
|R1||
|R2|R3:2|
|R3||
|R4|R5:4|
|R5||
|R6|R7:6|
|R7||
||**…**|
|R24|R25:24|
|R25||
|R26|R27:26|
|R27||
|R28|R29:28|
|R29 (SP)||
|R30 (FP)|R31:30 (LR:FP)|
|R31 (LR)||

## 2.2 Control registers

The Hexagon processor includes a set of 32-bit control registers that provide access to processor features such as the program counter, hardware loops, and vector predicates.

Unlike general registers, control registers are used as instruction operands only in the following cases:

- Instructions that require a specific control register as an operand

- Register transfer instructions

For example:

R2 = memw(R0++M1)   // Auto-increment addressing mode (M1)

- R9 = PC             // Get program counter (PC)

LC1 = R3            // Set hardware loop count (LC1)

- **NOTE:** When a control register is used in a register transfer, the other operand must be a general register.

**Figure 2-2 Control registers**

**==> picture [383 x 310] intentionally omitted <==**

**----- Start of picture text -----**<br>
LC0 SA0 UPCYCLELO<br>Loop registers Cycle count registers<br>LC1 SA1 UPCYCLEHI<br>PC Program counter  FRAMELIMIT Stack bounds register<br>USR User status register FRAMEKEY Stack smash register<br>M0 PKTCOUNTLO<br>Modifier registers Packet count registers<br>M1 PKTCOUNTHI<br>P3:0 Predicate registers UTIMERLO<br>Qtimer registers<br>UGP User general pointer UTIMERHI<br>GP Global pointer<br>CS0<br>Circular start registers<br>CS1<br>**----- End of picture text -----**<br>

## Aliased registers

The control registers have numeric aliases (C0 through C31).

**Table 2-3 Aliased control registers**

|**Register**|**Alias**|**Name**|
|---|---|---|
|SA0|C0|Loop start address register 0|
|LC0|C1|Loop count register 0|
|SA1|C2|Loop start address register 1|
|LC1|C3|Loop count register 1|
|P3:0|C4|Predicate registers 3:0|
|reserved|C5|-|
|M0|C6|Modifier register 0|
|M1|C7|Modifier register 1|
|USR|C8|User status register|
|PC|C9|Program counter|
|UGP|C10|User general pointer|
|GP|C11|Global pointer|
|CS0|C12|Circular start register 0|
|CS1|C13|Circular start register 1|
|UPCYCLELO|C14|Cycle count register (low)|
|UPCYCLEHI|C15|Cycle count register (high)|
|UPCYCLE|C15:14|Cycle count register|
|FRAMELIMIT|C16|Frame limit register|
|FRAMEKEY|C17|Frame key register|
|PKTCOUNTLO|C18|Packet count register (low)|
|PKTCOUNTHI|C19|Packet count register (high)|
|PKTCOUNT|C19:18|Packet count register|
|reserved|C20-29|-|
|UTIMERLO|C30|QTimer register (low)|
|UTIMERHI|C31|QTimer register (high)|
|UTIMER|C31:30|QTimer register|

**NOTE:** The control register numbers (0 through 31) specify the control registers in Instruction encodings.

## Register pairs

The control registers can be specified as register pairs that represent a single 64-bit register. Control registers specified as pairs must use their numeric aliases. For example:

   - C1:0 = R5:4 // C1:0 specifies the LC0/SA0 register pair

- **NOTE:** The first register in a control register pair must always be odd-numbered, and the second must be the next lower register.

**Table 2-4 Control register pairs**

|**Register**|**Register pair**|
|---|---|
|C0|C1:0|
|C1||
|C2|C3:2|
|C3||
|C4|C5:4|
|C5||
|C6|C7:6|
|C7||
||**…**|
|C30|C31:30|
|C31||

### 2.2.1 Program counter

The program counter (PC) register points to the next instruction packet to execute (Section 3.3). Instruction execution implicitly modifies the PC register, but PC an be read directly. For example:

R7 = PC      // Get program counter

**NOTE:** The PC register is read only: writing to it has no effect.

### 2.2.2 Loop registers

The Hexagon processor includes two sets of loop registers to support nested Hardware loops. Each hardware loop is implemented with a pair of registers containing the loop count and loop start address. The loop instruction implicitly modifies the loop registers, but the loop instructions can also be accessed directly. For example:

loop0(start, R4)  // Modifies LC0 and SA0 (LC0 = R4, SA0 = &start) LC1 = R22         // Set loop1 count R9 = SA1          // Get loop1 start address

**Table 2-5 Loop registers**

|**Register**|**Name**|**Description**|
|---|---|---|
|LC0, LC1|Loop count|Number of loop iterations to execute.|
|SA0, SA1|Loop start address|Address of first instruction in loop.|

### 2.2.3 User status register

The user status register (USR) stores processor status and control bits that user programs can access. The status bits contain the status results of certain instructions, while the control bits contain user-settable processor modes for hardware prefetching. For example:

R9:8 = vaddw(R9:8, R3:2):sat     // Vector add words R6 = USR                         // Get saturation status

USR stores the following status and control values:

- Cache prefetch types supported by the Hexagon processor enable

- Cache prefetch status

- Floating point modes

- Floating point status

- Hardware loop configuration (Section 8.2)

- Sticky Saturation overflow

**NOTE:** A user control register transfer to USR cannot be grouped in an instruction packet with a Floating point instruction.

When a transfer to USR changes the enable trap bits [29:25], an isync instruction (Section 5.11) must execute before the new exception programming can take effect.

**Table 2-6 User status register**

|**Name**|**RW**|**Bits**|**Field**|**Description**|
|---|---|---|---|---|
|USR||32||User status register|
||R|31|PFA|L2 prefetch active.<br>1: l2fetch instruction in progress<br>0: l2fetch finished (or inactive)<br>Set when the nonblocking l2fetch instruction prefetches<br>therequested data.<br>Remains set until thel2fetch prefetch operation<br>completes (or is inactive).|
||R|30|reserved|Return 0 if read.<br>Reserved for future expansion. To remain compatible<br>with future processor versions, software should always<br>write this field with the same value read from the field.|
||RW|29|FPINEE|Enable trap on IEEE inexact.|
||RW|28|FPUNFE|Enable trap on IEEE underflow.|
||RW|27|FPOVFE|Enable trap on IEEE overflow.|
||RW|26|FPDBZE|Enable trap on IEEE divide-by-zero.|
||RW|25|FPINVE|Enable trap on IEEE invalid.|
||R|24|reserved|Reserved|
||RW|23:22|FPRND|Rounding mode for floating point instructions.<br>00: Round to nearest, ties to even (default)<br>01: Toward zero<br>10: Downward (toward negative infinity)<br>11: Upward (toward positive infinity)|
||R|21:20|reserved|Return 0 if read.<br>Reserved for future expansion. To remain compatible<br>with future processor versions, software should always<br>write this field with the same value read from the field.|
||R|19:18|reserved|Reserved|
||R|17|reserved|Return 0 if read.<br>Reserved for future expansion. To remain compatible<br>with future processor versions, software should always<br>write this field with the same value read from the field.|
||RW|16:15|HFI|L1 instruction prefetch.<br>00: Disable<br>01: Enable (1 line)<br>10: Enable (2 lines)|
||RW|14:13|HFD|L1 data cache prefetch.<br>Four levels are defined from disabled to aggressive.<br>Implementation defines how to interpret these levels.<br>00: disable<br>01: conservative<br>10: moderate<br>11: aggressive|
||RW|12|PCMME|Enable packet counting in Monitor mode.|
||RW|11|PCGME|Enable packet counting in Guest mode.|
||RW|10|PCUME|Enable packet counting in User mode.|
||RW|9:8|LPCFGE|Hardware loop configuration.<br>Number of loop iterations (0 to 3) that remain before the<br>pipeline predicate should be set.|
||R|7:6|reserved|Return 0 if read.<br>Reserved for future expansion. To remain compatible<br>with future processor versions, software should always<br>write this field with the same value read from the field.|
||RW|5|FPINPF|Floating-point IEEE inexact sticky fag.|
||RW|4|FPUNFF|Floating-point IEEE underflow sticky flag.|
||RW|3|FPOVFF|Floating-point IEEE overflow sticky flag.|
||RW|2|FPDBZF|Floating-point IEEE divide-by-zero sticky flag.|
||RW|1|FPINVF|Floating-point IEEE invalid sticky flag.|
||RW|0|OVF|Sticky saturation overflow.<br>1: saturation occurred<br>0: no saturation<br>Set when saturation occurs while executing an<br>instruction that specifies optional saturation.<br>Remains set until explicitly cleared by a USR = Rs<br>instruction.|

### 2.2.4 Modifier registers

The modifier registers (M0 to M1) are used in the following addressing modes.

## Indirect auto-increment

In Indirect with auto-increment register addressing, the modifier registers store a signed 32-bit value that specifies the increment (or decrement) value.

**Table 2-7 Modifier registers used in indirect auto-increment addressing**

|**Register**|**Name**|**Description**|
|---|---|---|
|M0, M1|Increment|Signed auto-increment value.|

## Circular

In circular addressing (Section 5.8.10), the modifier registers store the circular buffer length and related "I" values.

**Table 2-8 Modifier registers as used in circular addressing**

|**Name**|**RW**|**Bits**|**Field**|**Description**|
|---|---|---|---|---|
|M0, M1||32||Circular buffer specifier.|
||RW|31:28|I[10:7]|I value (MSB - seeSection 5.8.11)|
||RW|27:24||0x0|
||RW|23:17|I[6:0]|I value (LSB)|
||RW|16:0|Length|Circular buffer length|

## Bit-reversed

In bit-reversed addressing (Section 5.8.12), the modifier registers store a signed 32-bit value that specifies the increment (or decrement) value.

**Table 2-9 Modifier registers as used in bit-reversed addressing**

|**Register**|**Name**|**Description**|
|---|---|---|
|M0, M1|Increment|Signed auto-increment value.|

### 2.2.5 Predicate registers

The predicate registers (P0 through P3) store the status results of the scalar and vector compare instructions (Chapter 6). For example:

P1 = cmp.eq(R2, R3)       // Scalar compare if (P1) jump end          // Jump to address (conditional) R8 = P1                   // Get compare status (P1 only) P3:0 = R4                 // Set compare status (P0 through P3)

The four predicate registers can be specified as a register quadruple (P3:0) that represents a single 32-bit register.

**NOTE:** Unlike the other control registers, the predicate registers are only eight bits wide because vector compares return a maximum of eight status results.

**Table 2-10 Predicate registers**

|**Register**|**Bits**|**Description**|
|---|---|---|
|P0, P1, P2, P3|8|Compare status results.|
|P3:0|32|Compare status results.|
||31:24|P3 register|
||23:16|P2 register|
||15:8|P1 register|
||7:0|P0 register|

### 2.2.6 Circular start registers

The circular start registers (CS0 through CS1) store the start address of a circular buffer in circular addressing (Section 5.8.10). For example:

CS0 = R5                     // Set circ start register M0 = R7                      // Set modifier register R0 = memb(R2++#4:circ(M0))   // Load from circ buffer pointed // to by CS0 with size/K vals in M0

**Table 2-11 Circular start registers**

|**Register**|**Name**|**Description**|
|---|---|---|
|CS0, CS1|Circular start|Circular buffer start address.|

### 2.2.7 User general pointer register

The user general pointer (UGP) register is a general-purpose control register. For example:

R9 = UGP      // Get UGP UGP = R3      // Set UGP

**NOTE:** UGP typically stores the address of thread local storage.

**Table 2-12 User general pointer register**

|**Register**|**Name**|**Description**|
|---|---|---|
|UGP|User general pointer|General-purpose control register.|

### 2.2.8 Global pointer

The global pointer (GP) is used in GP-relative addressing. For example:

GP = R7 // Set GP R2 = memw(GP+#200)     // GP-relative load

**Table 2-13 Global pointer register**

|**Name**|**R/W**|**Bits**|**Field**|**Description**|
|---|---|---|---|---|
|GP||32||Global pointer register|
||R/W|31:6|GDP|Global data pointer (Section 5.8.4).|
||R|5:0|reserved|Return 0 if read.<br>Reserved for future expansion. To remain forward-<br>compatible with future processor versions, software<br>should always write this field with the same value<br>read from the field.|

### 2.2.9 Cycle count registers

The cycle count registers (UPCYCLELO to UPCYCLEHI) store a 64-bit value containing the current number of processor cycles executed since the Hexagon processor was last reset. For example:

R5 = UPCYCLEHI     // Get cycle count (high) R4 = UPCYCLELO     // Get cycle count (low) R5:4 = UPCYCLE // Get cycle count

**NOTE:** The RTOS must grant permission to access these registers. Without this permission, reading these registers from user code returns zero.

**Table 2-14 Cycle count registers**

|**Register**|**Name**|**Description**|
|---|---|---|
|UPCYCLELO|Cycle count (low)|Processor cycle count (low 32 bits)|
|UPCYCLEHI|Cycle count (high)|Processor cycle count (high 32 bits)|
|UPCYCLE|Cycle count|Processor cycle count (64 bits)|

### 2.2.10 Frame limit register

The frame limit register (FRAMELIMIT) stores the low address of the memory area reserved for the software stack (Section 7.3.1). For example:

R9 = FRAMELIMIT // Get frame limit register FRAMELIMIT = R3      // Set frame limit register

**Table 2-15 Frame limit register**

|**Register**|**Name**|**Description**|
|---|---|---|
|FRAMELIMIT|Frame limit|Low address of software stack area.|

### 2.2.11 Frame key register

The frame key register (FRAMEKEY) stores the key value that XOR-scrambles return addresses when they are stored on the software stack (Section 7.3.2). For example:

R2 = FRAMEKEY // Get frame key register FRAMEKEY = R1 // Set frame key register

**Table 2-16 Frame key register**

|**Register**|**Name**|**Description**|
|---|---|---|
|FRAMEKEY|Frame key|Key to scramble return addresses<br>stored on the software stack.|

### 2.2.12 Packet count registers

The packet count registers (PKTCOUNTLO to PKTCOUNTHI) store a 64-bit value containing the current number of instruction packets executed since a PKTCOUNT register was last written to. For example:

R9 = PKTCOUNTHI     // Get packet count (high) R8 = PKTCOUNTLO     // Get packet count (low) R9:8 = PKTCOUNT // Get packet count

Packet counting can be configured to operate only in specific sets of processor modes (for example, User mode only, or Guest and Monitor modes only). Bits [12:10] in the User status register control the configuration for each mode.

Packets with exceptions are not counted as committed packets.

**NOTE:** Each hardware thread has its own set of packet count registers.

The RTOS must grant permission to access these registers. Without this permission, reading these registers from user code returns zero.

When a value is written to a PKTCOUNT register, the 64-bit packet count value is incremented before the value is stored in the register.

**Table 2-17 Packet count registers**

|**Register**|**Name**|**Description**|
|---|---|---|
|PKTCOUNTLO|Packet count (low)|Processor packet count (low 32 bits)|
|PKTCOUNTHI|Packet count (high)|Processor packet count (high 32 bits)|
|PKTCOUNT|Cycle count|Processor packet count (64 bits)|

### 2.2.13 QTimer registers

The QTimer registers (UTIMERLO to UTIMERHI) provide access to the QTimer global reference count value. QTimer registers enable Hexagon software to read the 64-bit time value without having to perform an expensive advanced high-performance bus (AHB) load. For example:

R5 = UTIMERHI     // Get QTimer reference count (high) R4 = UTIMERLO     // Get QTimer reference count (low) R5:4 = UTIMER // Get QTimer reference count

These registers are read only - hardware automatically updates these registers to contain the current QTimer value.

**NOTE:** The RTOS must grant permission to access these registers. Without this permission, reading these registers from user code returns zero.

**Table 2-18 QTimer registers**

|**Register**|**Name**|**Description**|
|---|---|---|
|UTIMERLO|QTimer (low)|QTimer global reference count (low 32 bits)|
|UTIMERHI|QTimer (high)|QTimer global reference count (high 32 bits)|
|UTIMER|QTimer|QTimer global reference count (64 bits)|
