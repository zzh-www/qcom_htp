<!-- chapter: 6 Conditional execution -->
<!-- source: 80-N2040-57_AB_Hexagon_V75_Programmers_Reference_Manual.pdf -->
<!-- pages: 93-101 (1-based as printed in PDF) -->

# 6 Conditional execution

The Hexagon processor uses a conditional execution model based on compare instructions that set predicate bits in one of four 8-bit predicate registers (P0 through P3). These predicate bits can conditionally execute certain instructions.

Conditional scalar operations examine only the least-significant bit in a predicate register, while conditional vector operations examine multiple bits in the register.

Branch instructions are the main consumers of the predicate registers.

## 6.1  Scalar predicates

Scalar predicates are 8-bit values in conditional instructions to represent truth values:

- 0xFF represents true

- 0x00 represents false

The Hexagon processor provides the four 8-bit predicate registers P0 through P3 to hold scalar predicates (Section 2.2.5). These registers are assigned values by the predicate-generating instructions, and examined by the predicate-consuming instructions.

### 6.1.1  Generating scalar predicates

The following instructions generate scalar predicates:

- Compare byte, halfword, word, doubleword

- Compare single- and double-precision floating point

- Classify floating-point value

- Compare bitmask

- Bounds check

- TLB match

- Store conditional

**Table 6-1 Scalar predicate-generating instructions**

|**Syntax**|**Operation**|
|---|---|
|Pd = cmpb.eq(Rs,{Rt,#u8})<br>Pd = cmph.eq(Rs,{Rt,#s8})<br>Pd = [!]cmp.eq(Rs,{Rt,#s10})<br>Pd = cmp.eq(Rss,Rtt)<br>Pd = sfcmp.eq(Rs,Rt)<br>Pd = dfcmp.eq(Rss,Rtt)|Equal (signed).<br>Compare register Rs to Rt or a signed immediate for equality.<br>Assign Pd the resulting truth value.|
|Pd = cmpb.gt(Rs,{Rt,#s8}<br>Pd = cmph.gt(Rs,{Rt,#s8})<br>Pd = [!]cmp.gt(Rs,{Rt,#s10})<br>Pd = cmp.gt(Rss,Rtt)<br>Pd = sfcmp.gt(Rs,Rt)<br>Pd = dfcmp.gt(Rss,Rtt)|Greater than (signed).<br>Compare register Rs to Rt or a signed immediate for signed<br>greater than. Assign Pd the resulting truth value.|
|Pd = cmpb.gtu(Rs,{Rt,#u7})<br>Pd = cmph.gtu(Rs,{Rt,#u7})<br>Pd = [!]cmp.gtu(Rs,{Rt,#u9})<br>Pd = cmp.gtu(Rss,Rtt)|Greater than (unsigned).<br>Compare register Rs to Rt or an unsigned immediate for<br>unsigned greater than. Assign Pd the resulting truth value.|
|Pd = cmp.ge(Rs,#s8)<br>Pd = sfcmp.ge(Rs,Rt)<br>Pd = dfcmp.ge(Rss,Rtt)|Greater than or equal (signed).<br>Compare register Rs to Rt or a signed immediate for signed<br>greater than or equal. Assign Pd the resulting truth value.|
|Pd = cmp.geu(Rs,#u8)|Greater than or equal (unsigned).<br>Compare register Rs to an unsigned immediate for unsigned<br>greater than or equal. Assign Pd the resulting truth value.|
|Pd = cmp.lt(Rs,Rt)|Less than (signed).<br>Compare register Rs to Rt for signed less than. Assign Pd the<br>resulting truth value.|
|Pd = cmp.ltu(Rs,Rt)|Less than (unsigned).<br>Compare register Rs to Rt for unsigned less than. Assign Pd<br>the resulting truth value.|
|Pd = sfcmp.uo(Rs,Rt)<br>Pd = dfcmp.uo(Rss,Rtt)|Unordered (signed).<br>Determine if register Rs or Rt is set to the value NaN. Assign<br>Pd the resulting truth value.|
|Pd=sfclass(Rs,#u5)<br>Pd=dfclass(Rss,#u5)|Classify value (signed).<br>Determine if register Rs is set to any of the specified classes.<br>Assign Pd the resulting truth value.|
|Pd = [!]tstbit(Rs,{Rt,#u5})|Test if bit set.<br>Rt or an unsigned immediate specifies a bit position.<br>Test if the bit in Rs that is specified by the bit position is set.<br>Assign Pd the resulting truth value.|
|Pd = [!]bitsclr(Rs,{Rt,#u6})|Test if bits clear.<br>Rt or an unsigned immediate specifies a bitmask.<br>Test if the bits in Rs that are specified by the bitmask are all<br>clear. Assign Pd the resulting truth value.|
|Pd = [!]bitsset(Rs,Rt)|Test if bits set.<br>Rt specifies a bitmask.<br>Test if the bits in Rs that are specified by the bitmask are all<br>set. Assign Pd the resulting truth value.|
|memw_locked(Rs,Pd) = Rt<br>memd_locked(Rs,Pd) = Rtt|Store conditional.<br>If no other atomic operation has been performed at the<br>address (atomicity is ensured), perform the store to the word<br>at address Rs. Assign Pd the resulting truth value.|
|Pd = boundscheck(Rs,Rtt)|Bounds check.<br>Determine if Rs falls in the numeric range defined by Rtt.<br>Assign Pd the resulting truth value.|
|Pd = tlbmatch(Rss,Rt)|Determine if TLB entry in Rss matches the ASID:PPN<br>specified in Rt. Assign Pd the resulting truth value.|

- **NOTE:** One of the compare instructions (cmp.eq) includes a variant that stores a binary predicate value (0 or 1) in a general register not a predicate register.

### 6.1.2  Consuming scalar predicates

Certain instructions can be conditionally executed based on the value of a scalar predicate (or alternatively specify a scalar predicate as an input to their operation).

The conditional instructions that consume scalar predicates examine only the least-significant bit of the predicate value. In the simplest case, this bit value directly determines whether the instruction executes:

- 1 indicates that the instruction executes

- 0 indicates that the instruction does not execute

If a conditional instruction includes the operator ! in its predicate expression, the logical negation of the bit value determines whether the instruction is executed.

Conditional instructions are expressed in assembly language with the instruction prefix if ( _pred_expr_ ) , where _pred_expr_ specifies the predicate expression. For example:

if (P0) jump target         // Jump if P0 is true if (!P2) R2 = R5            // Assign register if !P2 is true if (P1) R0 = sub(R2,R3)     // Conditionally subtract if P1 if (P2) R0 = memw(R2)       // Conditionally load word if P2

The following instructions can be used as conditional instructions:

- Jumps and calls (Section 8.3)

- Many load and store instructions (Section 5.9)

- Logical instructions (including AND/OR/XOR)

- Shift halfword

- 32-bit add/subtract by register or short immediate

- Sign and zero extend

- 32-bit register transfer and 64-bit combine word

- Register transfer immediate

- Deallocate frame and return

When a conditional load or store executes and the predicate expression is false, the instruction is canceled (including any exceptions that might occur). For example, if a conditional load uses an address with a memory permission violation, and the predicate expression is false, the load does not execute and the exception is not raised.

The mux instruction accepts a predicate as one of its basic operands:

Rd = mux(Ps,Rs,Rt)

The mux instruction selects either Rs or Rt based on the least significant bit in Ps. If the leastsignificant bit in Ps is a 1, Rd is set to Rs, otherwise it is set to Rt.

### 6.1.3  Auto-AND predicates

If multiple compare instructions in a packet write to the same predicate register, the result is the logical AND of the individual compare results. For example:

{ P0 = cmp(A)                      // If A && B,  jump P0 = cmp(B) if (P0.new) jump:T taken_path }

To perform the corresponding OR operation, the following instructions can compute the negation of an existing compare (using De Morgan's law):

- Pd = !cmp.{eq,gt}(Rs, {#s10,Rt} )

- Pd = !cmp.gtu(Rs, {#u9,Rt} )

- Pd = !tstbit(Rs, {#u5,Rt} )

- Pd = !bitsclr(Rs, {#u6,Rt} )

- Pd = !bitsset(Rs,Rt)

Auto-AND predicates have the following restrictions:

   - If a packet contains endloopN , it cannot perform an auto-AND with predicate register P3.

   - If a packet contains a register transfer from a general register to a predicate register, no other instruction in the packet can write to the same predicate register. As a result, a register transfer to P3:0 or C5:4 cannot be grouped with any other predicate-writing instruction.

   - The instructions spNloop0, decbin, tlbmatch, memw_locked, memd_locked, a, sub:carry, sfcmp , and dfcmp cannot be grouped with another instruction that sets the same predicate register.

- **NOTE:** A register transfer from a predicate register to a predicate register has the same auto-AND behavior as a compare instruction.

### 6.1.4  Dot-new predicates

The Hexagon processor can generate and use a scalar predicate in the same instruction packet (Section 3.3). This feature is expressed in assembly language by appending the suffix ".new" to the specified predicate register. For example:

if (P0.new) R3 = memw(R4)

The following C statement and the corresponding assembly code that is generated from it by the compiler is an example of how to use dot-new predicates.

## C statement

if (R2 == 4) R3 = *R4; else R5 = 5;

## Assembly code

{ P0 = cmp.eq(R2,#4) if (P0.new) R3 = memw(R4) if (!P0.new) R5 = #5 }

In the assembly code, a scalar predicate is generated and then consumed twice within the same instruction packet.

The following conditions apply to using dot-new predicates:

- The predicate must be generated by an instruction in the same packet. The assembler normally enforces this restriction, but if the processor executes a packet that violates this restriction, the execution result is undefined.

- A single packet can contain both the dot-new and normal forms of predicates. The normal form examines the old value in the predicate register, rather than the newly-generated value. For example:

{ P0 = cmp.eq(R2,#4) if (P0.new) R3 = memw(R4)  // Use newly-generated P0 value if (P0) R5 = #5            // Use previous P0 value }

### 6.1.5  Dependency constraints

Two instructions in an instruction packet should not write to the same destination register (Section 3.3.5). An exception to this rule is when the two instructions are conditional, and only one of them ever has the predicate expression value true when the packet executes.

For example, the following packet is valid as long as P2 and P3 never both evaluate to true when the packet is executed:

{ if (P2) R3 = #4      // P2, P3, or both must be false if (P3) R3 = #7 }

Because predicate values change at runtime, the programmer is responsible for ensuring that such packets are always valid during program execution. If they are invalid, the processor takes the following actions:

- When writing to general registers, an error exception is raised.

- When writing to predicate or control registers, the result is undefined.

## 6.2  Vector predicates

The predicate registers are also used for conditional vector operations. Unlike scalar predicates, vector predicates contain multiple truth values which are generated by vector predicategenerating operations.

For example, a vector compare instruction compares each element of a vector and assigns the compare results to a predicate register. Each bit in the predicate vector contains a truth value indicating the outcome of a separate compare performed by the vector instruction.

The vector mux instruction uses a vector predicate to selectively merge elements from two separate vectors into a single destination vector. This operation is useful for enabling the vectorization of loops with control flow (branches).

The vector instructions that use predicates are described in the following sections.

### 6.2.1  Vector compare

A vector compare instruction inputs two 64-bit vectors, performs separate compares for each pair of vector elements, and generates a predicate value which contains a bit vector of truth values.

In Figure 6-1 two 64-bit vectors of bytes (contained in Rss and Rtt) are being compared. The result is assigned as a vector predicate to the destination register Pd.

In the example vector predicate shown in Figure 6-1, every other compare result in the predicate is true (for example, 1).

**==> picture [316 x 152] intentionally omitted <==**

**----- Start of picture text -----**<br>
Rss<br>Rtt<br>cmp cmp cmp cmp cmp cmp cmp cmp<br>1 0 1 0 1 0 1 0 Pd<br>7 0<br>**----- End of picture text -----**<br>

**Figure 6-1 Vector byte compare**

Figure 6-2 shows how a vector halfword compare generates a vector predicate. Two 64-bit vectors of halfwords are being compared. The result is assigned as a vector predicate to the destination register Pd.

Because a vector halfword compare yields only four truth values, each truth value is encoded as two bits in the generated vector predicate.

**==> picture [316 x 173] intentionally omitted <==**

**----- Start of picture text -----**<br>
Rss<br>Rtt<br>cmp cmp cmp cmp<br>1 1 0 0 1 1 0 0 Pd<br>7 0<br>**----- End of picture text -----**<br>

**Figure 6-2 Vector halfword compare**

### 6.2.2  Vector mux instruction

A vector mux instruction conditionally selects the elements from two vectors. The instruction takes as input two source vectors and a predicate register. For each byte in the vector, the corresponding bit in the predicate register is used to choose from one of the two input vectors. The combined result is written to the destination register.

**==> picture [317 x 129] intentionally omitted <==**

**----- Start of picture text -----**<br>
Rss<br>Rtt<br>mux mux mux mux mux mux mux mux<br>P[7] P[6] P[5] P[4] P[3] P[2] P[1] P[0]<br>Rdd<br>**----- End of picture text -----**<br>

**Figure 6-3 Vector mux instruction**

**Table 6-2 Vector mux instruction**

|**Syntax**|**Operation**|
|---|---|
|Rdd = vmux(Ps,Rss,Rtt)|Select bytes from Rss and Rtt|

Changing the order of the source operands in a mux instruction enables formation of both senses of the result. For example:

R1:0 = vmux(P0,R3:2,R5:4)    // Choose bytes from R3:2 if true R1:0 = vmux(P0,R5:4,R3:2)    // Choose bytes from R3:2 if false

**NOTE:** By replicating the predicate bits generated by word or halfword compares, the vector mux instruction can select words or halfwords.

### 6.2.3  Using vector conditionals

Vector conditional support is used to vectorize loops with conditional statements.

Consider the following C statement:

for (i=0; i<8; i++) { if (A[i]) { B[i] = C[i]; } }

Assuming arrays of bytes, this code can be vectorized as follows:

R1:0 = memd(R_A)            // R1:0 holds A[7]-A[0] R3 = #0 // Clear R3:2 R2 = #0 P0 = vcmpb.eq(R1:0,R3:2)    // Compare bytes in A to zero

R5:4 = memd(R_B)            // R5:4 holds B[7]-B[0] R7:6 = memd(R_C)            // R7:6 holds C[7]-C[0] R3:2 = vmux(P0,R7:6,R5:4)   // if (A[i]) B[i]=C[i] memd(R_B) = R3:2            // store B[7]-B[0]

## 6.3  Predicate operations

The Hexagon processor provides a set of operations for manipulating and moving predicate registers.

**Table 6-3 Predicate register instructions**

|**Syntax**|**Operation**|
|---|---|
|Pd = Ps|Transfer predicate Ps to Pd|
|Pd = Rs|Transfer register Rs to predicate Pd|
|Rd = Ps|Transfer predicate Ps to register Rd|
|Pd = and(Ps,[!]Pt)|Set Pd to bitwise AND of Ps and [NOT] Pt|
|Pd = or(Ps,[!]Pt)|Set Pd to bitwise OR of Ps and [NOT] Pt|
|Pd = and(Ps, and(Pt,[!]Pu)|Set Pd to AND of Ps and (AND of Pt and [NOT] Pu)|
|Pd = and(Ps, or(Pt,[!]Pu)|Set Pd to AND of Ps and (OR of Pt and [NOT] Pu)|
|Pd = or(Ps, and(Pt,[!]Pu)|Set Pd to OR of Ps and (AND of Pt and [NOT] Pu)|
|Pd = or(Ps, or(Pt,[!]Pu)|Set Pd to OR of Ps and (OR of Pt and [NOT] Pu)|
|Pd = not(Ps)|Set Pd to bitwise inversion of Ps|
|Pd = xor(Ps,Pt)|Set Pd to bitwise exclusive OR of Ps and Pt|
|Pd = any8(Ps)|Set Pd to 0xFF if any bit in Ps is 1, 0x00 otherwise|
|Pd = all8(Ps)|Set Pd to 0x00 if any bit in Ps is 0, 0xFF otherwise|

**NOTE:** These instructions belong to instruction class CR.

Predicate registers can be transferred to and from the general registers either individually or as register quadruples (Section 2.2.5).
