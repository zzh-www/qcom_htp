<!-- chapter: Front matter (cover, TOC, revision history) -->
<!-- source: 80-N2040-58_AB_Hexagon_V75_HVX_Programmers_Reference_Manual.pdf -->
<!-- pages: 1-7 (1-based as printed in PDF) -->

# Qualcomm Hexagon V75 HVX Programmer's Reference Manual

80-N2040-58 Rev. AB

January 24, 2024

All Qualcomm products mentioned herein are products of Qualcomm Technologies, Inc. and/or its subsidiaries.

Qualcomm and Hexagon are trademarks or registered trademarks of Qualcomm Incorporated. Other product and brand names may be trademarks or registered trademarks of their respective owners.

This technical data may be subject to U.S. and international export, re-export, or transfer ("export") laws. Diversion contrary to U.S. and international law is strictly prohibited.

Qualcomm Technologies, Inc.
5775 Morehouse Drive
San Diego, CA 92121
U.S.A.

© 2023-2024 Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.

## Contents

- **1 Introduction** — 8
  - 1.1 SIMD coprocessor — 8
  - 1.2 HVX features — 9
    - 1.2.1 Vector length — 9
    - 1.2.2 Vector contexts — 9
    - 1.2.3 Memory access — 10
    - 1.2.4 Vector registers — 11
    - 1.2.5 Vector compute instructions — 12
  - 1.3 Technical assistance — 12
- **2 Registers** — 13
  - 2.1 Vector data registers — 13
    - 2.1.1 Unaligned vector pairs — 13
    - 2.1.2 VRF to GRF transfers — 14
  - 2.2 Vector predicate registers — 14
- **3 Memory** — 15
  - 3.1 Alignment — 15
  - 3.2 HVX local memory: VTCM — 15
  - 3.3 Scatter and gather — 16
  - 3.4 Memory type — 17
  - 3.5 Nontemporal — 17
  - 3.6 Permissions — 17
  - 3.7 Ordering — 17
  - 3.8 Atomicity — 18
  - 3.9 Maximizing performance of the vector memory system — 18
    - 3.9.1 Minimize VMEM access — 18
    - 3.9.2 Use aligned data — 18
    - 3.9.3 Avoid store to load stalls — 19
    - 3.9.4 L2FETCH — 19
    - 3.9.5 Access data contiguously — 19
    - 3.9.6 Use nontemporal for final data — 19
    - 3.9.7 Scalar processing of vector data — 19
    - 3.9.8 Avoid scatter/gather stalls — 20
- **4 Vector instructions** — 21
  - 4.1 VLIW packing rules — 21
    - 4.1.1 Double vector instructions — 21
    - 4.1.2 Vector instruction resource usage — 22
    - 4.1.3 Vector instruction — 22
  - 4.2 Vector load/store — 23
  - 4.3 Scatter and gather — 24
  - 4.4 Memory instruction slot combinations — 25
  - 4.5 Special instructions — 25
    - 4.5.1 Histogram — 25
  - 4.6 Qfloat — 26
    - 4.6.1 QFloat best practices — 27
  - 4.7 Instruction latency — 27
  - 4.8 Slot/resource/latency summary — 28
- **5 HVX PMU events** — 29
- **6 HVX instruction set** — 31
  - 6.1 HVX ALL compute resource — 33
    - Histogram — 33
    - Weighted histogram — 36
  - 6.2 HVX ALU double resource — 40
    - Predicate operations — 40
    - Combine — 42
    - In-lane shuffle — 43
    - Swap — 45
    - Sign/zero extension — 47
    - Arithmetic — 50
  - 6.3 HVX ALU resource — 54
    - Predicate operations — 54
    - Byte-conditional vector assign — 55
    - Min/max — 56
    - Absolute value — 59
    - Arithmetic — 61
    - Arithmetic with carry bit — 64
    - Logical operations — 66
    - Copy — 67
    - Temporary assignment — 69
    - Average — 70
    - Compare vectors — 74
    - Conditional accumulate — 82
    - Mux select — 85
    - Saturation — 87
    - In-lane shuffle — 89
  - 6.4 HVX debug — 91
    - Extract vector element — 91
  - 6.5 HVX gather double resource — 93
    - Vector gather — 93
  - 6.6 HVX gather — 95
    - Vector gather — 95
  - 6.7 HVX load — 98
    - Load - aligned — 98
    - Load - immediate use — 104
    - Load - temporary immediate use — 107
    - Load - unaligned — 110
  - 6.8 HVX MPY double resource — 112
    - 3 × 3 multiply for 2 × 2 tile — 112
    - Arithmetic widening — 125
    - Multiply with 2-wide reduction — 128
    - Lookup table for piecewise from 64-bit scalar — 133
    - Multiply with piecewise add/sub from 64-bit scalar — 134
    - Multiply-add — 135
    - Multiply - vector by scalar — 140
    - Multiply - vector by vector — 144
    - Multiply - half precision vector by vector — 148
    - Integer multiply - vector by vector — 151
    - Integer multiply (32×16) — 153
    - Integer multiply accumulate even/odd — 155
    - Multiply - single precision vector by vector — 157
    - Multiply (32 ×16) — 158
    - Multiply bytes with 4-wide reduction - vector by scalar — 161
    - Multiply by byte with accumulate and 4-wide reduction - vector by vector — 165
    - Multiply with 3-wide reduction — 167
    - Sum of reduction of absolute differences halfwords — 172
    - Sum of absolute differences byte — 174
  - 6.9 HVX MPY resource — 177
    - Multiply by byte with 2-wide reduction — 177
    - Multiply by halfword with 2-wide reduction — 179
    - Multiply - vector by scalar non-widening — 181
    - Multiply - vector by vector - non-widening — 183
    - Multiply half of the elements (16 ×16) — 184
    - Integer multiply by byte — 185
    - Multiply half of the elements with scalar (16 ×16) — 187
    - Multiply bytes with 4-wide reduction - vector by scalar — 188
    - Multiply by byte with 4-wide reduction - vector by vector — 190
    - Splat from scalar — 192
    - Vector to predicate transfer — 194
    - Predicate to vector transfer — 195
    - Absolute value of difference — 196
    - Insert element — 198
  - 6.10 HVX permute resource — 199
    - Byte alignment — 199
    - General permute network — 202
    - Shuffle - deal — 207
    - Pack — 210
    - Set predicate — 213
    - Vector in-lane lookup table — 214
  - 6.11 HVX permute shift resource — 220
    - Vector ASR overlay — 220
    - Vector shuffle and deal cross-lane — 222
    - Vector in-lane lookup table — 227
    - 8-bit elements — 227
    - Unpack — 235
  - 6.12 HVX scatter double resource — 237
    - Vector scatter — 237
  - 6.13 HVX scatter — 240
    - Vector scatter — 240
  - 6.14 HVX shift resource — 244
    - Narrowing shift — 244
    - Compute contiguous offsets for valid positions — 248
    - Add - half precision vector by vector — 250
    - Add - single precision vector by vector — 252
    - Shift and add — 254
    - Shift — 257
    - Narrowing shift by vector — 263
    - Convert IEEE floating point to IEEE integer — 265
    - Convert IEEE integer to IEEE floating point — 266
    - Convert Qfloat to IEEE floating point — 267
    - Round to next smaller element size — 268
    - Vector rotate right word — 271
    - Subtract - half precision vector by vector — 272
    - Subtract - single precision vector by vector — 274
    - Bit counting — 276
  - 6.15 HVX store — 278
    - Store - byte-enabled aligned — 278
    - Store - new — 281
    - Store - aligned — 284
    - Store - unaligned — 291

## Figures

| Figure | Title | Page |
|---|---|---|
| Figure 1-1 | Hexagon core with attached SIMD coprocessor | 8 |
| Figure 1-2 | Registers using 128 byte with a vector length of 1024 bits | 9 |
| Figure 1-3 | Four hardware threads (two HVX-enabled threads and two scalar-only threads) | 10 |
| Figure 1-4 | 1024-bit SIMD register | 11 |
| Figure 4-1 | Qfloat format | 26 |
| Figure 6-1 | vdelta example | 202 |
| Figure 6-2 | vrdelta example | 203 |
| Figure 6-3 | Vlut16 with even bytes used to look up a table value | 217 |
| Figure 6-4 | vshuff/vdeal (Vy,Vx, Rt) N = 64/Rt Rt = 2i | 222 |
| Figure 6-5 | vshuff(Vy, Vx, Rt) Rt = -8 == 32 + 16 + 8 | 223 |
| Figure 6-6 | vdeal(Vy,Vx, Rt) Rt = -8 or 56 | 224 |

## Tables

| Table | Title | Page |
|---|---|---|
| Table 2-1 | VRF to GRF transfer instructions | 14 |
| Table 3-1 | Atomicity of types of memory accesses | 18 |
| Table 3-2 | Peak scatter/gather performance for v75 | 20 |
| Table 4-1 | HVX execution resource usage | 22 |
| Table 4-2 | HVX instruction to Hexagon slots mapping | 22 |
| Table 4-3 | Sources for noncontiguous accesses: (Rt, Mu, Vv) | 24 |
| Table 4-4 | Basic scatter and gather instructions | 24 |
| Table 4-5 | Valid VMEM load/store and scatter/gather combinations | 25 |
| Table 4-6 | Differences between IEEE and Qfloat | 26 |
| Table 4-7 | HVX slot/resource/latency summary | 28 |
| Table 6-1 | Instruction syntax symbols | 31 |
| Table 6-2 | Instruction operand symbols | 31 |
| Table 6-3 | Instruction behavior symbols | 32 |
