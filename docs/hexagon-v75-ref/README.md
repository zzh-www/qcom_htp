# Qualcomm Hexagon V75 Programmer's Reference Manual

The reference is the PDF checked in alongside this file:
[`Hexagon_V75_Programmers_Reference_Manual.pdf`](Hexagon_V75_Programmers_Reference_Manual.pdf)
(661 pages) — the core scalar + DSP manual, sibling to the HVX manual.

> A per-chapter Markdown conversion (`pymupdf4llm` extraction + deterministic
> boilerplate strip / cross-page table merge / heading normalization) was
> prepared during reverse-engineering but is **not checked in** — the PDF is the
> canonical source and the split would only bloat the repo.  The page map below
> is kept so you can jump straight to the right pages in the PDF.

## Page map

| Chapter | Pages |
|---|---|
| Front matter (cover, TOC, revision history) | 1–16 |
| 1 Introduction (processor architecture overview) | 17–21 |
| 2 Registers (general + control) | 22–34 |
| 3 Instructions (syntax, classes, packets, duplexes) | 35–47 |
| 4 Data processing (data types, XTYPE/ALU32/vector/CR overview, H.264 CABAC) | 48–69 |
| 5 Memory | 70–92 |
| 6 Conditional execution | 93–101 |
| 7 Software stack | 102–105 |
| 8 Program flow | 106–124 |
| 9 PMU events | 125–141 |
| 10 Instruction encoding | 142–155 |
| 11.1 ALU32 | 156–192 |
| 11.2 CR | 193–203 |
| 11.3 JR | 204–208 |
| 11.4 J | 209–219 |
| 11.5 LD | 220–262 |
| 11.6 MEMOP | 263–265 |
| 11.7 NV (new-value) | 266–281 |
| 11.8 ST | 282–304 |
| 11.9 System | 305–324 |
| 11.10.1 XTYPE ALU | 325–394 |
| 11.10.2 XTYPE bit | 395–413 |
| 11.10.3 XTYPE complex | 414–443 |
| 11.10.4 XTYPE FP | 444–464 |
| 11.10.5 XTYPE MPY | 465–515 |
| 11.10.6 XTYPE perm | 516–536 |
| 11.10.7 XTYPE PRED | 537–555 |
| 11.10.8 XTYPE SHIFT | 556–640 |
| Instruction Index (alphabetical mnemonic → page) | 640–661 |
