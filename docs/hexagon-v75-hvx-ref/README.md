# Qualcomm Hexagon V75 HVX Programmer's Reference Manual

The reference is the PDF checked in alongside this file:
[`Hexagon_V75_HVX_Programmers_Reference_Manual.pdf`](Hexagon_V75_HVX_Programmers_Reference_Manual.pdf)
(304 pages).

> A per-chapter Markdown conversion (`pymupdf4llm` extraction + deterministic
> boilerplate strip / cross-page table merge / heading normalization / image
> embedding) was prepared during reverse-engineering but is **not checked in** —
> the PDF is the canonical source and the split would only bloat the repo.  The
> page map below is kept so you can jump straight to the right pages in the PDF.

## Page map

| Chapter | Pages |
|---|---|
| Front matter (cover, TOC, revision history) | 1–7 |
| 1 Introduction | 8–12 |
| 2 Registers | 13–14 |
| 3 Memory | 15–20 |
| 4 Vector instructions | 21–28 |
| 5 HVX PMU events | 29–30 |
| 6.1 HVX ALL compute resource (histogram, weighted histogram) | 31–39 |
| 6.2 HVX ALU double resource (predicate, combine, shuffle, swap, ext, arith) | 40–53 |
| 6.3 HVX ALU resource (predicate, conditional assign, min/max, abs, compare) | 54–90 |
| 6.4 HVX debug (extract vector element) | 91–92 |
| 6.5 HVX gather double resource | 93–94 |
| 6.6 HVX gather | 95–97 |
| 6.7 HVX load | 98–111 |
| 6.8 HVX MPY double resource (largest section; HMA/HMX matrix-accumulator) | 112–176 |
| 6.9 HVX MPY resource | 177–198 |
| 6.10 HVX permute resource | 199–219 |
| 6.11 HVX permute shift resource | 220–236 |
| 6.12 HVX scatter double resource | 237–239 |
| 6.13 HVX scatter | 240–243 |
| 6.14 HVX shift resource | 244–277 |
| 6.15 HVX store | 278–304 |
