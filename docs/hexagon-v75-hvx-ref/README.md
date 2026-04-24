# Qualcomm Hexagon V75 HVX Programmer's Reference Manual

Markdown conversion of `80-N2040-58_AB_Hexagon_V75_HVX_Programmers_Reference_Manual.pdf` (304 pages). Extracted with `pymupdf4llm` and polished deterministically (boilerplate strip, cross-page table merge, heading normalization, image embedding).

Source PDF: `tools/hexagon-sdk/docs/pdf/80-N2040-58_AB_Hexagon_V75_HVX_Programmers_Reference_Manual.pdf`

## Chapters

- [Front matter (cover, TOC, revision history)](00_front_matter.md) — p1-7
- [1 Introduction](01_introduction.md) — p8-12
- [2 Registers](02_registers.md) — p13-14
- [3 Memory](03_memory.md) — p15-20
- [4 Vector instructions](04_vector_instructions.md) — p21-28
- [5 HVX PMU events](05_hvx_pmu_events.md) — p29-30

### 6 HVX instruction set (p31-304)

- [6.1 HVX ALL compute resource](06_01_hvx_all_compute.md) — p31-39 (histogram, weighted histogram)
- [6.2 HVX ALU double resource](06_02_alu_double.md) — p40-53 (predicate ops, combine, shuffle, swap, sign/zero-ext, arithmetic)
- [6.3 HVX ALU resource](06_03_alu.md) — p54-90 (predicate ops, byte conditional assign, min/max, abs, arithmetic, logical, compare, etc.)
- [6.4 HVX debug](06_04_debug.md) — p91-92 (extract vector element)
- [6.5 HVX gather double resource](06_05_gather_double.md) — p93-94
- [6.6 HVX gather](06_06_gather.md) — p95-97
- [6.7 HVX load](06_07_load.md) — p98-111
- [6.8 HVX MPY double resource](06_08_mpy_double.md) — p112-176 (largest section; HMA/HMX matrix-accumulator instructions)
- [6.9 HVX MPY resource](06_09_mpy.md) — p177-198
- [6.10 HVX permute resource](06_10_permute.md) — p199-219
- [6.11 HVX permute shift resource](06_11_permute_shift.md) — p220-236
- [6.12 HVX scatter double resource](06_12_scatter_double.md) — p237-239
- [6.13 HVX scatter](06_13_scatter.md) — p240-243
- [6.14 HVX shift resource](06_14_shift.md) — p244-277
- [6.15 HVX store](06_15_store.md) — p278-304

## Images

Extracted figures are under `images/pNNN_K.png`, where `NNN` is the 1-based PDF page number and `K` distinguishes multiple figures on one page. Each chapter file references its figures inline via `![caption](images/...)`.

## Provenance

Each chapter file begins with an HTML-comment header recording the source PDF filename and the 1-based page range it covers. Technical content (Syntax / Behavior / Class / Notes / Encoding / Field-name tables) is preserved verbatim; only formatting/boilerplate was rewritten.
