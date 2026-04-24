# Qualcomm Hexagon V75 Programmer's Reference Manual

Markdown conversion of `80-N2040-57_AB_Hexagon_V75_Programmers_Reference_Manual.pdf` (661 pages) — the core (scalar + DSP) manual, sibling to the HVX manual. Extracted with `pymupdf4llm` and polished deterministically (boilerplate strip, cross-page table merge, heading normalization).

Source PDF: `tools/hexagon-sdk/docs/pdf/80-N2040-57_AB_Hexagon_V75_Programmers_Reference_Manual.pdf`

## Chapters

- [Front matter (cover, TOC, revision history)](00_front_matter.md) — p1-16
- [1 Introduction](01_introduction.md) — p17-21 (processor architecture overview)
- [2 Registers](02_registers.md) — p22-34 (general + control registers)
- [3 Instructions](03_instructions.md) — p35-47 (syntax, classes, packets, duplexes)
- [4 Data processing](04_data_processing.md) — p48-69 (data types, options, XTYPE/ALU32/vector/CR operation overview, H.264 CABAC)
- [5 Memory](05_memory.md) — p70-92
- [6 Conditional execution](06_conditional_execution.md) — p93-101
- [7 Software stack](07_software_stack.md) — p102-105
- [8 Program flow](08_program_flow.md) — p106-124
- [9 PMU events](09_pmu_events.md) — p125-141 (single merged PMU-event table)
- [10 Instruction encoding](10_instruction_encoding.md) — p142-155

### 11 Instruction set (p156-640)

- [11.1 ALU32](11_01_alu32.md) — p156-192
- [11.2 CR](11_02_cr.md) — p193-203
- [11.3 JR](11_03_jr.md) — p204-208
- [11.4 J](11_04_j.md) — p209-219
- [11.5 LD](11_05_ld.md) — p220-262
- [11.6 MEMOP](11_06_memop.md) — p263-265
- [11.7 NV (new-value)](11_07_nv.md) — p266-281
- [11.8 ST](11_08_st.md) — p282-304
- [11.9 System](11_09_system.md) — p305-324

#### 11.10 XTYPE

- [11.10.1 XTYPE ALU](11_10_01_xtype_alu.md) — p325-394 (~70 p)
- [11.10.2 XTYPE bit](11_10_02_xtype_bit.md) — p395-413
- [11.10.3 XTYPE complex](11_10_03_xtype_complex.md) — p414-443
- [11.10.4 XTYPE FP (floating-point)](11_10_04_xtype_fp.md) — p444-464
- [11.10.5 XTYPE MPY](11_10_05_xtype_mpy.md) — p465-515
- [11.10.6 XTYPE perm](11_10_06_xtype_perm.md) — p516-536
- [11.10.7 XTYPE PRED](11_10_07_xtype_pred.md) — p537-555
- [11.10.8 XTYPE SHIFT](11_10_08_xtype_shift.md) — p556-~640

### Back matter

- [Instruction Index (alphabetical mnemonic → page)](99_instruction_index.md) — ~p640-661 (PDF's back-of-book index; not in the source TOC)

## Provenance

Each chapter file begins with an HTML-comment header recording the source PDF filename and the 1-based page range it covers. Technical content — Syntax / Behavior / Class / Notes / Intrinsics / Encoding / Field-name tables — is preserved verbatim. Only formatting (heading levels, cross-page table seams, running-header/-footer boilerplate) was rewritten. Known quirk: some pages' running footer says "Hexagon V73 …" (likely a Qualcomm edit-history artifact); those were stripped along with the V75 footers.
