#!/usr/bin/env python3
"""Dump the x86 HTP prepare A16 drain/control pack path for W8A16.

This is a focused reverse-engineering helper for the W8A16 per-channel native
match blocker.  It runs qnn-context-binary-generator under gdb and captures the
libHtpPrepare.so Q6-style pack cluster that produces 128-byte A16 sidecar
chunks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


GDB_PY = r'''
import gdb
from pathlib import Path

OUT = Path("__OUT_DIR__")
MAX_HITS = __MAX_HITS__
CONST_ONLY = __CONST_ONLY__
OUT.mkdir(parents=True, exist_ok=True)
BASES = {}
HITS = {}

PREPARE_CALLS = [
    (0x21aed66, "prepare_vw_stream0"),
    (0x21aeddc, "prepare_vw_stream1"),
    (0x21af10c, "prepare_vw_stream2"),
    (0x21af196, "prepare_vw_stream3"),
    (0x21af1d5, "prepare_vh_pack0"),
    (0x21af208, "prepare_vh_pack1"),
    (0x21af239, "prepare_vub_final128"),
    (0x21A441F, "prepare_a16_qf32_pack_entry"),
    (0x219F4D5, "prepare_a16_qf32_pack_helper0"),
    (0x21A0954, "prepare_a16_qf32_pack_helper1"),
    (0x219C56E, "prepare_a16_qf32_pack_helper2"),
    (0x219D833, "prepare_a16_qf32_pack_helper3"),
]

Q6_DRAIN_PACK_CALLS = [
    (0x21aed66, "q6_a16_vw_stream0"),
    (0x21aeddc, "q6_a16_vw_stream1"),
    (0x21af10c, "q6_a16_vw_stream2"),
    (0x21af196, "q6_a16_vw_stream3"),
    (0x21af1d5, "q6_a16_vh_pack0"),
    (0x21af208, "q6_a16_vh_pack1"),
    (0x21af239, "q6_a16_vub_final128"),
]

QNN_CALLS = [
    (0x66E780, "memcpy_plt"),
    # OptFunction evaluator wrappers observed in ConvLayer.opt.bias_to_vtcm
    # OP_INNER records.  These are the currently active ctxgen path; the
    # 0x21b... Q6/HVX body pointers are often nested in records but are not
    # necessarily called directly during context generation.
    (0x10B4990, "optfunc_opref_exemplar_entry"),
    (0x10B49AE, "optfunc_opref_call_eval0"),
    (0x10B4A0B, "optfunc_opref_call_eval18"),
    (0x10B4C50, "optfunc_gen_node_entry"),
    (0x10B4CB8, "optfunc_gen_node_call10"),
    (0x10B4DCC, "optfunc_multi_call00"),
    (0x10B4DD8, "optfunc_multi_call10"),
    (0x10B4DE4, "optfunc_multi_call20"),
    (0x10B4E34, "optfunc_multi_call30"),
    (0x10B4130, "graph_gen_const_i32_array_entry"),
    (0x10B4190, "graph_gen_const_i32_common_bytes_entry"),
    (0x10B421F, "graph_gen_const_i32_common_bytes_call"),
    (0x0F8B4D0, "graphprepare_gen_const_i32_common_entry"),
    (0x0F8A990, "graphprepare_internal_const_node_entry"),
    (0x0F850F0, "graphprepare_append_const_node_entry"),
    (0x0F7D620, "graphprepare_do_append_const_node_entry"),
    (0x0F7DB20, "graphprepare_remap_id_entry"),
    (0x0F7FEC0, "graphprepare_replace_opdef_with_opconst_entry"),
    (0x10A2BF0, "opdef_const_ctor_tensor_entry"),
    (0x10A2DE0, "opdef_generate_entry"),
    (0x10A7CB3, "op_factory_candidate_call"),
    (0x10A29D0, "opdef_const_ctor_bytes_entry"),
    (0x0DB1910, "opertag_lookup_entry"),
    (0x128BE60, "serializer_fwrite_entry"),
    (0x128BD80, "serializer_buf_withlen_entry"),
    (0x128C890, "serializer_rewrite_auxdata_entry"),
    (0x21A441F, "a16_qf32_pack_entry"),
    (0x219F4D5, "a16_qf32_pack_helper0"),
    (0x21A0954, "a16_qf32_pack_helper1"),
    (0x219C56E, "a16_qf32_pack_helper2"),
    (0x219D833, "a16_qf32_pack_helper3"),
    *Q6_DRAIN_PACK_CALLS,
]


def append_trace(text):
    print(text, end="" if text.endswith("\n") else "\n")
    with (OUT / "pack_trace.log").open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def short_bt(limit=10):
    frame = gdb.newest_frame()
    rows = []
    depth = 0
    while frame is not None and depth < limit:
        try:
            rows.append(f"#{depth} 0x{int(frame.pc()):x} {frame.name() or '?'}")
        except Exception as exc:
            rows.append(f"#{depth} <bt-error {exc}>")
        frame = frame.older()
        depth += 1
    return "\n".join(rows)


def read_u64(addr):
    return int(gdb.parse_and_eval("*(unsigned long*)0x%x" % addr))


def read_u32(addr):
    return int(gdb.parse_and_eval("*(unsigned int*)0x%x" % addr))


def reg_u64(name):
    return int(gdb.parse_and_eval(f"(unsigned long)${name}"))


def mapping_base(needle):
    text = gdb.execute("info proc mappings", to_string=True)
    bases = []
    for line in text.splitlines():
        if needle not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            start = int(parts[0], 16)
            offset = int(parts[3], 16)
        except Exception:
            continue
        bases.append(start - offset)
    if not bases:
        return None
    return min(bases)


class AfterCall(gdb.Breakpoint):
    def __init__(self, spec, name, hit, pc, dst, args):
        super().__init__(spec, internal=True, temporary=True)
        self.name = name
        self.hit = hit
        self.call_pc = pc
        self.dst = dst
        self.args = args

    def stop(self):
        raw = OUT / f"{self.name}_{self.hit:03d}_dst_{self.dst:x}.raw"
        rows = [
            f"AFTER {self.name} hit={self.hit} call_pc=0x{self.call_pc:x} dst=0x{self.dst:x} dump={raw.name}",
            "  args " + " ".join(f"{k}=0x{v:x}" for k, v in self.args.items()),
        ]
        try:
            gdb.execute(f"dump binary memory {raw} 0x{self.dst:x} 0x{self.dst + 128:x}")
        except Exception as exc:
            rows.append(f"  dump_error={exc}")
        append_trace("\n".join(rows) + "\n")
        return False


class AfterTensorConstCtor(gdb.Breakpoint):
    def __init__(self, spec, hit, obj, const_id, qnn_base):
        super().__init__(spec, internal=True, temporary=True)
        self.hit = hit
        self.obj = obj
        self.const_id = const_id
        self.qnn_base = qnn_base

    def call_u64(self, rva):
        addr = self.qnn_base + rva
        return int(
            gdb.parse_and_eval(
                "((unsigned long (*)(void *))0x%x)((void*)0x%x)" % (addr, self.obj)
            )
        )

    def stop(self):
        rows = [
            f"AFTER_TENSOR_CONST hit={self.hit} id=0x{self.const_id:x} obj=0x{self.obj:x}",
        ]
        rows.append("  skipped inferior function call; tensor is dumped at ctor entry")
        append_trace("\n".join(rows) + "\n")
        return False


class AfterFactoryCandidate(gdb.Breakpoint):
    def __init__(self, spec, hit, op_id, selected, fn, out_slot):
        super().__init__(spec, internal=True, temporary=True)
        self.hit = hit
        self.op_id = op_id
        self.selected = selected
        self.fn = fn
        self.out_slot = out_slot

    def stop(self):
        rows = [
            f"AFTER_FACTORY_CANDIDATE hit={self.hit} id=0x{self.op_id:x} "
            f"selected=0x{self.selected:x} fn=0x{self.fn:x} out_slot=0x{self.out_slot:x}",
        ]
        try:
            op_ptr = read_u64(self.out_slot)
            rows.append(f"  op_ptr=0x{op_ptr:x}")
            if op_ptr:
                raw = OUT / f"factory_candidate_{self.hit:03d}_id_{self.op_id:x}_op.raw"
                gdb.execute(f"dump binary memory {raw} 0x{op_ptr:x} 0x{op_ptr + 512:x}")
                rows.append(f"  op_dump={raw.name}")
                try:
                    vtable = read_u64(op_ptr)
                    rows.append(
                        f"  op_vtable=0x{vtable:x} "
                        + gdb.execute(f"info symbol 0x{vtable:x}", to_string=True).strip()
                    )
                except Exception as exc:
                    rows.append(f"  op_vtable_error={exc}")
        except Exception as exc:
            rows.append(f"  out_slot_error={exc}")
        append_trace("\n".join(rows) + "\n")
        return False


class CallBreakpoint(gdb.Breakpoint):
    def __init__(self, libname, base, rva, name):
        super().__init__("*0x%x" % (base + rva), internal=False)
        self.libname = libname
        self.rva = rva
        self.name = name

    def stop(self):
        hit = HITS.get(self.name, 0) + 1
        HITS[self.name] = hit
        if hit > MAX_HITS:
            self.enabled = False
            return False
        try:
            pc = reg_u64("pc")
        except Exception:
            pc = int(gdb.selected_frame().pc())
        try:
            insn = gdb.selected_frame().architecture().disassemble(pc, count=1)[0]
            next_pc = pc + int(insn["length"])
        except Exception:
            next_pc = pc + 5
        args = {}
        for reg in ("rdi", "rsi", "rdx", "rcx", "r8", "r9"):
            try:
                args[reg] = reg_u64(reg)
            except Exception:
                args[reg] = 0
        dst = args.get("rdi", 0)
        try:
            ret = read_u64(reg_u64("rsp"))
        except Exception:
            ret = 0
        if self.name == "opdef_const_ctor_tensor_entry":
            obj = args.get("rdi", 0)
            const_id = args.get("rdx", 0)
            unique_ptr = args.get("rcx", 0)
            tensor_ptr = 0
            try:
                tensor_ptr = read_u64(unique_ptr)
            except Exception:
                tensor_ptr = 0
            rows = [
                f"TENSOR_CONST_CTOR hit={hit} id=0x{const_id:x} obj=0x{obj:x} "
                f"unique_ptr=0x{unique_ptr:x} tensor=0x{tensor_ptr:x} ret=0x{ret:x}",
                "  args " + " ".join(f"{k}=0x{v:x}" for k, v in args.items()),
                short_bt(10),
            ]
            if tensor_ptr:
                raw = OUT / f"tensor_const_ctor_{hit:03d}_id_{const_id:x}_tensor_obj.raw"
                try:
                    gdb.execute(
                        f"dump binary memory {raw} 0x{tensor_ptr:x} 0x{tensor_ptr + 512:x}"
                    )
                    rows.append(f"  tensor_obj_dump={raw.name}")
                except Exception as exc:
                    rows.append(f"  tensor_obj_dump_error={exc}")
                try:
                    vtable = read_u64(tensor_ptr)
                    rows.append(f"  tensor_vtable=0x{vtable:x}")
                    for slot in range(0, 0x90, 8):
                        fn = read_u64(vtable + slot)
                        sym = gdb.execute(f"info symbol 0x{fn:x}", to_string=True).strip()
                        rows.append(f"    vtbl+0x{slot:02x}=0x{fn:x} {sym}")
                except Exception as exc:
                    rows.append(f"  tensor_vtable_error={exc}")
                try:
                    raw_data = read_u64(tensor_ptr + 0x18)
                    rows.append(f"  flat_raw_data=0x{raw_data:x}")
                    if raw_data:
                        for dump_len in (512, 2048):
                            raw = OUT / (
                                f"tensor_const_ctor_{hit:03d}_id_{const_id:x}_"
                                f"raw_data_{raw_data:x}_dump_{dump_len}.raw"
                            )
                            try:
                                gdb.execute(
                                    f"dump binary memory {raw} 0x{raw_data:x} 0x{raw_data + dump_len:x}"
                                )
                                rows.append(f"  raw_data_dump={raw.name}")
                            except Exception:
                                pass
                except Exception as exc:
                    rows.append(f"  tensor_raw_data_error={exc}")
            append_trace("\n".join(rows) + "\n")
            if hit >= MAX_HITS:
                self.enabled = False
            return False
        if self.name == "opdef_generate_entry":
            interesting = {
                0x104600000019,
                0x104800000019,
                0x105300000019,
                0x106600000019,
            }
            candidates = []
            for reg in ("rdi", "rsi", "rdx"):
                ptr = args.get(reg, 0)
                if not ptr:
                    continue
                try:
                    obj_id = read_u64(ptr + 0x18)
                except Exception:
                    continue
                if CONST_ONLY or obj_id in interesting:
                    candidates.append((reg, ptr, obj_id))
            if not candidates:
                return False
            rows = [
                f"OPDEF_GENERATE hit={hit} pc=0x{pc:x} ret=0x{ret:x}",
                "  args " + " ".join(f"{k}=0x{v:x}" for k, v in args.items()),
            ]
            for reg, ptr, obj_id in candidates:
                rows.append(f"  candidate {reg}=0x{ptr:x} id=0x{obj_id:x}")
                try:
                    raw = OUT / f"opdef_generate_{hit:03d}_id_{obj_id:x}_{reg}_obj.raw"
                    gdb.execute(f"dump binary memory {raw} 0x{ptr:x} 0x{ptr + 512:x}")
                    rows.append(f"    obj_dump={raw.name}")
                except Exception as exc:
                    rows.append(f"    obj_dump_error={exc}")
                try:
                    vec = read_u64(ptr + 0x28)
                    rows.append(f"    field_28=0x{vec:x}")
                    vraw = OUT / f"opdef_generate_{hit:03d}_id_{obj_id:x}_{reg}_field28.raw"
                    gdb.execute(f"dump binary memory {vraw} 0x{vec:x} 0x{vec + 256:x}")
                    rows.append(f"    field28_dump={vraw.name}")
                except Exception as exc:
                    rows.append(f"    field28_error={exc}")
            rows.append(short_bt(12))
            append_trace("\n".join(rows) + "\n")
            if hit >= MAX_HITS:
                self.enabled = False
            return False
        if self.name == "op_factory_candidate_call":
            try:
                op_id = reg_u64("r13")
            except Exception:
                op_id = 0
            if op_id not in {0x104800000019, 0x105300000019, 0x106600000019}:
                return False
            try:
                selected = reg_u64("r14")
                fn = read_u64(selected + 0x28)
                out_slot = reg_u64("rsp") + 0x10
            except Exception as exc:
                append_trace(f"FACTORY_CANDIDATE_READ_ERROR hit={hit} id=0x{op_id:x} {exc}\n")
                return False
            rows = [
                f"FACTORY_CANDIDATE hit={hit} id=0x{op_id:x} selected=0x{selected:x} "
                f"fn=0x{fn:x} {gdb.execute(f'info symbol 0x{fn:x}', to_string=True).strip()}",
                f"  out_slot=0x{out_slot:x} r12=0x{reg_u64('r12'):x}",
                short_bt(12),
            ]
            try:
                raw = OUT / f"factory_candidate_{hit:03d}_id_{op_id:x}_selected.raw"
                gdb.execute(f"dump binary memory {raw} 0x{selected:x} 0x{selected + 256:x}")
                rows.append(f"  selected_dump={raw.name}")
            except Exception as exc:
                rows.append(f"  selected_dump_error={exc}")
            append_trace("\n".join(rows) + "\n")
            if hit >= MAX_HITS:
                self.enabled = False
            return False
        if self.name in {
            "graph_gen_const_i32_common_bytes_call",
            "graphprepare_gen_const_i32_common_entry",
            "opdef_const_ctor_bytes_entry",
            "graphprepare_internal_const_node_entry",
            "graphprepare_append_const_node_entry",
            "graphprepare_do_append_const_node_entry",
        }:
            if self.name == "opdef_const_ctor_bytes_entry":
                data = args.get("r8", 0)
                length = args.get("r9", 0)
                dump_len = length * 4 if length in {128, 1024} else length
                id_text = f"id0=0x{args.get('rdx', 0):x}"
            elif self.name == "graphprepare_internal_const_node_entry":
                data = args.get("r8", 0)
                length = args.get("r9", 0)
                dump_len = length * 4 if length in {128, 1024} else length
                id_text = (
                    f"id0=0x{args.get('rsi', 0):x} "
                    f"id1=0x{args.get('rdx', 0):x}"
                )
            elif self.name in {
                "graphprepare_append_const_node_entry",
                "graphprepare_do_append_const_node_entry",
            }:
                data = args.get("rcx", 0)
                length = args.get("r8", 0)
                dump_len = length * 4 if length in {128, 1024} else length
                id_text = f"id0=0x{args.get('rsi', 0) & 0xffffffff:x}"
            else:
                data = args.get("rcx", 0)
                length = args.get("r8", 0)
                dump_len = length
                id_text = ""
            rows = [
                f"CONST_BYTES {self.name} hit={hit} lib={self.libname} rva=0x{self.rva:x} "
                f"pc=0x{pc:x} ret=0x{ret:x} {id_text} data=0x{data:x} len={length} dump_len={dump_len}",
                "  args " + " ".join(f"{k}=0x{v:x}" for k, v in args.items()),
                short_bt(10),
            ]
            if data and 0 < dump_len <= 8192:
                raw = OUT / f"{self.name}_{hit:03d}_len_{length}_dump_{dump_len}.raw"
                try:
                    gdb.execute(
                        f"dump binary memory {raw} 0x{data:x} 0x{data + dump_len:x}"
                    )
                    rows.append(f"  dump={raw.name}")
                except Exception as exc:
                    rows.append(f"  dump_error={exc}")
            append_trace("\n".join(rows) + "\n")
            if hit >= MAX_HITS:
                self.enabled = False
            return False
        if self.name in {
            "serializer_fwrite_entry",
            "serializer_buf_withlen_entry",
            "serializer_rewrite_auxdata_entry",
        }:
            if self.name == "serializer_rewrite_auxdata_entry":
                data = args.get("rcx", 0)
                count = args.get("r8", 0)
                dump_len = count * 4 if count <= 4096 else count
                size_text = f"count=0x{count:x} dump_len={dump_len}"
            else:
                data = args.get("rsi", 0)
                dump_len = args.get("rdx", 0)
                size_text = f"len={dump_len}"
            rows = [
                f"SERIALIZER {self.name} hit={hit} lib={self.libname} rva=0x{self.rva:x} "
                f"pc=0x{pc:x} ret=0x{ret:x} data=0x{data:x} {size_text}",
                "  args " + " ".join(f"{k}=0x{v:x}" for k, v in args.items()),
                short_bt(10),
            ]
            if data and 0 < dump_len <= 8192 and (
                dump_len in {128, 512, 8192} or dump_len % 512 == 0
            ):
                raw = OUT / f"{self.name}_{hit:03d}_len_{dump_len}.raw"
                try:
                    gdb.execute(
                        f"dump binary memory {raw} 0x{data:x} 0x{data + dump_len:x}"
                    )
                    rows.append(f"  dump={raw.name}")
                except Exception as exc:
                    rows.append(f"  dump_error={exc}")
            append_trace("\n".join(rows) + "\n")
            if hit >= MAX_HITS:
                self.enabled = False
            return False
        if self.name == "graphprepare_remap_id_entry":
            old_id = args.get("rsi", 0)
            new_id = args.get("rdx", 0)
            rows = [
                f"REMAP_ID hit={hit} old=0x{old_id:x} new=0x{new_id:x} "
                f"pc=0x{pc:x} ret=0x{ret:x}",
                "  args " + " ".join(f"{k}=0x{v:x}" for k, v in args.items()),
                short_bt(10),
            ]
            append_trace("\n".join(rows) + "\n")
            if hit >= MAX_HITS:
                self.enabled = False
            return False
        if self.name == "graphprepare_replace_opdef_with_opconst_entry":
            opdef = args.get("rsi", 0)
            new_uptr = args.get("rdx", 0)
            rows = [
                f"REPLACE_OPDEF_CONST hit={hit} opdef=0x{opdef:x} new_uptr=0x{new_uptr:x} "
                f"pc=0x{pc:x} ret=0x{ret:x}",
                "  args " + " ".join(f"{k}=0x{v:x}" for k, v in args.items()),
                short_bt(10),
            ]
            try:
                old_id = read_u64(opdef + 0x18)
                vtable = read_u64(opdef)
                rows.append(f"  old_id=0x{old_id:x} opdef_vtable=0x{vtable:x}")
                try:
                    typeinfo = read_u64(vtable - 8)
                    rows.append(
                        f"  opdef_typeinfo=0x{typeinfo:x} "
                        + gdb.execute(f"info symbol 0x{typeinfo:x}", to_string=True).strip()
                    )
                    name_ptr = read_u64(typeinfo + 8)
                    rows.append(
                        "  opdef_typeinfo_name="
                        + gdb.execute(f"x/s 0x{name_ptr:x}", to_string=True).strip()
                    )
                except Exception as exc:
                    rows.append(f"  opdef_typeinfo_error={exc}")
                for slot in range(0, 0x80, 8):
                    fn = read_u64(vtable + slot)
                    sym = gdb.execute(f"info symbol 0x{fn:x}", to_string=True).strip()
                    rows.append(f"    opdef_vtbl+0x{slot:02x}=0x{fn:x} {sym}")
                raw = OUT / f"replace_opdef_{hit:03d}_old_{old_id:x}_obj.raw"
                try:
                    gdb.execute(f"dump binary memory {raw} 0x{opdef:x} 0x{opdef + 512:x}")
                    rows.append(f"  opdef_obj_dump={raw.name}")
                except Exception as exc:
                    rows.append(f"  opdef_obj_dump_error={exc}")
                if old_id in {0x105300000019, 0x11AE00000019, 0x11B000000019}:
                    for off in range(0, 0x200, 8):
                        try:
                            ptr = read_u64(opdef + off)
                        except Exception:
                            continue
                        if not (0x10000 <= ptr < 0x800000000000):
                            continue
                        rows.append(f"  field+0x{off:x}=0x{ptr:x}")
                        try:
                            pvtable = read_u64(ptr)
                            psym = gdb.execute(f"info symbol 0x{pvtable:x}", to_string=True).strip()
                            rows.append(f"    pointee_vtable=0x{pvtable:x} {psym}")
                            ptypeinfo = read_u64(pvtable - 8)
                            rows.append(
                                f"    pointee_typeinfo=0x{ptypeinfo:x} "
                                + gdb.execute(
                                    f"info symbol 0x{ptypeinfo:x}", to_string=True
                                ).strip()
                            )
                            pname = read_u64(ptypeinfo + 8)
                            rows.append(
                                "    pointee_typeinfo_name="
                                + gdb.execute(f"x/s 0x{pname:x}", to_string=True).strip()
                            )
                        except Exception as exc:
                            rows.append(f"    pointee_vtable_error={exc}")
                        try:
                            praw = OUT / (
                                f"replace_opdef_{hit:03d}_old_{old_id:x}_"
                                f"field_{off:03x}_ptr_{ptr:x}.raw"
                            )
                            gdb.execute(f"dump binary memory {praw} 0x{ptr:x} 0x{ptr + 256:x}")
                            rows.append(f"    pointee_dump={praw.name}")
                        except Exception as exc:
                            rows.append(f"    pointee_dump_error={exc}")
            except Exception as exc:
                rows.append(f"  opdef_read_error={exc}")
            append_trace("\n".join(rows) + "\n")
            if hit >= MAX_HITS:
                self.enabled = False
            return False
        if self.name == "memcpy_plt":
            length = args.get("rdx", 0)
            if length not in {512, 2048, 8192}:
                return False
            rows = [
                f"MEMCPY hit={hit} len={length} dst=0x{args.get('rdi',0):x} src=0x{args.get('rsi',0):x} "
                f"ret=0x{ret:x}",
                "  args " + " ".join(f"{k}=0x{v:x}" for k, v in args.items()),
                short_bt(10),
            ]
            for label, addr in [("src", args.get("rsi", 0)), ("dst", args.get("rdi", 0))]:
                if addr:
                    raw = OUT / f"memcpy_{label}_{hit:03d}_len_{length}.raw"
                    try:
                        gdb.execute(
                            f"dump binary memory {raw} 0x{addr:x} 0x{addr + min(length, 8192):x}"
                        )
                        rows.append(f"  {label}_dump={raw.name}")
                    except Exception as exc:
                        rows.append(f"  {label}_dump_error={exc}")
            append_trace("\n".join(rows) + "\n")
            if hit >= MAX_HITS:
                self.enabled = False
            return False
        append_trace(
            f"CALL {self.name} hit={hit} lib={self.libname} rva=0x{self.rva:x} pc=0x{pc:x} next=0x{next_pc:x} ret=0x{ret:x} dst=0x{dst:x}\n"
            + "  args "
            + " ".join(f"{k}=0x{v:x}" for k, v in args.items())
            + "\n"
            + short_bt(8)
            + "\n"
        )
        if dst:
            AfterCall("*0x%x" % next_pc, self.name, hit, pc, dst, args)
        return False


def find_dequant_symbol_addr():
    for expr in [
        "(unsigned long)&_Z15dequantize_biasR11ReplacementRK5OpRefS3_S3_",
        "(unsigned long)&'dequantize_bias(Replacement&, OpRef const&, OpRef const&, OpRef const&)'",
    ]:
        try:
            addr = int(gdb.parse_and_eval(expr))
            if addr:
                return addr
        except Exception:
            pass
    return None


def install_breakpoints_for(libname, calls, source, forced_base=None):
    if libname in BASES:
        return
    base = forced_base
    if base is None:
        base = mapping_base(libname)
    if base is None:
        return
    BASES[libname] = base
    append_trace(f"{libname} base 0x{base:x} via {source}\n")
    for rva, name in calls:
        CallBreakpoint(libname, base, rva, name)


def install_breakpoints(source):
    if not CONST_ONLY:
        install_breakpoints_for("libHtpPrepare.so", PREPARE_CALLS, source)
        install_breakpoints_for("libQnnHtp.so", QNN_CALLS, source)
        return
    install_breakpoints_for(
        "libQnnHtp.so",
        [
            (0x10A2BF0, "opdef_const_ctor_tensor_entry"),
            (0x0F7FEC0, "graphprepare_replace_opdef_with_opconst_entry"),
        ],
        source,
    )


def on_new_objfile(event):
    filename = getattr(event.new_objfile, "filename", "") or ""
    if filename.endswith("libHtpPrepare.so"):
        if not CONST_ONLY:
            install_breakpoints_for("libHtpPrepare.so", PREPARE_CALLS, "new_objfile")
    elif filename.endswith("libQnnHtp.so"):
        if CONST_ONLY:
            install_breakpoints_for(
                "libQnnHtp.so",
                [
                    (0x10A2BF0, "opdef_const_ctor_tensor_entry"),
                    (0x0F7FEC0, "graphprepare_replace_opdef_with_opconst_entry"),
                ],
                "new_objfile",
            )
        else:
            install_breakpoints_for("libQnnHtp.so", QNN_CALLS, "new_objfile")


class MainEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("main", internal=False)

    def stop(self):
        install_breakpoints("main")
        return False


class DequantEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("_Z15dequantize_biasR11ReplacementRK5OpRefS3_S3_", internal=False)

    def stop(self):
        try:
            pc = int(gdb.selected_frame().pc())
            install_breakpoints_for(
                "libQnnHtp.so",
                [
                    (0x10A2BF0, "opdef_const_ctor_tensor_entry"),
                    (0x0F7FEC0, "graphprepare_replace_opdef_with_opconst_entry"),
                ]
                if CONST_ONLY
                else QNN_CALLS,
                "dequantize_bias",
                pc - 0x1BA81B2,
            )
        except Exception as exc:
            append_trace(f"BASE_FROM_DEQUANT_ERROR {exc}\n")
        return False


gdb.events.new_objfile.connect(on_new_objfile)
MainEntry()
DequantEntry()
'''


def compare_dumps(out_dir: Path, sidecar_raw: Path | None) -> None:
    if sidecar_raw is None or not sidecar_raw.exists():
        return
    sidecar = sidecar_raw.read_bytes()
    chunks = [sidecar[i : i + 128] for i in range(0, len(sidecar), 128)]
    rows = []
    for raw in sorted(out_dir.glob("*.raw")):
        data = raw.read_bytes()
        best = (-1, -1)
        for idx, chunk in enumerate(chunks):
            score = sum(a == b for a, b in zip(data, chunk))
            if score > best[0]:
                best = (score, idx)
        full_score = sum(a == b for a, b in zip(data, sidecar))
        best512 = (-1, -1)
        if len(data) >= len(sidecar):
            for off in range(0, len(data) - len(sidecar) + 1, 16):
                score = sum(a == b for a, b in zip(data[off : off + len(sidecar)], sidecar))
                if score > best512[0]:
                    best512 = (score, off)
        rows.append(
            f"{raw.name}: best_chunk={best[1]} byte_match={best[0]}/128 "
            f"full512={full_score}/{len(sidecar)} best512_off={best512[1]} "
            f"best512={best512[0]}/{len(sidecar)}"
        )
    if rows:
        (out_dir / "native_sidecar_chunk_match.txt").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--native-sidecar-raw", type=Path)
    parser.add_argument("--max-hits", type=int, default=8)
    parser.add_argument(
        "--const-only",
        action="store_true",
        help="only trace tensor const construction and OpDef::generate",
    )
    parser.add_argument(
        "--qnn-sdk-root",
        type=Path,
        default=Path(os.environ.get("QNN_SDK_ROOT", "tools/qnn-sdk")),
    )
    args = parser.parse_args()

    native_dir = args.native_dir.resolve()
    out_dir = args.out_dir.resolve()
    qnn = args.qnn_sdk_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_dir = out_dir / "ctx"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(
            GDB_PY.replace("__OUT_DIR__", str(out_dir)).replace(
                "__MAX_HITS__", str(args.max_hits)
            ).replace("__CONST_ONLY__", "True" if args.const_only else "False")
        )
        gdb_py = Path(f.name)

    env = os.environ.copy()
    lib_dir = qnn / "lib/x86_64-linux-clang"
    env["LD_LIBRARY_PATH"] = str(lib_dir) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )

    cmd = [
        "gdb",
        "-q",
        "--batch",
        "-ex",
        "set pagination off",
        "-ex",
        "set breakpoint pending on",
        "-ex",
        f"source {gdb_py}",
        "-ex",
        "run",
        "--args",
        str(qnn / "bin/x86_64-linux-clang/qnn-context-binary-generator"),
        "--backend",
        str(qnn / "lib/x86_64-linux-clang/libQnnHtp.so"),
        "--dlc_path",
        str(native_dir / "case.dlc"),
        "--binary_file",
        "case_native_ctx_gdb",
        "--output_dir",
        str(ctx_dir),
        "--config_file",
        str(native_dir / "htp_config.json"),
        "--profiling_level",
        "detailed",
        "--profiling_option",
        "optrace",
        "--save_backend_op_mapping",
    ]
    log_path = out_dir / "gdb.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    compare_dumps(out_dir, args.native_sidecar_raw)
    print(f"wrote {log_path}")
    if (out_dir / "pack_trace.log").exists():
        print(f"wrote {out_dir / 'pack_trace.log'}")
    if (out_dir / "native_sidecar_chunk_match.txt").exists():
        print(f"wrote {out_dir / 'native_sidecar_chunk_match.txt'}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
