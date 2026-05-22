#!/usr/bin/env python3
"""Run ctxgen under gdb and dump u8i8 HTP bias-prepare internals.

This is a focused reverse-engineering helper for the x86 ctxgen path.  It uses
symbols exported by libQnnHtp.so to dump:

- dequantize_bias output float buffer just before GraphPrepare emits the const;
- find_bias_scale output register just before GraphPrepare emits the scale const.
- targeted graph-prepare builder callsites for convert_bias / adjust_bias /
  bias_scale_shuff, so the active lowering chain can be checked by inputs.
- candidate runtime conversion bodies, when they actually execute during ctxgen.
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
TARGET_SIDEcar_BYTES = __SIDECAR_BYTES__
OUT.mkdir(parents=True, exist_ok=True)
LIB_BASE = None
BIAS_CONVERT_ACTIVE = False
BIAS_CONVERT_ROWS = []
BIAS_CONVERT_BODY_COUNT = 0
BIAS_CONVERT_CAPTURED_BODY = None
INT32_CONST_INDEX = 0
INT32_COMMON_INDEX = 0
FLOAT_COMMON_INDEX = 0
FLOAT_ARRAY_INDEX = 0
FLOAT_REPEAT_INDEX = 0
OPDEF_CONST_SIDEcar_INDEX = 0
CONST_DATA_SIDEcar_INDEX = 0
CONST_DATA_SIDEcar_THIS = set()
SERIALIZE_SIDEcar_INDEX = 0
PROBE_HIT_COUNTS = {}
TRACE_BREAKS_INSTALLED = False
PREPARE_BREAKS_INSTALLED = False
WRAPPER_TRACE_HITS = 0
WITH_TRACE_HITS = 0
OP_INNER_TRACE_HITS = 0
CALLSITE_TRACE_HITS = {}
SCALE_NORMALIZE_TRACE_HITS = 0
REQUANT_BIAS_ACTIVE = False
REQUANT_BIAS_CAPTURED = False
REQUANT_BIAS_ROWS = {}

WITH_OUTPUT_LIKE_RET_SITES = {
    0x1BA8F4B: "rule_a.with_output_like_1",
    0x1BA8F66: "rule_a.with_output_like_2",
    0x1BA93A8: "rule_b.with_output_like_1",
    0x1BA93C9: "rule_b.with_output_like_2",
    0x1BA94DE: "rule_b.with_output_like_3",
    0x1BA94FC: "rule_b.with_output_like_4",
    0x1BA9601: "rule_b.with_output_like_5",
}

OP_INNER_RET_SITES = {
    0x1BA963C: "rule_b.op_inner",
}


class OneShotDump(gdb.Breakpoint):
    def __init__(self, spec, name, cb):
        super().__init__(spec, internal=False, temporary=True)
        self.name = name
        self.cb = cb

    def stop(self):
        try:
            self.cb()
        except Exception as exc:
            print(f"GDB_DUMP_ERROR {self.name}: {exc}")
        return False


class TempReturn(gdb.Breakpoint):
    def __init__(self, ret_addr, name, cb):
        super().__init__("*0x%x" % ret_addr, internal=True, temporary=True)
        self.name = name
        self.cb = cb

    def stop(self):
        try:
            self.cb()
        except Exception as exc:
            print(f"GDB_RETURN_ERROR {self.name}: {exc}")
        return False


class Entry(gdb.Breakpoint):
    def __init__(self, spec, rel, name, cb):
        super().__init__(spec, internal=False)
        self.rel = rel
        self.name = name
        self.cb = cb
        self.done = False

    def stop(self):
        try:
            if not self.done:
                self.done = True
                pc = int(gdb.selected_frame().pc())
                spec = "*0x%x" % (pc + self.rel)
                OneShotDump(spec, self.name, self.cb)
                print(f"armed {self.name} at {spec}")
        except Exception as exc:
            print(f"GDB_ENTRY_ERROR {self.name}: {exc}")
        return False


class RelBreakpoint(gdb.Breakpoint):
    def __init__(self, rva, name, cb, temporary=False):
        super().__init__("*0x%x" % (LIB_BASE + rva), internal=False, temporary=temporary)
        self.name = name
        self.cb = cb

    def stop(self):
        try:
            self.cb()
        except Exception as exc:
            print(f"GDB_DUMP_ERROR {self.name}: {exc}")
        return False


def u64(addr):
    return int(gdb.parse_and_eval("*(unsigned long*)0x%x" % addr))


def read_cstr(addr, limit=160):
    if addr == 0:
        return "<null>"
    try:
        data = bytes(gdb.selected_inferior().read_memory(addr, limit))
        return data.split(b"\x00", 1)[0].decode("utf-8", "replace")
    except Exception as exc:
        return f"<cstr-error 0x{addr:x}: {exc}>"


def symbol_name(addr):
    try:
        text = gdb.execute("info symbol 0x%x" % addr, to_string=True).strip()
        return text if text else "<no-symbol>"
    except Exception as exc:
        return f"<symbol-error {exc}>"


def append_trace(name, text):
    print(text, end="" if text.endswith("\n") else "\n")
    with (OUT / name).open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def maybe_ptr_string(addr):
    if addr == 0:
        return ""
    s = read_cstr(addr, 96)
    if s.startswith("<") or not s:
        return ""
    printable = sum(1 for ch in s if 32 <= ord(ch) < 127)
    if printable < max(1, len(s) * 3 // 4):
        return ""
    return s


def dump_qwords(addr, count=6):
    vals = []
    for i in range(count):
        try:
            v = u64(addr + 8 * i)
            s = maybe_ptr_string(v)
            vals.append(f"+0x{8*i:x}=0x{v:x}" + (f" '{s}'" if s else ""))
        except Exception as exc:
            vals.append(f"+0x{8*i:x}=<err {exc}>")
    return " ".join(vals)


def dump_callsite(name, max_hits=8):
    count = CALLSITE_TRACE_HITS.get(name, 0) + 1
    CALLSITE_TRACE_HITS[name] = count
    if count > max_hits:
        return
    regs = {}
    for reg in ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rsp"]:
        try:
            regs[reg] = int(gdb.parse_and_eval(f"(unsigned long)${reg}"))
        except Exception:
            regs[reg] = 0
    rows = [f"CALLSITE {name} hit={count}"]
    for reg in ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]:
        val = regs[reg]
        s = maybe_ptr_string(val)
        rows.append(f"  {reg}=0x{val:x}" + (f" '{s}'" if s else ""))
        if val and not s:
            rows.append(f"  {reg}_qwords {dump_qwords(val, 8)}")
    rows.append(f"  rsp_qwords {dump_qwords(regs['rsp'], 12)}")
    rows.append(short_bt(14))
    append_trace("target_callsites_trace.log", "\n".join(rows) + "\n")


def dump_scale_normalize_point(stage, max_hits=24):
    global SCALE_NORMALIZE_TRACE_HITS
    if stage == "entry":
        SCALE_NORMALIZE_TRACE_HITS += 1
    if SCALE_NORMALIZE_TRACE_HITS > max_hits:
        return
    rows = [f"SCALE_NORMALIZE {stage} hit={SCALE_NORMALIZE_TRACE_HITS}"]
    if stage == "entry":
        try:
            rdi = int(gdb.parse_and_eval("(unsigned long)$rdi"))
            factor = f32_at(rdi)
            rows.append(f"  rdi=0x{rdi:x} factor_f32={factor:.12g}")
            rows.append(f"  rdi_qwords {dump_qwords(rdi, 8)}")
        except Exception as exc:
            rows.append(f"  entry_error={exc}")
    elif stage == "pre_convert":
        try:
            rows.append(f"  xmm1={gdb.parse_and_eval('$xmm1')}")
            rows.append(f"  factor_stack_f32={f32_at(int(gdb.parse_and_eval('(unsigned long)$rsp')) + 0xc):.12g}")
        except Exception as exc:
            rows.append(f"  pre_convert_error={exc}")
    elif stage == "exit":
        try:
            rows.append(f"  eax={int(gdb.parse_and_eval('(int)$eax'))}")
            rows.append(f"  rax={int(gdb.parse_and_eval('(long)$rax'))}")
        except Exception as exc:
            rows.append(f"  exit_error={exc}")
    rows.append(short_bt(10))
    append_trace("scale_normalize_trace.log", "\n".join(rows) + "\n")


def arm_requant_bias_capture():
    """Capture the internal requant_bias scalar loop recovered at 0x1ba8116."""
    global REQUANT_BIAS_ACTIVE, REQUANT_BIAS_CAPTURED, REQUANT_BIAS_ROWS
    if REQUANT_BIAS_CAPTURED:
        return
    try:
        opdef = int(gdb.parse_and_eval("(unsigned long)*(unsigned long*)($rsi + 0x10)"))
        n = int(gdb.parse_and_eval("(unsigned long)*(unsigned long*)(0x%x + 0x20)" % opdef))
    except Exception as exc:
        print(f"requant_bias_entry_arg_error: {exc}")
        return
    if n not in (32, 160, 256):
        return
    REQUANT_BIAS_ACTIVE = True
    REQUANT_BIAS_CAPTURED = True
    REQUANT_BIAS_ROWS = {}
    append_trace("requant_bias_numeric_trace.log", f"REQUANT_BIAS_ENTRY n={n}\n{short_bt(16)}\n")


def dump_requant_bias_pre_nearby():
    if not REQUANT_BIAS_ACTIVE:
        return
    try:
        idx = int(gdb.parse_and_eval("(unsigned long)$rbx"))
        rdx = int(gdb.parse_and_eval("(unsigned long)$rdx"))
        rsp = int(gdb.parse_and_eval("(unsigned long)$rsp"))
        in_f32 = f32_at(rsp + 0x0C)
        enc_offset_i32 = int(gdb.parse_and_eval("*(int*)0x%x" % (rdx + 0x04)))
        enc_mul_f32 = f32_at(rdx + 0x0C)
        pre_nearby_f32 = float(gdb.parse_and_eval("$xmm1.v4_float[0]"))
    except Exception as exc:
        append_trace("requant_bias_numeric_trace.log", f"REQUANT_BIAS_PRE_ERROR {exc}\n")
        return
    REQUANT_BIAS_ROWS[idx] = {
        "index": idx,
        "input_f32": in_f32,
        "encoding_offset_i32": enc_offset_i32,
        "encoding_mul_f32": enc_mul_f32,
        "pre_nearby_f32": pre_nearby_f32,
    }


def dump_requant_bias_write():
    if not REQUANT_BIAS_ACTIVE:
        return
    try:
        idx = int(gdb.parse_and_eval("(unsigned long)$rbx"))
        out_i32 = int(gdb.parse_and_eval("(int)$eax"))
    except Exception as exc:
        append_trace("requant_bias_numeric_trace.log", f"REQUANT_BIAS_WRITE_ERROR {exc}\n")
        return
    row = REQUANT_BIAS_ROWS.setdefault(idx, {"index": idx})
    row["out_i32"] = out_i32


def dump_requant_bias_exit():
    global REQUANT_BIAS_ACTIVE
    if not REQUANT_BIAS_ACTIVE:
        return
    REQUANT_BIAS_ACTIVE = False
    path = OUT / "requant_bias_numeric.csv"
    with path.open("w", encoding="utf-8") as f:
        f.write("index,input_f32,encoding_offset_i32,encoding_mul_f32,pre_nearby_f32,out_i32\n")
        for idx in sorted(REQUANT_BIAS_ROWS):
            row = REQUANT_BIAS_ROWS[idx]
            f.write(
                f"{idx},{row.get('input_f32', float('nan')):.12g},"
                f"{row.get('encoding_offset_i32', '')},"
                f"{row.get('encoding_mul_f32', float('nan')):.12g},"
                f"{row.get('pre_nearby_f32', float('nan')):.12g},"
                f"{row.get('out_i32', '')}\n"
            )
    append_trace(
        "requant_bias_numeric_trace.log",
        f"REQUANT_BIAS_EXIT rows={len(REQUANT_BIAS_ROWS)} path={path}\n",
    )


def dump_wrapper_descriptor(kind, operand_count):
    global WRAPPER_TRACE_HITS
    desc = int(gdb.parse_and_eval("(unsigned long)$rdi"))
    repl = int(gdb.parse_and_eval("(unsigned long)$rsi"))
    outdef = int(gdb.parse_and_eval("(unsigned long)$rdx"))
    try:
        ret = u64(int(gdb.parse_and_eval("(unsigned long)$rsp")))
    except Exception:
        ret = 0
    fn = u64(desc)
    fn_sym = symbol_name(fn)
    # Keep this focused on the two functions under investigation.
    if "dequantize_bias" not in fn_sym and "find_bias_scale" not in fn_sym:
        return
    WRAPPER_TRACE_HITS += 1
    rows = [
        f"WRAPPER {kind} hit={WRAPPER_TRACE_HITS} ret_rva=0x{ret - LIB_BASE:x} "
        f"desc=0x{desc:x} repl=0x{repl:x} outdef=0x{outdef:x}",
        f"  fn=0x{fn:x} {fn_sym}",
    ]
    for i in range(operand_count):
        ptr = u64(desc + 8 + 8 * i)
        rows.append(f"  tag{i}=0x{ptr:x} '{read_cstr(ptr)}'")
    rows.append(short_bt(12))
    append_trace("replacement_wrapper_trace.log", "\n".join(rows) + "\n")


def dump_with_output_like():
    global WITH_TRACE_HITS
    try:
        ret = u64(int(gdb.parse_and_eval("(unsigned long)$rsp")))
    except Exception:
        return
    ret_rva = ret - LIB_BASE
    site = WITH_OUTPUT_LIKE_RET_SITES.get(ret_rva)
    if not site:
        return
    WITH_TRACE_HITS += 1
    repl_func_or_operand = int(gdb.parse_and_eval("(unsigned long)$rdi"))
    optfunc = int(gdb.parse_and_eval("(unsigned long)$rsi"))
    out_idx = int(gdb.parse_and_eval("(int)$edx"))
    rows = [
        f"WITH_OUTPUT_LIKE hit={WITH_TRACE_HITS} site={site} ret_rva=0x{ret_rva:x} "
        f"out_idx={out_idx} arg0=0x{repl_func_or_operand:x} optfunc=0x{optfunc:x}",
        f"  arg0_qwords {dump_qwords(repl_func_or_operand, 6)}",
        f"  optfunc_qwords {dump_qwords(optfunc, 6)}",
        short_bt(10),
    ]
    append_trace("replacement_builder_trace.log", "\n".join(rows) + "\n")


def dump_op_inner():
    global OP_INNER_TRACE_HITS
    try:
        ret = u64(int(gdb.parse_and_eval("(unsigned long)$rsp")))
    except Exception:
        return
    ret_rva = ret - LIB_BASE
    site = OP_INNER_RET_SITES.get(ret_rva)
    op_pkg = int(gdb.parse_and_eval("(unsigned long)$rdi"))
    op_name = int(gdb.parse_and_eval("(unsigned long)$rsi"))
    pkg_s = read_cstr(op_pkg)
    name_s = read_cstr(op_name)
    interesting = site is not None or any(
        term in (pkg_s + " " + name_s)
        for term in [
            "bias_to_vtcm",
            "bias_scale_shuff",
            "convert_bias",
            "requant_bias",
            "adjust_bias",
        ]
    )
    if not interesting:
        return
    OP_INNER_TRACE_HITS += 1
    n_inputs = int(gdb.parse_and_eval("(int)$edx"))
    optfuncs = int(gdb.parse_and_eval("(unsigned long)$rcx"))
    rows = [
        f"OP_INNER hit={OP_INNER_TRACE_HITS} site={site or '<unmapped>'} ret_rva=0x{ret_rva:x} "
        f"pkg='{pkg_s}' name='{name_s}' n_inputs={n_inputs} "
        f"optfuncs=0x{optfuncs:x}",
        f"  optfuncs_qwords {dump_qwords(optfuncs, 10)}",
    ]
    for i in range(max(0, min(n_inputs, 8))):
        try:
            fn = u64(optfuncs + 16 * i)
            rec = u64(optfuncs + 16 * i + 8)
            rows.append(f"  input{i}_fn=0x{fn:x} {symbol_name(fn)} rec=0x{rec:x}")
            rows.append(f"  input{i}_rec_qwords {dump_qwords(rec, 12)}")
        except Exception as exc:
            rows.append(f"  input{i}_decode_error {exc}")
    rows.append(short_bt(10))
    append_trace("replacement_builder_trace.log", "\n".join(rows) + "\n")


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
    try:
        text = gdb.execute(
            "info address _Z15dequantize_biasR11ReplacementRK5OpRefS3_S3_",
            to_string=True,
        )
        import re

        m = re.search(r"0x[0-9a-fA-F]+", text)
        if m:
            return int(m.group(0), 16)
    except Exception:
        pass
    return None


def install_prepare_breakpoints():
    global PREPARE_BREAKS_INSTALLED
    if PREPARE_BREAKS_INSTALLED or LIB_BASE is None:
        return
    PREPARE_BREAKS_INSTALLED = True
    # Candidate runtime conversion bodies.
    RelBreakpoint(0x1509906, "bias_convert_entry_a", lambda: bias_convert_entry(0x1509906))
    RelBreakpoint(0x1509B6C, "bias_convert_write_main_a", lambda: bias_convert_write_main(0x1509906))
    RelBreakpoint(0x1509BC2, "bias_convert_write_tail_a", lambda: bias_convert_write_tail(0x1509906))
    RelBreakpoint(0x1509BCD, "bias_convert_exit_a", lambda: bias_convert_exit(0x1509906))
    RelBreakpoint(0x1509DAD, "bias_convert_entry_b", lambda: bias_convert_entry(0x1509DAD))
    RelBreakpoint(0x150A013, "bias_convert_write_main_b", lambda: bias_convert_write_main(0x1509DAD))
    RelBreakpoint(0x150A069, "bias_convert_write_tail_b", lambda: bias_convert_write_tail(0x1509DAD))
    RelBreakpoint(0x150A074, "bias_convert_exit_b", lambda: bias_convert_exit(0x1509DAD))
    RelBreakpoint(0x0F8BB50, "gen_const_i32_array", dump_int32_const_array)
    RelBreakpoint(
        0x0F8B4D0,
        "graphprepare_gen_const_i32_common",
        lambda: dump_int32_common_const("GraphPrepare"),
    )
    RelBreakpoint(
        0x10AF3C0,
        "replacement_gen_const_i32_common",
        lambda: dump_int32_common_const("Replacement"),
    )
    RelBreakpoint(
        0x0F8B8D0,
        "graphprepare_gen_const_float_common",
        lambda: dump_float_common_const("GraphPrepare"),
    )
    RelBreakpoint(
        0x10AF3D0,
        "replacement_gen_const_float_common",
        lambda: dump_float_common_const("Replacement"),
    )
    RelBreakpoint(
        0x0F8BBB0,
        "graphprepare_gen_const_f32_array",
        lambda: dump_float_array_const("GraphPrepare"),
    )
    RelBreakpoint(
        0x10AF3A0,
        "replacement_gen_const_f32_array",
        lambda: dump_float_array_const("Replacement"),
    )
    RelBreakpoint(
        0x0F8BC50,
        "graphprepare_gen_const_f32_repeat",
        lambda: dump_float_repeat_const("GraphPrepare"),
    )
    RelBreakpoint(
        0x10A29D0,
        "opdef_const_from_bytes",
        dump_opdef_const_from_bytes,
    )
    RelBreakpoint(0x10A2EE0, "opdef_const_data_len", arm_const_data_len_return)
    RelBreakpoint(0x10A2EC0, "opdef_const_data_ptr", arm_const_data_ptr_return)
    RelBreakpoint(
        0x128BE60,
        "serializer_fwrite",
        lambda: dump_serializer_sidecar("serialize_fwrite", "rsi", "rdx"),
    )
    RelBreakpoint(
        0x128BD80,
        "serializer_buf_withlen",
        lambda: dump_serializer_sidecar("serialize_buf_withlen", "rsi", "rdx"),
    )
    RelBreakpoint(
        0x128C890,
        "serializer_rewrite_auxdata",
        lambda: dump_serializer_sidecar("rewrite_auxdata", "rcx", "r8"),
    )
    for name, rva in [
        ("convert_bias_simple_builder", 0x1B8133C),
        ("convert_bias_opt_builder", 0x1B813B5),
        ("adjust_bias_builder_a", 0x1B80AD3),
        ("v73_convert_bias_builder_a", 0x1B80D73),
        ("adjust_bias_builder_b", 0x1B80E6D),
        ("v73_convert_bias_builder_b", 0x1B80F65),
        ("bias_scale_shuff_builder_a", 0x1B87D78),
        ("bias_scale_shuff_builder_b", 0x1B8851D),
        ("scale_normalizing_conv2d_a", 0x1B642DA),
        ("requant_bias_conv2d_a", 0x1B64311),
        ("scale_normalizing_conv2d_b", 0x1B643DF),
        ("requant_bias_conv2d_b", 0x1B6445F),
        ("scale_normalizing_conv_a", 0x1BAE788),
        ("requant_bias_conv_a", 0x1BAE7BF),
        ("scale_normalizing_conv_b", 0x1BAE88D),
        ("requant_bias_conv_b", 0x1BAE90D),
    ]:
        RelBreakpoint(rva, f"callsite_{name}", lambda name=name: dump_callsite(name))
    RelBreakpoint(0x0DF81C0, "scale_normalize_entry", lambda: dump_scale_normalize_point("entry"))
    RelBreakpoint(0x0DF8228, "scale_normalize_pre_convert", lambda: dump_scale_normalize_point("pre_convert"))
    RelBreakpoint(0x0DF8245, "scale_normalize_exit", lambda: dump_scale_normalize_point("exit"))
    RelBreakpoint(0x1BA8116, "requant_bias_entry", arm_requant_bias_capture)
    RelBreakpoint(0x1BA817A, "requant_bias_pre_nearby", dump_requant_bias_pre_nearby)
    RelBreakpoint(0x1BA8196, "requant_bias_write", dump_requant_bias_write)
    RelBreakpoint(0x1BA81A2, "requant_bias_exit", dump_requant_bias_exit)
    # Candidate bodies recovered from op-name/typeinfo tables around
    # convert_bias, bias_scale_shuff, and bias_to_vtcm.  These are hit-probes
    # first; dedicated dumps should be added only after the active ctxgen path is
    # proven.
    for name, rva in [
        ("sumweight_wrapper", 0x14DD7C8),
        ("sumweight_exec", 0x14DD90E),
        ("bias_pack_a", 0x14DDA6E),
        ("bias_pack_const", 0x14DDFC3),
        ("bias_pack_b", 0x14DE000),
        ("weight_shuffle_a", 0x14E0C51),
        ("weight_shuffle_b", 0x14E0D19),
        ("weight_shuffle_c", 0x14E107F),
        ("bias_convert_v73_i32_guard", 0x150A11C),
        ("bias_convert_v73_call", 0x150A1BC),
        ("bias_scale_shuff_fi_fi_fs_impl_a0", 0x3F5FC70),
        ("bias_scale_shuff_fi_fi_fs_impl_a1", 0x3F76330),
        ("bias_scale_shuff_fi_fi_fs_impl_a2", 0x3F76350),
        ("bias_scale_shuff_fi_fi_fs_impl_a3", 0x3F76370),
        ("bias_scale_shuff_fi_fi_fs_impl_b0", 0x40A3DF0),
        ("bias_scale_shuff_fi_fi_fs_impl_b1", 0x40DB420),
        ("bias_scale_shuff_fi_fi_fs_impl_b2", 0x40DB460),
        ("bias_scale_shuff_fi_fi_fs_impl_b3", 0x40DB4A0),
        ("bias_scale_shuff_fi_fi_fs_impl_c0", 0x42B5F30),
        ("bias_scale_shuff_fi_fi_fs_impl_c1", 0x42DFD30),
        ("bias_scale_shuff_fi_fi_fs_impl_c2", 0x42DFD70),
        ("bias_scale_shuff_fi_fi_fs_impl_c3", 0x42DFDB0),
        ("bias_scale_shuff_fi_fi_fs_impl_d0", 0x44C9B00),
        ("bias_scale_shuff_fi_fi_fs_impl_d1", 0x450A700),
        ("bias_scale_shuff_fi_fi_fs_impl_d2", 0x450A720),
        ("bias_scale_shuff_fi_fi_fs_impl_d3", 0x450A740),
        ("bias_scale_shuff_fi_fi_fs_impl_e0", 0x46A81E0),
        ("bias_scale_shuff_fi_fi_fs_impl_e1", 0x46DAF10),
        ("bias_scale_shuff_fi_fi_fs_impl_e2", 0x46DAF40),
        ("bias_scale_shuff_fi_fi_fs_impl_e3", 0x46DAF70),
    ]:
        RelBreakpoint(rva, f"probe_{name}", lambda name=name, rva=rva: log_probe_hit(name, rva))


def install_trace_breakpoints(source):
    global LIB_BASE, TRACE_BREAKS_INSTALLED
    if TRACE_BREAKS_INSTALLED:
        install_prepare_breakpoints()
        return
    addr = find_dequant_symbol_addr()
    if addr is None:
        return
    LIB_BASE = addr - 0x1BA81B2
    TRACE_BREAKS_INSTALLED = True
    print(f"libQnnHtp base 0x{LIB_BASE:x} via {source}")
    RelBreakpoint(0x1AF9D64, "wrapper_three_oprefs", lambda: dump_wrapper_descriptor("three_oprefs", 3))
    RelBreakpoint(0x1B5650C, "wrapper_one_opref", lambda: dump_wrapper_descriptor("one_opref", 1))
    RelBreakpoint(0x66BB60, "with_output_like_plt", dump_with_output_like)
    RelBreakpoint(0x66FAD0, "op_inner_plt", dump_op_inner)
    install_prepare_breakpoints()


def on_new_objfile(event):
    filename = getattr(event.new_objfile, "filename", "") or ""
    if filename.endswith("libQnnHtp.so"):
        install_trace_breakpoints("new_objfile")


def short_bt(limit=8):
    frame = gdb.newest_frame()
    rows = []
    depth = 0
    while frame is not None and depth < limit:
        try:
            pc = int(frame.pc())
            name = frame.name() or "?"
            rows.append(f"#{depth} 0x{pc:x} {name}")
        except Exception as exc:
            rows.append(f"#{depth} <bt-error {exc}>")
        frame = frame.older()
        depth += 1
    return "\n".join(rows)


def log_probe_hit(name, rva, max_hits=3):
    count = PROBE_HIT_COUNTS.get(name, 0) + 1
    PROBE_HIT_COUNTS[name] = count
    if count > max_hits:
        return
    regs = {}
    for reg in ["rdi", "rsi", "rdx", "rcx", "r8", "r9", "rsp"]:
        try:
            regs[reg] = int(gdb.parse_and_eval(f"(unsigned long)${reg}"))
        except Exception:
            regs[reg] = None
    msg = [
        f"PROBE_HIT {name} rva=0x{rva:x} count={count}",
        "regs " + " ".join(
            f"{reg}=0x{val:x}" if val is not None else f"{reg}=<err>"
            for reg, val in regs.items()
        ),
        short_bt(),
    ]
    text = "\n".join(msg) + "\n"
    print(text, end="")
    with (OUT / "candidate_hits.log").open("a") as f:
        f.write(text + "\n")


class BaseEntry(gdb.Breakpoint):
    def __init__(self):
        super().__init__("_Z15dequantize_biasR11ReplacementRK5OpRefS3_S3_", internal=False)
        self.done = False

    def stop(self):
        global LIB_BASE
        try:
            if not self.done:
                self.done = True
                pc = int(gdb.selected_frame().pc())
                LIB_BASE = pc - 0x1BA81B2
                print(f"libQnnHtp base 0x{LIB_BASE:x}")
                install_trace_breakpoints("dequant_entry")
        except Exception as exc:
            print(f"GDB_BASE_ERROR: {exc}")
        return False


def dump_dequant_out():
    n = int(gdb.parse_and_eval("(unsigned long)$rcx"))
    p = int(gdb.parse_and_eval("(unsigned long)$rdx"))
    end = p + n * 4
    print(f"dequantize_bias_out len={n} ptr=0x{p:x}")
    (OUT / "dequantize_bias_bt.txt").write_text(short_bt(24) + "\n")
    gdb.execute(f"dump binary memory {OUT / 'dequantize_bias_out_float32.raw'} 0x{p:x} 0x{end:x}")


def dump_find_scale():
    txt = str(gdb.parse_and_eval("$xmm0"))
    print(f"find_bias_scale_xmm0 {txt}")
    (OUT / "find_bias_scale_xmm0.txt").write_text(txt + "\n")
    (OUT / "find_bias_scale_bt.txt").write_text(short_bt(24) + "\n")


def dump_int32_const_array():
    global INT32_CONST_INDEX
    try:
        ptr = int(gdb.parse_and_eval("(unsigned long)$rdx"))
        length = int(gdb.parse_and_eval("(unsigned long)$rcx"))
    except Exception as exc:
        print(f"gen_const_i32_arg_error: {exc}")
        return
    if length <= 0 or length > 4096:
        return
    INT32_CONST_INDEX += 1
    raw = OUT / ("gen_const_i32_%03d_len_%d.raw" % (INT32_CONST_INDEX, length))
    end = ptr + length * 4
    print(f"gen_const_i32 index={INT32_CONST_INDEX} len={length} ptr=0x{ptr:x}")
    gdb.execute(f"dump binary memory {raw} 0x{ptr:x} 0x{end:x}")


def dump_int32_common_const(kind):
    global INT32_COMMON_INDEX
    candidates = []
    for ptr_reg, len_reg in [("rcx", "r8"), ("rdx", "rcx")]:
        try:
            ptr = int(gdb.parse_and_eval(f"(unsigned long)${ptr_reg}"))
            length = int(gdb.parse_and_eval(f"(unsigned long)${len_reg}"))
        except Exception:
            continue
        if ptr and 0 < length <= 65536 and length % 4 == 0:
            candidates.append((ptr_reg, len_reg, ptr, length))
    if not candidates:
        return
    INT32_COMMON_INDEX += 1
    rows = [f"INT32_COMMON {kind} index={INT32_COMMON_INDEX}"]
    for ptr_reg, len_reg, ptr, length in candidates:
        raw = OUT / (
            "gen_const_int32_common_%03d_%s_%s_len_%d.raw"
            % (INT32_COMMON_INDEX, ptr_reg, len_reg, length)
        )
        try:
            gdb.execute(f"dump binary memory {raw} 0x{ptr:x} 0x{ptr + length:x}")
            rows.append(f"  {ptr_reg}/{len_reg} ptr=0x{ptr:x} len={length} dump={raw.name}")
        except Exception as exc:
            rows.append(f"  {ptr_reg}/{len_reg} ptr=0x{ptr:x} len={length} dump_error={exc}")
    rows.append(short_bt(16))
    append_trace("gen_const_int32_common_trace.log", "\n".join(rows) + "\n")


def dump_float_common_const(kind):
    global FLOAT_COMMON_INDEX
    try:
        ptr = int(gdb.parse_and_eval("(unsigned long)$rcx"))
        length = int(gdb.parse_and_eval("(unsigned long)$r8"))
    except Exception:
        return
    if not ptr or length <= 0 or length > 65536 or length % 4 != 0:
        return
    FLOAT_COMMON_INDEX += 1
    raw = OUT / ("gen_const_float_common_%03d_len_%d.raw" % (FLOAT_COMMON_INDEX, length))
    rows = [f"FLOAT_COMMON {kind} index={FLOAT_COMMON_INDEX} ptr=0x{ptr:x} len={length} dump={raw.name}"]
    try:
        gdb.execute(f"dump binary memory {raw} 0x{ptr:x} 0x{ptr + length:x}")
    except Exception as exc:
        rows.append(f"  dump_error={exc}")
    rows.append(short_bt(16))
    append_trace("gen_const_float_common_trace.log", "\n".join(rows) + "\n")


def dump_float_array_const(kind):
    global FLOAT_ARRAY_INDEX
    try:
        ptr = int(gdb.parse_and_eval("(unsigned long)$rdx"))
        count = int(gdb.parse_and_eval("(unsigned long)$rcx"))
    except Exception:
        return
    if not ptr or count <= 0 or count > 65536:
        return
    FLOAT_ARRAY_INDEX += 1
    raw = OUT / ("gen_const_f32_array_%03d_count_%d.raw" % (FLOAT_ARRAY_INDEX, count))
    rows = [f"F32_ARRAY {kind} index={FLOAT_ARRAY_INDEX} ptr=0x{ptr:x} count={count} dump={raw.name}"]
    try:
        gdb.execute(f"dump binary memory {raw} 0x{ptr:x} 0x{ptr + count * 4:x}")
    except Exception as exc:
        rows.append(f"  dump_error={exc}")
    rows.append(short_bt(16))
    append_trace("gen_const_f32_trace.log", "\n".join(rows) + "\n")


def dump_float_repeat_const(kind):
    global FLOAT_REPEAT_INDEX
    try:
        count = int(gdb.parse_and_eval("(unsigned long)$rdx"))
        value = str(gdb.parse_and_eval("$xmm0"))
    except Exception:
        return
    if count <= 0 or count > 65536:
        return
    FLOAT_REPEAT_INDEX += 1
    rows = [
        f"F32_REPEAT {kind} index={FLOAT_REPEAT_INDEX} count={count} xmm0={value}",
        short_bt(16),
    ]
    append_trace("gen_const_f32_trace.log", "\n".join(rows) + "\n")


def dump_opdef_const_from_bytes():
    """Capture only consts whose byte length matches the final bias sidecar."""
    global OPDEF_CONST_SIDEcar_INDEX
    try:
        this = int(gdb.parse_and_eval("(unsigned long)$rdi"))
        graph_prepare = int(gdb.parse_and_eval("(unsigned long)$rsi"))
        const_id = int(gdb.parse_and_eval("(unsigned long long)$rdx"))
        outdef = int(gdb.parse_and_eval("(unsigned long)$rcx"))
        data = int(gdb.parse_and_eval("(unsigned long)$r8"))
        length = int(gdb.parse_and_eval("(unsigned long)$r9"))
    except Exception as exc:
        print(f"opdef_const_arg_error: {exc}")
        return
    if length != TARGET_SIDEcar_BYTES or data == 0:
        return
    OPDEF_CONST_SIDEcar_INDEX += 1
    raw = OUT / ("opdef_const_sidecar_%03d_len_%d.raw" % (OPDEF_CONST_SIDEcar_INDEX, length))
    rows = [
        f"OPDEF_CONST_SIDECAR index={OPDEF_CONST_SIDEcar_INDEX} "
        f"this=0x{this:x} graph_prepare=0x{graph_prepare:x} id=0x{const_id:x} "
        f"outdef=0x{outdef:x} data=0x{data:x} len={length} dump={raw.name}",
        f"  outdef_qwords {dump_qwords(outdef, 12)}",
    ]
    try:
        gdb.execute(f"dump binary memory {raw} 0x{data:x} 0x{data + length:x}")
    except Exception as exc:
        rows.append(f"  dump_error={exc}")
    rows.append(short_bt(24))
    append_trace("opdef_const_sidecar_trace.log", "\n".join(rows) + "\n")


def arm_const_data_len_return():
    try:
        this = int(gdb.parse_and_eval("(unsigned long)$rdi"))
        ret = u64(int(gdb.parse_and_eval("(unsigned long)$rsp")))
    except Exception:
        return

    def cb(this=this, ret=ret):
        try:
            length = int(gdb.parse_and_eval("(unsigned long)$rax"))
        except Exception:
            return
        if length != TARGET_SIDEcar_BYTES:
            return
        CONST_DATA_SIDEcar_THIS.add(this)
        rows = [
            f"CONST_DATA_LEN_SIDECAR this=0x{this:x} len={length} ret=0x{ret:x} ret_rva=0x{ret - LIB_BASE:x}",
            f"  this_qwords {dump_qwords(this, 24)}",
            short_bt(24),
        ]
        append_trace("const_data_sidecar_trace.log", "\n".join(rows) + "\n")

    TempReturn(ret, "const_data_len_ret", cb)


def arm_const_data_ptr_return():
    try:
        this = int(gdb.parse_and_eval("(unsigned long)$rdi"))
        ret = u64(int(gdb.parse_and_eval("(unsigned long)$rsp")))
    except Exception:
        return

    def cb(this=this, ret=ret):
        global CONST_DATA_SIDEcar_INDEX
        if this not in CONST_DATA_SIDEcar_THIS:
            return
        try:
            ptr = int(gdb.parse_and_eval("(unsigned long)$rax"))
        except Exception:
            return
        if ptr == 0:
            return
        CONST_DATA_SIDEcar_INDEX += 1
        raw = OUT / ("const_data_sidecar_%03d_len_%d.raw" % (CONST_DATA_SIDEcar_INDEX, TARGET_SIDEcar_BYTES))
        rows = [
            f"CONST_DATA_PTR_SIDECAR index={CONST_DATA_SIDEcar_INDEX} this=0x{this:x} "
            f"ret=0x{ret:x} ret_rva=0x{ret - LIB_BASE:x} ptr=0x{ptr:x} dump={raw.name}",
            short_bt(24),
        ]
        try:
            gdb.execute(f"dump binary memory {raw} 0x{ptr:x} 0x{ptr + TARGET_SIDEcar_BYTES:x}")
        except Exception as exc:
            rows.append(f"  dump_error={exc}")
        append_trace("const_data_sidecar_trace.log", "\n".join(rows) + "\n")

    TempReturn(ret, "const_data_ptr_ret", cb)


def dump_serializer_sidecar(name, ptr_reg, len_reg):
    global SERIALIZE_SIDEcar_INDEX
    try:
        ptr = int(gdb.parse_and_eval(f"(unsigned long)${ptr_reg}"))
        length = int(gdb.parse_and_eval(f"(unsigned long)${len_reg}"))
    except Exception:
        return
    if ptr == 0 or length != TARGET_SIDEcar_BYTES:
        return
    SERIALIZE_SIDEcar_INDEX += 1
    raw = OUT / ("serializer_sidecar_%03d_%s_len_%d.raw" % (SERIALIZE_SIDEcar_INDEX, name, length))
    rows = [
        f"SERIALIZE_SIDECAR index={SERIALIZE_SIDEcar_INDEX} name={name} "
        f"ptr=0x{ptr:x} len={length} dump={raw.name}",
        short_bt(32),
    ]
    try:
        gdb.execute(f"dump binary memory {raw} 0x{ptr:x} 0x{ptr + length:x}")
    except Exception as exc:
        rows.append(f"  dump_error={exc}")
    append_trace("serializer_sidecar_trace.log", "\n".join(rows) + "\n")


def f32_at(addr):
    return float(gdb.parse_and_eval("*((float*)0x%x)" % addr))


def i64_at(addr):
    return int(gdb.parse_and_eval("*((long*)0x%x)" % addr))


def bias_convert_entry(body_rva):
    global BIAS_CONVERT_ACTIVE, BIAS_CONVERT_ROWS, BIAS_CONVERT_BODY_COUNT, BIAS_CONVERT_CAPTURED_BODY
    n = int(gdb.parse_and_eval("(unsigned long)*(unsigned long*)($rsi + 0x20)"))
    shift = int(gdb.parse_and_eval("*(int*)($r9 + 0xc)"))
    BIAS_CONVERT_BODY_COUNT += 1
    print(f"bias_convert_entry body=0x{body_rva:x} count={BIAS_CONVERT_BODY_COUNT} n={n} shift={shift}")
    if n == 256 and not BIAS_CONVERT_ACTIVE and not BIAS_CONVERT_ROWS:
        BIAS_CONVERT_ACTIVE = True
        BIAS_CONVERT_CAPTURED_BODY = body_rva
        BIAS_CONVERT_ROWS = []
        (OUT / "bias_convert_entry.txt").write_text(
            f"body_rva=0x{body_rva:x}\ncount={BIAS_CONVERT_BODY_COUNT}\nn={n}\nshift={shift}\n"
        )


def append_bias_convert_row(index, out_i32):
    rsp = int(gdb.parse_and_eval("(unsigned long)$rsp"))
    row = {
        "index": int(index),
        "numerator_f32": f32_at(rsp + 0x108),
        "current_input_scale_f32": f32_at(rsp + 0x10C),
        "ratio_f32": f32_at(rsp + 0x118),
        "shift": i64_at(rsp + 0x110),
        "out_i32": int(out_i32),
    }
    BIAS_CONVERT_ROWS.append(row)


def bias_convert_write_main(body_rva):
    if not BIAS_CONVERT_ACTIVE or BIAS_CONVERT_CAPTURED_BODY != body_rva:
        return
    idx = int(gdb.parse_and_eval("(unsigned long)$rbp"))
    out = int(gdb.parse_and_eval("(int)$r13"))
    append_bias_convert_row(idx, out)


def bias_convert_write_tail(body_rva):
    if not BIAS_CONVERT_ACTIVE or BIAS_CONVERT_CAPTURED_BODY != body_rva:
        return
    idx = int(gdb.parse_and_eval("(unsigned long)$rbx"))
    out = int(gdb.parse_and_eval("(int)$eax"))
    append_bias_convert_row(idx, out)


def bias_convert_exit(body_rva):
    global BIAS_CONVERT_ACTIVE
    if not BIAS_CONVERT_ACTIVE or BIAS_CONVERT_CAPTURED_BODY != body_rva:
        return
    BIAS_CONVERT_ACTIVE = False
    path = OUT / "bias_convert_writes.csv"
    with path.open("w") as f:
        f.write("index,numerator_f32,current_input_scale_f32,ratio_f32,shift,out_i32\n")
        for row in BIAS_CONVERT_ROWS:
            f.write(
                f"{row['index']},{row['numerator_f32']:.12g},"
                f"{row['current_input_scale_f32']:.12g},{row['ratio_f32']:.12g},"
                f"{row['shift']},{row['out_i32']}\n"
            )
    print(f"bias_convert_writes rows={len(BIAS_CONVERT_ROWS)} path={path}")


gdb.events.new_objfile.connect(on_new_objfile)
BaseEntry()
Entry("_Z15dequantize_biasR11ReplacementRK5OpRefS3_S3_", 0x1FE, "dequantize_bias_out", dump_dequant_out)
Entry("_Z15find_bias_scaleR11ReplacementRK5OpRef", 0xFD, "find_bias_scale_out", dump_find_scale)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--qnn-sdk-root",
        type=Path,
        default=Path(os.environ.get("QNN_SDK_ROOT", "tools/qnn-sdk")),
    )
    parser.add_argument(
        "--sidecar-bytes",
        type=int,
        default=2048,
        help="bias/control sidecar byte length to capture from ctxgen",
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
                "__SIDECAR_BYTES__", str(args.sidecar_bytes)
            )
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
    print(f"wrote {log_path}")
    for path in sorted(out_dir.glob("*")):
        if path.is_file() and path.name != "gdb.log":
            print(f"wrote {path}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
