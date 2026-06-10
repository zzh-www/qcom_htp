#!/usr/bin/env python3
"""Emulate the native set_hmx_params_conv1x1 helper for probe planning.

The implementation follows the V75 skel helper at
libQnnHtpV75Skel.so@0x287dc0.  It is intentionally small and only reports the
16 u32 words written into the hmx_params buffer so W16A16/W8A16 probe tuples can
be compared without another device run.
"""

from __future__ import annotations

import argparse


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def _extractu(value: int, width: int, offset: int) -> int:
    return (value >> offset) & ((1 << width) - 1)


def _bitsclr(value: int, mask: int) -> bool:
    return (value & mask) == 0


def conv1x1_words(arg1: int, arg2: int, arg3: int, arg4: int, arg5: int) -> list[int]:
    words = [0] * 16

    p3 = arg2 > 0x20
    r2_plus = _u32(arg2 + 0x1F)
    r7 = _extractu(r2_plus, 3, 2)
    r13 = _extractu(arg5, 5, 8)
    p_r13_eq_2 = r13 == 2
    p_r13_gt_2 = r13 > 2
    r2_field = _extractu(arg1, 6, 5)
    p_arg5_bit20_clear = _bitsclr(arg5, 0x20)
    r6 = _extractu(arg1, 1, 2) + 1
    r12 = 0x7 if p3 else r7

    if p_r13_gt_2:
        r9 = (r6 * r13) << 5
        r8 = r13 - 1
    else:
        r8 = 0x80 + (r7 << 7)
        r9 = r8 * r6
        r15 = r9 << 1
        if p3:
            r15 = 0x800
        # The r9 span-doubling applies to the non-dilate (`:deep`, arg5 bit7
        # clear) weight feed. In dilate mode (arg5 & 0x80, e.g. W16A16 int16
        # weight = 2 byte planes), the activation is read in pairs so the rt
        # span stays halved: r9 keeps r8*r6 (verified byte-exact vs the real
        # skel set_hmx_params_conv1x1 dump: arg5=0x80 -> word6=0x3ff).
        dilate = (arg5 & 0x80) != 0
        if not p_r13_eq_2 and not dilate:
            r9 = r15
        if p_r13_eq_2 and r6 == 2:
            r9 = (r6 * r13) << 5
            r8 = r13 - 1

    r13_selector = 2
    neg_arg4 = _u32(-arg4)
    if not p_arg5_bit20_clear:
        r7 = r12
    if r2_field <= 0x3B:
        r13_selector = 4
        if r2_field == 0x20:
            r13_selector = 5
        elif r2_field != 0x30:
            r13_selector = 3

    r12 = neg_arg4 & 7
    word1 = _u32(r2_field << 5)
    r9 = _u32(r9 - 1)
    words[6] = r9

    word3 = _u32(word1 | (r7 << 2) | 3)
    word2 = _u32((r2_field & (r12 << r13_selector)) << 5)
    words[2] = word2
    words[3] = word3

    if p_r13_gt_2 or (p_r13_eq_2 and r6 == 2):
        word3 = _u32(r8 | word1)
        arg5 = _u32(arg5 & 0xFFFFE0FF)
        word2 = _u32(0x1F ^ (word2 | r8))
        words[2] = word2
        words[3] = word3

    r6_from_arg3 = _u32(arg3 << 5)
    words[12] = _u32(arg5)
    words[1] = word1
    words[0] = _extractu(r6_from_arg3, 11, 0)

    if _bitsclr(arg1, 0x3):
        return words

    if arg1 & 1:
        w2_low = _extractu(word2, 2, 5)
        w2_high = word2 & ~0x7F
        w2_shift = (word2 << 2) & 0x7C
        w3_high = word3 & ~0x7F
        w3_low = _extractu(word3, 2, 5)
        w3_shift = (word3 << 2) & 0x7C
        words[2] = _u32(w2_low | w2_shift | w2_high)
        words[3] = _u32(w3_low | w3_shift | w3_high)

    if arg1 & 0x2:
        words[0] = _u32((r6_from_arg3 & 0x780) | (arg3 & 0x3))
        words[1] = _u32((r2_field & 0x3) | (word1 & 0x780))

    return words


def convw4b1x1_words(
    arg1: int,
    arg2: int,
    arg3: int,
    arg4: int,
    arg5: int,
    arg6: int,
) -> list[int]:
    """Return the decoded W4 1x1 HMX mask words.

    This follows libQnnHtpV75Skel.so
    ``_Z25set_hmx_params_convw4b1x1P10hmx_paramsmmmmmm`` at 0x289380.  The
    canonical W4A16 tuple is still ``(0x70b, 256, 0, 0, 0, 0xa0)``, but keeping
    the field derivation here prevents route-word diagnostics from depending on
    a hard-coded tuple shortcut.
    """

    words = [0] * 16
    r6 = _extractu(arg1, 6, 5)
    r8 = arg6 & 0x20
    if (not _bitsclr(arg1, 0x8)) and (arg6 & 0x40) == 0:
        route_span = 0x200 if r8 == 0 else 0x400
        route_sub = 0x7
    else:
        route_sub = _extractu(arg2 + 0x1F, 3, 2)
        lane_count = _extractu(arg1, 1, 2) + 1
        route_span = (0x80 + (route_sub << 7)) * lane_count
        if r8 != 0:
            route_span <<= 1
            if arg2 > 0x20:
                route_sub = 0x7
                route_span = 0x800

    selector = 2
    if r6 <= 0x3B:
        selector = 4
        if r6 == 0x20:
            selector = 5
        elif r6 != 0x30:
            selector = 3
    elif r6 != 0x3C:
        selector = 1

    word1 = _u32(r6 << 5)
    words[6] = _u32(route_span - 1)
    words[12] = _u32(arg6)
    neg_arg5_low = _u32(-arg5) & 0x7
    route_word2 = r6 & (neg_arg5_low << selector)
    selector_mask = _u32(-1 << selector)
    route_word3_base = _u32(word1 | (route_sub << 2))
    arg3_shift = _u32(arg3 << 5)
    words[1] = word1
    words[0] = _extractu(arg3_shift, 11, 0)
    route_word2 = _u32(route_word2 | (arg4 & _u32(~selector_mask)))
    words[2] = _u32(route_word2 << 5)
    words[3] = _u32(route_word3_base | 0x3)

    if _bitsclr(arg1, 0x3):
        return words

    if arg1 & 1:
        word3_shift = _u32(words[3] << 2)
        word3_high = route_word3_base & ~0x7F
        word3_low = r6 & 0x3
        word2_high = words[2] & 0x7F80
        word2_low = route_word2 & 0x3
        words[2] = _u32(word2_high | word2_low)
        words[3] = _u32(word3_high | word3_low | (word3_shift & 0x7C))

    if arg1 & 0x2:
        words[1] = _u32((r6 & 0x3) | (word1 & 0x780))
        words[0] = _u32((arg3_shift & 0x780) | (arg3 & 0x3))
    return words


def _parse_int(text: str) -> int:
    return int(text, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tuple", nargs="+", help="five integers: arg1 arg2 arg3 arg4 arg5")
    args = parser.parse_args()
    if len(args.tuple) != 5:
        parser.error("expected exactly five arguments")
    values = [_parse_int(v) for v in args.tuple]
    words = conv1x1_words(*values)
    print("tuple:", " ".join(hex(v) for v in values))
    print("words:", "[" + ", ".join(hex(v) for v in words) + "]")


if __name__ == "__main__":
    main()
