#!/usr/bin/env python3
"""QNN HTP U8 accumulator drain sidecar helpers."""

from __future__ import annotations

import numpy as np

QNN_HTP_U8_DRAIN_CONTROL_BASE = np.uint16(0x0040)
QNN_HTP_U8_DRAIN_CONTROL_MID_QUARTERS = np.uint16(0x8040)
QNN_HTP_U8_DRAIN_FP16_MAX = np.float32(65504.0)
QNN_HTP_U8_DRAIN_OVERFLOW_SCALE = np.float32(2.0**32)
QNN_HTP_W8A16_DRAIN_SCALE_CTL = 0x8000


def qnn_htp_w8a16_drain_scale(
    act_scale: float | np.float32,
    weight_scale: np.ndarray,
    output_scale: float | np.float32,
) -> np.ndarray:
    """Recreate QNN's W8A16 per-channel drain scale const.

    QNN does not materialize the W8A16 drain scale as the direct per-channel
    expression ``act_scale * weight_scale / output_scale``.  The observed HTP
    prepare graph first normalizes the weight scales by their channel maximum,
    then multiplies by the maximum channel's drain scale.  Keeping those two
    float32 rounding points is required for bit-exact control/drain packing.
    """
    weight = np.asarray(weight_scale, dtype=np.float32)
    max_weight = np.max(weight).astype(np.float32)
    if max_weight == np.float32(0.0):
        return np.zeros(weight.shape, dtype=np.float32)
    normalized = (weight / max_weight).astype(np.float32)
    max_channel_scale = (
        (max_weight * np.float32(act_scale)) / np.float32(output_scale)
    ).astype(np.float32)
    return (normalized * max_channel_scale).astype(np.float32)


def qnn_htp_u8_drain_scale_control(exact_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode QNN native U8 drain fp16 scale and control fields.

    Native Conv1x1 A8 sidecars store each output channel as:

        uint16 scale_f16_bits, uint16 control

    The normal path rounds within the enclosing fp16 interval using QNN's
    quarter split: the middle two quarters keep the lower endpoint and set
    control bit 15, while the upper quarter selects the upper endpoint.

    The overflow path is part of ConvLayer.opt.bias_scale_shuff lowering: when
    the exact drain scale is outside finite fp16 range, QNN first divides the
    scale by 2^32 and then applies the same quarter split.  The base control
    word remains 0x0040; bit 15 is still only the quarter-compensation bit.
    """
    exact = exact_scale.astype(np.float32)
    encoded = np.where(
        exact > QNN_HTP_U8_DRAIN_FP16_MAX,
        (exact / QNN_HTP_U8_DRAIN_OVERFLOW_SCALE).astype(np.float32),
        exact,
    ).astype(np.float32)

    nearest = encoded.astype(np.float16)
    nearest_f32 = nearest.astype(np.float32)
    lower_u16 = nearest.view(np.uint16).astype(np.int32)
    lower_u16[nearest_f32 > encoded] -= 1
    lower_u16 = lower_u16.astype(np.uint16)
    upper_u16 = (lower_u16.astype(np.int32) + 1).astype(np.uint16)

    lower_f32 = lower_u16.view(np.float16).astype(np.float32)
    upper_f32 = upper_u16.view(np.float16).astype(np.float32)
    frac = (encoded - lower_f32) / (upper_f32 - lower_f32)

    scale_u16 = np.where(frac >= 0.75, upper_u16, lower_u16).astype(np.uint16)
    control_u16 = np.where(
        (frac >= 0.25) & (frac < 0.75),
        QNN_HTP_U8_DRAIN_CONTROL_MID_QUARTERS,
        QNN_HTP_U8_DRAIN_CONTROL_BASE,
    ).astype(np.uint16)
    return scale_u16, control_u16


def qnn_htp_a16_drain_mantissa_scale_control(
    exact_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the nested A16 drain mantissa helper used in 512B records.

    A16 records split the primary drain scale into an fp16 interval index and a
    secondary mantissa-control scale.  The secondary scale uses the same 0x0040
    / 0x8040 control words as U8 drain records, but QNN treats exact quarter
    boundaries differently: 0.25 remains low-control, while 0.75 remains the
    lower fp16 word with high-control.
    """
    encoded = exact_scale.astype(np.float32)
    nearest = encoded.astype(np.float16)
    lower_u16 = nearest.view(np.uint16).astype(np.int32)
    lower_u16[nearest.astype(np.float32) > encoded] -= 1
    lower_u16 = lower_u16.astype(np.uint16)
    upper_u16 = (lower_u16.astype(np.int32) + 1).astype(np.uint16)

    lower_f32 = lower_u16.view(np.float16).astype(np.float32)
    upper_f32 = upper_u16.view(np.float16).astype(np.float32)
    frac = (encoded - lower_f32) / (upper_f32 - lower_f32)

    scale_u16 = np.where(frac > 0.75, upper_u16, lower_u16).astype(np.uint16)
    control_u16 = np.where(
        (frac > 0.25) & (frac <= 0.75),
        QNN_HTP_U8_DRAIN_CONTROL_MID_QUARTERS,
        QNN_HTP_U8_DRAIN_CONTROL_BASE,
    ).astype(np.uint16)
    return scale_u16, control_u16


def qnn_htp_w8a16_drain_control_words(
    exact_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode QNN native W8A16 A16 drain/control words.

    This mirrors the QNN HTP x86 prepare path:

        0x1048 -> 0x1064
        factory candidate 0x14ea3e0
        materializer 0x14e37cf / helper 0x14e3843

    For W8A16 per-channel native records the active path uses 32 float scales
    to produce 64 uint32 words per N32 tile.  The scalar control inputs are
    equivalent to ``scale_ctl=0x8000``, ``mode=0``, and ``split=1``.  The code
    below keeps the same integer bit operations as the materializer so boundary
    cases follow QNN's emitted context bytes.
    """
    bits = exact_scale.astype(np.float32).view(np.uint32).reshape(-1)
    if bits.size % 32:
        raise ValueError(f"W8A16 drain scales must be N32-aligned, got {bits.size}")

    word01_by_channel = np.zeros(bits.size, dtype=np.uint32)
    word23_by_channel = np.zeros(bits.size, dtype=np.uint32)
    mode = 0
    scale_ctl = (QNN_HTP_W8A16_DRAIN_SCALE_CTL << 4) | 0x8

    mode_bits = mode & 0x1C010
    select = 1 if mode_bits != 0x8010 else 0
    base_shift = 8
    if (mode & 0x40) == 0:
        base_shift = -2 + select + select
    exp_mode = mode & 0x1C000
    if exp_mode == 0x8000:
        base_shift = -1 if (mode & 0x300000) == 0x100000 else base_shift
    elif exp_mode != 0:
        base_shift = (mode >> 14) | -8
    base_shift += -103

    word0_base = (scale_ctl << 19) & 0x7FC00000
    word3_base64 = ((scale_ctl & 0xFF000) << 43) & 0xFFFFFFFFFFFFFFFF

    out_stride = 0
    lane_base = 0
    for src_idx, raw in enumerate(bits.astype(np.uint32)):
        rounded = (int(raw) + 2) & 0xFFFFFFFF
        mant = ((rounded >> 2) & 0x1FFFFF) | 0x200000
        exponent = (rounded >> 23) & 0xFF
        exponent_shift = base_shift + exponent

        right_shift = exponent_shift - 31
        if (right_shift & 0xFFFFFFFF) >= 0x16:
            right_shift = 0x16

        exp_bits = (exponent_shift & 0x1F) << 10
        if exponent_shift < 0x20:
            right_shift = 0
        if exponent_shift >= 0x20:
            exp_bits = 0x7C00

        mant >>= right_shift
        word3_64 = (((~mant) & 0x200000) << 27) & 0xFFFFFFFFFFFFFFFF
        word3_64 |= word3_base64
        word3_64 |= ((mant & 0x1FF800) << 21) & 0xFFFFFFFFFFFFFFFF

        mant_ror = ((mant >> 1) | ((mant & 1) << 31)) & 0xFFFFFFFF
        word01 = (mant_ror & 0x800003FF) | exp_bits | word0_base

        word01_by_channel[src_idx] = np.uint32(word01)
        word23_by_channel[src_idx] = np.uint32((word3_64 >> 32) & 0xFFFFFFFF)

        lane_base += 2
        out_stride = (out_stride + 0x20) & 0xFFFFFFFF

    words_u16 = (
        np.stack([word01_by_channel, word23_by_channel], axis=1)
        .astype(np.uint32)
        .reshape(-1)
        .view(np.uint16)
        .reshape(-1, 4)
    )
    return (
        words_u16[:, 0].copy(),
        words_u16[:, 1].copy(),
        words_u16[:, 2].copy(),
        words_u16[:, 3].copy(),
    )
