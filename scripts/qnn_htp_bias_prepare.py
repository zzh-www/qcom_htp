#!/usr/bin/env python3
"""Recovered QNN HTP prepare bias-sidecar helpers.

These helpers model the per-channel native Conv1x1 prepare path that turns the
quantized DLC static Conv bias into the final HTP `bias_to_vtcm` sidecar bias
field.  They intentionally use explicit float32 steps; small changes in where
values are rounded alter +/-1 sidecar bytes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PerChannelBiasPrepareStages:
    """Observable stages in QNN HTP per-channel bias prepare."""

    global_bias_scale: np.float32
    dequantized_bias_f32: np.ndarray
    find_bias_scale_f32: np.float32
    requant_mul_f32: np.float32
    requant_bias_pre_nearby_f32: np.ndarray
    requant_bias_expanded_i32: np.ndarray
    bias_scale_shuff_restore_f32: np.ndarray
    final_bias_i32: np.ndarray


def qnn_htp_perchannel_global_bias_scale(
    act_scale: np.float32 | float,
    weight_scale: np.ndarray,
) -> np.float32:
    """Return the global scale used by QNN `dequantize_bias`.

    QNN scans the weight-scale tensor with `fmaxf`; it does not use
    `act_scale * weight_scale[channel]` for this prepare stage.
    """
    return (np.float32(act_scale) * np.max(weight_scale.astype(np.float32))).astype(np.float32)


def qnn_htp_perchannel_find_bias_scale(dequantized_bias_f32: np.ndarray) -> np.float32:
    """Return QNN `find_bias_scale` for a float32 bias vector."""
    max_abs = np.max(np.abs(dequantized_bias_f32.astype(np.float32))).astype(np.float32)
    if max_abs == np.float32(0.0):
        return np.float32(0.0)
    return (max_abs * np.float32(16.0) / np.float32(2.0**32)).astype(np.float32)


def qnn_htp_perchannel_prepare_bias_stages(
    bias_q_i32: np.ndarray,
    act_scale: np.float32 | float,
    weight_scale: np.ndarray,
) -> PerChannelBiasPrepareStages:
    """Reproduce QNN HTP prepare's per-channel sidecar-bias conversion.

    Recovered rule:

        global_bias_scale = float32(act_scale) * max(float32(weight_scale))
        dequant = float32(bias_q * global_bias_scale)
        find_bias_scale = float32(max(abs(dequant)) * 16 / 2^32)
        expanded = nearbyintf(float32(dequant * float32(1 / find_bias_scale)))
        final = trunc(float32(float32(expanded * find_bias_scale) / global_bias_scale))

    `np.rint` matches the observed `nearbyintf` behavior under the default
    round-to-nearest-even floating-point environment used by ctxgen.
    """
    bias_q = bias_q_i32.astype(np.int32, copy=False)
    global_bias_scale = qnn_htp_perchannel_global_bias_scale(act_scale, weight_scale)
    dequantized_bias_f32 = (bias_q.astype(np.float32) * global_bias_scale).astype(np.float32)
    find_bias_scale_f32 = qnn_htp_perchannel_find_bias_scale(dequantized_bias_f32)

    if find_bias_scale_f32 == np.float32(0.0):
        zeros_i32 = np.zeros_like(bias_q, dtype=np.int32)
        zeros_f32 = np.zeros_like(bias_q, dtype=np.float32)
        return PerChannelBiasPrepareStages(
            global_bias_scale=global_bias_scale,
            dequantized_bias_f32=dequantized_bias_f32,
            find_bias_scale_f32=find_bias_scale_f32,
            requant_mul_f32=np.float32(0.0),
            requant_bias_pre_nearby_f32=zeros_f32,
            requant_bias_expanded_i32=zeros_i32,
            bias_scale_shuff_restore_f32=zeros_f32,
            final_bias_i32=zeros_i32,
        )

    requant_mul_f32 = (np.float32(1.0) / find_bias_scale_f32).astype(np.float32)
    requant_bias_pre_nearby_f32 = (dequantized_bias_f32 * requant_mul_f32).astype(np.float32)
    requant_bias_expanded_i32 = np.rint(requant_bias_pre_nearby_f32)
    requant_bias_expanded_i32 = np.clip(
        requant_bias_expanded_i32,
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    ).astype(np.int32)

    bias_scale_shuff_restore_f32 = (
        (requant_bias_expanded_i32.astype(np.float32) * find_bias_scale_f32).astype(np.float32)
        / global_bias_scale
    ).astype(np.float32)
    final_bias_i32 = np.trunc(bias_scale_shuff_restore_f32).astype(np.int32)

    return PerChannelBiasPrepareStages(
        global_bias_scale=global_bias_scale,
        dequantized_bias_f32=dequantized_bias_f32,
        find_bias_scale_f32=find_bias_scale_f32,
        requant_mul_f32=requant_mul_f32,
        requant_bias_pre_nearby_f32=requant_bias_pre_nearby_f32,
        requant_bias_expanded_i32=requant_bias_expanded_i32,
        bias_scale_shuff_restore_f32=bias_scale_shuff_restore_f32,
        final_bias_i32=final_bias_i32,
    )


def qnn_htp_perchannel_prepare_bias_q(
    bias_q_i32: np.ndarray,
    act_scale: np.float32 | float,
    weight_scale: np.ndarray,
) -> np.ndarray:
    """Return the prepared int32 bias before family-specific sidecar projection."""
    return qnn_htp_perchannel_prepare_bias_stages(
        bias_q_i32,
        act_scale,
        weight_scale,
    ).final_bias_i32


def qnn_htp_perchannel_a8_sidecar_bias_q(
    bias_q_i32: np.ndarray,
    act_scale: np.float32 | float,
    weight_scale: np.ndarray,
) -> np.ndarray:
    """Return the common prepared sidecar bias for per-channel U8-output kernels.

    For u8i8 this value is inserted directly into the 256-byte U8-output bias
    record.  W4A8 uses the same prepared bias, then projects the final folded
    effective-bias field into its 16x W4 HMX accumulator domain.
    """
    return qnn_htp_perchannel_prepare_bias_q(bias_q_i32, act_scale, weight_scale)


def qnn_htp_perchannel_w4a8_sidecar_effective_i32(
    bias_q_i32: np.ndarray,
    act_scale: np.float32 | float,
    weight_scale: np.ndarray,
    weight_q_kn: np.ndarray,
    act_zp: int = 128,
) -> np.ndarray:
    """Return QNN native W4A8 per-channel effective-bias field.

    W4A8's ConvLayer sidecar stores the effective-bias field in the same
    accumulator domain as the W4 HMX body: 16x the U8I8-style folded value.
    The paired drain scale is correspondingly divided by 16.
    """
    sidecar_bias = qnn_htp_perchannel_a8_sidecar_bias_q(
        bias_q_i32,
        act_scale,
        weight_scale,
    )
    folded = (-int(act_zp)) * weight_q_kn.astype(np.int32).sum(axis=0).astype(np.int32)
    return ((folded + sidecar_bias).astype(np.int64) * 16).astype(np.int32)


def qnn_htp_perchannel_a16_sidecar_bias_q(
    bias_q_i32: np.ndarray,
    act_scale: np.float32 | float,
    weight_scale: np.ndarray,
) -> np.ndarray:
    """Return the final sidecar bias for per-channel U16-output kernels.

    The observed w8a16 and w4a16 per-channel native records use the prepared
    bias after a signed truncation by 256 before adding the record's
    `-128 * sum(weight)` folding term.
    """
    prepared = qnn_htp_perchannel_prepare_bias_q(bias_q_i32, act_scale, weight_scale)
    return np.trunc(prepared.astype(np.float32) / np.float32(256.0)).astype(np.int32)


# Backward-compatible u8i8 names used by existing probes and generators.
U8I8BiasPrepareStages = PerChannelBiasPrepareStages
qnn_htp_u8i8_global_bias_scale = qnn_htp_perchannel_global_bias_scale
qnn_htp_u8i8_find_bias_scale = qnn_htp_perchannel_find_bias_scale
qnn_htp_u8i8_prepare_bias_stages = qnn_htp_perchannel_prepare_bias_stages
qnn_htp_u8i8_prepare_bias_q = qnn_htp_perchannel_a8_sidecar_bias_q
