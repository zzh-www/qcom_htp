#!/usr/bin/env python3
"""Generate custom HMX quantized MatMul/FC-as-Conv1d test graphs."""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DOMAIN = "hmx"


@dataclass(frozen=True)
class Family:
    name: str
    op: str
    act_bits: int
    weight_bits: int
    out_bits: int


FAMILIES = {
    "w4a8": Family("w4a8", "HmxU8I4ToU8MatMul", 8, 4, 8),
    "w8a16": Family("w8a16", "HmxU16I8ToU16MatMul", 16, 8, 16),
    "w4a16": Family("w4a16", "HmxU16I4ToU16MatMul", 16, 4, 16),
    "w16a16": Family("w16a16", "HmxU16I16ToU16MatMul", 16, 16, 16),
}


def _uint_dtype(bits: int):
    return np.uint8 if bits == 8 else np.uint16


def _uint_tensorproto(bits: int):
    return TensorProto.UINT8 if bits == 8 else TensorProto.UINT16


def _uint_storage(bits: int) -> str:
    if bits == 8:
        return "uint8"
    if bits == 16:
        return "uint16_le"
    raise ValueError(f"unsupported unsigned storage bitwidth: {bits}")


def _signed_dtype(bits: int):
    if bits <= 8:
        return np.int8
    return np.int16


def _activation_encoding(bits: int) -> dict:
    max_value = float((1 << bits) - 1)
    return {
        "bitwidth": bits,
        "dtype": "int",
        "is_symmetric": "False",
        "scale": 1.0,
        "offset": 0,
        "min": 0.0,
        "max": max_value,
    }


def _native_a16_encoding() -> dict:
    return {
        "bitwidth": 16,
        "dtype": "int",
        "is_symmetric": "True",
        "scale": 1.0 / 32767.0,
        "offset": -32768,
        "min": -1.0,
        "max": 1.0,
    }


def _is_native_a16_family(family: Family) -> bool:
    return family.act_bits == 16 and family.out_bits == 16


def _symmetric_encoding(bits: int) -> dict:
    qmax = (1 << (bits - 1)) - 1
    return {
        "bitwidth": bits,
        "dtype": "int",
        "is_symmetric": "True",
        "scale": 1.0,
        "offset": -(1 << (bits - 1)),
        "min": float(-qmax - 1),
        "max": float(qmax),
    }


def _legacy_encoding_to_v1(name: str, enc: dict) -> dict:
    """Convert the repo's legacy override record to QAIRT v1.0.0 format."""
    bits = int(enc["bitwidth"])
    offset = int(enc.get("offset", 0))
    dtype = "UINT" if offset == 0 and float(enc.get("min", 0.0)) >= 0.0 else "INT"
    is_sym = enc.get("is_symmetric", False)
    if isinstance(is_sym, str):
        is_sym = is_sym.lower() == "true"
    return {
        "name": name,
        "enc_type": "PER_TENSOR",
        "bw": bits,
        "dtype": dtype,
        "is_sym": bool(is_sym),
        "scale": [enc.get("scale", 1.0)],
        "offset": [offset],
        "min": [enc.get("min")],
        "max": [enc.get("max")],
    }


def _w4_lpbq_encoding(name: str, weight_dims: tuple[int, ...], block_size: int) -> dict:
    """Build a deterministic LPBQ override for the packed W4 carrier tensor.

    QAIRT v1.0.0 represents LPBQ as BLOCKWISE_EXPANSION while still accepting
    typed custom-op tensors.  The converter maps the scale axis to the last
    dimension and the block axis to dimension 2 for these packed carriers:
    W4A8 [1,1,K,N/2] and W4A16 [1,1,K/2,N].
    """
    if len(weight_dims) != 4:
        raise ValueError(f"W4 LPBQ requires a 4D weight carrier, got {weight_dims}")
    block_axis_size = int(weight_dims[2])
    channels = int(weight_dims[3])
    if block_axis_size <= 0 or channels <= 0:
        raise ValueError(f"invalid W4 LPBQ weight shape: {weight_dims}")
    if block_size <= 0 or block_axis_size % block_size:
        raise ValueError(
            f"W4 LPBQ block size {block_size} must divide packed weight axis-2 size {block_axis_size}"
        )
    blocks_per_channel = block_axis_size // block_size
    return {
        "name": name,
        "dtype": "INT",
        "bw": 8,
        "is_sym": True,
        "compressed_bw": 4,
        "enc_type": "LPBQ",
        "block_size": block_size,
        "scale": [1.0] * channels,
        "offset": [-128] * channels,
        "per_block_int_scale": [[1] * blocks_per_channel for _ in range(channels)],
    }


def _make_quant_overrides(
    activation_encodings: dict[str, list[dict]],
    param_encodings: dict[str, list[dict]],
    *,
    lpbq_v1: bool,
) -> dict:
    if not lpbq_v1:
        return {
            "activation_encodings": activation_encodings,
            "param_encodings": param_encodings,
        }

    lpbq_params = [
        enc_list[0]
        for enc_list in param_encodings.values()
        if enc_list and enc_list[0].get("enc_type") == "LPBQ"
    ]
    v1_activations = [
        _legacy_encoding_to_v1(name, enc_list[0])
        for name, enc_list in activation_encodings.items()
        if enc_list and enc_list[0].get("enc_type") != "LPBQ"
    ]
    v1_params = [
        _legacy_encoding_to_v1(name, enc_list[0])
        for name, enc_list in param_encodings.items()
        if enc_list and enc_list[0].get("enc_type") != "LPBQ"
    ]
    return {
        "version": "1.0.0",
        "activation_encodings": v1_activations,
        "param_encodings": [*v1_params, *lpbq_params],
    }


def _make_activation(
    family: Family,
    m: int,
    k: int,
    idx: int,
    activation_mode: str = "default",
    activation_k: int = 0,
) -> np.ndarray:
    dtype = _uint_dtype(family.act_bits)
    mask = (1 << family.act_bits) - 1
    zp = 1 << (family.act_bits - 1)
    if activation_mode == "zp":
        return np.full((1, 1, m, k), zp, dtype=dtype)
    if activation_mode == "k_impulse":
        arr = np.full((1, 1, m, k), zp, dtype=dtype)
        arr[:, :, :, activation_k % k] = np.array(zp + 1, dtype=dtype)
        return arr
    seed = (idx + 1) * 374761393
    return np.array(
        [((i * 37 + seed) & mask) for i in range(m * k)],
        dtype=dtype,
    ).reshape(1, 1, m, k)


def _make_weight(family: Family, k: int, n: int, limit: int | None = None) -> np.ndarray:
    qmax = (1 << (family.weight_bits - 1)) - 1
    if limit is not None:
        qmax = max(1, min(qmax, int(limit)))
    k_idx, n_idx = np.meshgrid(np.arange(k), np.arange(n), indexing="ij")
    return (((k_idx * 31 + n_idx * 13) % (2 * qmax + 1)) - qmax).astype(np.int32)


def _pack_w8_kmajor(w_raw_kn: np.ndarray) -> np.ndarray:
    """Pack logical [K, N] int8 weights into 32x32 K-major HMX tiles."""
    k, n = w_raw_kn.shape
    if k % 32 or n % 32:
        raise ValueError("W8 HMX pack requires K and N multiples of 32")
    packed = np.zeros(k * n, dtype=np.int8)
    out = 0
    for kt in range(k // 32):
        k_base = kt * 32
        for nt in range(n // 32):
            n_base = nt * 32
            tile = np.zeros(1024, dtype=np.int8)
            for r in range(32):
                for c in range(32):
                    dst = (r // 4) * 128 + c * 4 + (r % 4)
                    tile[dst] = w_raw_kn[k_base + r, n_base + c]
            packed[out:out + 1024] = tile
            out += 1024
    return packed


def _pack_w8_kmajor_split128(w_raw_kn: np.ndarray) -> np.ndarray:
    """Pack W8 as contiguous 128-output-channel K-major HMX streams."""
    k, n = w_raw_kn.shape
    if n % 128:
        raise ValueError("split128 W8 HMX pack requires N multiple of 128")
    parts = []
    for n_base in range(0, n, 128):
        parts.append(_pack_w8_kmajor(w_raw_kn[:, n_base:n_base + 128]))
    return np.concatenate(parts).astype(np.int8, copy=False)


def _pack_w4_native_kpair_linear(nib: np.ndarray, order: str) -> np.ndarray:
    """Pack W4 as QNN native Conv W4 sidecar shape [1, 1, K/2, N]."""
    k, n = nib.shape
    if k % 2:
        raise ValueError("native W4 K-pair pack requires even K")
    lo = nib[0::2, :]
    hi = nib[1::2, :]
    if order == "native_kpair_lohi":
        packed = lo | (hi << 4)
    elif order == "native_kpair_hilo":
        packed = (lo << 4) | hi
    else:
        raise ValueError(f"unsupported native W4 K-pair order: {order}")
    return packed.astype(np.uint8, copy=False)


def _pack_w4_native_nmajor_kpair_linear(nib: np.ndarray, order: str) -> np.ndarray:
    """Pack W4 K-pairs in Conv weight memory order: N-major, then K/2."""
    k, n = nib.shape
    if k % 2:
        raise ValueError("native W4 K-pair pack requires even K")
    lo = nib[0::2, :]
    hi = nib[1::2, :]
    if order == "native_nmajor_kpair_lohi":
        packed_kn = lo | (hi << 4)
    elif order == "native_nmajor_kpair_hilo":
        packed_kn = (lo << 4) | hi
    else:
        raise ValueError(f"unsupported native N-major W4 K-pair order: {order}")
    return packed_kn.T.reshape(k // 2, n).astype(np.uint8, copy=False)


def _pack_w4_native_nmajor_k4_linear(nib: np.ndarray, order: str) -> np.ndarray:
    """Pack native W4 as N32-tiled, then K8-grouped k/k+4 pairs."""
    k, n = nib.shape
    if k % 8 or n % 32:
        raise ValueError("native W4 K4 pack requires K multiple of 8 and N multiple of 32")
    raw = np.zeros((k * n) // 2, dtype=np.uint8)
    out = 0
    for n_base in range(0, n, 32):
        for kg in range(k // 8):
            k_base = kg * 8
            for n_idx in range(n_base, n_base + 32):
                for kr in range(4):
                    lo = nib[k_base + kr, n_idx]
                    hi = nib[k_base + kr + 4, n_idx]
                    if order == "native_nmajor_k4_lohi":
                        raw[out] = lo | (hi << 4)
                    elif order == "native_nmajor_k4_hilo":
                        raw[out] = (lo << 4) | hi
                    else:
                        raise ValueError(f"unsupported native K4 W4 order: {order}")
                    out += 1
    return raw.reshape(k // 2, n)


def _pack_w4_native_kblock32_nmajor_k4_linear(nib: np.ndarray, order: str) -> np.ndarray:
    """Pack native W4 as K32-block-major, then N32-tiled k/k+4 pairs.

    Clean QNN W4A16 Conv sidecars use the same 512-byte tile payload as
    native_nmajor_k4_lohi, but transpose the outer 8x8 grid for 256^3:
    K32 block first, N32 tile second.
    """
    k, n = nib.shape
    if k % 32 or n % 32:
        raise ValueError("native W4 K32-block K4 pack requires K and N multiples of 32")
    raw = np.zeros((k * n) // 2, dtype=np.uint8)
    out = 0
    for kb in range(k // 32):
        for n_base in range(0, n, 32):
            for kg in range(4):
                k_base = kb * 32 + kg * 8
                for n_idx in range(n_base, n_base + 32):
                    for kr in range(4):
                        lo = nib[k_base + kr, n_idx]
                        hi = nib[k_base + kr + 4, n_idx]
                        if order == "native_kblock32_nmajor_k4_lohi":
                            raw[out] = lo | (hi << 4)
                        elif order == "native_kblock32_nmajor_k4_hilo":
                            raw[out] = (lo << 4) | hi
                        else:
                            raise ValueError(f"unsupported native K32-block K4 order: {order}")
                        out += 1
    return raw.reshape(k // 2, n)


def _pack_w4_native_npair_nmajor(nib: np.ndarray, order: str) -> np.ndarray:
    """Pack W4 output-channel pairs in prepared Conv sidecar shape [1,1,N/2,K]."""
    k, n = nib.shape
    if n % 2:
        raise ValueError("native W4 N-pair pack requires even N")
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    if order == "native_npair_adjacent_nmajor_lohi":
        packed_kn2 = lo | (hi << 4)
    elif order == "native_npair_adjacent_nmajor_hilo":
        packed_kn2 = (lo << 4) | hi
    else:
        raise ValueError(f"unsupported native N-pair adjacent order: {order}")
    return packed_kn2.T.astype(np.uint8, copy=False)


def _pack_w4_native_npair_tile32_nmajor(nib: np.ndarray, order: str) -> np.ndarray:
    """Pack W4 output-channel pairs as n,n+32 within each 64-column tile."""
    k, n = nib.shape
    if n % 64:
        raise ValueError("native W4 tile32 N-pair pack requires N multiple of 64")
    packed = np.zeros((n // 2, k), dtype=np.uint8)
    for nb in range(n // 64):
        n_base = nb * 64
        for nc in range(32):
            lo = nib[:, n_base + nc]
            hi = nib[:, n_base + nc + 32]
            if order == "native_npair_tile32_nmajor_lohi":
                value = lo | (hi << 4)
            elif order == "native_npair_tile32_nmajor_hilo":
                value = (lo << 4) | hi
            else:
                raise ValueError(f"unsupported native N-pair tile32 order: {order}")
            packed[nb * 32 + nc, :] = value
    return packed.astype(np.uint8, copy=False)


def _pack_native_a16_bias(family: Family, w_raw_kn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack native A16 bias/control records for 32-column HMX tiles."""
    k, n = w_raw_kn.shape
    if k % 32 or n % 32:
        raise ValueError("native A16 bias pack requires K and N multiples of 32")
    sum_w = w_raw_kn.astype(np.int32).sum(axis=0)
    effective_i32 = (-128 * sum_w).astype(np.int32)
    packed = np.zeros((n // 32, 512), dtype=np.uint8)
    if family.weight_bits == 4:
        const_words = [0x5524, 0x8040, 0x0092, 0x4000]
    elif family.weight_bits == 8:
        const_words = [0x4440, 0x8040, 0x0008, 0x4000]
    else:
        raise ValueError("native A16 bias pack is not decoded for this weight width")
    const = np.array(const_words, dtype=np.uint16).view(np.uint8)
    for nt in range(n // 32):
        for parity in (0, 1):
            half_base = parity * 256
            for lane, c in enumerate(range(parity, 32, 2)):
                col = nt * 32 + c
                lane_base = half_base + 8 * lane
                packed[nt, lane_base:lane_base + 8] = const
                packed[nt, half_base + 128 + 8 * lane:half_base + 132 + 8 * lane] = (
                    np.array([int(effective_i32[col])], dtype=np.int32).view(np.uint8)
                )
    return packed, effective_i32


def _weight_initializer(
    family: Family,
    w_raw_kn: np.ndarray,
    w4_pack_order: str,
    w4_nibble_encoding: str,
    w4_native_sidecar_raw: str | None,
    w8_pack_order: str,
    w8_carrier_dtype: str,
):
    if family.weight_bits == 4:
        if w4_native_sidecar_raw:
            if family.name != "w4a16":
                raise ValueError("--w4-native-sidecar-raw is currently decoded only for w4a16")
            k, n = w_raw_kn.shape
            if k % 2:
                raise ValueError("native W4 sidecar import requires even K")
            raw = np.fromfile(w4_native_sidecar_raw, dtype=np.uint8)
            expected = (k // 2) * n
            if raw.size != expected:
                raise ValueError(
                    f"native W4 sidecar byte count mismatch: got {raw.size}, want {expected}"
                )
            carrier = np.bitwise_xor(raw, np.uint8(0x80))
            return numpy_helper.from_array(
                carrier.view(np.int8).reshape(1, 1, k // 2, n).copy(),
                name="weight",
            )
        if w4_pack_order == "native_full_codes":
            if family.name != "w4a16":
                raise ValueError("native_full_codes is currently decoded only for w4a16")
            if w4_nibble_encoding == "biased":
                codes = (w_raw_kn.astype(np.int32) + 8).astype(np.uint8) & 0x0F
            else:
                codes = (w_raw_kn.astype(np.int8) & 0x0F).astype(np.uint8)
            arr = codes.view(np.int8).reshape(1, 1, w_raw_kn.shape[0], w_raw_kn.shape[1])
            return numpy_helper.from_array(arr, name="weight")

        if (
            w4_pack_order.startswith("native_npair_adjacent_nmajor_") or
            w4_pack_order.startswith("native_npair_tile32_nmajor_")
        ):
            if family.name != "w4a16":
                raise ValueError("native N-pair W4 pack order is currently decoded only for w4a16")
            if w_raw_kn.shape[0] % 32 or w_raw_kn.shape[1] % 64:
                raise ValueError("native W4 N-pair pack requires K multiple of 32 and N multiple of 64")
            if w4_nibble_encoding == "biased":
                nib = (w_raw_kn.astype(np.int32) + 8).astype(np.uint8) & 0x0F
            else:
                nib = (w_raw_kn.astype(np.int8) & 0x0F).astype(np.uint8)
            if w4_pack_order.startswith("native_npair_adjacent_nmajor_"):
                packed = _pack_w4_native_npair_nmajor(nib, w4_pack_order)
            else:
                packed = _pack_w4_native_npair_tile32_nmajor(nib, w4_pack_order)
            carrier = np.bitwise_xor(packed.reshape(-1), np.uint8(0x80))
            arr = carrier.view(np.int8).reshape(1, 1, w_raw_kn.shape[1] // 2, w_raw_kn.shape[0])
            return numpy_helper.from_array(arr, name="weight")

        if (
            w4_pack_order.startswith("native_kpair_") or
            w4_pack_order.startswith("native_nmajor_kpair_") or
            w4_pack_order.startswith("native_nmajor_k4_") or
            w4_pack_order.startswith("native_kblock32_nmajor_k4_")
        ) and family.name != "w4a16":
            raise ValueError("native W4 pack order is currently decoded only for w4a16")
        if (
            w4_pack_order.startswith("native_kpair_") or
            w4_pack_order.startswith("native_nmajor_kpair_") or
            w4_pack_order.startswith("native_nmajor_k4_") or
            w4_pack_order.startswith("native_kblock32_nmajor_k4_")
        ):
            if w_raw_kn.shape[0] % 32 or w_raw_kn.shape[1] % 32:
                raise ValueError("native W4 HMX pack requires K and N multiples of 32")
            if w4_nibble_encoding == "biased":
                nib = (w_raw_kn.astype(np.int32) + 8).astype(np.uint8) & 0x0F
            else:
                nib = (w_raw_kn.astype(np.int8) & 0x0F).astype(np.uint8)
            if w4_pack_order.startswith("native_kblock32_nmajor_k4_"):
                packed = _pack_w4_native_kblock32_nmajor_k4_linear(nib, w4_pack_order)
            elif w4_pack_order.startswith("native_nmajor_k4_"):
                packed = _pack_w4_native_nmajor_k4_linear(nib, w4_pack_order)
            elif w4_pack_order.startswith("native_nmajor_kpair_"):
                packed = _pack_w4_native_nmajor_kpair_linear(nib, w4_pack_order)
            else:
                packed = _pack_w4_native_kpair_linear(nib, w4_pack_order)
            carrier = np.bitwise_xor(packed.reshape(-1), np.uint8(0x80))
            arr = carrier.view(np.int8).reshape(1, 1, w_raw_kn.shape[0] // 2, w_raw_kn.shape[1])
            return numpy_helper.from_array(arr, name="weight")

        # Native HTP stores W4 MatMul/FC weights as an int8 carrier with the N
        # dimension halved.  The VTCM tile is K-major over 32x64 logical tiles.
        # Within each tile, one byte holds output channels n and n+32 for the
        # same K row; adjacent output channels are not paired.
        if w_raw_kn.shape[0] % 32 or w_raw_kn.shape[1] % 64:
            raise ValueError("W4 HMX pack requires K multiple of 32 and N multiple of 64")
        if w4_nibble_encoding == "biased":
            nib = (w_raw_kn.astype(np.int32) + 8).astype(np.uint8) & 0x0F
        else:
            nib = (w_raw_kn.astype(np.int8) & 0x0F).astype(np.uint8)
        k, n = w_raw_kn.shape
        packed = np.zeros(k * n // 2, dtype=np.uint8)
        out = 0
        for kt in range(k // 32):
            k_base = kt * 32
            for nb in range(n // 64):
                n_base = nb * 64
                tile = np.zeros(1024, dtype=np.uint8)
                for kg in range(8):
                    for nc in range(32):
                        for kr in range(4):
                            k_idx = k_base + kg * 4 + kr
                            lo = nib[k_idx, n_base + nc]
                            hi = nib[k_idx, n_base + nc + 32]
                            if w4_pack_order == "lohi":
                                value = lo | (hi << 4)
                            else:
                                value = (lo << 4) | hi
                            tile[kg * 128 + nc * 4 + kr] = value
                packed[out:out + 1024] = tile
                out += 1024
        # The custom-op constant still passes through QNN's generic
        # weights_to_vtcm sidecar before QHPI exposes qhpi_tensor_raw_data().
        # For this INT8 carrier path the sidecar toggles the sign bit, so store
        # the inverse representation in the DLC and let ctxgen/runtime produce
        # the native byte stream above.
        carrier = np.bitwise_xor(packed, np.uint8(0x80))
        arr = carrier.view(np.int8).reshape(1, 1, k, n // 2)
        return numpy_helper.from_array(arr, name="weight")

    if family.weight_bits == 8:
        if family.name == "w8a16":
            if w8_pack_order == "kmajor_split128":
                signed_payload = _pack_w8_kmajor_split128(w_raw_kn)
            elif w8_pack_order == "kmajor":
                # Diagnostic path: try the validated u8i8 V73DEEP W8 tile stream.
                # The default remains raw because QNN's custom-op QUInt8 sidecar
                # does not currently match native signed shflSWeights preparation.
                signed_payload = _pack_w8_kmajor(w_raw_kn)
            else:
                signed_payload = w_raw_kn.astype(np.int8).reshape(-1)
            if w8_carrier_dtype == "int8":
                arr = signed_payload.reshape(1, 1, *w_raw_kn.shape)
            else:
                arr = signed_payload.view(np.uint8).reshape(1, 1, *w_raw_kn.shape)
        else:
            carrier = w_raw_kn.astype(np.int8).view(np.uint8)
            if family.name != "w8a16":
                carrier = np.bitwise_xor(carrier, np.uint8(0x80))
            arr = carrier.reshape(1, 1, *w_raw_kn.shape)
    else:
        arr = w_raw_kn.astype(np.int16).view(np.uint16).reshape(1, 1, *w_raw_kn.shape)
    return numpy_helper.from_array(arr, name="weight")


def _bias_initializer(
    family: Family,
    w_raw_kn: np.ndarray,
    scale_value: float,
    baseline_value: int,
    bias_layout: str,
    bias_fold: str,
) -> tuple:
    n = w_raw_kn.shape[1]
    n_t = n // 32
    act_zp = 1 << (family.act_bits - 1)
    sum_w = w_raw_kn.astype(np.int32).sum(axis=0)
    if bias_layout == "native_a16":
        if not _is_native_a16_family(family):
            raise ValueError("native_a16 bias layout is only valid for A16/U16 families")
        bias_q = np.zeros(n, dtype=np.int32)
        bias_fold_bytes, effective_i32 = _pack_native_a16_bias(family, w_raw_kn)
        init = numpy_helper.from_array(
            bias_fold_bytes.view(np.int32).reshape(1, n_t, 1, 128).copy(),
            name="bias",
        )
        return init, bias_q, effective_i32
    if bias_layout == "native_a16_w4compact":
        if family.name != "w4a16":
            raise ValueError("native_a16_w4compact is currently decoded only for w4a16")
        bias_q = np.zeros(n, dtype=np.int32)
        bias_fold_bytes, effective_i32 = _pack_native_a16_bias(family, w_raw_kn)
        init = numpy_helper.from_array(
            bias_fold_bytes[:, :256].view(np.int32).reshape(1, n_t, 1, 64).copy(),
            name="bias",
        )
        return init, bias_q, effective_i32
    if bias_layout == "native_a16_nobias":
        if family.name != "w4a16":
            raise ValueError("native_a16_nobias is currently decoded only for w4a16")
        bias_q = np.zeros(n, dtype=np.int32)
        effective_i32 = np.zeros(n, dtype=np.int32)
        bias_i32 = np.full((1, n_t, 1, 64), 0x80008000, dtype=np.uint32).view(np.int32)
        init = numpy_helper.from_array(bias_i32.copy(), name="bias")
        return init, bias_q, effective_i32
    if bias_layout == "zero":
        bias_q = np.zeros(n, dtype=np.int32)
        effective_i32 = np.zeros(n, dtype=np.int32)
    else:
        bias_q = np.arange(1, n + 1, dtype=np.int32)
    if bias_layout == "zero":
        pass
    elif bias_fold == "bias_only":
        effective_i32 = bias_q.copy()
    else:
        effective_i32 = (-act_zp) * sum_w + bias_q

    # Native A16 Conv1x1 paths prepare one 512B bias/control record per
    # 32-output-channel tile.  The HTP graph exposes this as
    # Int32 [1, N/32, 1, 128].
    bias_tile_bytes = 512 if _is_native_a16_family(family) else 256
    bias_fold_bytes = np.zeros((n_t, bias_tile_bytes), dtype=np.uint8)
    if bias_layout == "zero":
        bias_i32 = bias_fold_bytes.view(np.int32)
        bias_shape = (
            (1, n_t, 1, 128)
            if _is_native_a16_family(family)
            else (1, n_t, 1, bias_tile_bytes // 4)
        )
        init = numpy_helper.from_array(
            bias_i32.reshape(*bias_shape).copy(),
            name="bias",
        )
        return init, bias_q, effective_i32
    for nt in range(n_t):
        for c in range(32):
            col = nt * 32 + c
            scale_u16 = np.float16(scale_value).view(np.uint16).item()
            baseline_u16 = int(baseline_value) & 0xFFFF
            if bias_layout == "a16_planes":
                if family.name != "w8a16":
                    raise ValueError("a16_planes bias layout is only valid for w8a16")
                for base in (0, 128):
                    bias_fold_bytes[nt, base + 4 * c:base + 4 * c + 2] = (
                        np.array([scale_u16], np.uint16).view(np.uint8)
                    )
                    bias_fold_bytes[nt, base + 4 * c + 2:base + 4 * c + 4] = (
                        np.array([baseline_u16], np.uint16).view(np.uint8)
                    )
                for base in (256, 384):
                    bias_fold_bytes[nt, base + 4 * c:base + 4 * c + 4] = (
                        np.array([int(effective_i32[col])], np.int32).view(np.uint8)
                    )
                continue
            if bias_layout == "a16_eff_all":
                if family.name != "w8a16":
                    raise ValueError("a16_eff_all bias layout is only valid for w8a16")
                for base in (0, 128, 256, 384):
                    bias_fold_bytes[nt, base + 4 * c:base + 4 * c + 4] = (
                        np.array([int(effective_i32[col])], np.int32).view(np.uint8)
                    )
                continue

            for base in range(0, bias_tile_bytes, 256):
                if bias_layout == "swapped":
                    bias_fold_bytes[nt, base + 4 * c:base + 4 * c + 4] = (
                        np.array([int(effective_i32[col])], np.int32).view(np.uint8)
                    )
                    bias_fold_bytes[nt, base + 128 + 4 * c:base + 128 + 4 * c + 2] = (
                        np.array([scale_u16], np.uint16).view(np.uint8)
                    )
                    bias_fold_bytes[nt, base + 128 + 4 * c + 2:base + 128 + 4 * c + 4] = (
                        np.array([baseline_u16], np.uint16).view(np.uint8)
                    )
                else:
                    bias_fold_bytes[nt, base + 4 * c:base + 4 * c + 2] = (
                        np.array([scale_u16], np.uint16).view(np.uint8)
                    )
                    bias_fold_bytes[nt, base + 4 * c + 2:base + 4 * c + 4] = (
                        np.array([baseline_u16], np.uint16).view(np.uint8)
                    )
                    bias_fold_bytes[nt, base + 128 + 4 * c:base + 128 + 4 * c + 4] = (
                        np.array([int(effective_i32[col])], np.int32).view(np.uint8)
                    )
    bias_i32 = bias_fold_bytes.view(np.int32)
    if _is_native_a16_family(family):
        # Native A16 lowers bias as one 512B record per N tile:
        # chrometrace reports Int32 dims [1, N/32, 1, 128].  Keep that shape so
        # QNN's Flat4 direct layout matches the native prepared payload.
        bias_shape = (1, n_t, 1, 128)
    else:
        bias_shape = (1, n_t, 1, bias_tile_bytes // 4)
    init = numpy_helper.from_array(
        bias_i32.reshape(*bias_shape).copy(),
        name="bias",
    )
    return init, bias_q, effective_i32


def _make_reference(family: Family, a_raw: np.ndarray, w_raw_kn: np.ndarray, bias_q: np.ndarray) -> np.ndarray:
    act_zp = 1 << (family.act_bits - 1)
    max_out = (1 << family.out_bits) - 1
    a = a_raw.reshape(w_raw_kn.shape[0], w_raw_kn.shape[0]).astype(np.int32) - act_zp
    out = a @ w_raw_kn.astype(np.int32) + bias_q
    return np.clip(out, 0, max_out).astype(_uint_dtype(family.out_bits))


def _make_reference_step(
    family: Family,
    cur: np.ndarray,
    w_raw_kn: np.ndarray,
    bias_q: np.ndarray,
    reference_contract: str,
) -> np.ndarray:
    max_out = (1 << family.out_bits) - 1
    if reference_contract == "native":
        if not _is_native_a16_family(family):
            raise ValueError("native reference contract is only valid for A16/U16 families")
        qmax = (1 << (family.weight_bits - 1)) - 1
        if qmax <= 0:
            raise ValueError("native reference contract requires signed weights")
        if family.name == "w4a16":
            acc = (cur.astype(np.int64) - 32768) @ w_raw_kn.astype(np.int64)
            acc = np.floor_divide(acc, 256) * 256
            y = acc.astype(np.float64) / float(qmax)
            return np.clip(np.rint(y + 32768.0), 0, max_out).astype(_uint_dtype(family.out_bits))
        y = ((cur.astype(np.float64) - 32768.0) @ w_raw_kn.astype(np.float64)) / float(qmax)
        return np.clip(np.rint(y + 32768.0), 0, max_out).astype(_uint_dtype(family.out_bits))
    return np.clip(
        (cur.astype(np.int32) - (1 << (family.act_bits - 1))) @ w_raw_kn.astype(np.int32) + bias_q,
        0,
        max_out,
    ).astype(_uint_dtype(family.out_bits))


def generate(family: Family, args: argparse.Namespace) -> None:
    assert args.M % 32 == 0 and args.K % 32 == 0 and args.N % 32 == 0
    if args.chain != 1 and not (args.M == args.K == args.N):
        raise ValueError("non-square M/K/N is currently supported only for chain=1")

    m, k, n = args.M, args.K, args.N
    w4a8_native_tiled_layout = family.name == "w4a8" and args.op_input_layout == "native"
    row_tile = 32 if w4a8_native_tiled_layout or family.name != "w4a8" else 16
    assert m % row_tile == 0
    row_groups = m // row_tile
    chain = max(1, int(args.chain))
    w4_native_conv_surface_layout = family.name == "w4a16" and args.op_input_layout == "native_conv_surface"
    w16_native_layout = family.name == "w16a16" and args.op_input_layout in ("native", "native_split")
    w16_native_split_layout = family.name == "w16a16" and args.op_input_layout == "native_split"
    native_act_layout = (
        family.name == "w8a16" and args.op_input_layout in (
            "native",
            "native_split",
            "native_in_tiled_out",
            "native_in_tiled_out_split",
            "native_in_row4_out",
        )
    ) or (family.name == "w4a16" and args.op_input_layout == "native") or w4_native_conv_surface_layout or w16_native_layout
    native_out_layout = (
        family.name == "w8a16" and args.op_input_layout in ("native", "native_split")
    ) or (family.name == "w4a16" and args.op_input_layout == "native") or w4_native_conv_surface_layout or w16_native_layout
    native_op_layout = native_act_layout or native_out_layout
    native_split_layout = (
        family.name == "w8a16" and args.op_input_layout in (
            "native_split",
            "native_in_tiled_out_split",
        )
    ) or w16_native_split_layout
    w4_native_conv_input_layout = family.name == "w4a16" and args.op_input_layout == "native_conv"
    if args.op_input_layout == "native_conv" and family.name != "w4a16":
        raise ValueError("native_conv input layout is currently a w4a16-only diagnostic")
    if family.name == "w4a8" and args.op_input_layout not in ("tiled", "native"):
        raise ValueError("w4a8 supports only tiled and native op input layouts")
    if family.name == "w4a16" and args.op_input_layout not in ("tiled", "native", "native_conv", "native_conv_surface"):
        raise ValueError("w4a16 supports only tiled, native, native_conv, and native_conv_surface op input layouts")
    if native_split_layout and chain != 1:
        raise ValueError("native_split currently supports chain=1 only")
    if native_split_layout and args.mode not in ("chain", "chain_float", "chain_qdq"):
        raise ValueError("native_split currently supports chain, chain_float, and chain_qdq modes only")
    if native_split_layout and n % 2:
        raise ValueError("native_split requires even N")
    if args.native_split_output_mode == "separate" and not native_split_layout:
        raise ValueError("--native-split-output-mode=separate requires a native split layout")
    if w4_native_conv_input_layout and args.mode != "chain_qdq":
        raise ValueError("w4a16 native_conv input layout currently supports chain_qdq only")
    graph_act_shape = (
        [1, k, 1, m] if w4_native_conv_input_layout else
        ([1, m, 1, k] if w4_native_conv_surface_layout else [1, 1, m, k])
    )
    op_act_shape = (
        [1, m, 1, k] if w4_native_conv_surface_layout else
        ([1, 1, m, k] if native_act_layout else [1, row_groups, row_tile, k])
    )
    if w4_native_conv_surface_layout:
        op_out_shape = [1, m, 1, n]
    elif native_out_layout:
        op_out_shape = [1, 1, m, n]
    elif family.name == "w8a16" and args.op_input_layout == "native_in_row4_out":
        op_out_shape = [1, m // 4, 4, n]
    else:
        op_out_shape = [1, row_groups, row_tile, n]
    np.random.seed(args.seed)

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out if os.path.isabs(args.out) else os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    w_raw_kn = _make_weight(family, k, n, args.weight_limit)
    reference_contract = args.reference_contract
    if reference_contract == "auto":
        reference_contract = "native" if _is_native_a16_family(family) and args.a16_quant_contract == "native" else "legacy"
    split_count = 2 if native_split_layout else 1
    split_n = n // split_count
    wt_inits = []
    bias_inits = []
    bias_q_parts = []
    effective_i32_parts = []
    for split_idx in range(split_count):
        col0 = split_idx * split_n
        col1 = col0 + split_n
        w_part = w_raw_kn if w16_native_split_layout else w_raw_kn[:, col0:col1]
        bias_w_part = w_raw_kn[:, col0:col1]
        if family.name == "w16a16" and args.w16_weight_carrier_dtype in ("int8", "uint8"):
            w16_payload_shape = w_part.shape
            if args.w16_weight_sidecar_raw:
                raw = np.fromfile(args.w16_weight_sidecar_raw, dtype=np.uint8)
                expected = int(np.prod(w_part.shape))
                if raw.size == expected * split_count:
                    raw = raw[split_idx * expected:(split_idx + 1) * expected]
                elif split_count == 1 and raw.size == expected * 2:
                    # Diagnostic import for the tiled public surface with
                    # HMX_W16A16_INTERNAL_SPLIT_N128.  QNN's native lowering
                    # stores two full 64K QInt8 sidecars for N[0:128] and
                    # N[128:256]; the wrapper derives N from the output/bias
                    # contract, so the weight tensor can over-carry both raw
                    # sidecars without changing the public output shape.
                    w16_payload_shape = (w_part.shape[0], w_part.shape[1] * 2)
                if raw.size != expected:
                    if not (split_count == 1 and raw.size == expected * 2):
                        raise ValueError(
                            f"W16 sidecar raw byte count mismatch: got {raw.size}, want {expected}"
                        )
                signed_w16_payload = raw.view(np.int8).reshape(w16_payload_shape)
            elif args.w16_weight_pack_order == "clip":
                signed_w16_payload = np.clip(w_part, -128, 127).astype(np.int8)
            elif args.w16_weight_pack_order == "hi8":
                signed_w16_payload = (w_part.astype(np.int32) >> 8).astype(np.int8)
            elif args.w16_weight_pack_order == "lo8":
                signed_w16_payload = w_part.astype(np.int8)
            else:
                raise ValueError(f"unsupported W16 8-bit pack order: {args.w16_weight_pack_order}")
            if args.w16_weight_carrier_dtype == "uint8":
                signed_w16_payload = signed_w16_payload.view(np.uint8)
            wt_part = numpy_helper.from_array(
                signed_w16_payload.reshape(1, 1, *w16_payload_shape),
                name="weight",
            )
        else:
            wt_part = _weight_initializer(
                family,
                w_part,
                args.w4_pack_order,
                args.w4_nibble_encoding,
                args.w4_native_sidecar_raw,
                args.w8_pack_order,
                args.w8_carrier_dtype,
            )
        bias_part, bias_q_part, effective_i32_part = _bias_initializer(
            family,
            bias_w_part,
            args.bias_scale,
            args.bias_baseline,
            args.bias_layout,
            args.bias_fold,
        )
        if family.name == "w16a16" and args.w16_bias_sidecar_raw:
            raw = np.fromfile(args.w16_bias_sidecar_raw, dtype=np.uint8)
            bias_shape = tuple(int(dim) for dim in bias_part.dims)
            expected = int(np.prod(bias_shape)) * np.dtype(np.int32).itemsize
            if raw.size == expected * split_count:
                raw = raw[split_idx * expected:(split_idx + 1) * expected]
            if raw.size != expected:
                raise ValueError(
                    f"W16 bias sidecar raw byte count mismatch: got {raw.size}, want {expected}"
                )
            bias_payload = raw.view(np.int32).reshape(bias_shape)
            bias_part = numpy_helper.from_array(bias_payload.copy(), name=bias_part.name)
        if native_split_layout:
            wt_part.name = f"weight_{split_idx}"
            bias_part.name = f"bias_{split_idx}"
        wt_inits.append(wt_part)
        bias_inits.append(bias_part)
        bias_q_parts.append(bias_q_part)
        effective_i32_parts.append(effective_i32_part)
    bias_q = np.concatenate(bias_q_parts).astype(np.int32)
    effective_i32 = np.concatenate(effective_i32_parts).astype(np.int32)
    if family.name == "w4a16":
        scratch_input_name = "control"
        scratch_init = numpy_helper.from_array(np.array([1], dtype=np.int32), name=scratch_input_name)
    elif family.name == "w16a16" and args.w16_scratch_sidecar_raw:
        raw = np.fromfile(args.w16_scratch_sidecar_raw, dtype=np.uint8)
        scratch_input_name = "scratch"
        scratch_init = numpy_helper.from_array(
            raw.reshape(1, 1, 1, raw.size).copy(),
            name=scratch_input_name,
        )
    else:
        scratch_input_name = "scratch"
        scratch_init = numpy_helper.from_array(
            np.zeros((1, 1, 1, 2048), dtype=np.uint8),
            name=scratch_input_name,
        )
    in_reshape_dims = numpy_helper.from_array(np.array(op_act_shape, dtype=np.int64), name="in_reshape_dims")
    if w4_native_conv_surface_layout and args.final_output_rank == "4d":
        graph_out_shape = [1, m, 1, n]
    else:
        graph_out_shape = [1, m, n] if args.final_output_rank == "3d" else [1, 1, m, n]
    out_reshape_dims = numpy_helper.from_array(np.array(graph_out_shape, dtype=np.int64), name="out_reshape_dims")

    act_tp = _uint_tensorproto(family.act_bits)
    out_tp = _uint_tensorproto(family.out_bits)
    initializers = [*wt_inits, *bias_inits, scratch_init, in_reshape_dims, out_reshape_dims]
    if args.mode == "chain_qdq":
        act_q_scale = 1.0
        if _is_native_a16_family(family) and args.a16_quant_contract == "native":
            act_q_scale = 1.0 / 32767.0
        initializers.extend([
            numpy_helper.from_array(np.array([act_q_scale], dtype=np.float32), name="act_q_scale"),
            numpy_helper.from_array(np.array([1 << (family.act_bits - 1)], dtype=_uint_dtype(family.act_bits)), name="act_q_zp"),
        ])
    inputs_info = []
    outputs_info = []
    value_infos = []
    nodes = []

    if args.mode in ("chain", "chain_float", "chain_qdq"):
        graph_act_tp = TensorProto.FLOAT if args.mode in ("chain_float", "chain_qdq") else act_tp
        graph_out_tp = out_tp
        inputs_info.append(helper.make_tensor_value_info("act_raw", graph_act_tp, graph_act_shape))
        if native_split_layout and args.native_split_output_mode == "separate":
            split_graph_out_shape = (
                [1, m, split_n] if args.final_output_rank == "3d" else
                ([1, 1, m, split_n] if native_out_layout else [1, row_groups, row_tile, split_n])
            )
            for split_idx in range(split_count):
                outputs_info.append(helper.make_tensor_value_info(
                    f"out_part{split_idx}", graph_out_tp, split_graph_out_shape))
        else:
            outputs_info.append(helper.make_tensor_value_info("out", graph_out_tp, graph_out_shape))
        reshape_input = "act_raw"
        if args.mode == "chain_qdq":
            nodes.append(helper.make_node("QuantizeLinear", ["act_raw", "act_q_scale", "act_q_zp"], ["act_q"], name="quantize_act"))
            value_infos.append(helper.make_tensor_value_info("act_q", act_tp, graph_act_shape))
            reshape_input = "act_q"
        if w4_native_conv_input_layout:
            nodes.append(helper.make_node(
                "Transpose",
                [reshape_input],
                ["act_conv_nhwc"],
                name="transpose_native_conv_act",
                perm=[0, 2, 3, 1],
            ))
            value_infos.append(helper.make_tensor_value_info("act_conv_nhwc", act_tp, [1, 1, m, k]))
            reshape_input = "act_conv_nhwc"
        if native_act_layout:
            prev = reshape_input
        else:
            nodes.append(helper.make_node("Reshape", [reshape_input, "in_reshape_dims"], ["act_4d"], name="reshape_in"))
            prev = "act_4d"
        if native_split_layout:
            split_outputs = []
            split_shape = [1, 1, m, split_n] if native_out_layout else [1, row_groups, row_tile, split_n]
            for split_idx in range(split_count):
                out_name = f"hmx_{family.name}_part{split_idx}"
                nodes.append(helper.make_node(
                    family.op,
                    inputs=[f"bias_{split_idx}", f"weight_{split_idx}", prev, scratch_input_name],
                    outputs=[out_name],
                    name=f"hmx_{family.name}_split{split_idx}",
                    domain=DOMAIN,
                ))
                value_infos.append(helper.make_tensor_value_info(out_name, graph_out_tp, split_shape))
                split_outputs.append(out_name)
            if args.native_split_output_mode == "separate":
                if native_out_layout and args.final_output_rank == "3d":
                    for split_idx, split_output in enumerate(split_outputs):
                        split_reshape_dims = f"out_part{split_idx}_reshape_dims"
                        initializers.append(numpy_helper.from_array(
                            np.array([1, m, split_n], dtype=np.int64),
                            name=split_reshape_dims,
                        ))
                        nodes.append(helper.make_node(
                            "Reshape",
                            [split_output, split_reshape_dims],
                            [f"out_part{split_idx}"],
                            name=f"reshape_out_part{split_idx}",
                        ))
                else:
                    for split_idx, split_output in enumerate(split_outputs):
                        nodes.append(helper.make_node(
                            "Identity",
                            [split_output],
                            [f"out_part{split_idx}"],
                            name=f"export_out_part{split_idx}",
                        ))
            else:
                concat_out = "out" if native_out_layout and args.final_output_rank == "4d" else "hmx_concat"
                nodes.append(helper.make_node("Concat", split_outputs, [concat_out], name="concat_out", axis=3))
                if concat_out != "out":
                    concat_shape = [1, 1, m, n] if native_out_layout else op_out_shape
                    value_infos.append(helper.make_tensor_value_info("hmx_concat", graph_out_tp, concat_shape))
                    nodes.append(helper.make_node("Reshape", ["hmx_concat", "out_reshape_dims"], ["out"], name="reshape_out"))
        else:
            for i in range(chain):
                if i < chain - 1:
                    out_name = f"hmx_{family.name}_{i}"
                else:
                    out_name = f"hmx_{family.name}"
                nodes.append(helper.make_node(
                    family.op,
                    inputs=["bias", "weight", prev, scratch_input_name],
                    outputs=[out_name],
                    name=f"hmx_{family.name}_chain{i}",
                    domain=DOMAIN,
                ))
                value_infos.append(helper.make_tensor_value_info(out_name, graph_out_tp, op_out_shape))
                prev = out_name
            nodes.append(helper.make_node("Reshape", [f"hmx_{family.name}", "out_reshape_dims"], ["out"], name="reshape_out"))
    elif args.mode in ("direct", "direct_flat"):
        inputs_info.append(helper.make_tensor_value_info("act_raw", act_tp, op_act_shape))
        if args.mode == "direct_flat" or native_out_layout:
            outputs_info.append(helper.make_tensor_value_info("out", out_tp, graph_out_shape))
        else:
            outputs_info.append(helper.make_tensor_value_info("out", out_tp, op_out_shape))
        prev = "act_raw"
        direct_needs_reshape = (args.mode == "direct_flat" and not native_out_layout) or (
            native_out_layout and args.final_output_rank == "3d"
        )
        for i in range(chain):
            if direct_needs_reshape:
                out_name = f"hmx_{family.name}_{i}" if i < chain - 1 else f"hmx_{family.name}"
            else:
                out_name = f"hmx_{family.name}_{i}" if i < chain - 1 else "out"
            nodes.append(helper.make_node(
                family.op,
                inputs=["bias", "weight", prev, scratch_input_name],
                outputs=[out_name],
                name=f"hmx_{family.name}_direct{i}",
                domain=DOMAIN,
            ))
            if i < chain - 1:
                value_infos.append(helper.make_tensor_value_info(out_name, out_tp, op_out_shape))
            prev = out_name
        if direct_needs_reshape:
            value_infos.append(helper.make_tensor_value_info(f"hmx_{family.name}", out_tp, op_out_shape))
            nodes.append(helper.make_node("Reshape", [f"hmx_{family.name}", "out_reshape_dims"], ["out"], name="reshape_out"))
    else:
        for i in range(chain):
            in_name = "act_raw" if i == 0 else f"act_raw_{i}"
            out_name = f"out_{i}"
            act4d_name = f"act_4d_{i}"
            mm_name = f"hmx_{family.name}_{i}"
            in_shape_name = "in_reshape_dims" if i == 0 else f"in_reshape_dims_{i}"
            out_shape_name = "out_reshape_dims" if i == 0 else f"out_reshape_dims_{i}"
            if i > 0:
                initializers.append(numpy_helper.from_array(np.array([1, row_groups, row_tile, k], dtype=np.int64), name=in_shape_name))
                initializers.append(numpy_helper.from_array(np.array(graph_out_shape, dtype=np.int64), name=out_shape_name))
            inputs_info.append(helper.make_tensor_value_info(in_name, act_tp, [1, 1, m, k]))
            outputs_info.append(helper.make_tensor_value_info(out_name, out_tp, graph_out_shape))
            nodes.append(helper.make_node("Reshape", [in_name, in_shape_name], [act4d_name], name=f"reshape_in_{i}"))
            nodes.append(helper.make_node(
                family.op,
                inputs=["bias", "weight", act4d_name, scratch_input_name],
                outputs=[mm_name],
                name=f"hmx_{family.name}_indep{i}",
                domain=DOMAIN,
            ))
            value_infos.append(helper.make_tensor_value_info(mm_name, out_tp, [1, row_groups, row_tile, n]))
            nodes.append(helper.make_node("Reshape", [mm_name, out_shape_name], [out_name], name=f"reshape_out_{i}"))

    graph = helper.make_graph(
        nodes,
        name=f"custom_{family.name}_{args.mode}",
        inputs=inputs_info,
        outputs=outputs_info,
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name=f"custom_{family.name}_{m}x{k}x{n}",
        opset_imports=[helper.make_opsetid("", 13), helper.make_opsetid(DOMAIN, 1)],
    )
    model.ir_version = 8
    onnx.save(model, out_path)

    if _is_native_a16_family(family) and args.a16_quant_contract == "native":
        act_enc = _native_a16_encoding()
        out_enc = _native_a16_encoding()
    else:
        act_enc = _activation_encoding(family.act_bits)
        out_enc = _activation_encoding(family.out_bits)
    bias_enc = _symmetric_encoding(32)
    if args.mode in ("chain", "chain_float", "chain_qdq", "direct", "direct_flat"):
        activation_encodings = {"act_raw": [act_enc]}
        if native_split_layout and args.native_split_output_mode == "separate":
            for split_idx in range(split_count):
                activation_encodings[f"out_part{split_idx}"] = [out_enc]
        else:
            activation_encodings["out"] = [out_enc]
        if args.mode in ("chain", "chain_float", "chain_qdq"):
            if not native_act_layout:
                activation_encodings["act_4d"] = [act_enc]
            if args.mode == "chain_qdq":
                activation_encodings["act_q"] = [act_enc]
            if w4_native_conv_input_layout:
                activation_encodings["act_conv_nhwc"] = [act_enc]
            if native_split_layout:
                for split_idx in range(split_count):
                    activation_encodings[f"hmx_{family.name}_part{split_idx}"] = [out_enc]
            else:
                for i in range(chain):
                    if i < chain - 1:
                        out_name = f"hmx_{family.name}_{i}"
                    else:
                        out_name = f"hmx_{family.name}"
                    activation_encodings[out_name] = [out_enc]
    else:
        activation_encodings = {}
        for i in range(chain):
            activation_encodings["act_raw" if i == 0 else f"act_raw_{i}"] = [act_enc]
            activation_encodings[f"out_{i}"] = [out_enc]

    if native_split_layout:
        param_encodings = {f"bias_{split_idx}": [bias_enc] for split_idx in range(split_count)}
    else:
        param_encodings = {"bias": [bias_enc]}
    use_lpbq_v1_quant_overrides = False
    if family.weight_bits == 4:
        if args.w4_encoding == "lpbq":
            if split_count != 1:
                raise ValueError("W4 LPBQ custom flow currently supports only unsplit weights")
            weight_dims = tuple(int(dim) for dim in wt_inits[0].dims)
            param_encodings["weight"] = [
                _w4_lpbq_encoding("weight", weight_dims, args.w4_lpbq_block_size)
            ]
            use_lpbq_v1_quant_overrides = True
        else:
            param_encodings["weight"] = [_symmetric_encoding(4)]
    if family.name == "w16a16" and args.w16_weight_carrier_dtype == "int8":
        if native_split_layout:
            for split_idx in range(split_count):
                param_encodings[f"weight_{split_idx}"] = [_symmetric_encoding(8)]
        else:
            param_encodings["weight"] = [_symmetric_encoding(8)]

    overrides = _make_quant_overrides(
        activation_encodings,
        param_encodings,
        lpbq_v1=use_lpbq_v1_quant_overrides,
    )
    with open(os.path.join(out_dir, "quant_overrides.json"), "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2)

    runtime_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "runtime_inputs_u8")
    os.makedirs(runtime_dir, exist_ok=True)
    runtime_input_list = os.path.join(out_dir, "runtime_input_list.txt")
    native_input_name = f"act_{family.name}.raw"
    native_input_rel = f"runtime_inputs_u8/{native_input_name}"
    native_input_storage = _uint_storage(family.act_bits)
    if args.mode in ("chain", "chain_float", "chain_qdq", "direct_flat"):
        a0 = _make_activation(family, m, k, 0, args.activation_mode, args.activation_k)
        if w4_native_conv_input_layout:
            act_nchw = a0.reshape(m, k).T.reshape(1, k, 1, m)
            act_nchw.tofile(os.path.join(runtime_dir, native_input_name))
        elif args.mode == "chain_float":
            native_input_storage = "float32_le"
            a0.astype(np.float32).tofile(os.path.join(runtime_dir, native_input_name))
        else:
            a0.tofile(os.path.join(runtime_dir, native_input_name))
        with open(runtime_input_list, "w", encoding="utf-8") as f:
            f.write(f"act_raw:={native_input_rel}\n")
        cur = a0.reshape(m, k).astype(np.int32)
        for _ in range(chain):
            cur = _make_reference_step(family, cur, w_raw_kn, bias_q, reference_contract)
        out_ref = cur
    else:
        input_parts = []
        for i in range(chain):
            a_i = _make_activation(family, m, k, i, args.activation_mode, args.activation_k)
            fname = f"act_{family.name}.raw" if i == 0 else f"act_{family.name}_{i}.raw"
            a_i.tofile(os.path.join(runtime_dir, fname))
            name = "act_raw" if i == 0 else f"act_raw_{i}"
            input_parts.append(f"{name}:=runtime_inputs_u8/{fname}")
        with open(runtime_input_list, "w", encoding="utf-8") as f:
            f.write(" ".join(input_parts) + "\n")
        a0 = _make_activation(family, m, k, 0, args.activation_mode, args.activation_k).reshape(m, k)
        if reference_contract == "legacy":
            out_ref = _make_reference(family, a0, w_raw_kn, bias_q)
        else:
            out_ref = _make_reference_step(family, a0, w_raw_kn, bias_q, reference_contract)

    with open(os.path.join(out_dir, "native_io.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_name": "act_raw" if args.mode != "independent" else [
                    "act_raw" if i == 0 else f"act_raw_{i}" for i in range(chain)
                ],
                "output_name": (
                    [f"out_part{i}" for i in range(split_count)]
                    if native_split_layout and args.native_split_output_mode == "separate"
                    else ("out" if args.mode != "independent" else [f"out_{i}" for i in range(chain)])
                ),
                "native_input": native_input_rel,
                "runtime_input_list": "runtime_input_list.txt",
                "native_input_storage": native_input_storage,
                "native_input_bytes": int(os.path.getsize(os.path.join(out_dir, native_input_rel))),
                "expected_native_output_storage": _uint_storage(family.out_bits),
                "expected_native_output_bytes": int(out_ref.size * np.dtype(_uint_dtype(family.out_bits)).itemsize),
                "shape_mkn": [m, k, n],
                "chain": chain,
                "graph_input_shape": graph_act_shape,
                "graph_output_shape": graph_out_shape,
                "op_input_layout": args.op_input_layout,
                "a16_quant_contract": args.a16_quant_contract,
                "reference_contract": reference_contract,
            },
            f,
            indent=2,
        )

    np.save(out_path + ".wRaw_KN.npy", w_raw_kn)
    np.save(out_path + ".bias_q_int32.npy", bias_q)
    np.save(out_path + ".effective_int32.npy", effective_i32)
    np.save(out_path + f".out_ref_u{family.out_bits}.npy", out_ref)

    print(f"  -> {out_path}")
    print(f"  graph: {family.op} x {chain} ({args.mode})")
    print(f"  shape: M={m} K={k} N={n}; W{family.weight_bits} A{family.act_bits}; weight encodings are carried by the native packer")
    print(f"  ref out[0..3,0]: {out_ref[:4, 0].tolist()}")


def main(family_name: str) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=256)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--seed", type=int, default=0xB17E)
    p.add_argument("--chain", type=int, default=8)
    p.add_argument("--mode", choices=["chain", "chain_float", "chain_qdq", "direct", "direct_flat", "independent"], default="chain")
    p.add_argument("--bias-scale", type=float, default=512.0)
    p.add_argument("--bias-baseline", type=int, default=0)
    p.add_argument(
        "--bias-layout",
        choices=[
            "legacy",
            "swapped",
            "a16_planes",
            "a16_eff_all",
            "zero",
            "native_a16",
            "native_a16_w4compact",
            "native_a16_nobias",
        ],
        default="legacy",
    )
    p.add_argument("--bias-fold", choices=["zero_point", "bias_only"], default="zero_point")
    p.add_argument("--weight-limit", type=int, default=None)
    p.add_argument("--activation-mode", choices=["default", "zp", "k_impulse"], default="default")
    p.add_argument("--activation-k", type=int, default=0)
    p.add_argument(
        "--w4-pack-order",
        choices=[
            "lohi",
            "hilo",
            "native_kpair_lohi",
            "native_kpair_hilo",
            "native_nmajor_kpair_lohi",
            "native_nmajor_kpair_hilo",
            "native_nmajor_k4_lohi",
            "native_nmajor_k4_hilo",
            "native_kblock32_nmajor_k4_lohi",
            "native_kblock32_nmajor_k4_hilo",
            "native_full_codes",
            "native_npair_adjacent_nmajor_lohi",
            "native_npair_adjacent_nmajor_hilo",
            "native_npair_tile32_nmajor_lohi",
            "native_npair_tile32_nmajor_hilo",
        ],
        default="lohi",
    )
    p.add_argument("--w4-nibble-encoding", choices=["twos", "biased"], default="twos")
    p.add_argument(
        "--w4-encoding",
        choices=["symmetric", "lpbq"],
        default="symmetric",
        help="weight encoding metadata: legacy int4 symmetric or QAIRT v1 LPBQ",
    )
    p.add_argument(
        "--w4-lpbq-block-size",
        type=int,
        default=32,
        help="packed weight axis-2 block size for QAIRT v1 LPBQ overrides",
    )
    p.add_argument(
        "--w4-native-sidecar-raw",
        default=None,
        help="import a prepared native W4 sidecar byte stream for w4a16 diagnostics",
    )
    p.add_argument("--w8-pack-order", choices=["raw", "kmajor", "kmajor_split128"], default="raw")
    p.add_argument("--w8-carrier-dtype", choices=["uint8", "int8"], default="uint8")
    p.add_argument("--w16-weight-carrier-dtype", choices=["uint16", "int8", "uint8"], default="uint16")
    p.add_argument("--w16-weight-pack-order", choices=["clip", "hi8", "lo8"], default="clip")
    p.add_argument("--w16-weight-sidecar-raw", default=None)
    p.add_argument("--w16-bias-sidecar-raw", default=None)
    p.add_argument("--w16-scratch-sidecar-raw", default=None)
    p.add_argument("--a16-quant-contract", choices=["legacy", "native"], default="legacy")
    p.add_argument("--reference-contract", choices=["auto", "legacy", "native"], default="auto")
    p.add_argument("--final-output-rank", choices=["4d", "3d"], default="4d")
    p.add_argument("--native-split-output-mode", choices=["concat", "separate"], default="concat")
    p.add_argument(
        "--op-input-layout",
        choices=[
            "tiled",
            "native",
            "native_split",
            "native_in_tiled_out",
            "native_in_tiled_out_split",
            "native_in_row4_out",
            "native_conv",
            "native_conv_surface",
        ],
        default="tiled",
    )
    p.add_argument("-o", "--out", default=f"{family_name}_chain.onnx")
    args = p.parse_args()
    generate(FAMILIES[family_name], args)
