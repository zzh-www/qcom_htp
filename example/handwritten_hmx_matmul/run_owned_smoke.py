#!/usr/bin/env python3
"""Generate and run an owned-runtime smoke artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import prepare_owned_inputs


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent
FORBIDDEN = (
    "qnn-context-binary-generator",
    "qnn-net-run",
    "qairt-converter",
    "qairt-quantizer",
    "QnnHmxMatMul",
)


def load_oracles() -> dict:
    with (EXAMPLE / "oracles.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def first_mismatch(left: bytes, right: bytes) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def write_owned_output_compare(family: str, out_dir: Path, oracle: dict) -> None:
    output_path = out_dir / "device_out" / "out.raw"
    native_path = ROOT / oracle["raw_output"]["path"]
    run_path = out_dir / "owned_run.json"
    run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    output = output_path.read_bytes() if output_path.is_file() else b""
    native = native_path.read_bytes() if native_path.is_file() else b""
    byte_diffs = sum(1 for a, b in zip(output, native) if a != b)
    byte_diffs += abs(len(output) - len(native))
    payload = {
        "schema": "handwritten_hmx_matmul_owned_output_compare.v1",
        "family": family,
        "qnn_used": False,
        "acceptance_role": "diagnostic_copy_smoke_not_milestone4_acceptance",
        "runtime_kind": run.get("runtime_kind"),
        "device_execution": run.get("device_execution"),
        "compute_backend": run.get("compute_backend"),
        "hmx_body_entered": run.get("hmx_body_entered"),
        "accepted_for_milestone4_compute_gate": run.get("accepted_for_milestone4_compute_gate"),
        "output": {
            "path": rel(output_path),
            "bytes": len(output),
            "sha256": sha256_bytes(output) if output else None,
        },
        "native_raw": {
            "path": rel(native_path),
            "bytes": len(native),
            "sha256": sha256_bytes(native) if native else None,
        },
        "exact": bool(output) and output == native,
        "byte_differences": byte_diffs,
        "first_mismatch_offset": first_mismatch(output, native),
        "note": (
            "The current owned runtime is a copy-smoke path.  This compare is "
            "diagnostic only and must not be promoted until the owned runtime "
            "enters HMX bodies on device."
        ),
    }
    analysis = out_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "owned_output_compare.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_prepared_state(family: str, out_dir: Path, oracle: dict) -> dict:
    del oracle
    return prepare_owned_inputs.write_prepared_state(family, out_dir)


def runtime_sources() -> list[Path]:
    return [
        EXAMPLE / "src" / "handwritten_hmx_matmul.cpp",
        EXAMPLE / "tools" / "owned_smoke.cpp",
        EXAMPLE / "include" / "handwritten_hmx_matmul.h",
    ]


def binary_is_stale(binary: Path) -> bool:
    if not binary.is_file():
        return True
    binary_mtime = binary.stat().st_mtime
    return any(source.stat().st_mtime > binary_mtime for source in runtime_sources())


def ensure_host_binary() -> Path:
    binary = EXAMPLE / "build" / "host" / "owned_smoke"
    if binary_is_stale(binary):
        subprocess.run([str(EXAMPLE / "build_host.sh")], cwd=ROOT, check=True)
    return binary


def ensure_android_binary() -> Path:
    binary = EXAMPLE / "build" / "android-aarch64" / "owned_smoke"
    if binary_is_stale(binary):
        subprocess.run([str(EXAMPLE / "build_android.sh")], cwd=ROOT, check=True)
    return binary


def assert_no_forbidden(command: list[str]) -> None:
    joined = " ".join(command)
    hits = [token for token in FORBIDDEN if token in joined]
    if hits:
        raise SystemExit(f"forbidden QNN runtime token in owned command: {hits}")


def ssh_cat_to(device: str, remote_path: str, data: bytes) -> None:
    subprocess.run(
        ["ssh", device, f"cat > {remote_path}"],
        input=data,
        check=True,
    )


def ssh_cat_from(device: str, remote_path: str) -> bytes:
    result = subprocess.run(
        ["ssh", device, f"cat {remote_path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def run_on_device(
    args: argparse.Namespace,
    oracle: dict,
    out_dir: Path,
    prepared: Path,
    m: int,
    k: int,
    n: int,
) -> None:
    binary = ensure_android_binary()
    remote = args.remote_dir.rstrip("/")
    device = args.device
    subprocess.run(["ssh", device, f"rm -rf {remote} && mkdir -p {remote}/prepared_state {remote}/device_out"], check=True)
    ssh_cat_to(device, f"{remote}/owned_smoke", binary.read_bytes())
    subprocess.run(["ssh", device, f"chmod +x {remote}/owned_smoke"], check=True)
    for name in (
        "activation",
        "packed_weight",
        "folded_bias",
        "control",
        "extra_control",
        "activation_table",
        "output_table",
        "descriptor",
        "mask_control",
        "output_surface",
    ):
        ssh_cat_to(device, f"{remote}/prepared_state/{name}.raw", (prepared / f"{name}.raw").read_bytes())
    ssh_cat_to(
        device,
        f"{remote}/prepared_state/manifest.json",
        (prepared / "manifest.json").read_bytes(),
    )

    remote_command = [
        f"./owned_smoke",
        "--family",
        args.family,
        "--m",
        str(m),
        "--k",
        str(k),
        "--n",
        str(n),
        "--chain",
        str(oracle["chain"]),
        "--runtime-kind",
        "device_smoke",
        "--device-execution",
        "1",
        "--activation",
        "prepared_state/activation.raw",
        "--packed-weight",
        "prepared_state/packed_weight.raw",
        "--folded-bias",
        "prepared_state/folded_bias.raw",
        "--control",
        "prepared_state/control.raw",
        "--extra-control",
        "prepared_state/extra_control.raw",
        "--activation-table",
        "prepared_state/activation_table.raw",
        "--output-table",
        "prepared_state/output_table.raw",
        "--descriptor",
        "prepared_state/descriptor.raw",
        "--mask-control",
        "prepared_state/mask_control.raw",
        "--output",
        "device_out/out.raw",
        "--output-bytes",
        str(oracle["raw_output"]["bytes"]),
        "--owned-run-json",
        "owned_run.json",
    ]
    assert_no_forbidden(remote_command)
    subprocess.run(["ssh", device, f"cd {remote} && {' '.join(remote_command)}"], check=True)

    (out_dir / "device_out").mkdir(exist_ok=True)
    (out_dir / "device_out" / "out.raw").write_bytes(
        ssh_cat_from(device, f"{remote}/device_out/out.raw")
    )
    run = json.loads(ssh_cat_from(device, f"{remote}/owned_run.json").decode("utf-8"))
    run["artifact_dir"] = rel(out_dir)
    run["oracle_manifest"] = "example/handwritten_hmx_matmul/oracles.json"
    run["prepared_state_manifest"] = "prepared_state/manifest.json"
    run["device"] = device
    run["remote_dir"] = remote
    (out_dir / "owned_run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default="u8i8", choices=sorted(load_oracles()["families"]))
    parser.add_argument("--out-dir", default="/tmp/handwritten_hmx_matmul_owned_smoke")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--device", help="SSH alias for device execution")
    parser.add_argument(
        "--remote-dir",
        default="handwritten_hmx_matmul_owned_smoke",
        help="Remote directory used with --device",
    )
    args = parser.parse_args()

    if args.rebuild:
        subprocess.run([str(EXAMPLE / "build_host.sh")], cwd=ROOT, check=True)
        if args.device:
            subprocess.run([str(EXAMPLE / "build_android.sh")], cwd=ROOT, check=True)
    binary = ensure_host_binary()
    manifest = load_oracles()
    oracle = manifest["families"][args.family]
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    build_prepared_state(args.family, out_dir, oracle)

    output = out_dir / "device_out" / "out.raw"
    owned_run = out_dir / "owned_run.json"
    prepared = out_dir / "prepared_state"
    m, k, n = oracle["shape_mkn"]
    if args.device:
        run_on_device(args, oracle, out_dir, prepared, m, k, n)
        write_owned_output_compare(args.family, out_dir, oracle)
        print(f"owned device smoke artifact: {out_dir}")
        return 0

    command = [
        str(binary),
        "--family",
        args.family,
        "--m",
        str(m),
        "--k",
        str(k),
        "--n",
        str(n),
        "--chain",
        str(oracle["chain"]),
        "--runtime-kind",
        "host_smoke",
        "--device-execution",
        "0",
        "--activation",
        str(prepared / "activation.raw"),
        "--packed-weight",
        str(prepared / "packed_weight.raw"),
        "--folded-bias",
        str(prepared / "folded_bias.raw"),
        "--control",
        str(prepared / "control.raw"),
        "--extra-control",
        str(prepared / "extra_control.raw"),
        "--activation-table",
        str(prepared / "activation_table.raw"),
        "--output-table",
        str(prepared / "output_table.raw"),
        "--descriptor",
        str(prepared / "descriptor.raw"),
        "--mask-control",
        str(prepared / "mask_control.raw"),
        "--output",
        str(output),
        "--output-bytes",
        str(oracle["raw_output"]["bytes"]),
        "--owned-run-json",
        str(owned_run),
    ]
    assert_no_forbidden(command)
    subprocess.run(command, cwd=ROOT, check=True)

    run = json.loads(owned_run.read_text(encoding="utf-8"))
    run["artifact_dir"] = rel(out_dir)
    run["oracle_manifest"] = "example/handwritten_hmx_matmul/oracles.json"
    run["prepared_state_manifest"] = "prepared_state/manifest.json"
    owned_run.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_owned_output_compare(args.family, out_dir, oracle)
    print(f"owned smoke artifact: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
