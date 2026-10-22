#!/usr/bin/env python3
"""Run all sparse GQA kernels and print a comparable timing summary."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(label: str, command: list[str]) -> tuple[dict[str, str], bool]:
    print(f"\n=== {label} ===", flush=True)
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        print(f"Could not start: {exc}")
        return {"status": f"ERROR: {exc}"}, False

    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    result: dict[str, str] = {"status": "OK" if proc.returncode == 0 else f"EXIT {proc.returncode}"}

    event_ms = re.search(r"Kernel Latency:\s*([\d.]+)\s*ms/iter", proc.stdout)
    host_ms = re.search(r"Enqueue\+sync wall time:\s*([\d.]+)\s*ms/iter", proc.stdout)
    e2e_ms = re.search(r"E2E Latency\s*:\s*([\d.]+)\s*ms/iter", proc.stdout)
    tflops = re.search(r"Throughput\s*:\s*([\d.]+)\s*TFLOP/s", proc.stdout)
    validation = re.search(r"PASS(?:\s*\([^)]*\))?\s*:\s*(True|False)", proc.stdout)

    if event_ms:
        result["time"] = f"{float(event_ms.group(1)):.4f} ms"
        result["method"] = "CUDA events"
    elif host_ms:
        result["time"] = f"{float(host_ms.group(1)):.3f} ms"
        result["method"] = "enqueue + sync"
    if e2e_ms:
        result["e2e"] = f"{float(e2e_ms.group(1)):.3f} ms"
    if tflops:
        result["rate"] = f"{float(tflops.group(1)):.3f} TFLOP/s"
    if validation:
        result["validation"] = "PASS" if validation.group(1) == "True" else "FAIL"
    elif "validation=NOT CHECKED" in proc.stdout or "Reference check: unavailable" in proc.stdout:
        result["validation"] = "not checked"
    else:
        result["validation"] = "not reported"

    return result, proc.returncode == 0 and result.get("validation") != "FAIL"


def show_table(rows: list[tuple[str, dict[str, str]]]) -> None:
    columns = ["Kernel", "Time", "Timing method", "E2E", "Derived rate", "Validation", "Run"]
    values = [
        [
            name,
            result.get("time", "—"),
            result.get("method", "—"),
            result.get("e2e", "—"),
            result.get("rate", "—"),
            result.get("validation", "not reported"),
            result["status"],
        ]
        for name, result in rows
    ]
    widths = [max(len(columns[i]), *(len(row[i]) for row in values)) for i in range(len(columns))]
    print("\n=== Comparison ===")
    print(" | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))))
    print("-+-".join("-" * width for width in widths))
    for row in values:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(columns))))
    print("\nCUDA-event times are device timings. Mojo's time includes host enqueue and synchronization.")


def main() -> int:
    rows: list[tuple[str, dict[str, str]]] = []
    succeeded = True

    for label, script in (
        ("Optimized CuTe", "sm89_cutedsl.py"),
        ("Baseline CuTe", "backup.py"),
        ("Triton", "sm89_triton.py"),
    ):
        result, ok = run(label, [sys.executable, str(ROOT / script)])
        rows.append((label, result))
        succeeded &= ok

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        rows.append(("Raw CUDA", {"status": "ERROR: nvcc not found"}))
        succeeded = False
    else:
        with tempfile.TemporaryDirectory(prefix="qsa-sm89-") as temp_dir:
            executable = Path(temp_dir) / "sm89_cuda"
            compile_result, compile_ok = run(
                "Compile raw CUDA",
                [nvcc, "-O3", "-arch=sm_89", "-lineinfo", str(ROOT / "sm89_cuda.cu"), "-o", str(executable)],
            )
            if compile_ok:
                result, ok = run("Raw CUDA", [str(executable)])
            else:
                result, ok = {"status": compile_result["status"]}, False
            rows.append(("Raw CUDA", result))
            succeeded &= compile_ok and ok

    mojo = ROOT / ".venv" / "bin" / "mojo"
    if not mojo.is_file():
        mojo = Path(shutil.which("mojo") or "")
    if not mojo.is_file():
        rows.append(("Mojo / MAX", {"status": "ERROR: mojo not found"}))
        succeeded = False
    else:
        result, ok = run("Mojo / MAX", [str(mojo), str(ROOT / "sm89_mojo.mojo")])
        rows.append(("Mojo / MAX", result))
        succeeded &= ok

    show_table(rows)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
