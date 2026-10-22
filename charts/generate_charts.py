"""Regenerate the checked-in benchmark comparison figures with Matplotlib."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


OUT = Path(__file__).parent
NAMES = ["Triton", "Raw CUDA C++", "Optimized CuTe DSL", "Baseline CuTe DSL", "PyTorch eager baseline"]
LATENCY_MS = [2.0830, 2.7381, 3.0718, 15.5152, 27.8097]
TFLOPS = [1.031, 0.784, 0.699, 0.138, 0.077]
COLORS = ["#6554C0", "#167D9A", "#2E8B57", "#D97706", "#667085"]

PROFILE_NAMES = ["Optimized CuTe", "Baseline CuTe", "Raw CUDA", "Mojo / MAX", "Triton"]
OCCUPANCY = [95.15, 98.86, 94.98, 94.01, 57.81]
DRAM_THROUGHPUT = [27.23, 15.46, 40.16, 8.05, 41.35]
PROFILE_COLORS = ["#2E8B57", "#D97706", "#167D9A", "#8993A4", "#6554C0"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=190, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def latency_chart():
    fig, ax = plt.subplots(figsize=(10.5, 5.4), layout="constrained")
    fig.patch.set_facecolor("#FBFCFE")
    ax.set_facecolor("#FBFCFE")
    y = np.arange(len(NAMES))
    bars = ax.barh(y, LATENCY_MS, color=COLORS, height=0.58, edgecolor="none")
    ax.invert_yaxis()
    ax.set_yticks(y, NAMES, fontsize=11)
    ax.set_xlim(0, 33)
    ax.set_xticks(range(0, 34, 5))
    ax.set_xlabel("CUDA-event kernel latency (ms)", fontsize=11, labelpad=9)
    ax.set_title("Sparse GQA FP8 implementation latency", loc="left", fontsize=16, weight="bold", pad=16, color="#182230")
    ax.text(0, 1.015, "RTX 4060 Laptop GPU · 256 requests · Top-K 512", transform=ax.transAxes, fontsize=10, color="#526174")
    ax.grid(axis="x", color="#DCE2EA", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8, colors="#344054")
    for bar, ms, tflops in zip(bars, LATENCY_MS, TFLOPS):
        ax.text(ms + 0.16, bar.get_y() + bar.get_height() / 2,
                f"{ms:.4f} ms · {tflops:.3f} TFLOP/s", va="center", ha="left",
                fontsize=9.5, color="#182230")
    ax.text(0, -0.20, "One full-suite run, sorted by CUDA-event elapsed time. PyTorch measures its eager-op sequence; Mojo is omitted because it uses host enqueue + sync.",
            transform=ax.transAxes, fontsize=8.5, color="#526174")
    save(fig, "latency_comparison")


def throughput_chart():
    fig, ax = plt.subplots(figsize=(10.5, 5.1), layout="constrained")
    fig.patch.set_facecolor("#FBFCFE")
    ax.set_facecolor("#FBFCFE")
    y = np.arange(len(NAMES))
    bars = ax.barh(y, TFLOPS, color=COLORS, height=0.58, edgecolor="none")
    ax.invert_yaxis()
    ax.set_yticks(y, NAMES, fontsize=11)
    ax.set_xlim(0, 1.16)
    ax.set_xticks(np.arange(0, 1.21, 0.2))
    ax.set_xlabel("Derived throughput (TFLOP/s)", fontsize=11, labelpad=9)
    ax.set_title("Derived compute rate", loc="left", fontsize=16, weight="bold", pad=16, color="#182230")
    ax.text(0, 1.015, "Calculated from the same CUDA-event latencies and nominal operation count", transform=ax.transAxes, fontsize=10, color="#526174")
    ax.grid(axis="x", color="#DCE2EA", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8, colors="#344054")
    for bar, value in zip(bars, TFLOPS):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=10, color="#182230")
    ax.text(0, -0.18, "Mojo is excluded because its rate uses host enqueue-plus-synchronization time.", transform=ax.transAxes, fontsize=8.5, color="#526174")
    save(fig, "throughput_comparison")


def profile_chart():
    fig, ax = plt.subplots(figsize=(10.5, 5.3), layout="constrained")
    fig.patch.set_facecolor("#FBFCFE")
    ax.set_facecolor("#FBFCFE")
    x = np.arange(len(PROFILE_NAMES))
    width = 0.36
    occ = ax.bar(x - width / 2, OCCUPANCY, width, label="Achieved occupancy", color="#4776A8")
    dram = ax.bar(x + width / 2, DRAM_THROUGHPUT, width, label="DRAM throughput", color="#D49A36")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percent of peak (%)", fontsize=10)
    ax.set_xticks(x, PROFILE_NAMES, fontsize=9, rotation=12, ha="right")
    ax.set_title("Nsight Compute utilization counters", loc="left", fontsize=16, weight="bold", pad=16, color="#182230")
    ax.text(0, 1.015, "Basic metric set · counters are peak-relative, not application efficiency", transform=ax.transAxes, fontsize=10, color="#526174")
    ax.grid(axis="y", color="#DCE2EA", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=7, colors="#344054")
    ax.legend(frameon=False, loc="upper right", ncols=2)
    ax.bar_label(occ, fmt="%.1f", padding=2, fontsize=8, color="#344054")
    ax.bar_label(dram, fmt="%.1f", padding=2, fontsize=8, color="#344054")
    ax.text(0, -0.25, "Raw CUDA uses the refreshed specialized-kernel capture; other bars use the checked-in 256-request captures.", transform=ax.transAxes, fontsize=8.5, color="#526174")
    save(fig, "ncu_utilization")


def box(ax, xy, w, h, text, fc, ec="#D0D7E2", fs=9, color="#182230", weight="normal"):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.025,rounding_size=0.04", facecolor=fc, edgecolor=ec, linewidth=1.0)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight)


def arrow(ax, start, end, color="#526174", lw=1.4, rad=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}"))


def data_movement_diagram():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("#FBFCFE")
    fig.suptitle("PyTorch gather reuses K/V rows; current GPU kernels reload per query head", x=0.06, y=0.98, ha="left", fontsize=15, weight="bold", color="#182230")
    fig.text(0.06, 0.91, "Logical selected K/V payload per workload invocation, inferred from the implementation mappings", fontsize=10, color="#526174")
    fig.subplots_adjust(top=0.79, bottom=0.16, left=0.04, right=0.98, wspace=0.08)
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_facecolor("#FBFCFE")

    ax = axes[0]
    ax.set_title("PyTorch eager · batched gather", loc="left", fontsize=11, weight="bold", pad=10, color="#182230")
    box(ax, (0.25, 6.3), 2.05, 1.0, "K/V cache\n256 MiB allocated", "#EAF2F8", fs=9, weight="bold")
    boundary = FancyBboxPatch((3.1, 4.2), 6.2, 4.2, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor="#F2F8F4", edgecolor="#8CC6A3", linewidth=1.2)
    ax.add_patch(boundary)
    ax.text(6.2, 8.05, "One gathered K/V tensor reused by 4 Q heads", ha="center", fontsize=8.5, weight="bold", color="#236B43")
    for i in range(4):
        y = 7.05 - i * 0.72
        box(ax, (5.25, y), 3.1, 0.52, f"Batched matmul for Q head {i}", "#E8F5EE", ec="#8CC6A3", fs=8)
        arrow(ax, (2.35, 6.8), (5.15, y + 0.26), "#2E8B57", 0.9, rad=(i - 1.5) * 0.055)
    box(ax, (0.5, 1.35), 8.8, 1.15, "≈ 256 MiB logical K/V payload · gather once per KV head", "#E8F5EE", ec="#8CC6A3", fs=9.2, color="#236B43", weight="bold")

    ax = axes[1]
    ax.set_title("Fused CUDA / CuTe / Triton / Mojo mappings", loc="left", fontsize=11, weight="bold", pad=10, color="#182230")
    box(ax, (0.25, 6.3), 2.05, 1.0, "Same K/V cache\n256 MiB allocated", "#EAF2F8", fs=9, weight="bold")
    boundary = FancyBboxPatch((3.0, 4.2), 6.5, 4.2, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor="#FFF9F0", edgecolor="#E8B56A", linewidth=1.2)
    ax.add_patch(boundary)
    ax.text(6.25, 8.05, "4 query-head warps / programs each load K/V", ha="center", fontsize=8.5, weight="bold", color="#9A5700")
    for i in range(4):
        y = 7.35 - i * 0.72
        box(ax, (5.15, y), 3.7, 0.52, f"Q head {i} · independent K/V loads", "#FFF3E2", ec="#E8B56A", fs=8)
        arrow(ax, (2.35, 6.8), (5.05, y + 0.26), "#D97706", 0.9, rad=(i - 1.5) * 0.06)
    box(ax, (0.5, 1.35), 8.8, 1.15, "≈ 1 GiB logical K/V payload · all 4 fused kernels", "#FFF3E2", ec="#E8B56A", fs=10, color="#9A5700", weight="bold")
    fig.text(0.06, 0.04, "Physical DRAM traffic may be lower due to cache reuse. PyTorch has large gathered FP32 intermediates.", fontsize=9, color="#526174")
    save(fig, "kv_data_movement")


if __name__ == "__main__":
    latency_chart()
    throughput_chart()
    profile_chart()
    data_movement_diagram()
