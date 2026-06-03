"""Publication figure for the transit_learning APSP GPU sweep.

Data baked in from the A100 run on 2026-06-01 (results/transit_learning_gpu/
gpu_throughput_20260601T0823*.txt). Regenerate the figure with:

    python plot_results.py

Three panels:
  (a) Speedup (FW / matrix-squaring) vs batch, per real city -- the win is
      huge at batch 1 and decays as batch grows.
  (b) Absolute throughput vs batch (synthetic N=100): matrix-squaring saturates
      early; Floyd-Warshall keeps climbing as batching fills idle GPU.
  (c) Peak memory vs N at batch 256: torch eager scales O(B*N^3) (the
      (B,N,N,N) intermediate XLA fused away in TopoGraph) -- the memory wall.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "results" / "transit_learning_gpu"
OUT.mkdir(parents=True, exist_ok=True)

# --- measured data (A100-40GB, median of 30 calls) ------------------------- #
# real cities: speedup = FW_time / MS_time at batch [1, 64, 256, 1024]
BATCH_REAL = [1, 64, 256, 1024]
CITY_SPEEDUP = {
    "Mandl (N=15)":    [19.8, 23.8, 23.8, 23.9],
    "Mumford0 (N=30)": [33.1, 40.5, 31.3, 9.2],
    "Mumford1 (N=70)": [63.0, 18.8, 5.8, 2.3],
    "Mumford2 (N=110)":[98.4, 8.7, 2.9, 1.8],
    "Mumford3 (N=127)":[117.4, 6.8, 2.5, 1.7],
}

# synthetic N=100 absolute throughput (APSP solves/sec)
BATCH_SYN = [1, 16, 64, 256, 1024]
FW_APS_N100 = [28, 393, 1563, 5212, 8918]
MS_APS_N100 = [2540, 13699, 16582, 17494, 17669]

# peak memory (MB) at batch 256 vs N (synthetic + real cities, all eager)
MEM_N   = [15, 25, 30, 70, 100, 110, 127, 225]
MEM_MB  = [4, 18, 30, 367, 1055, 1401, 2148, 11822]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def main():
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 200})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) speedup vs batch, per city
    for (name, ys), c in zip(CITY_SPEEDUP.items(), COLORS):
        ax1.plot(BATCH_REAL, ys, "o-", color=c, label=name, lw=2, ms=6)
    ax1.axhline(1.0, color="gray", ls="--", lw=1)
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xticks(BATCH_REAL); ax1.set_xticklabels(BATCH_REAL)
    ax1.set_xlabel("batch size (graphs)")
    ax1.set_ylabel("speedup  (FW time / matrix-squaring time)")
    ax1.set_title("(a) Matrix-squaring vs Floyd-Warshall\nbiggest at batch 1, decays with batch")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.annotate("117x", (1, 117.4), textcoords="offset points", xytext=(6, -2),
                 fontsize=8, color=COLORS[4])

    # (b) throughput saturation
    ax2.plot(BATCH_SYN, MS_APS_N100, "o-", color="#C44E52", lw=2, ms=6,
             label="matrix-squaring (TopoGraph kernel)")
    ax2.plot(BATCH_SYN, FW_APS_N100, "s-", color="#4C72B0", lw=2, ms=6,
             label="Floyd-Warshall (transit_learning)")
    ax2.set_xscale("log", base=2); ax2.set_yscale("log")
    ax2.set_xticks(BATCH_SYN); ax2.set_xticklabels(BATCH_SYN)
    ax2.set_xlabel("batch size (graphs)")
    ax2.set_ylabel("throughput (APSP solves / sec)")
    ax2.set_title("(b) Throughput, synthetic N=100\nMS saturates early; FW keeps climbing")
    ax2.legend(fontsize=8, loc="lower right")

    # (c) memory wall: peak MB vs N at batch 256, with N^2 and N^3 guides
    ax3.plot(MEM_N, MEM_MB, "o-", color="#55A868", lw=2, ms=6,
             label="measured peak (batch 256)")
    # guide lines anchored at N=100
    n0, m0 = 100, 1055
    g3 = [m0 * (n / n0) ** 3 for n in MEM_N]
    g2 = [m0 * (n / n0) ** 2 for n in MEM_N]
    ax3.plot(MEM_N, g3, "--", color="gray", lw=1.2, label=r"$O(B\cdot N^3)$ (eager)")
    ax3.plot(MEM_N, g2, ":", color="gray", lw=1.2, label=r"$O(B\cdot N^2)$ (XLA-fused)")
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlabel("graph size N (nodes)")
    ax3.set_ylabel("peak GPU memory (MB)")
    ax3.set_title("(c) Memory wall at batch 256\neager materializes the (B,N,N,N) tensor")
    ax3.legend(fontsize=8, loc="upper left")

    fig.suptitle("TopoGraph batched APSP kernel applied to transit_learning's shortest-path step "
                 "(A100-40GB)", fontsize=12, y=1.02)
    fig.tight_layout()
    dest = OUT / "gpu_throughput_figure.png"
    fig.savefig(dest, bbox_inches="tight")
    print(f"saved: {dest}")


if __name__ == "__main__":
    main()
