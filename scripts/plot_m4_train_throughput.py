"""Plot the M4 end-to-end throughput study (the RQ3 / Figure-1b chart).

Reads the latest `results/m4_train_throughput/m4_train_throughput_*.json`
(written by `scripts/m4_train_throughput.py`) and draws env-steps/sec vs.
batch size for the three on-device drivers — simulator-only,
policy-inference, and full train-step — on the 10x10 reference grid. The
gap between the curves is the answer to RQ3: how much of the simulator's
throughput survives once the policy and a gradient update are in the loop.

One point, baked into the title: the % of simulator throughput the
train-step retains at the sweet-spot batch.

No data hand-editing: every number comes from the JSON artifact. Robust to
partial runs (plots whatever batches/drivers are present).

Run:  python scripts/plot_m4_train_throughput.py
      python scripts/plot_m4_train_throughput.py path/to/specific.json
Writes:  results/m4_train_throughput/m4_train_throughput_<ts>.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "m4_train_throughput"

# Driver -> (label, color, marker). Colors match the talk palette
# (sim = gray reference, policy = amber, training = cherry).
DRIVERS = {
    "sim_only": ("Simulator only", "#7F8C8D", "o"),
    "policy_infer": ("+ MLP inference", "#E08E0B", "s"),
    "train_step": ("+ REINFORCE train step", "#C0392B", "^"),
}
REFERENCE_GRID = "10x10"


def _latest_json() -> Path | None:
    if not RESULTS_DIR.exists():
        return None
    files = sorted(RESULTS_DIR.glob("m4_train_throughput_*.json"))
    return files[-1] if files else None


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("ERROR: matplotlib not installed. Install the figures extra:")
        print('  uv pip install -e ".[figures]"')
        sys.exit(1)

    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_json()
    if json_path is None or not json_path.exists():
        print("No M4 results JSON found. Run scripts/m4_train_throughput.py first")
        print(f"(looked in {RESULTS_DIR.relative_to(REPO_ROOT)}/).")
        sys.exit(1)

    payload = json.loads(json_path.read_text())
    rows = payload["results"]

    # Group reference-grid rows by driver -> {batch: env_steps_per_sec}.
    def grid_str(r):
        gs = r["grid_shape"]
        return f"{gs[0]}x{gs[1]}" if isinstance(gs, list) else str(gs)

    series: dict[str, dict[int, float]] = {d: {} for d in DRIVERS}
    for r in rows:
        if grid_str(r) != REFERENCE_GRID or r["driver"] not in series:
            continue
        series[r["driver"]][int(r["batch_size"])] = float(r["env_steps_per_sec"])

    if not any(series.values()):
        print(f"No {REFERENCE_GRID} rows in {json_path.name}; nothing to plot.")
        sys.exit(1)

    plt.rcParams.update({
        "font.size": 15, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.3, "figure.dpi": 200,
    })
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for driver, (label, color, marker) in DRIVERS.items():
        pts = sorted(series[driver].items())
        if not pts:
            continue
        xs = [b for b, _ in pts]
        ys = [v for _, v in pts]
        ax.plot(xs, ys, marker + "-", color=color, lw=3, ms=9, label=label, zorder=3)

    ax.set_xscale("log", base=2)
    all_batches = sorted({b for s in series.values() for b in s})
    ax.set_xticks(all_batches)
    ax.set_xticklabels([str(b) for b in all_batches])
    ax.set_xlabel("Batch size  (# environments)")
    ax.set_ylabel("Env-steps / sec")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper left", frameon=False, fontsize=13)

    # The one point: retained throughput at the train-step sweet spot.
    title = f"End-to-end training throughput on {REFERENCE_GRID}"
    verdict = payload.get("verdict", {})
    sweet_b = verdict.get("train_step_sweet_spot_batch")
    per_batch = verdict.get("per_batch", {})
    if sweet_b is not None and str(sweet_b) in {str(k) for k in per_batch}:
        key = sweet_b if sweet_b in per_batch else str(sweet_b)
        retained = per_batch[key]["train_over_sim"] * 100
        title = (f"Training keeps {retained:.0f}% of simulator throughput "
                 f"(batch {sweet_b}, {REFERENCE_GRID})")
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=12)

    fig.tight_layout()
    out_png = json_path.with_suffix(".png")
    fig.savefig(out_png, bbox_inches="tight")
    try:
        shown = out_png.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_png  # json supplied from outside the repo (e.g. a test path)
    print(f"Wrote: {shown}")


if __name__ == "__main__":
    main()
