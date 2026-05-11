"""Per-component time breakdown for the CPU baseline step.

Produces the "where does the time go" data point the M1 checkpoint asks
for and the M5 ablation will compare against the GPU port. The output
(JSON + CSV + optional PNG) lands in ``results/profiling/`` so the
checkpoint writeup and figure scripts can pull from a stable location.

Implementation notes:

* **We re-implement `step` here as `step_profiled`** rather than threading
  timing through the production code path. The duplication is small and
  isolates profiling from a hot path — production `step` stays free of
  ``time.perf_counter`` calls that would distort the throughput
  benchmark in `runner.py`.
* **Each component is timed with one `perf_counter` call between calls.**
  This is the lowest-overhead instrumentation that still attributes time
  to named components. The two `compute_accessibility` calls (pre- and
  post-growth) are reported separately so we can see whether the post-
  growth call is meaningful overhead worth folding into a single matmul.
* **Bookkeeping** (validation, step counter, ``State`` construction) is
  bucketed as its own component so we can see how much of step time is
  Python/dataclass overhead vs. real numerical work — this matters when
  comparing CPU vs. GPU breakdowns at the same operating point in M5.
"""

from __future__ import annotations

import csv
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from topograph.policies import OneStepGreedyPolicy, RandomLegalPolicy
from topograph.sim_cpu import (
    Action,
    GridCityConfig,
    Policy,
    State,
    apply_action,
    compute_accessibility,
    compute_demand,
    compute_travel_times,
    compute_welfare,
    make_world,
    reset,
    update_activity,
)

PROFILE_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Inputs to a single profiling run.

    Defaults match the project plan's representative workload: 10x10 grid,
    horizon 15, 100 measured episodes (so 1500 step samples) after a brief
    warmup, random policy. Greedy is too expensive at this episode count
    because it calls `step` once per legal action per step.
    """

    grid_shape: tuple[int, int] = (10, 10)
    horizon: int = 15
    candidate_k_nearest: int = 8
    initial_budget: float = 50.0
    policy: str = "random"
    n_episodes: int = 100
    warmup_episodes: int = 5
    world_seed: int = 0
    rng_seed: int = 0


# ---------------------------------------------------------------------------
# Policy factory (kept inline to avoid importing the throughput-bench
# factory and getting tangled in its CLI-specific error messages)
# ---------------------------------------------------------------------------


class _NoOpPolicy:
    def __call__(self, state: State, rng: np.random.Generator) -> int:
        del rng
        return state.world.no_op_action


_POLICY_REGISTRY: dict[str, Any] = {
    "random": RandomLegalPolicy,
    "greedy": OneStepGreedyPolicy,
    "noop": _NoOpPolicy,
}


def _make_policy(name: str) -> Policy:
    cls = _POLICY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"unknown policy {name!r}; valid: {sorted(_POLICY_REGISTRY)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Instrumented step
# ---------------------------------------------------------------------------

# Ordered tuple of component names — used as the canonical column order in
# the CSV and the canonical bar order in the figure. Listed in *execution*
# order so the table reads top-to-bottom like a flow of computation.
COMPONENTS: tuple[str, ...] = (
    "apply_action",
    "compute_travel_times",
    "compute_accessibility_pre",
    "update_activity",
    "compute_accessibility_post",
    "compute_welfare",
    "compute_demand",
    "bookkeeping",
)


def step_profiled(
    state: State, action: Action
) -> tuple[State, float, bool, dict[str, Any]]:
    """Like `topograph.sim_cpu.step`, but accumulates per-component
    timings into ``info['timings_s']``.

    The shape of the composition is identical to production `step`; only
    the instrumentation is added. If the production composition changes,
    update this in lockstep and rerun the profile.
    """
    if not (0 <= action <= state.world.no_op_action):
        raise ValueError(
            f"action {action} out of range [0, {state.world.no_op_action}]"
        )
    if state.done:
        raise RuntimeError("step_profiled called on a terminated state")

    timings: dict[str, float] = {}
    t = time.perf_counter()

    new_mask, new_budget, did_activate = apply_action(state, action)
    t, prev = time.perf_counter(), t
    timings["apply_action"] = t - prev

    travel_times = compute_travel_times(state.world, new_mask)
    t, prev = time.perf_counter(), t
    timings["compute_travel_times"] = t - prev

    accessibility = compute_accessibility(state.world, state.activity, travel_times)
    t, prev = time.perf_counter(), t
    timings["compute_accessibility_pre"] = t - prev

    new_activity = update_activity(
        state.activity, accessibility, state.world.growth_rate
    )
    t, prev = time.perf_counter(), t
    timings["update_activity"] = t - prev

    final_acc = compute_accessibility(state.world, new_activity, travel_times)
    t, prev = time.perf_counter(), t
    timings["compute_accessibility_post"] = t - prev

    reward = compute_welfare(new_activity, final_acc)
    t, prev = time.perf_counter(), t
    timings["compute_welfare"] = t - prev

    demand = compute_demand(state.world, new_activity, travel_times)
    t, prev = time.perf_counter(), t
    timings["compute_demand"] = t - prev

    next_step = state.step + 1
    next_done = next_step >= state.world.horizon
    next_state = State(
        world=state.world,
        edge_mask=new_mask,
        activity=new_activity,
        budget_remaining=new_budget,
        step=next_step,
        done=next_done,
    )
    t, prev = time.perf_counter(), t
    timings["bookkeeping"] = t - prev

    info: dict[str, Any] = {
        "did_activate": did_activate,
        "travel_times": travel_times,
        "accessibility": final_acc,
        "demand": demand,
        "timings_s": timings,
    }
    return next_state, reward, next_done, info


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------


def run_profile(cfg: ProfileConfig) -> dict[str, Any]:
    """Run the workload and return a JSON-ready breakdown dict.

    Time per component is summed across all measured step calls (warmup
    excluded). Fractions are computed against the sum of component time
    so they total to 1.0 — *not* against the wall-clock duration of the
    whole loop. This means the "fraction" column reports time inside
    instrumented step components only; Python loop overhead, policy
    calls, and `reset` are accounted for in the wall-clock total but not
    in the fractions.
    """
    world_cfg = GridCityConfig(
        grid_shape=cfg.grid_shape,
        horizon=cfg.horizon,
        candidate_k_nearest=cfg.candidate_k_nearest,
        initial_budget=cfg.initial_budget,
    )
    world = make_world(seed=cfg.world_seed, cfg=world_cfg)
    policy = _make_policy(cfg.policy)

    # Warmup: identical machinery, discarded timings.
    for i in range(cfg.warmup_episodes):
        state = reset(world)
        rng = np.random.default_rng(cfg.rng_seed + 10_000_000 + i)
        while not state.done:
            a = int(policy(state, rng))
            state, _r, _done, _info = step_profiled(state, a)

    totals: dict[str, float] = defaultdict(float)
    policy_time = 0.0
    n_steps = 0
    overall_t0 = time.perf_counter()
    for i in range(cfg.n_episodes):
        state = reset(world)
        rng = np.random.default_rng(cfg.rng_seed + i)
        while not state.done:
            t_pol = time.perf_counter()
            a = int(policy(state, rng))
            policy_time += time.perf_counter() - t_pol
            state, _r, _done, info = step_profiled(state, a)
            timings = info["timings_s"]
            for name in COMPONENTS:
                totals[name] += timings[name]
            n_steps += 1
    overall_t1 = time.perf_counter()
    wall_clock_total_s = overall_t1 - overall_t0

    component_total_s = float(sum(totals[name] for name in COMPONENTS))
    components_out: dict[str, dict[str, float]] = {}
    for name in COMPONENTS:
        components_out[name] = {
            "total_s": float(totals[name]),
            "mean_us_per_step": float(totals[name] / n_steps * 1e6) if n_steps else 0.0,
            "fraction": (
                float(totals[name] / component_total_s)
                if component_total_s > 0
                else 0.0
            ),
        }

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "kind": "topograph_profile",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "world_info": {
            "n_zones": world.n_zones,
            "n_candidate_edges": world.n_candidate_edges,
            "action_dim": world.action_dim,
        },
        "system_info": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "total_steps": n_steps,
        "wall_clock_total_s": wall_clock_total_s,
        "component_total_s": component_total_s,
        "policy_total_s": policy_time,
        "components": components_out,
    }


# ---------------------------------------------------------------------------
# Output: CSV, JSON, optional PNG
# ---------------------------------------------------------------------------


def write_csv(result: dict[str, Any], path: Path) -> None:
    """Write the per-component breakdown as CSV.

    Rows are sorted by fraction descending so the hot components are
    visible at the top of a `head` of the file. Column order matches the
    JSON's components dict (component, total_s, mean_us_per_step, fraction).
    """
    components = result["components"]
    rows = sorted(
        components.items(), key=lambda kv: -kv[1]["fraction"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "total_s", "mean_us_per_step", "fraction"])
        for name, info in rows:
            writer.writerow(
                [
                    name,
                    f"{info['total_s']:.6f}",
                    f"{info['mean_us_per_step']:.3f}",
                    f"{info['fraction']:.6f}",
                ]
            )


def write_figure(result: dict[str, Any], path: Path) -> bool:
    """Write a horizontal-bar PNG of the component fractions.

    Returns True if matplotlib was available and the figure was written,
    False otherwise. The CSV is the canonical artifact; the figure is a
    nice-to-have. This keeps matplotlib an optional dep.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    components = result["components"]
    items = sorted(components.items(), key=lambda kv: kv[1]["fraction"])  # asc for h-bar
    names = [n for n, _ in items]
    fracs = [v["fraction"] for _, v in items]
    means_us = [v["mean_us_per_step"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(names, fracs)
    for bar, mu in zip(bars, means_us, strict=True):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{mu:.1f} µs",
            va="center",
            ha="left",
            fontsize=9,
        )
    cfg = result["config"]
    g_r, g_c = cfg["grid_shape"]
    ax.set_xlabel("fraction of step time")
    ax.set_xlim(0, max(1.0, max(fracs) * 1.25))
    ax.set_title(
        f"CPU baseline step breakdown — {g_r}x{g_c} grid, horizon={cfg['horizon']}, "
        f"policy={cfg['policy']} ({result['total_steps']} step samples)"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def format_breakdown(result: dict[str, Any]) -> str:
    """Compact human-readable rendering, sorted hot-component-first."""
    cfg = result["config"]
    g_r, g_c = cfg["grid_shape"]
    lines = [
        f"topograph-profile  ({result['timestamp']})",
        f"  config:   grid={g_r}x{g_c}  horizon={cfg['horizon']}  "
        f"policy={cfg['policy']}  episodes={cfg['n_episodes']} (+{cfg['warmup_episodes']} warmup)",
        f"  workload: {result['total_steps']} step samples  "
        f"wall_clock_total={result['wall_clock_total_s']:.3f}s  "
        f"policy_total={result['policy_total_s']:.3f}s  "
        f"in-step total={result['component_total_s']:.3f}s",
        "",
        f"  {'component':<28} {'mean (µs/step)':>16} {'fraction':>12}",
        f"  {'-' * 28} {'-' * 16} {'-' * 12}",
    ]
    rows = sorted(
        result["components"].items(), key=lambda kv: -kv[1]["fraction"]
    )
    for name, info in rows:
        lines.append(
            f"  {name:<28} {info['mean_us_per_step']:>16,.2f} "
            f"{info['fraction']:>12.1%}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default output path
# ---------------------------------------------------------------------------


def default_output_stem(cfg: ProfileConfig, timestamp: str) -> str:
    """Stem for the default ``results/profiling/<stem>.{csv,json,png}`` paths."""
    ts_safe = timestamp.replace(":", "").replace("-", "").replace("+0000", "Z")
    g_r, g_c = cfg.grid_shape
    return f"cpu_breakdown_{cfg.policy}_{g_r}x{g_c}_h{cfg.horizon}_{ts_safe}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="topograph-profile",
        description=(
            "Per-component time breakdown for the CPU baseline `step`. "
            "Writes JSON + CSV (and PNG if matplotlib is installed) into "
            "results/profiling/."
        ),
    )
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--k-nearest", type=int, default=8, dest="k_nearest")
    parser.add_argument("--budget", type=float, default=50.0)
    parser.add_argument(
        "--policy", choices=["random", "greedy", "noop"], default="random"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--world-seed", type=int, default=0)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <repo>/results/profiling/.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print summary to stdout, do not write CSV/JSON/PNG.",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip the PNG figure even if matplotlib is available.",
    )
    return parser


_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "profiling"


def main(argv: list[str] | None = None) -> int:
    import json

    args = _build_parser().parse_args(argv)
    cfg = ProfileConfig(
        grid_shape=(args.grid, args.grid),
        horizon=args.horizon,
        candidate_k_nearest=args.k_nearest,
        initial_budget=args.budget,
        policy=args.policy,
        n_episodes=args.episodes,
        warmup_episodes=args.warmup,
        world_seed=args.world_seed,
        rng_seed=args.rng_seed,
    )
    result = run_profile(cfg)
    print(format_breakdown(result))

    if args.no_save:
        return 0

    out_dir = args.output_dir or _DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = default_output_stem(cfg, result["timestamp"])
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    png_path = out_dir / f"{stem}.png"

    json_path.write_text(json.dumps(result, indent=2) + "\n")
    write_csv(result, csv_path)
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")
    if not args.no_figure:
        ok = write_figure(result, png_path)
        if ok:
            print(f"wrote {png_path}")
        else:
            print("matplotlib not available; skipped PNG (install topograph[figures])")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
