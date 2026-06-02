"""Per-component time breakdown for the GPU `step`.

The GPU counterpart to `bench/profiling.py`. Produces the "where does the
time go" data point for the batched on-device simulator, so it can be set
side by side with the CPU breakdown (same component names, same JSON/CSV
schema). On the CPU baseline APSP is ~75% of step time; this script measures
what that profile looks like once the whole step is vmapped on the GPU.

**Methodology — and its one caveat.** The production GPU step fuses every
component into a single XLA program, so there is no per-component boundary to
put a timer across (unlike the CPU step, which is a sequence of separate
Python calls). Instead we time each component as its *own* jitted+vmapped op
at a fixed operating point, median over many runs with `block_until_ready`,
and report the fractions — exactly parallel to how the CPU profiler times
each component in isolation.

The caveat: isolated timing attributes each op's kernel-launch/dispatch
overhead to that component, so cheap components (welfare, apply_action) look
relatively larger than their true share inside the fused step. The *fused
whole-step* time is reported alongside as ground truth, and the
sum-of-components / fused-step ratio quantifies how much fusion buys. The
headline comparison — whether the travel-time (APSP) fraction shifts vs the
CPU's 75% — is robust to this because APSP dominates the compute.

Components mirror the CPU profiler. `compute_travel_times` here is
build-direct-distance + K-iteration APSP (the GPU equivalent of the CPU's
direct-distance + Floyd-Warshall); the apsp-only and build-only splits are
reported as extra detail. The policy is timed separately (it is not part of
`step`), matching the CPU profiler's `policy_total_s`.

Run on GPU:
  python -m topograph.bench.gpu_profiling --grid 10 --batch 256
  (or via scripts/modal_m5_fig3_gpu_breakdown.py)
"""

from __future__ import annotations

import csv
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:  # pragma: no cover
    print("ERROR: JAX not installed. Install the gpu extras: uv pip install -e \".[gpu]\"")
    print(f"Underlying error: {e}")
    raise

from topograph.sim_cpu import GridCityConfig, make_world
from topograph.sim_gpu import (
    DEFAULT_K_ITERATIONS,
    GPUState,
    apply_action_batched,
    apsp_batched,
    build_direct_distance_single,
    compute_accessibility_batched,
    compute_demand_batched,
    compute_welfare_batched,
    random_legal_action_batched,
    step_batched,
    update_activity_batched,
    valid_action_mask_batched,
    world_arrays_from_config,
)

PROFILE_SCHEMA_VERSION = 1

# Same component set and execution order as bench/profiling.py, so the CPU and
# GPU breakdowns line up row-for-row in the comparison figure.
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


@dataclass(frozen=True, slots=True)
class GpuProfileConfig:
    grid_shape: tuple[int, int] = (10, 10)
    batch_size: int = 256  # the throughput sweet spot
    horizon: int = 15
    k_iterations: int = DEFAULT_K_ITERATIONS
    mask_density: float = 0.4  # realistic mid-episode partial network
    n_timed: int = 50
    n_warmup: int = 3
    world_seed: int = 0


def _time_fn(fn: Callable[[], Any], n_warmup: int, n_timed: int) -> float:
    """Median wall time of `fn`, blocking on its (possibly pytree) output."""
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run(cfg: GpuProfileConfig) -> dict[str, Any]:
    world = make_world(cfg.world_seed, GridCityConfig(grid_shape=cfg.grid_shape, horizon=cfg.horizon))
    wa = world_arrays_from_config(world)
    B = cfg.batch_size
    N, E = world.n_zones, world.n_candidate_edges
    k = cfg.k_iterations
    rng = np.random.default_rng(cfg.world_seed)

    # Operating-point inputs: a realistic mid-episode partial network.
    mask = jnp.asarray(rng.random((B, E)) < cfg.mask_density)
    budget = jnp.full((B,), world.initial_budget * 0.5, dtype=jnp.float32)
    actions = jnp.asarray(rng.integers(0, E, size=B, dtype=np.int32))
    activity = jnp.broadcast_to(
        jnp.asarray(world.initial_activity, dtype=jnp.float32), (B, N)
    )
    decay, growth = wa.accessibility_decay, wa.growth_rate
    alpha = jnp.float32(world.gravity_alpha)
    beta = jnp.float32(world.gravity_beta)

    build_direct_batched = jax.jit(
        lambda m: jax.vmap(build_direct_distance_single, in_axes=(0, None))(m, wa)
    )
    travel_times_fn = jax.jit(
        lambda m: apsp_batched(
            jax.vmap(build_direct_distance_single, in_axes=(0, None))(m, wa),
            k_iterations=k,
        )
    )

    # Precompute intermediates once (untimed) to feed downstream components.
    direct = build_direct_batched(mask)
    tt = apsp_batched(direct, k_iterations=k)
    acc_pre = compute_accessibility_batched(activity, tt, decay)
    new_activity = update_activity_batched(activity, acc_pre, growth)
    final_acc = compute_accessibility_batched(new_activity, tt, decay)
    jax.block_until_ready((direct, tt, acc_pre, new_activity, final_acc))

    state = GPUState(
        edge_mask=mask, activity=activity, budget=budget,
        step=jnp.zeros((B,), jnp.int32), done=jnp.zeros((B,), bool),
    )

    nw, nt = cfg.n_warmup, cfg.n_timed
    timings: dict[str, float] = {
        "apply_action": _time_fn(
            lambda: apply_action_batched(mask, budget, actions, wa.cost_per_edge), nw, nt),
        "compute_travel_times": _time_fn(lambda: travel_times_fn(mask), nw, nt),
        "compute_accessibility_pre": _time_fn(
            lambda: compute_accessibility_batched(activity, tt, decay), nw, nt),
        "update_activity": _time_fn(
            lambda: update_activity_batched(activity, acc_pre, growth), nw, nt),
        "compute_accessibility_post": _time_fn(
            lambda: compute_accessibility_batched(new_activity, tt, decay), nw, nt),
        "compute_welfare": _time_fn(
            lambda: compute_welfare_batched(new_activity, final_acc), nw, nt),
        "compute_demand": _time_fn(
            lambda: compute_demand_batched(new_activity, tt, alpha, beta), nw, nt),
        # "Bookkeeping" on the GPU is just the step-counter / done update.
        "bookkeeping": _time_fn(
            lambda: (state.step + 1, state.step + 1 >= cfg.horizon), nw, nt),
    }

    # Extra detail: split travel_times into build vs APSP, and the policy cost.
    detail = {
        "apsp_only": _time_fn(lambda: apsp_batched(direct, k_iterations=k), nw, nt),
        "build_direct_distance": _time_fn(lambda: build_direct_batched(mask), nw, nt),
    }
    policy_s = _time_fn(
        lambda: random_legal_action_batched(
            jax.random.split(jax.random.PRNGKey(0), B),
            valid_action_mask_batched(mask, budget, wa.cost_per_edge),
        ),
        nw, nt,
    )
    fused_step_s = _time_fn(
        lambda: step_batched(state, actions, wa, horizon=cfg.horizon, k_iterations=k),
        nw, nt,
    )

    component_total_s = float(sum(timings[n] for n in COMPONENTS))
    components_out = {
        name: {
            "ms_per_batched_step": timings[name] * 1e3,
            "us_per_env_step": timings[name] * 1e6 / B,
            "fraction": (timings[name] / component_total_s) if component_total_s > 0 else 0.0,
        }
        for name in COMPONENTS
    }

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "kind": "topograph_gpu_profile",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "world_info": {"n_zones": N, "n_candidate_edges": E, "action_dim": world.action_dim},
        "system_info": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "platform": platform.platform(),
        },
        "method": (
            "each component timed as an isolated jitted+vmapped op (median); "
            "isolated timing attributes per-op launch overhead to each component, "
            "so the fused_step time below is the ground-truth whole-step cost"
        ),
        "components": components_out,
        "detail": {k_: {"ms_per_batched_step": v * 1e3, "us_per_env_step": v * 1e6 / B}
                   for k_, v in detail.items()},
        "component_sum_ms": component_total_s * 1e3,
        "fused_step_ms": fused_step_s * 1e3,
        "fusion_ratio_sum_over_fused": (component_total_s / fused_step_s) if fused_step_s > 0 else None,
        "policy_ms": policy_s * 1e3,
    }


def write_csv(result: dict[str, Any], path: Path) -> None:
    rows = sorted(result["components"].items(), key=lambda kv: -kv[1]["fraction"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "ms_per_batched_step", "us_per_env_step", "fraction"])
        for name, info in rows:
            w.writerow([name, f"{info['ms_per_batched_step']:.4f}",
                        f"{info['us_per_env_step']:.4f}", f"{info['fraction']:.6f}"])


def write_figure(result: dict[str, Any], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    items = sorted(result["components"].items(), key=lambda kv: kv[1]["fraction"])
    names = [n for n, _ in items]
    fracs = [v["fraction"] for _, v in items]
    labels = [v["us_per_env_step"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(names, fracs)
    for bar, mu in zip(bars, labels, strict=True):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{mu:.2f} µs/env", va="center", ha="left", fontsize=9)
    cfg = result["config"]
    g_r, g_c = cfg["grid_shape"]
    ax.set_xlabel("fraction of summed component time")
    ax.set_xlim(0, max(1.0, max(fracs) * 1.25))
    ax.set_title(f"GPU step breakdown — {g_r}x{g_c}, batch {cfg['batch_size']}, "
                 f"K={cfg['k_iterations']} ({result['system_info']['backend']})")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def format_breakdown(result: dict[str, Any]) -> str:
    cfg = result["config"]
    g_r, g_c = cfg["grid_shape"]
    lines = [
        f"topograph-gpu-profile  ({result['timestamp']})",
        f"  backend:  {result['system_info']['backend']}  "
        f"devices={result['system_info']['devices']}",
        f"  config:   grid={g_r}x{g_c}  batch={cfg['batch_size']}  "
        f"K={cfg['k_iterations']}  ({cfg['n_timed']} timed runs)",
        f"  fused whole-step: {result['fused_step_ms']:.3f} ms   "
        f"sum-of-components: {result['component_sum_ms']:.3f} ms   "
        f"(sum/fused = {result['fusion_ratio_sum_over_fused']:.2f}x)",
        f"  policy (separate): {result['policy_ms']:.3f} ms   "
        f"detail: apsp_only={result['detail']['apsp_only']['ms_per_batched_step']:.3f} ms, "
        f"build_direct={result['detail']['build_direct_distance']['ms_per_batched_step']:.3f} ms",
        "",
        f"  {'component':<28} {'µs/env-step':>14} {'fraction':>12}",
        f"  {'-' * 28} {'-' * 14} {'-' * 12}",
    ]
    for name, info in sorted(result["components"].items(), key=lambda kv: -kv[1]["fraction"]):
        lines.append(f"  {name:<28} {info['us_per_env_step']:>14,.3f} {info['fraction']:>12.1%}")
    return "\n".join(lines)


def default_output_stem(cfg: GpuProfileConfig, timestamp: str) -> str:
    ts_safe = timestamp.replace(":", "").replace("-", "").replace("+0000", "Z")
    g_r, g_c = cfg.grid_shape
    return f"gpu_breakdown_step_{g_r}x{g_c}_b{cfg.batch_size}_h{cfg.horizon}_{ts_safe}"


def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="topograph-gpu-profile",
        description="Per-component GPU step breakdown. Writes JSON + CSV (+ PNG if "
                    "matplotlib) into results/profiling/.",
    )
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--k", type=int, default=DEFAULT_K_ITERATIONS, dest="k")
    p.add_argument("--density", type=float, default=0.4)
    p.add_argument("--timed", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--world-seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-figure", action="store_true")
    return p


_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[3] / "results" / "profiling"


def main(argv: list[str] | None = None) -> int:
    import json
    args = _build_parser().parse_args(argv)
    cfg = GpuProfileConfig(
        grid_shape=(args.grid, args.grid),
        batch_size=args.batch,
        horizon=args.horizon,
        k_iterations=args.k,
        mask_density=args.density,
        n_timed=args.timed,
        n_warmup=args.warmup,
        world_seed=args.world_seed,
    )
    result = run(cfg)
    print(format_breakdown(result))

    if args.no_save:
        return 0
    out_dir = args.output_dir or _DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = default_output_stem(cfg, result["timestamp"])
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(result, out_dir / f"{stem}.csv")
    print(f"\nwrote {out_dir / f'{stem}.json'}")
    print(f"wrote {out_dir / f'{stem}.csv'}")
    if not args.no_figure and write_figure(result, out_dir / f"{stem}.png"):
        print(f"wrote {out_dir / f'{stem}.png'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
