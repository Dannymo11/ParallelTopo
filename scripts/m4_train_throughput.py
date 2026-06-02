"""M4 end-to-end training-throughput study (RQ3).

Does the simulator's throughput survive once a policy is in the loop, or
does the policy / host boundary dominate? This script answers that by
timing three on-device drivers on the *same* 10x10/horizon-15 workload and
batch sweep, and reporting the ratios between them:

  1. sim-only       — `rollout_random`  (the M5 Fig-1 driver, no policy net)
  2. policy-infer    — `rollout_mlp`     (simulator + 64-unit MLP forward)
  3. train-step      — `train_step`      (simulator + MLP + REINFORCE backward
                                          + an SGD update)

All three keep the policy and the environment on one device under one
compiled program (PureJaxRL-style), so the only thing that changes between
rows is how much policy work rides on top of the simulator. The headline
RQ3 number is `train-step env-steps/sec  /  sim-only env-steps/sec` at the
sweet-spot batch: how much of the simulator's throughput is left once you
are actually training. The batch at which the train-step curve stops rising
is the "policy becomes the bottleneck" point (pairs with M5 Figure 3).

Metric: env-steps/sec = B * HORIZON / wall_time_of_one_call. A rollout/
train-step call advances B envs for HORIZON steps on-device with nothing
transferred to the host mid-call.

NOT in scope here (the M4 spec's explicit non-goals): training to
convergence, tuning, or evaluating policy quality. And item 4 of the M4
plan — a multi-process SB3 `VectorEnv` CPU *training* baseline for an
end-to-end GPU-vs-CPU training speedup — is intentionally left as a
separate harness; see the note printed at the end. This script establishes
the on-device numbers, which are the upper bound that baseline is compared
against.

Outputs:
  results/m4_train_throughput/m4_train_throughput_<ts>.json
  results/m4_train_throughput/m4_train_throughput_<ts>.csv

Run on GPU:  python scripts/m4_train_throughput.py
             (or via scripts/modal_m4_train_throughput.py)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    print("ERROR: JAX not installed. Install the gpu extras:")
    print('  uv pip install -e ".[gpu]"')
    print(f"\nUnderlying error: {e}")
    sys.exit(1)

from topograph.sim_cpu import GridCityConfig, make_world
from topograph.sim_gpu import (
    DEFAULT_HIDDEN,
    DEFAULT_K_ITERATIONS,
    feature_dim,
    init_mlp_params,
    n_actions,
    reset_batched,
    rollout_mlp,
    rollout_random,
    train_step,
    world_arrays_from_config,
)

# Workload — matches the M1/M3 reference workload so the rows compose with
# the existing throughput tables.
HORIZON: int = 15
GRID_SHAPES: list[tuple[int, int]] = [(10, 10)]  # add (15, 15) for the size story
BATCH_SIZES: list[int] = [1, 16, 64, 256, 1024]
HIDDEN_UNITS: int = DEFAULT_HIDDEN  # 64, per the M4 spec
LEARNING_RATE: float = 1e-3

REFERENCE_GRID_SHAPE: tuple[int, int] = (10, 10)

WARMUP_CALLS: int = 2   # first call JIT-compiles for this (grid, batch, driver)
TIMED_CALLS: int = 5    # median over this many
SEED: int = 0

CPU_TEMP_BUDGET_GB: float = 8.0  # skip oversized cells when running on the host


def predicted_temp_gb(grid_shape: tuple[int, int], batch_size: int) -> float:
    n_zones = grid_shape[0] * grid_shape[1]
    return batch_size * (n_zones ** 3) * 4 / (1024 ** 3)


@dataclass(frozen=True)
class DriverResult:
    driver: str            # "sim_only" | "policy_infer" | "train_step"
    grid_shape: tuple[int, int]
    n_zones: int
    batch_size: int
    median_call_s: float
    rollouts_per_sec: float
    env_steps_per_sec: float

    def as_dict(self) -> dict:
        return {
            "driver": self.driver,
            "grid_shape": list(self.grid_shape),
            "n_zones": self.n_zones,
            "batch_size": self.batch_size,
            "median_call_s": self.median_call_s,
            "rollouts_per_sec": self.rollouts_per_sec,
            "env_steps_per_sec": self.env_steps_per_sec,
        }


def _time_median(fn, warmup: int, timed: int) -> float:
    """Median wall-time of `fn()`; `fn` must block on its own output."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(timed):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def benchmark_cell(grid_shape: tuple[int, int], batch_size: int) -> list[DriverResult]:
    """Time all three drivers for one (grid, batch) cell."""
    world = make_world(SEED, GridCityConfig(grid_shape=grid_shape, horizon=HORIZON))
    wa = world_arrays_from_config(world)
    budget0 = float(world.initial_budget)
    state = reset_batched(
        wa, batch_size,
        jnp.asarray(world.initial_activity, dtype=jnp.float32),
        budget0,
    )
    key = jax.random.PRNGKey(SEED)
    params = init_mlp_params(
        jax.random.PRNGKey(SEED + 1),
        in_dim=feature_dim(wa),
        out_dim=n_actions(wa),
        hidden=HIDDEN_UNITS,
    )

    def sim_only():
        _f, r, _a = rollout_random(state, key, wa, horizon=HORIZON,
                                   k_iterations=DEFAULT_K_ITERATIONS)
        r.block_until_ready()

    def policy_infer():
        _f, r, _a = rollout_mlp(params, state, key, wa, horizon=HORIZON,
                                k_iterations=DEFAULT_K_ITERATIONS, budget0=budget0)
        r.block_until_ready()

    def do_train_step():
        _p, loss = train_step(params, state, key, wa, horizon=HORIZON,
                              k_iterations=DEFAULT_K_ITERATIONS,
                              budget0=budget0, lr=LEARNING_RATE)
        loss.block_until_ready()

    drivers = [
        ("sim_only", sim_only),
        ("policy_infer", policy_infer),
        ("train_step", do_train_step),
    ]
    out: list[DriverResult] = []
    for name, fn in drivers:
        median_t = _time_median(fn, WARMUP_CALLS, TIMED_CALLS)
        rps = batch_size / median_t
        out.append(DriverResult(
            driver=name,
            grid_shape=grid_shape,
            n_zones=int(world.n_zones),
            batch_size=batch_size,
            median_call_s=median_t,
            rollouts_per_sec=rps,
            env_steps_per_sec=rps * HORIZON,
        ))
    return out


def _by(results: list[DriverResult], driver: str, grid, batch) -> DriverResult | None:
    return next(
        (r for r in results
         if r.driver == driver and r.grid_shape == grid and r.batch_size == batch),
        None,
    )


def print_results(results: list[DriverResult]) -> None:
    print(f"\n{'=' * 96}")
    print(f"  M4 end-to-end throughput  (HIDDEN={HIDDEN_UNITS} MLP, K={DEFAULT_K_ITERATIONS}, "
          f"fp32, HORIZON={HORIZON})")
    print("  env-steps/sec = batch * horizon / call-time;  policy & sim on one device")
    print("=" * 96)
    header = (f"\n  {'grid':<8}{'batch':>7}  "
              f"{'sim-only':>13}  {'policy-infer':>14}  {'train-step':>13}  "
              f"{'infer/sim':>10}  {'train/sim':>10}")
    print(header)
    print(f"  {'-' * 92}")
    grids = sorted({r.grid_shape for r in results})
    batches = sorted({r.batch_size for r in results})
    for grid in grids:
        for batch in batches:
            sim = _by(results, "sim_only", grid, batch)
            inf = _by(results, "policy_infer", grid, batch)
            trn = _by(results, "train_step", grid, batch)
            if not (sim and inf and trn):
                continue
            infer_ratio = inf.env_steps_per_sec / sim.env_steps_per_sec
            train_ratio = trn.env_steps_per_sec / sim.env_steps_per_sec
            print(
                f"  {str(grid):<8}{batch:>7}  "
                f"{sim.env_steps_per_sec:>13,.0f}  "
                f"{inf.env_steps_per_sec:>14,.0f}  "
                f"{trn.env_steps_per_sec:>13,.0f}  "
                f"{infer_ratio:>9.2f}x  {train_ratio:>9.2f}x"
            )


def rq3_verdict(results: list[DriverResult]) -> dict:
    """Summarize RQ3 on the reference grid: throughput retained under the
    policy, and the batch at which the train-step curve stops rising."""
    grid = REFERENCE_GRID_SHAPE
    batches = sorted({r.batch_size for r in results if r.grid_shape == grid})
    per_batch = {}
    train_curve: list[tuple[int, float]] = []
    for b in batches:
        sim = _by(results, "sim_only", grid, b)
        inf = _by(results, "policy_infer", grid, b)
        trn = _by(results, "train_step", grid, b)
        if not (sim and inf and trn):
            continue
        per_batch[b] = {
            "sim_only_env_steps_per_sec": sim.env_steps_per_sec,
            "policy_infer_env_steps_per_sec": inf.env_steps_per_sec,
            "train_step_env_steps_per_sec": trn.env_steps_per_sec,
            "infer_over_sim": inf.env_steps_per_sec / sim.env_steps_per_sec,
            "train_over_sim": trn.env_steps_per_sec / sim.env_steps_per_sec,
        }
        train_curve.append((b, trn.env_steps_per_sec))

    # "Sweet spot" = batch with the highest train-step env-steps/sec.
    sweet = max(train_curve, key=lambda kv: kv[1]) if train_curve else (None, None)
    return {
        "reference_grid_shape": list(grid),
        "per_batch": per_batch,
        "train_step_sweet_spot_batch": sweet[0],
        "train_step_sweet_spot_env_steps_per_sec": sweet[1],
    }


def main() -> None:
    devices = jax.devices()
    print("=" * 96)
    print("M4 end-to-end training-throughput study (RQ3)")
    print("=" * 96)
    print(f"  JAX devices: {devices}")
    print(f"  Default backend: {jax.default_backend()}")
    if jax.default_backend() == "cpu":
        print("\n  WARNING: running on CPU. Fine for a smoke test, but the RQ3")
        print("  throughput story is defined against a GPU. Numbers below are not it.")
    print(f"  Grid shapes:  {GRID_SHAPES}")
    print(f"  Batch sizes:  {BATCH_SIZES}")
    print(f"  MLP hidden:   {HIDDEN_UNITS}   LR: {LEARNING_RATE}")

    is_cpu = jax.default_backend() == "cpu"
    results: list[DriverResult] = []
    for grid_shape in GRID_SHAPES:
        for batch_size in BATCH_SIZES:
            if is_cpu and predicted_temp_gb(grid_shape, batch_size) > CPU_TEMP_BUDGET_GB:
                print(f"  (skipping grid={grid_shape} batch={batch_size}: over CPU budget)")
                continue
            try:
                results.extend(benchmark_cell(grid_shape, batch_size))
            except Exception as e:  # noqa: BLE001 — partial data is still useful
                print(f"\n  FAILED grid={grid_shape} batch={batch_size}: {e}")

    print_results(results)
    verdict = rq3_verdict(results)

    print(f"\n{'=' * 96}")
    print("  RQ3 verdict (reference grid 10x10)")
    print("=" * 96)
    for b, v in verdict["per_batch"].items():
        print(f"  batch {b:>5}:  train-step keeps {v['train_over_sim'] * 100:5.1f}% of "
              f"sim-only throughput  (inference keeps {v['infer_over_sim'] * 100:5.1f}%)")
    if verdict["train_step_sweet_spot_batch"] is not None:
        print(f"\n  Train-step sweet spot: batch {verdict['train_step_sweet_spot_batch']} "
              f"@ {verdict['train_step_sweet_spot_env_steps_per_sec']:,.0f} env-steps/sec")

    out_dir = REPO_ROOT / "results" / "m4_train_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    json_path = out_dir / f"m4_train_throughput_{ts}.json"
    json_path.write_text(json.dumps({
        "config": {
            "grid_shapes": GRID_SHAPES,
            "batch_sizes": BATCH_SIZES,
            "horizon": HORIZON,
            "k_iterations": DEFAULT_K_ITERATIONS,
            "hidden_units": HIDDEN_UNITS,
            "learning_rate": LEARNING_RATE,
            "drivers": ["sim_only (rollout_random)",
                        "policy_infer (rollout_mlp)",
                        "train_step (REINFORCE + SGD)"],
            "metric": "env_steps_per_sec = batch * horizon / call_time",
            "warmup_calls": WARMUP_CALLS,
            "timed_calls": TIMED_CALLS,
            "reference_grid_shape": list(REFERENCE_GRID_SHAPE),
            "backend": jax.default_backend(),
            "devices": [str(d) for d in devices],
            "scope_note": ("On-device numbers only. Item 4 of the M4 plan — a "
                           "multi-process SB3 VectorEnv CPU training baseline — is a "
                           "separate harness; these numbers are its upper bound."),
        },
        "results": [r.as_dict() for r in results],
        "verdict": verdict,
    }, indent=2))

    csv_path = out_dir / f"m4_train_throughput_{ts}.csv"
    if results:
        fieldnames = list(results[0].as_dict().keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = r.as_dict()
                row["grid_shape"] = f"{r.grid_shape[0]}x{r.grid_shape[1]}"
                writer.writerow(row)

    print(f"\n  Wrote: {json_path.relative_to(REPO_ROOT)}")
    if results:
        print(f"  Wrote: {csv_path.relative_to(REPO_ROOT)}")
    print("\n  NOTE (M4 item 4): the GPU-vs-CPU end-to-end *training* speedup needs a")
    print("  multi-process SB3 VectorEnv baseline on the same task — not yet built.")
    print("  The train-step env-steps/sec above is the upper bound it is measured against.")


if __name__ == "__main__":
    main()
