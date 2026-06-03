"""Run the FW-vs-matrix-squaring APSP throughput sweep on a Modal A100 GPU.

This is the GPU half of the transit_learning comparison: the CPU run can't show
TopoGraph's speedup because a CPU has no idle parallelism to reward batching and
torch eager materializes the (B, N, N, N) min-plus intermediate. An A100 fixes
the first problem (and the OOM-resilient sweep documents the second).

Mirrors ParallelTopo/scripts/modal_m3_throughput.py, but with a torch CUDA image
instead of jax, and it uploads transit_learning's torch_utils.py + the
Mandl/Mumford instances so the baseline FW and the real graphs are available
inside the container.

Prereqs (one time):  pip install modal  &&  modal setup
Run overnight:       modal run modal_throughput.py

Output: prints the full sweep and writes a timestamped log to
results/transit_learning_gpu/ alongside this folder.
"""

from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# transit_learning repo root (for torch_utils.py + dataset instances).
# Override by editing this if your transit_learning lives elsewhere.
TL_ROOT = Path("/Users/dannymo/dev/transit_learning")
INSTANCES = TL_ROOT / "datasets" / "mumford_dataset" / "Instances"

# torch's default linux wheels are CUDA (cu12) builds, so plain pip install is a
# self-contained GPU torch. numpy is the only other dep the harness needs.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy>=1.26")
    # the harness itself
    .add_local_file(str(HERE / "compare.py"), "/root/harness/compare.py")
    .add_local_file(str(HERE / "apsp_matrix_squaring.py"),
                    "/root/harness/apsp_matrix_squaring.py")
    .add_local_file(str(HERE / "data_loader.py"), "/root/harness/data_loader.py")
    # transit_learning's baseline FW (standalone, only needs torch)
    .add_local_file(str(TL_ROOT / "torch_utils.py"), "/root/tl/torch_utils.py")
    # the real graphs
    .add_local_dir(str(INSTANCES), "/root/tl/Instances")
)

app = modal.App("transit-learning-apsp-throughput", image=image)

# What the overnight sweep covers. Per-N batch lists are bounded so the eager
# (B, N, N, N) intermediate (~B*N^3*4 bytes) stays roughly under the 40 GB A100;
# anything that still OOMs is caught per-cell and reported, not fatal.
SYNTH_SWEEP = {
    25:  [1, 16, 64, 256, 1024, 4096],
    100: [1, 16, 64, 256, 1024],
    225: [1, 16, 64, 256],
    400: [1, 16, 64],
}
# Real cities, replicated to a batch (apples-to-apples with real topology).
REAL_CITIES = ["Mandl", "Mumford0", "Mumford1", "Mumford2", "Mumford3"]
REAL_BATCHES = [1, 64, 256, 1024]


@app.function(gpu="A100-40GB", timeout=7200)
def run_sweep() -> dict:
    import subprocess

    workdir = "/root/harness"
    common = ["--tl-root", "/root/tl", "--instances-dir", "/root/tl/Instances"]
    chunks: list[str] = []

    def run(args: list[str], banner: str) -> None:
        chunks.append("\n" + "=" * 78 + f"\n# {banner}\n" + "=" * 78)
        proc = subprocess.run(["python", "compare.py", *args], cwd=workdir,
                              capture_output=True, text=True)
        chunks.append(proc.stdout)
        if proc.stderr.strip():
            chunks.append("[stderr]\n" + proc.stderr)

    # 1. Correctness sanity on the GPU (cheap, confirms the port end-to-end).
    run(["equiv", *common], "Equivalence vs transit_learning FW (GPU)")
    run(["ksweep", *common], "Convergence-K vs FW")

    # 2. Synthetic graph-size x batch sweep (the main throughput curve).
    for n, batches in SYNTH_SWEEP.items():
        run(["throughput", *common, "--device", "cuda", "--synth-n", str(n),
             "--batch", *map(str, batches), "--iters", "30"],
            f"Throughput: synthetic N={n}, batches {batches}")

    # 3. Real cities, replicated to a batch.
    for city in REAL_CITIES:
        run(["throughput", *common, "--device", "cuda",
             "--throughput-city", city,
             "--batch", *map(str, REAL_BATCHES), "--iters", "30"],
            f"Throughput: real city {city}, batches {REAL_BATCHES}")

    return {"log": "\n".join(chunks)}


@app.local_entrypoint()
def main() -> None:
    result = run_sweep.remote()
    print(result["log"])

    out_dir = HERE / "results" / "transit_learning_gpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = out_dir / f"gpu_throughput_{ts}.txt"
    dest.write_text(result["log"])
    print(f"\n  saved: {dest}")
