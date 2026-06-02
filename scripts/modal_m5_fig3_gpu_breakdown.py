"""Run the GPU per-component step breakdown on a Modal A100.

Runs `python -m topograph.bench.gpu_profiling` on the GPU and copies the
JSON/CSV/PNG back into results/profiling/ (alongside the CPU breakdown, so
the two sit together for the comparison figure).

Prereqs (one time):  pip install modal  &&  modal setup
Run:                 modal run scripts/modal_m5_fig3_gpu_breakdown.py
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# jax pin rationale: see modal_m2_throughput.py (0.4.35 crashes at import on GPU).
# matplotlib included so the breakdown PNG is produced on the runner.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26",
        "scipy>=1.12",
        "networkx>=3.2",
        "matplotlib>=3.8",
        "jax[cuda12]==0.5.3",
    )
    .add_local_dir(str(REPO / "src"), "/root/ParallelTopo/src")
)

app = modal.App("topograph-m5-fig3-gpu-breakdown", image=image)


@app.function(gpu="A100-40GB", timeout=1200)
def run_breakdown(grid: int = 10, batch: int = 256) -> dict:
    import glob
    import subprocess

    workdir = "/root/ParallelTopo"
    proc = subprocess.run(
        ["python", "-m", "topograph.bench.gpu_profiling",
         "--grid", str(grid), "--batch", str(batch)],
        cwd=workdir, capture_output=True, text=True,
        env={"PYTHONPATH": f"{workdir}/src", "PATH": "/usr/local/bin:/usr/bin:/bin"},
    )
    stdout = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")

    files: dict[str, bytes] = {}
    out_dir = Path(workdir) / "results" / "profiling"
    for path in glob.glob(str(out_dir / "gpu_breakdown_*")):
        p = Path(path)
        files[p.name] = p.read_bytes()
    return {"stdout": stdout, "returncode": proc.returncode, "files": files}


@app.local_entrypoint()
def main(grid: int = 10, batch: int = 256) -> None:
    result = run_breakdown.remote(grid, batch)
    print(result["stdout"])

    out_dir = REPO / "results" / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        dest = out_dir / name
        dest.write_bytes(data)
        print(f"  saved: {dest.relative_to(REPO)}")

    if result["returncode"] != 0:
        print(f"\n  NOTE: breakdown exited with code {result['returncode']}.")
