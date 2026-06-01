"""Run the M5 Figure-4 variable-topology overhead bench on a Modal A100.

Same image/setup as the other Modal runners; runs scripts/m5_fig4_overhead.py
and writes results back to results/m5_fig4_overhead/.

Prereqs (one time):  pip install modal  &&  modal setup
Run:                 modal run scripts/modal_m5_fig4_overhead.py
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# jax pin rationale: see modal_m2_throughput.py (0.4.35 crashes at import on GPU).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26",
        "scipy>=1.12",
        "networkx>=3.2",
        "jax[cuda12]==0.5.3",
    )
    .add_local_dir(str(REPO / "src"), "/root/ParallelTopo/src")
    .add_local_file(
        str(REPO / "scripts" / "m5_fig4_overhead.py"),
        "/root/ParallelTopo/scripts/m5_fig4_overhead.py",
    )
)

app = modal.App("topograph-m5-fig4-overhead", image=image)


@app.function(gpu="A100-40GB", timeout=1800)
def run_benchmark() -> dict:
    import glob
    import subprocess

    workdir = "/root/ParallelTopo"
    proc = subprocess.run(
        ["python", "scripts/m5_fig4_overhead.py"],
        cwd=workdir, capture_output=True, text=True,
    )
    stdout = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")

    files: dict[str, bytes] = {}
    out_dir = Path(workdir) / "results" / "m5_fig4_overhead"
    for path in glob.glob(str(out_dir / "m5_fig4_overhead_*")):
        p = Path(path)
        files[p.name] = p.read_bytes()
    return {"stdout": stdout, "returncode": proc.returncode, "files": files}


@app.local_entrypoint()
def main() -> None:
    result = run_benchmark.remote()
    print(result["stdout"])

    out_dir = REPO / "results" / "m5_fig4_overhead"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        dest = out_dir / name
        dest.write_bytes(data)
        print(f"  saved: {dest.relative_to(REPO)}")

    if result["returncode"] != 0:
        print(f"\n  NOTE: bench exited with code {result['returncode']}.")
