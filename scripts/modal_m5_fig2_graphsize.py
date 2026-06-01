"""Run the M5 Figure-2 graph-size sweep on a Modal A100.

Sweeps N up to 30x30 at batch {256, 1024} to locate the throughput ceiling.
The largest cells (e.g. 30x30 @ batch 1024) may approach the 40 GB limit; if
one OOMs, the bench records it as the ceiling and continues. Switch to
"A100-80GB" below for extra headroom.

Prereqs (one time):  pip install modal  &&  modal setup
Run:                 modal run scripts/modal_m5_fig2_graphsize.py
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
        str(REPO / "scripts" / "m5_fig2_graphsize.py"),
        "/root/ParallelTopo/scripts/m5_fig2_graphsize.py",
    )
)

app = modal.App("topograph-m5-fig2-graphsize", image=image)


@app.function(gpu="A100-40GB", timeout=2400)  # use "A100-80GB" for the largest cells
def run_benchmark() -> dict:
    import glob
    import subprocess

    workdir = "/root/ParallelTopo"
    proc = subprocess.run(
        ["python", "scripts/m5_fig2_graphsize.py"],
        cwd=workdir, capture_output=True, text=True,
    )
    stdout = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")

    files: dict[str, bytes] = {}
    out_dir = Path(workdir) / "results" / "m5_fig2_graphsize"
    for path in glob.glob(str(out_dir / "m5_fig2_graphsize_*")):
        p = Path(path)
        files[p.name] = p.read_bytes()
    return {"stdout": stdout, "returncode": proc.returncode, "files": files}


@app.local_entrypoint()
def main() -> None:
    result = run_benchmark.remote()
    print(result["stdout"])

    out_dir = REPO / "results" / "m5_fig2_graphsize"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        dest = out_dir / name
        dest.write_bytes(data)
        print(f"  saved: {dest.relative_to(REPO)}")

    if result["returncode"] != 0:
        print(f"\n  NOTE: bench exited with code {result['returncode']} "
              "(a cell may have hit the ceiling — check the table above).")
