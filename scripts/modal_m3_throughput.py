"""Run the M3 FULL-simulator throughput benchmark on a Modal A100 GPU.

Same setup as scripts/modal_m2_throughput.py, but runs scripts/m3_throughput.py
(the full on-device rollout) instead of the APSP-only M2 bench. Produces the
M5 Figure 1 headline numbers and writes them back into your local
results/m3_throughput/.

Prereqs (one time):  pip install modal  &&  modal setup
Run:                 modal run scripts/modal_m3_throughput.py
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# See modal_m2_throughput.py for the jax pin rationale (jaxlib 0.4.35 crashes
# at `import jax` on GPU; 0.5.3 is a known-good self-contained CUDA build).
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
        str(REPO / "scripts" / "m3_throughput.py"),
        "/root/ParallelTopo/scripts/m3_throughput.py",
    )
)

app = modal.App("topograph-m3-throughput", image=image)


@app.function(gpu="A100-40GB", timeout=1800)  # use "A100-80GB" for 15x15 b=1024
def run_benchmark() -> dict:
    import glob
    import subprocess

    workdir = "/root/ParallelTopo"
    proc = subprocess.run(
        ["python", "scripts/m3_throughput.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")

    files: dict[str, bytes] = {}
    out_dir = Path(workdir) / "results" / "m3_throughput"
    for path in glob.glob(str(out_dir / "m3_throughput_*")):
        p = Path(path)
        files[p.name] = p.read_bytes()

    return {"stdout": stdout, "returncode": proc.returncode, "files": files}


@app.local_entrypoint()
def main() -> None:
    result = run_benchmark.remote()
    print(result["stdout"])

    out_dir = REPO / "results" / "m3_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        dest = out_dir / name
        dest.write_bytes(data)
        print(f"  saved: {dest.relative_to(REPO)}")

    if result["returncode"] != 0:
        print(f"\n  NOTE: bench exited with code {result['returncode']} "
              "(some cells may have OOM'd — check the table above).")
