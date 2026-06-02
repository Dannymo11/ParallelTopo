"""Run the M4 end-to-end training-throughput study on a Modal A100 GPU.

Same setup as scripts/modal_m3_throughput.py, but runs
scripts/m4_train_throughput.py — the RQ3 driver that times sim-only vs
policy-inference vs full-train-step on-device and writes the results back
into your local results/m4_train_throughput/.

Prereqs (one time):  pip install modal  &&  modal setup
Run:                 modal run scripts/modal_m4_train_throughput.py
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# Same proven image as the M2/M3 runners: jaxlib 0.4.35 crashes at `import
# jax` on GPU; 0.5.3 is a known-good self-contained CUDA build. The MLP
# policy is hand-rolled, so no flax/optax needed — the image is unchanged.
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
        str(REPO / "scripts" / "m4_train_throughput.py"),
        "/root/ParallelTopo/scripts/m4_train_throughput.py",
    )
)

app = modal.App("topograph-m4-train-throughput", image=image)


@app.function(gpu="A100-40GB", timeout=1800)  # use "A100-80GB" for 15x15 b=1024
def run_benchmark() -> dict:
    import glob
    import subprocess

    workdir = "/root/ParallelTopo"
    proc = subprocess.run(
        ["python", "scripts/m4_train_throughput.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")

    files: dict[str, bytes] = {}
    out_dir = Path(workdir) / "results" / "m4_train_throughput"
    for path in glob.glob(str(out_dir / "m4_train_throughput_*")):
        p = Path(path)
        files[p.name] = p.read_bytes()

    return {"stdout": stdout, "returncode": proc.returncode, "files": files}


@app.local_entrypoint()
def main() -> None:
    result = run_benchmark.remote()
    print(result["stdout"])

    out_dir = REPO / "results" / "m4_train_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        dest = out_dir / name
        dest.write_bytes(data)
        print(f"  saved: {dest.relative_to(REPO)}")

    if result["returncode"] != 0:
        print(f"\n  NOTE: bench exited with code {result['returncode']} "
              "(some cells may have OOM'd — check the table above).")
