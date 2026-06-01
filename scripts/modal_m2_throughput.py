"""Run the M2 APSP throughput benchmark on a Modal A100 GPU.

This ships the repo's `src/` and `scripts/m2_apsp_throughput.py` into a
CUDA-JAX container, runs the benchmark unchanged on an A100, and writes the
resulting JSON/CSV back into your local `results/m2_apsp_throughput/`.

Prereqs (one time):
    pip install modal
    modal setup            # opens browser, links your Modal account

Run:
    modal run scripts/modal_m2_throughput.py

"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]

# CUDA-JAX image. jax[cuda12] bundles the CUDA runtime via pip wheels, so the
# only host requirement is a recent NVIDIA driver, which Modal GPU workers
# provide.
#
# NOTE: jaxlib 0.4.35 has a CUDA-path autodetection bug — it calls
# pathlib.Path(nvidia.cuda_nvcc.__file__) without guarding against that wheel
# being a namespace package (__file__ is None), which crashes at `import jax`
# on GPU. Newer JAX guards it, so we pin a current self-contained CUDA build.
# The APSP kernel is plain jnp.min/broadcasting, so the JAX version has no
# effect on the throughput numbers.
#
# If a future version ever reintroduces a CUDA-wheel resolution problem, the
# robust fallback is a CUDA devel base image + the local-CUDA variant:
#   image = (modal.Image
#       .from_registry("nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04", add_python="3.12")
#       .pip_install("numpy>=1.26","scipy>=1.12","networkx>=3.2","jax[cuda12-local]")
#       ...)
# which links against the system toolkit and skips the pip nvcc-wheel probe.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26",
        "scipy>=1.12",
        "networkx>=3.2",
        "jax[cuda12]==0.5.3",
    )
    # Only the two things the bench needs — keeps the upload tiny and avoids
    # shipping .venv/.git/results.
    .add_local_dir(str(REPO / "src"), "/root/ParallelTopo/src")
    .add_local_file(
        str(REPO / "scripts" / "m2_apsp_throughput.py"),
        "/root/ParallelTopo/scripts/m2_apsp_throughput.py",
    )
)

app = modal.App("topograph-m2-throughput", image=image)


@app.function(gpu="A100-40GB", timeout=1800)  # use "A100-80GB" for 15x15 b=1024
def run_benchmark() -> dict:
    """Run the bench inside the container; return stdout + result file bytes."""
    import glob
    import subprocess

    workdir = "/root/ParallelTopo"
    proc = subprocess.run(
        ["python", "scripts/m2_apsp_throughput.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")

    files: dict[str, bytes] = {}
    out_dir = Path(workdir) / "results" / "m2_apsp_throughput"
    for path in glob.glob(str(out_dir / "m2_throughput_*")):
        p = Path(path)
        files[p.name] = p.read_bytes()

    return {"stdout": stdout, "returncode": proc.returncode, "files": files}


@app.local_entrypoint()
def main() -> None:
    result = run_benchmark.remote()

    print(result["stdout"])

    out_dir = REPO / "results" / "m2_apsp_throughput"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in result["files"].items():
        dest = out_dir / name
        dest.write_bytes(data)
        print(f"  saved: {dest.relative_to(REPO)}")

    if result["returncode"] != 0:
        print(f"\n  NOTE: bench exited with code {result['returncode']} "
              "(some cells may have OOM'd — check the table above).")
