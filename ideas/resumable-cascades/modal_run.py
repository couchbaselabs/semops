"""Run the resumable-cascade experiment on Modal: extract + timing on one NVIDIA L4
(the GPU FINDINGS.md §4.8 used), then the cascade analysis on CPUs.

    modal run --detach modal_run.py              # everything
    modal run --detach modal_run.py --skip-gpu   # re-run only the analysis

Features and results live in a Modal Volume, so a dropped client loses nothing:
    modal volume get semops-resumable-cascades results/analyze.json results/
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REMOTE = "/root/semops/ideas/resumable-cascades"

app = modal.App("semops-resumable-cascades")
vol = modal.Volume.from_name("semops-resumable-cascades", create_if_missing=True)

# this file is re-imported inside the container, where the local repo layout does not exist
SRC = str(HERE.parents[1] / "src") if modal.is_local() else "/root/semops/src"
EXP = str(HERE / "experiment.py") if modal.is_local() else f"{REMOTE}/experiment.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "transformers==5.16.1", "datasets", "scikit-learn", "numpy",
                 "huggingface_hub", "accelerate")
    .env({"HF_HOME": "/vol/hf", "RC_CACHE": "/vol/cache", "RC_RESULTS": "/vol/results",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(SRC, "/root/semops/src")
    .add_local_file(EXP, f"{REMOTE}/experiment.py")
)


def _experiment():
    import sys
    sys.path.insert(0, REMOTE)
    import experiment
    return experiment


@app.function(image=image, gpu=["L4", "A10G"], volumes={"/vol": vol}, timeout=4 * 3600, memory=49152)
def gpu_stage(force: bool = False) -> dict:
    import types
    E = _experiment()
    a = types.SimpleNamespace(force=force)
    E.extract(a)
    vol.commit()
    E.timing(a)
    vol.commit()
    return {k: (E.RESULTS / f"{k}.json").read_text() for k in ("extract", "timing")}


@app.function(image=image, cpu=32, memory=65536, volumes={"/vol": vol}, timeout=3 * 3600)
def cpu_stage(seeds: int, samples: list) -> str:
    import types
    vol.reload()
    E = _experiment()
    a = types.SimpleNamespace(seeds=seeds, samples=samples, recall=0.9, precision=0.9, delta=0.1, workers=32)
    E.analyze(a)
    vol.commit()
    return (E.RESULTS / "analyze.json").read_text()


@app.function(image=image, volumes={"/vol": vol}, timeout=8 * 3600)
def pipeline(seeds: int = 30, force: bool = False) -> str:
    """Server-side orchestration, so the run does not depend on the local client's
    connection. Deploy, then spawn:
        modal deploy modal_run.py
        python -c "import modal; modal.Function.from_name('semops-resumable-cascades', 'pipeline').spawn()"
    Results land in the Volume under results/."""
    gpu_stage.remote(force)
    return cpu_stage.remote(seeds, [100, 400])


@app.local_entrypoint()
def main(seeds: int = 30, skip_gpu: bool = False, force: bool = False):
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    if not skip_gpu:
        for k, v in gpu_stage.remote(force).items():
            (out / f"{k}.json").write_text(v)
            print("wrote", out / f"{k}.json")
    (out / "analyze.json").write_text(cpu_stage.remote(seeds, [100, 400]))
    print("wrote", out / "analyze.json")
