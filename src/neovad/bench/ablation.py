import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from pydantic import BaseModel

from neovad.bench.accuracy import AccuracyBenchmark
from neovad.bench.latency import LatencyBenchmark
from neovad.config import NeoVADConfig
from neovad.models.vad import VADModel

# Developer one-shot utility (not a CLI surface): trains a grid of configs with an
# identical budget, one process per GPU, and scores each run on the SAME three axes —
# neutral VoxConverse ROC-AUC, held-out synthetic metrics, and streaming RTF — so
# architecture/data choices are compared apples-to-apples.
#   queue : uv run python -m neovad.bench.ablation queue <study_dir>
#   worker: (spawned by queue)  ... worker <study_dir>/<run_name>


class AblationRun(BaseModel):
    name: str
    config: NeoVADConfig

    def dir(self, study_dir: Path) -> Path:
        return study_dir / self.name


class AblationStudy(BaseModel):
    runs: list[AblationRun]
    gpus: list[int] = [0, 1]
    vox_cache: str = "/disk/manual/voxconverse_eval.pt"
    vox_clips: int = 153

    def pending(self, study_dir: Path) -> list[AblationRun]:
        return [r for r in self.runs if not (r.dir(study_dir) / "results.json").exists()]

    def ensure_vox_cache(self) -> None:
        cache = Path(self.vox_cache)
        if cache.exists():
            return
        clips = AccuracyBenchmark.voxconverse_clips(self.vox_clips)
        torch.save(clips, cache)
        print(f"[ablation] cached {len(clips)} VoxConverse windows -> {cache}", flush=True)

    def execute(self, study_dir: Path) -> None:
        study_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_vox_cache()
        queue = self.pending(study_dir)
        print(f"[ablation] {len(queue)} runs pending on GPUs {self.gpus}", flush=True)
        active: dict[int, tuple[subprocess.Popen, AblationRun]] = {}
        while queue or active:
            for gpu in [g for g in self.gpus if g not in active]:
                if not queue:
                    break
                run = queue.pop(0)
                rdir = run.dir(study_dir)
                rdir.mkdir(parents=True, exist_ok=True)
                (rdir / "run.json").write_text(run.model_dump_json())
                (rdir / "study.json").write_text(self.model_dump_json())
                env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
                log = open(rdir / "worker.log", "w")
                proc = subprocess.Popen(
                    [sys.executable, "-m", "neovad.bench.ablation", "worker", str(rdir)],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                active[gpu] = (proc, run)
                print(f"[ablation] {run.name} -> GPU {gpu} (pid {proc.pid})", flush=True)
            for gpu, (proc, run) in list(active.items()):
                if proc.poll() is not None:
                    status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
                    print(f"[ablation] {run.name} finished: {status}", flush=True)
                    del active[gpu]
            time.sleep(10)
        self.summarize(study_dir)

    def summarize(self, study_dir: Path) -> list[dict]:
        rows = []
        for run in self.runs:
            path = run.dir(study_dir) / "results.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
        rows.sort(key=lambda r: -r.get("vox_auc", 0.0))
        (study_dir / "summary.json").write_text(json.dumps(rows, indent=2))
        cols = [
            "name",
            "vox_auc",
            "vox_f1",
            "synth_auc",
            "primary_f1",
            "false_fire",
            "rtf30",
            "params_m",
        ]
        print("\t".join(cols), flush=True)
        for r in rows:
            print(
                "\t".join(
                    f"{r.get(c):.4f}" if isinstance(r.get(c), float) else str(r.get(c))
                    for c in cols
                ),
                flush=True,
            )
        return rows


class AblationWorker:
    """Runs inside one GPU process: train with the run's config, then score."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.run = AblationRun.model_validate_json((run_dir / "run.json").read_text())
        self.study = AblationStudy.model_validate_json((run_dir / "study.json").read_text())

    def train(self) -> VADModel:
        from neovad.train.lit import NeoVADLit  # heavy import kept out of queue process

        torch.set_float32_matmul_precision("high")
        cfg = self.run.config.model_copy(deep=True)
        cfg.name = self.run.name
        cfg.train.output_dir = str(self.dir)
        return NeoVADLit.run(cfg)

    def score(self, model: VADModel) -> dict:
        model = model.eval().cpu()
        acc = AccuracyBenchmark()
        vox = acc.evaluate(model, torch.load(self.study.vox_cache, weights_only=False))[0]
        synth = acc.evaluate(
            model, AccuracyBenchmark.synthetic_clips(self.run.config.data.root, 60)
        )[0]
        fg = self.foreground_metrics(model)
        rtf = LatencyBenchmark(seconds=10, threads=1).bench_neovad(model, chunk_hops=3)
        return {
            "name": self.run.name,
            "vox_auc": vox.roc_auc,
            "vox_f1": vox.f1,
            "synth_auc": synth.roc_auc,
            "primary_f1": fg["primary_f1"],
            "false_fire": fg["false_fire"],
            "rtf30": rtf.rtf,
            "params_m": rtf.params_m,
        }

    @torch.no_grad()
    def foreground_metrics(self, model: VADModel) -> dict:
        from neovad.config import DataConfig
        from neovad.data.sources import Datasets
        from neovad.data.synth import MixtureSynthesizer
        from neovad.frontend.mel import FrontendConfig
        from neovad.nn.head import SpeechClass

        data_cfg = self.run.config.data
        root = data_cfg.root
        synth = MixtureSynthesizer(
            DataConfig(
                clip_seconds=6.0,
                speech_sources=data_cfg.speech_sources,
                noise_sources=data_cfg.noise_sources,
            ),
            FrontendConfig(),
            Datasets.files("speech", root, data_cfg.speech_sources),
            Datasets.files("noise", root, data_cfg.noise_sources)
            + Datasets.files("music", root, data_cfg.noise_sources),
            Datasets.files("rir", root),
            seed=555,
        )
        tp = fp = fn = fire = sec = 0.0
        for i in range(60):
            synth.reseed(800000 + i)
            mix = synth.mix_components()
            logits = model(mix.mix[None])
            prob = model.speech_probability(logits)[0]
            n = min(prob.shape[0], mix.labels.shape[0])
            pred = prob[:n] > 0.5
            tgt = mix.labels[:n] == int(SpeechClass.PRIMARY)
            secondary = mix.labels[:n] == int(SpeechClass.SECONDARY)
            tp += float((pred & tgt).sum())
            fp += float((pred & ~tgt).sum())
            fn += float((~pred & tgt).sum())
            fire += float((pred & secondary).sum())
            sec += float(secondary.sum())
        return {
            "primary_f1": 2 * tp / (2 * tp + fp + fn + 1e-9),
            "false_fire": fire / (sec + 1e-9),
        }

    def execute(self) -> None:
        model = self.train()
        results = self.score(model)
        (self.dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"[worker] {self.run.name}: {results}", flush=True)


def finalize(src: Path, dst: Path, top_k: int, epochs: int, steps: int) -> None:
    # Long-train the study's top-k configs (by neutral VoxConverse AUC) as a new study.
    study = AblationStudy.model_validate_json((src / "study.json").read_text())
    ranking = study.summarize(src)
    winners = {r["name"] for r in ranking[:top_k]}
    finals = []
    for run in study.runs:
        if run.name not in winners:
            continue
        cfg = run.config.model_copy(deep=True)
        cfg.train.max_epochs = epochs
        cfg.data.steps_per_epoch = steps
        cfg.train.warmup_steps = max(cfg.train.warmup_steps, steps)
        finals.append(AblationRun(name=f"{run.name}_final", config=cfg))
    final_study = AblationStudy(runs=finals, gpus=study.gpus, vox_cache=study.vox_cache)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "study.json").write_text(final_study.model_dump_json(indent=2))
    final_study.execute(dst)


def main(argv: list[str]) -> None:
    mode, target = argv[1], Path(argv[2])
    if mode == "worker":
        AblationWorker(target).execute()
    elif mode == "queue":
        study = AblationStudy.model_validate_json((target / "study.json").read_text())
        study.execute(target)
    elif mode == "finalize":
        finalize(target, Path(argv[3]), top_k=int(argv[4]), epochs=int(argv[5]), steps=int(argv[6]))
    else:
        raise SystemExit(f"unknown mode {mode!r}; use queue|worker|finalize")


if __name__ == "__main__":
    main(sys.argv)
