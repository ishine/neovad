import io

import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel

from neovad.config import DataConfig
from neovad.data.sources import Datasets
from neovad.data.synth import MixtureSynthesizer
from neovad.frontend.mel import FrontendConfig
from neovad.models.vad import VADModel


class AccuracyResult(BaseModel):
    name: str
    roc_auc: float  # threshold-free, the field-standard VAD metric
    f1: float  # best-threshold frame F1
    threshold: float  # threshold achieving that F1
    frames: int


class AccuracyBenchmark:
    """Apples-to-apples speech/non-speech accuracy: neovad and a reference VAD scored on
    the *same* audio against the *same* per-frame ground truth, on neovad's 10 ms grid.

    neovad is scored by its any-speech probability (1 - P(non-speech)) so it is judged on
    the generic task a speech/non-speech VAD does, not its foreground advantage. The
    headline metric is ROC-AUC (threshold-free, so neither model is favoured by threshold
    tuning); frame-F1 at the best threshold is reported alongside.

    A clip is ``(waveform_16k [S], speech_label_per_10ms_frame [T])``.
    """

    SILERO_SR = 16000
    SILERO_WINDOW = 512  # samples (~32 ms) — Silero's fixed chunk

    def __init__(self, hop_ms: float = 10.0, sample_rate: int = 16000):
        self.hop = int(sample_rate * hop_ms / 1000)
        self.sr = sample_rate

    @staticmethod
    def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        # Mann-Whitney U form: P(score(pos) > score(neg)), ties counted as 0.5.
        pos = labels == 1
        n_pos, n_neg = int(pos.sum()), int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        # tie-aware average ranks (1-based), then the Mann-Whitney U identity
        _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)
        avg_rank = (ends - counts + 1 + ends) / 2  # mean of the rank span of each value
        ranks = avg_rank[inv]
        return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    @staticmethod
    def best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            pred = scores >= t
            tp = float((pred & (labels == 1)).sum())
            fp = float((pred & (labels == 0)).sum())
            fn = float((~pred & (labels == 1)).sum())
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_f1, best_t

    @torch.no_grad()
    def neovad_scores(self, model: VADModel, wav: torch.Tensor) -> np.ndarray:
        prob = model.any_speech_probability(model(wav[None]))[0]
        return prob.float().cpu().numpy()  # [T] at 10 ms

    @torch.no_grad()
    def silero_scores(self, silero, wav: torch.Tensor, n_frames: int) -> np.ndarray:
        # one prob per 512-sample window, then mapped onto the 10 ms frame grid
        win = self.SILERO_WINDOW
        probs = []
        for i in range(0, wav.shape[0] - win + 1, win):
            probs.append(float(silero(wav[i : i + win], self.SILERO_SR)))
        probs = np.array(probs) if probs else np.zeros(1)
        idx = np.minimum((np.arange(n_frames) * self.hop) // win, len(probs) - 1)
        return probs[idx]

    def evaluate(self, model: VADModel, clips: list, silero=None) -> list[AccuracyResult]:
        model = model.eval()
        neo_s, neo_y, sil_s, sil_y = [], [], [], []
        for wav, labels in clips:
            y = labels.cpu().numpy()
            ns = self.neovad_scores(model, wav)
            n = min(len(ns), len(y))
            neo_s.append(ns[:n])
            neo_y.append(y[:n])
            if silero is not None:
                ss = self.silero_scores(silero, wav, n)
                sil_s.append(ss[:n])
                sil_y.append(y[:n])
        out = [self._score("neovad", np.concatenate(neo_s), np.concatenate(neo_y))]
        if silero is not None:
            out.append(self._score("silero-v6", np.concatenate(sil_s), np.concatenate(sil_y)))
        return out

    def _score(self, name: str, scores: np.ndarray, labels: np.ndarray) -> AccuracyResult:
        f1, t = self.best_f1(scores, labels)
        return AccuracyResult(
            name=name, roc_auc=self.roc_auc(scores, labels), f1=f1, threshold=t, frames=len(labels)
        )

    @staticmethod
    def synthetic_clips(root: str, n: int, seconds: float = 6.0, seed: int = 900000) -> list:
        # Held-out clips from the SAME synthesis pipeline -> in-distribution for neovad.
        fe = FrontendConfig()
        sp = Datasets.files("speech", root)
        ns = Datasets.files("noise", root) + Datasets.files("music", root)
        rir = Datasets.files("rir", root)
        synth = MixtureSynthesizer(DataConfig(clip_seconds=seconds), fe, sp, ns, rir, seed=seed)
        clips = []
        for i in range(n):
            synth.reseed(seed + i)
            mix = synth.mix_components()
            clips.append((mix.mix, (mix.labels > 0).long()))  # speech = primary OR secondary
        return clips

    @staticmethod
    def voxconverse_clips(n: int, window_s: int = 30, sr: int = 16000, hop: int = 160) -> list:
        # Neutral external benchmark: real conversational speech neither model trained on.
        from datasets import Audio, load_dataset  # optional (the `train` extra)

        ds = load_dataset("diarizers-community/voxconverse", split="test", streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        clips = []
        for ex in ds:
            raw = ex["audio"]["bytes"] or open(ex["audio"]["path"], "rb").read()
            data, _ = sf.read(io.BytesIO(raw), dtype="float32")
            if data.ndim > 1:
                data = data.mean(1)
            win = window_s * sr
            for i in range(0, len(data) - win + 1, win):
                nf = win // hop
                y = np.zeros(nf, dtype=np.int64)
                for s, e in zip(ex["timestamps_start"], ex["timestamps_end"], strict=False):
                    a, b = int((s - i / sr) * sr / hop), int(np.ceil((e - i / sr) * sr / hop))
                    y[max(0, a) : min(nf, b)] = 1
                clips.append((torch.from_numpy(data[i : i + win]), torch.from_numpy(y)))
                if len(clips) >= n:
                    return clips
        return clips

    @staticmethod
    def report(results: list[AccuracyResult]) -> None:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="speech/non-speech accuracy (same audio, same labels)")
        for col in ["model", "ROC-AUC", "frame-F1", "@threshold", "frames"]:
            table.add_column(col, justify="right")
        for r in results:
            table.add_row(
                r.name, f"{r.roc_auc:.4f}", f"{r.f1:.4f}", f"{r.threshold:.2f}", str(r.frames)
            )
        Console().print(table)
