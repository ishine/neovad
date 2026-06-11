from enum import IntEnum
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor, nn

from neovad.registry import Registry


class SpeechClass(IntEnum):
    """Per-frame target classes. The whole point of the library: gate on PRIMARY
    only, so background noise (NON_SPEECH) and interfering voices (SECONDARY) do not
    trigger activation."""

    NON_SPEECH = 0
    PRIMARY = 1
    SECONDARY = 2


class VADHead(Registry, nn.Module, root=True):
    """Per-frame classification head. Pointwise over time (no streaming state), so the
    parallel and streaming paths call the identical ``forward``."""

    kind: ClassVar[str] = ""

    def __init__(self, dim: int, n_classes: int):
        super().__init__()
        self.dim = dim
        self.n_classes = n_classes

    def speech_probability(self, logits: Tensor) -> Tensor:
        # Probability that a frame is the *foreground* speaker — the gating signal.
        if self.n_classes == 1:
            return logits.squeeze(-1).sigmoid()
        return logits.softmax(-1)[..., int(SpeechClass.PRIMARY)]

    def any_speech_probability(self, logits: Tensor) -> Tensor:
        # Probability of *any* speech (primary OR secondary) = 1 - P(non-speech).
        # This is the signal to compare against a generic speech/non-speech VAD.
        if self.n_classes == 1:
            return logits.squeeze(-1).sigmoid()
        return 1.0 - logits.softmax(-1)[..., int(SpeechClass.NON_SPEECH)]

    def any_speech_logit(self, logits: Tensor) -> Tensor:
        # log(P(speech)/P(non-speech)) — sigmoid of this equals any_speech_probability.
        # Autocast-safe target for BCEWithLogits (prob-space BCE is banned under bf16).
        if self.n_classes == 1:
            return logits.squeeze(-1)
        ns = int(SpeechClass.NON_SPEECH)
        speech = torch.cat([logits[..., :ns], logits[..., ns + 1 :]], dim=-1)
        return torch.logsumexp(speech, dim=-1) - logits[..., ns]


class HeadConfig(BaseModel):
    kind: str
    n_classes: int = 3  # NON_SPEECH / PRIMARY / SECONDARY

    def build(self, dim: int) -> VADHead:
        return VADHead.by_name(self.kind)(dim, self)


class LinearHeadConfig(HeadConfig):
    kind: Literal["linear"] = "linear"


class LinearHead(VADHead):
    """Plain dense projection to per-frame class logits."""

    kind = "linear"

    def __init__(self, dim: int, cfg: LinearHeadConfig):
        super().__init__(dim, cfg.n_classes)
        self.proj = nn.Linear(dim, cfg.n_classes)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)


class AttractorHeadConfig(HeadConfig):
    kind: Literal["attractor"] = "attractor"
    proj_dim: int = 0  # 0 -> dim


class AttractorHead(VADHead):
    """Per-frame cosine similarity to learnable primary/secondary speaker attractors
    plus a learnable non-speech bias.

    A lightweight, streaming-safe stand-in for EEND-SAA attractors: it gives the
    foreground/background decision its own embedding geometry (the frame is compared
    to a 'primary speaker' direction) rather than collapsing it into one dense matrix,
    which empirically sharpens primary-vs-secondary separation. Requires >= 2 classes.
    """

    kind = "attractor"

    def __init__(self, dim: int, cfg: AttractorHeadConfig):
        if cfg.n_classes < 2:
            raise ValueError("attractor head needs n_classes >= 2 (non-speech + >=1 speaker)")
        super().__init__(dim, cfg.n_classes)
        d = cfg.proj_dim or dim
        self.proj = nn.Linear(dim, d)
        self.attractors = nn.Parameter(torch.randn(cfg.n_classes - 1, d))
        self.non_speech = nn.Parameter(torch.zeros(1))
        self.temp = nn.Parameter(torch.tensor(10.0))

    def forward(self, h: Tensor) -> Tensor:
        e = F.normalize(self.proj(h), dim=-1)  # [B, T, d]
        a = F.normalize(self.attractors, dim=-1)  # [C-1, d]
        sim = torch.einsum("btd,cd->btc", e, a) * self.temp  # [B, T, C-1]
        ns = self.non_speech.expand(h.shape[0], h.shape[1], 1)  # [B, T, 1]
        return torch.cat([ns, sim], dim=-1)
