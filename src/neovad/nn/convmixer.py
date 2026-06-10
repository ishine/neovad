from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neovad.nn.conv import CausalDepthwiseConv1d, ConvState
from neovad.nn.mixer import MixerConfig, StreamingMixer
from neovad.nn.norm import RMSNorm


class ConvMixerConfig(MixerConfig):
    kind: Literal["convmixer"] = "convmixer"
    kernel: int = 15  # ~150 ms left receptive field per layer at 10 ms hop
    expand: int = 2


class ConvMixer(StreamingMixer):
    """Causal Conformer convolution module (Gulati et al., 2020), streaming-adapted.

    ``pointwise(2x, GLU) -> causal depthwise conv -> RMSNorm -> SiLU -> pointwise``.
    BatchNorm from the paper is replaced by RMSNorm (no batch statistics, so the
    streaming path is identical to training). Local-context-only by construction —
    the strong audio prior; its receptive field grows linearly with depth and its
    streaming state is just each layer's conv tail.
    """

    kind = "convmixer"

    def __init__(self, dim: int, cfg: ConvMixerConfig, depth: int = 1):
        super().__init__(dim)
        d_inner = cfg.expand * dim
        self.pw_in = nn.Linear(dim, 2 * d_inner, bias=False)  # GLU halves back to d_inner
        self.conv = CausalDepthwiseConv1d(d_inner, cfg.kernel, bias=True)
        self.norm = RMSNorm(d_inner)
        self.pw_out = nn.Linear(d_inner, dim, bias=False)

    def mix(self, x: Tensor, conv_out: Tensor) -> Tensor:
        return self.pw_out(F.silu(self.norm(conv_out)))

    def forward(self, x: Tensor) -> Tensor:
        z = F.glu(self.pw_in(x), dim=-1)
        return self.mix(x, self.conv(z))

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> ConvState:
        return self.conv.init_state(batch, device, dtype)

    def step(self, x: Tensor, state: ConvState) -> Tensor:
        z = F.glu(self.pw_in(x), dim=-1)
        return self.mix(x, self.conv.step(z, state))
