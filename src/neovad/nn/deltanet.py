from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from neovad.nn.conv import CausalDepthwiseConv1d, ConvState
from neovad.nn.mixer import MixerConfig, ModuleState, StreamingMixer
from neovad.nn.norm import RMSNormGated


class GatedDeltaNetState(ModuleState):
    s: Tensor  # [B, H, d_k, d_v] fast-weight memory
    conv: ConvState  # shared short-conv tail over the packed q/k/v projections


class GatedDeltaNetConfig(MixerConfig):
    kind: Literal["gdn"] = "gdn"
    n_heads: int = 4
    d_conv: int = 4


class GatedDeltaNetMixer(StreamingMixer):
    """Gated DeltaNet (Yang et al., 2024) — gated delta-rule fast-weight memory.

    Per head, the state is a small associative matrix updated by erase-then-write:
    ``S_t = alpha_t * (S_{t-1} - beta_t (S_{t-1} k_t) k_t^T) + beta_t v_t k_t^T``,
    ``o_t = S_t^T q_t`` — an online gradient step on ``||S k - v||^2`` with a
    Mamba-style decay ``alpha`` and learning-rate gate ``beta``. The targeted
    overwrite is exactly the behaviour wanted for foreground tracking: replace the
    stale speaker/noise estimate addressed by ``k_t`` instead of blending into it.

    The training path runs the same recurrence as a vectorized sequential scan over
    time (exactness-first; the chunked WY form is a future optimization) — at VAD clip
    lengths (~400 frames) this stays cheap. ``q/k`` are short-conv'd, SiLU'd and
    L2-normalized per head as in the paper.
    """

    kind = "gdn"

    def __init__(self, dim: int, cfg: GatedDeltaNetConfig, depth: int = 1):
        if dim % cfg.n_heads != 0:
            raise ValueError(f"dim {dim} not divisible by n_heads {cfg.n_heads}")
        super().__init__(dim)
        self.h = cfg.n_heads
        self.dk = dim // cfg.n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.conv = CausalDepthwiseConv1d(3 * dim, cfg.d_conv, bias=True)
        self.w_alpha = nn.Linear(dim, cfg.n_heads, bias=True)
        self.a_log = nn.Parameter(torch.zeros(cfg.n_heads))
        self.w_beta = nn.Linear(dim, cfg.n_heads, bias=True)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNormGated(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def projections(self, x: Tensor, conv_out: Tensor) -> tuple[Tensor, ...]:
        q, k, v = rearrange(F.silu(conv_out), "b t (three h d) -> three b t h d", three=3, h=self.h)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        alpha = torch.exp(-F.softplus(self.w_alpha(x)) * torch.exp(self.a_log))  # [B,T,H]
        beta = torch.sigmoid(self.w_beta(x))  # [B,T,H]
        return q, k, v, alpha, beta

    def scan(self, q, k, v, alpha, beta, s: Tensor) -> tuple[Tensor, Tensor]:
        # sequential delta-rule scan; all ops vectorized over [B, H]
        outs = []
        for t in range(q.shape[1]):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]  # [B,H,d]
            at = alpha[:, t][..., None, None]  # [B,H,1,1]
            bt = beta[:, t][..., None]  # [B,H,1]
            sk = torch.einsum("bhkv,bhk->bhv", s, kt)  # k^T S -> currently stored value
            erase = torch.einsum("bhv,bhk->bhkv", bt * sk, kt)
            write = torch.einsum("bhv,bhk->bhkv", bt * vt, kt)
            s = at * (s - erase) + write  # decay applies to the erased memory, not the write
            outs.append(torch.einsum("bhkv,bhk->bhv", s, qt))
        return rearrange(torch.stack(outs, dim=1), "b t h d -> b t (h d)"), s

    def init_memory(self, batch: int, device, dtype) -> Tensor:
        return torch.zeros(batch, self.h, self.dk, self.dk, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v, alpha, beta = self.projections(x, self.conv(self.qkv(x)))
        out, _ = self.scan(q, k, v, alpha, beta, self.init_memory(x.shape[0], x.device, x.dtype))
        return self.out_proj(self.norm(out, self.gate(x)))

    def init_state(
        self, batch: int, device: torch.device, dtype: torch.dtype
    ) -> GatedDeltaNetState:
        return GatedDeltaNetState(
            s=self.init_memory(batch, device, dtype),
            conv=self.conv.init_state(batch, device, dtype),
        )

    def step(self, x: Tensor, state: GatedDeltaNetState) -> Tensor:
        q, k, v, alpha, beta = self.projections(x, self.conv.step(self.qkv(x), state.conv))
        out, state.s = self.scan(q, k, v, alpha, beta, state.s)
        return self.out_proj(self.norm(out, self.gate(x)))
