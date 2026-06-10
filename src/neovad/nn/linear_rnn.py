import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neovad.nn.conv import CausalDepthwiseConv1d, ConvState
from neovad.nn.mixer import MixerConfig, ModuleState, StreamingMixer
from neovad.nn.scan import diagonal_chunked_scan


class LinearRNNState(ModuleState):
    h: Tensor  # [B, d_inner] diagonal recurrence state
    conv: ConvState | None = None  # short-conv tail (RG-LRU branch only)


class RGLRUConfig(MixerConfig):
    kind: Literal["rglru"] = "rglru"
    d_conv: int = 4
    c: float = 8.0  # gate sharpness from the Griffin paper
    chunk: int = 32


class RGLRUMixer(StreamingMixer):
    """Hawk recurrent block around the RG-LRU (De et al., 2024, "Griffin").

    The Real-Gated Linear Recurrent Unit:
    ``r_t = sigmoid(W_a x)``, ``i_t = sigmoid(W_x x)``,
    ``log a_t = -c * softplus(Lambda) * r_t``,
    ``h_t = a_t h_{t-1} + sqrt(1 - a_t^2) * (i_t * x_t)`` — a per-channel gated decay,
    O(1) streaming state, no softmax anywhere. Wrapped Hawk-style: a GeLU gate branch
    multiplies the (causal-conv -> RG-LRU) branch. Training uses the exact diagonal
    chunked scan; ``step`` is the recurrence itself.
    """

    kind = "rglru"

    def __init__(self, dim: int, cfg: RGLRUConfig, depth: int = 1):
        super().__init__(dim)
        self.c = cfg.c
        self.chunk = cfg.chunk
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.conv = CausalDepthwiseConv1d(dim, cfg.d_conv, bias=True)
        self.w_r = nn.Linear(dim, dim, bias=True)
        self.w_i = nn.Linear(dim, dim, bias=True)
        # Lambda init so decay a spans ~(0.9, 0.999) at r=1 (Griffin appendix)
        self.lam = nn.Parameter(
            torch.linspace(
                math.log(math.expm1(0.001 / self.c)), math.log(math.expm1(0.1 / self.c)), dim
            )
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def gates(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # log a_t (negative) and the scaled input u_t
        r = torch.sigmoid(self.w_r(x))
        i = torch.sigmoid(self.w_i(x))
        log_a = -self.c * F.softplus(self.lam) * r
        scale = torch.sqrt(1.0 - torch.exp(2.0 * log_a) + 1e-6)
        return log_a, scale * (i * x)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.gelu(self.gate_proj(x))
        z = self.conv(self.in_proj(x))
        log_a, u = self.gates(z)
        h, _ = diagonal_chunked_scan(u, log_a, chunk=self.chunk)
        return self.out_proj(h * gate)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> LinearRNNState:
        return LinearRNNState(
            h=torch.zeros(batch, self.dim, device=device, dtype=dtype),
            conv=self.conv.init_state(batch, device, dtype),
        )

    def step(self, x: Tensor, state: LinearRNNState) -> Tensor:
        gate = F.gelu(self.gate_proj(x))
        z = self.conv.step(self.in_proj(x), state.conv)
        log_a, u = self.gates(z)
        hs = []
        h = state.h
        for t in range(x.shape[1]):
            h = torch.exp(log_a[:, t]) * h + u[:, t]
            hs.append(h)
        state.h = h
        return self.out_proj(torch.stack(hs, dim=1) * gate)


class MinGRUConfig(MixerConfig):
    kind: Literal["mingru"] = "mingru"
    expand: int = 2
    chunk: int = 32


class MinGRUMixer(StreamingMixer):
    """minGRU (Feng et al., 2024, "Were RNNs All We Needed?").

    ``z_t = sigmoid(W_z x)``, ``h~_t = W_h x``,
    ``h_t = (1 - z_t) h_{t-1} + z_t h~_t`` — the candidate state drops its dependence
    on ``h_{t-1}``, which makes the recurrence diagonal-linear and parallel-scannable
    while keeping GRU-style gating. The cheapest stateful mixer here by FLOPs.
    """

    kind = "mingru"

    def __init__(self, dim: int, cfg: MinGRUConfig, depth: int = 1):
        super().__init__(dim)
        self.chunk = cfg.chunk
        self.d_inner = cfg.expand * dim
        self.w_z = nn.Linear(dim, self.d_inner, bias=True)
        self.w_h = nn.Linear(dim, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def gates(self, x: Tensor) -> tuple[Tensor, Tensor]:
        z = torch.sigmoid(self.w_z(x))
        log_decay = torch.log1p(-z.clamp(max=1 - 1e-6))  # log(1 - z)
        return log_decay, z * self.w_h(x)

    def forward(self, x: Tensor) -> Tensor:
        log_decay, u = self.gates(x)
        h, _ = diagonal_chunked_scan(u, log_decay, chunk=self.chunk)
        return self.out_proj(h)

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> LinearRNNState:
        return LinearRNNState(h=torch.zeros(batch, self.d_inner, device=device, dtype=dtype))

    def step(self, x: Tensor, state: LinearRNNState) -> Tensor:
        log_decay, u = self.gates(x)
        hs = []
        h = state.h
        for t in range(x.shape[1]):
            h = torch.exp(log_decay[:, t]) * h + u[:, t]
            hs.append(h)
        state.h = h
        return self.out_proj(torch.stack(hs, dim=1))
