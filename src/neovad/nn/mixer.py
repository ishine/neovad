from typing import ClassVar, Self

import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from neovad.registry import Registry


class ModuleState(BaseModel):
    """Per-module streaming state carrying tensors, mutated in place across steps.

    Streaming inference keeps one of these per streamable submodule; ``forward``
    (the parallel training path) never touches it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StreamingMixer(Registry, nn.Module, root=True):
    """A sequence mixer: maps ``[B, T, D] -> [B, T, D]`` causally.

    Two equivalent execution paths share one set of weights:

    * ``forward`` — the parallel, full-sequence path used for training; must be
      causal so that frame ``t`` only attends to frames ``<= t``.
    * ``init_state`` + ``step`` — the recurrent path used for streaming inference.
      ``step`` consumes one or more new frames ``[B, n, D]`` (n >= 1) plus the mutable
      :class:`ModuleState` and returns ``[B, n, D]``. Recurrent mixers implement the
      single-frame ``step_frame`` and inherit the multi-frame loop; mixers whose
      recurrence natively consumes sequences (GRU) override ``step`` directly.

    Equivalence between the two paths (within float tolerance) is a hard contract,
    verified by tests; it is what lets us train in parallel and serve frame-by-frame.
    Concrete mixers set ``kind`` and read hyper-parameters from their paired
    :class:`MixerConfig`.
    """

    kind: ClassVar[str] = ""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def init_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> ModuleState:
        raise NotImplementedError

    def step(self, x: Tensor, state: ModuleState) -> Tensor:
        if x.shape[1] == 1:
            return self.step_frame(x, state)
        return torch.cat(
            [self.step_frame(x[:, t : t + 1], state) for t in range(x.shape[1])], dim=1
        )

    def step_frame(self, x: Tensor, state: ModuleState) -> Tensor:
        raise NotImplementedError


class MixerConfig(BaseModel):
    """Typed, discriminated config for a mixer. Owns construction of its module.

    The concrete config (e.g. ``GQAConfig``) lives next to the module it configures
    and pins ``kind`` to a ``Literal``; the discriminated union over all of them is
    assembled in :mod:`neovad.config`.
    """

    kind: str

    def build(self, dim: int, depth: int = 1) -> StreamingMixer:
        # depth = 1-based layer index, contextual construction info that schedule-based
        # mixers (e.g. Differential Attention's lambda-init) consume; others ignore it.
        return StreamingMixer.by_name(self.kind)(dim, self, depth)

    def with_overrides(self, **kwargs) -> Self:
        return self.model_copy(update=kwargs)
