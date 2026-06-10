import torch
from pydantic import BaseModel
from torch import Tensor, nn


class SpecAugmentConfig(BaseModel):
    enabled: bool = True
    freq_masks: int = 2
    freq_width: int = 12  # of n_mels bins
    time_masks: int = 2
    time_width: int = 20  # of 10 ms frames (200 ms)


class SpecAugment(nn.Module):
    """Mel-domain masking (Park et al., 2019), train-time only.

    Applied between frontend and backbone in the parallel ``forward`` when the module
    is in train mode; eval/streaming never see it, so the forward-vs-step equivalence
    contract is untouched. Time masks teach the temporal state to bridge dropouts;
    frequency masks stop the model keying on single bands (a noise-robustness win).
    """

    def __init__(self, cfg: SpecAugmentConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, mel: Tensor) -> Tensor:
        # mel: [B, T, M]
        if not (self.training and self.cfg.enabled):
            return mel
        b, t, m = mel.shape
        out = mel.clone()
        for _ in range(self.cfg.freq_masks):
            width = torch.randint(1, self.cfg.freq_width + 1, (b,), device=mel.device)
            start = (torch.rand(b, device=mel.device) * (m - width).clamp(min=1)).long()
            for i in range(b):
                out[i, :, start[i] : start[i] + width[i]] = 0.0
        for _ in range(self.cfg.time_masks):
            width = torch.randint(1, self.cfg.time_width + 1, (b,), device=mel.device)
            start = (torch.rand(b, device=mel.device) * (t - width).clamp(min=1)).long()
            for i in range(b):
                out[i, start[i] : start[i] + width[i]] = 0.0
        return out
