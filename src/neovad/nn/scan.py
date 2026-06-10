import torch
from torch import Tensor


def diagonal_chunked_scan(
    u: Tensor, log_decay: Tensor, chunk: int = 32, h0: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Exact parallel evaluation of the diagonal linear recurrence
    ``h_t = exp(log_decay_t) * h_{t-1} + u_t`` for per-channel decays.

    Within each chunk the solution is the masked quadratic form
    ``h_t = sum_{s<=t} exp(cum_t - cum_s) u_s`` (cum = running log-decay), with the
    carry-in state propagated by ``exp(cum_t) * h0``; chunks run sequentially. Masking
    happens in log space *before* exp (the same overflow lesson as the Mamba-2 SSD
    forward), and the chunk size bounds the quadratic memory at
    ``[B, chunk, chunk, D]``. Returns ``(h [B,T,D], h_last [B,D])``.

    This is the shared training-path engine for every per-channel linear-recurrence
    mixer (RG-LRU, minGRU); their streaming ``step`` is the recurrence itself.
    """
    b, t, d = u.shape
    h = h0 if h0 is not None else torch.zeros(b, d, device=u.device, dtype=u.dtype)
    out = []
    for c0 in range(0, t, chunk):
        ld = log_decay[:, c0 : c0 + chunk].float()
        uu = u[:, c0 : c0 + chunk].float()
        n = ld.shape[1]
        cum = ld.cumsum(dim=1)  # [B, n, D]
        rel = cum[:, :, None] - cum[:, None, :]  # [B, n_t, n_s, D] = cum_t - cum_s
        mask = torch.ones(n, n, device=u.device, dtype=torch.bool).tril()
        rel = rel.masked_fill(~mask[None, :, :, None], float("-inf"))
        inner = torch.einsum("btsd,bsd->btd", torch.exp(rel), uu)
        hs = inner + torch.exp(cum) * h.float()[:, None]
        out.append(hs.to(u.dtype))
        h = out[-1][:, -1]
    return torch.cat(out, dim=1), h
