import torch
import torch.nn.functional as F
from typing import Iterable, Tuple, Optional

EPS = 1e-8

def _reduce_per_freq(x: torch.Tensor, freq_dim: int = 1) -> torch.Tensor:
    """Mean over all dims except freq → (K,)"""
    dims = [d for d in range(x.ndim) if d != freq_dim]
    return x.mean(dim=dims)

def _apply_weights(per_k: torch.Tensor, weights: Optional[torch.Tensor], reduce: str) -> torch.Tensor:
    if weights is None:
        return per_k.mean() if reduce == "mean" else per_k.sum()
    w = weights.to(per_k.device).reshape(-1)
    if w.numel() != per_k.numel():
        raise ValueError("weights must have shape (K,)")
    return (per_k * w).sum() / (w.sum() + EPS) if reduce == "mean" else (per_k * w).sum()

# ---------------------------
# Magnitude-domain losses
# A_hat, A : magnitude spectra, shape (B,K,L) (non-negative)
# ---------------------------

def magMSE(A_hat: torch.Tensor, A: torch.Tensor, *, freq_dim=1,
           weights: Optional[torch.Tensor]=None, reduce="mean") -> Tuple[torch.Tensor, torch.Tensor]:
    per = _reduce_per_freq((A_hat - A).pow(2), freq_dim)
    return _apply_weights(per, weights, reduce), per

def magMAE(A_hat: torch.Tensor, A: torch.Tensor, *, freq_dim=1,
           weights: Optional[torch.Tensor]=None, reduce="mean"):
    per = _reduce_per_freq((A_hat - A).abs(), freq_dim)
    return _apply_weights(per, weights, reduce), per

def LSD(A_hat: torch.Tensor, A: torch.Tensor, *, freq_dim=1,
        weights: Optional[torch.Tensor]=None, reduce="mean"):
    # ⟨ | log10Â - log10A |^2 ⟩
    lh = torch.log10(A_hat.clamp_min(EPS))
    lt = torch.log10(A.clamp_min(EPS))
    per = _reduce_per_freq((lh - lt).pow(2), freq_dim)
    return _apply_weights(per, weights, reduce), per

def wLSD(A_hat: torch.Tensor, A: torch.Tensor, W_lsd: torch.Tensor, *,
         freq_dim=1, reduce="mean"):
    # weighted log-MSE over freq (W_lsd is (K,))
    return LSD(A_hat, A, freq_dim=freq_dim, weights=W_lsd, reduce=reduce)

def magComp(A_hat: torch.Tensor, A: torch.Tensor, *, c: float = 0.3,  # 0<c<1
            freq_dim=1, weights: Optional[torch.Tensor]=None, reduce="mean"):
    # ⟨ |Â^c - A^c|^2 ⟩
    Ahc = A_hat.clamp_min(0).pow(c)
    Ac  = A.clamp_min(0).pow(c)
    per = _reduce_per_freq((Ahc - Ac).pow(2), freq_dim)
    return _apply_weights(per, weights, reduce), per

def SNR_mag(A_hat: torch.Tensor, A: torch.Tensor, *, freq_dim=1,
            weights: Optional[torch.Tensor]=None, reduce="mean"):
    # −log10( ⟨A^2⟩ / ⟨|Â−A|^2⟩ )
    num = _reduce_per_freq(A.pow(2), freq_dim) + EPS
    den = _reduce_per_freq((A_hat - A).pow(2), freq_dim) + EPS
    per = -torch.log10(num / den)
    return _apply_weights(per, weights, reduce), per

def magCorr(A_hat: torch.Tensor, A: torch.Tensor, *, freq_dim=1,
            weights: Optional[torch.Tensor]=None, reduce="mean"):
    # − ⟨ÂA⟩^2 / (⟨Â^2⟩ ⟨A^2⟩)
    num = _reduce_per_freq((A_hat * A), freq_dim)
    den = torch.sqrt(_reduce_per_freq(A_hat.pow(2), freq_dim) * _reduce_per_freq(A.pow(2), freq_dim) + EPS)
    per = -(num / (den + EPS)).pow(2)
    return _apply_weights(per, weights, reduce), per

def lsd_from_logpower(Y_hat, Y):
    """LSD on magnitude via log-power (base-10), scalar."""
    return 0.25 * (Y_hat - Y).pow(2).mean()

def wlsd_from_logpower(Y_hat, Y, W):  # W: (K,) per-freq weights
    per = 0.25 * (Y_hat - Y).pow(2).mean(dim=(0,2))  # mean over B and L → (K,)
    return (per * W).sum() / (W.sum() + EPS)

def logpower_mse(Y_hat, Y):
    """Plain MSE in log-power space (not the same as magMSE)."""
    return (Y_hat - Y).pow(2).mean()

def logpower_mae(Y_hat, Y):
    return (Y_hat - Y).abs().mean()
