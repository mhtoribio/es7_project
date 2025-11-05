from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F  # for padding

def complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    """
    x: complex tensor of shape (...,)
    returns: real-valued tensor of shape (..., 2) with [..., 0]=Re, [...,1]=Im.
    """
    return torch.stack((x.real, x.imag), dim=-1)

class PSDDataset(Dataset):
    def __init__(
        self,
        root: Path,
        *,
        N_max: int,
        L_max: int,
        dtype: torch.dtype = torch.float32,
    ):
        """
        root  : directory with scenario .npz files
        N_max : maximum number of speakers in the model (pad as needed)
        L_max : fixed number of STFT frames (time dimension) for all samples
        """
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.root}")
        self.N_max = N_max
        self.L_max = int(L_max)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path, allow_pickle=False)

        # Y: (F, L, M), S_early: (N, F, L)
        Y_np = data["Y"]
        S_np = data["S_early"]

        # Convert to torch with a fresh storage
        Y = torch.tensor(Y_np, dtype=torch.complex64)       # (F, L, M)
        S_early = torch.tensor(S_np, dtype=torch.complex64) # (N, F, L)

        F_bins, L, M = Y.shape
        N = S_early.shape[0]
        L_max = self.L_max

        # --------------------------------------------------------------
        # Make time dimension uniform: crop or pad to L_max
        # --------------------------------------------------------------
        if L > L_max:
            # Always crop to first L_max frames (you could random-crop here)
            Y = Y[:, :L_max, :]              # (F, L_max, M)
            S_early = S_early[:, :, :L_max]  # (N, F, L_max)
            L = L_max
        elif L < L_max:
            pad = L_max - L
            # Y: (F, L, M), pad along time dim (dim=1) at the end
            # F.pad order for 3D: (D3_before, D3_after, D2_before, D2_after, D1_before, D1_after)
            # Here: D1=F, D2=L, D3=M -> pad D2_after with 'pad'
            Y = F.pad(Y, (0, 0, 0, pad, 0, 0))           # (F, L_max, M)
            # S_early: (N, F, L), pad last dim (L) at the end
            S_early = F.pad(S_early, (0, pad))          # (N, F, L_max)
            L = L_max

        assert L == L_max

        # --------------------------------------------------------------
        # Inputs: real/imag split → (C, F, L_max)
        # --------------------------------------------------------------
        # Y: (F, L, M) complex
        Y_ri = complex_to_ri(Y)                 # (F, L, M, 2), real
        Y_ri = Y_ri.to(self.dtype)

        # Flatten mic+RI into channels: C = 2 * M
        Y_ri = Y_ri.view(F_bins, L_max, M * 2)  # (F, L, C)
        Y_in = Y_ri.permute(2, 0, 1).contiguous()  # (C, F, L)

        # --------------------------------------------------------------
        # Targets: Φ_s_n(f,ℓ) ≈ |S_early|^2 → (N_max, F, L_max)
        # --------------------------------------------------------------
        Phi = (S_early.abs() ** 2).to(self.dtype)  # (N, F, L)

        if N < self.N_max:
            pad_shape = (self.N_max - N, F_bins, L_max)
            Phi_pad = torch.zeros(pad_shape, dtype=self.dtype)
            Phi = torch.cat([Phi, Phi_pad], dim=0)  # (N_max, F, L)
            speaker_mask = torch.cat(
                [torch.ones(N, dtype=torch.bool),
                 torch.zeros(self.N_max - N, dtype=torch.bool)],
                dim=0,
            )
        elif N == self.N_max:
            speaker_mask = torch.ones(self.N_max, dtype=torch.bool)
        else:
            # More speakers than N_max: truncate
            Phi = Phi[:self.N_max]
            speaker_mask = torch.ones(self.N_max, dtype=torch.bool)

        # final shapes:
        #   Y_in: (C, F, L_max)
        #   Phi : (N_max, F, L_max)
        #   speaker_mask: (N_max,)
        return Y_in, Phi, speaker_mask
