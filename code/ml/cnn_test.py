#######this from markus 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

###save_scenario_features placed in the script creating the scenario?
def save_scenario_features(
    out_dir: Path,
    scenario_id: str,
    Y: np.ndarray,           # shape (F, L, M), complex64
    S_early: np.ndarray,     # shape (N, F, L), complex64
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario_id}.npz"
    # If N varies per scenario, we just store native N and will pad later
    np.savez_compressed(out_path, Y=Y.astype(np.complex64),
                        S_early=S_early.astype(np.complex64))

#####complex split to re and im 
def complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    """
    x: complex tensor of shape (...,)
    returns: real-valued tensor of shape (..., 2) with [..., 0]=Re, [...,1]=Im.
    """
    return torch.stack((x.real, x.imag), dim=-1)

##### prepare psd dataset?
class PSDDataset(Dataset):
    def __init__(
        self,
        root: Path,
        *,
        N_max: int,
        dtype: torch.dtype = torch.float32,
    ):
        """
        root: directory with scenario .npz files 
        N_max: maximum number of speakers in the model (pad as needed)
        """
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {self.root}")
        self.N_max = N_max
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path, allow_pickle=False)
        Y = data["Y"]        # (F, L, M), complex64
        S_early = data["S_early"]  # (N, F, L), complex64

        # Convert to torch
        Y = torch.from_numpy(Y)      # complex64 → torch.complex64
        S_early = torch.from_numpy(S_early)

        # Shapes
        F, L, M = Y.shape
        N = S_early.shape[0]

        # ----- Inputs: real/imag split -----
        # math: y(ℓ,k,m) -> code: Y[f,l,m]
        # We'll go to (F, L, M, 2), then maybe to (C, F, L) for the model.
        Y_ri = complex_to_ri(Y)              # (F, L, M, 2), float
        Y_ri = Y_ri.to(self.dtype)

        # Flatten mic+RI into a channel dimension: C = M * 2
        Y_ri = Y_ri.view(F, L, M * 2)        # (F, L, C)
        # For convolution-friendly layout: (C, F, L)
        Y_in = Y_ri.permute(2, 0, 1)         # (C, F, L)

        # ----- Targets: PSD per speaker -----
        # S_early: (N, F, L), complex
        # φ_s_n(f,ℓ) ≈ |S_n(f,ℓ)|^2
        Phi = (S_early.abs() ** 2).to(self.dtype)  # (N, F, L), real

        # Pad speakers to N_max along speaker dim, if needed
        if N < self.N_max:
            pad_shape = (self.N_max - N, F, L)
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
            # More speakers than N_max: just truncate.
            Phi = Phi[:self.N_max]
            speaker_mask = torch.ones(self.N_max, dtype=torch.bool)

        # final shapes:
        #   Y_in: (C, F, L)
        #   Phi : (N_max, F, L)
        #   speaker_mask: (N_max,)
        return Y_in, Phi, speaker_mask


#####simple psd with re and im as input
#### no maxpool or linear layres because we do not want to reduce time or freq solution
class SimplePSDNet(nn.Module):
    def __init__(self, C_in: int, N_max: int, hidden: int = 64):
        """
        C_in: number of input channels (M * 2)
        N_max: number of output speaker PSD maps
        """
        super().__init__()
        # We treat (F, L) as a 2D image, channels = C_in
        self.net = nn.Sequential(
            nn.Conv2d(C_in, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, N_max, kernel_size=1),  # output N_max channels
            nn.Softplus(),  # ensure non-negative PSD (smooth relu)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, F, L)
        returns: (B, N_max, F, L)
        """
        return self.net(x)


###training function for model
def train_psd_net(
    data_root: Path,
    *,
    N_max: int,
    batch_size: int = 8,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
):
    ds = PSDDataset(data_root, N_max=N_max)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

    # One sample to get C_in, F, L
    Y_in0, Phi0, _ = ds[0]
    C_in = Y_in0.shape[0]

    model = SimplePSDNet(C_in=C_in, N_max=N_max).to(device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for Y_in, Phi_gt, mask in dl:
            # Y_in: (B, C_in, F, L)
            # Phi_gt: (B, N_max, F, L)
            # mask: (B, N_max)
            Y_in = Y_in.to(device)
            Phi_gt = Phi_gt.to(device)      # ground truth PSD
            mask = mask.to(device)          # speaker presence mask

            optim_.zero_grad()

            Phi_hat = model(Y_in)           # (B, N_max, F, L)

            # Mask out padded speakers in the loss
            # Expand mask to broadcast over F,L:
            #   mask: (B, N_max) -> (B, N_max, 1, 1)
            mask_exp = mask[:, :, None, None]
            diff = (Phi_hat - Phi_gt) * mask_exp

            # MSE averaged over valid speakers, freq, frames, batch
            mse = (diff ** 2).sum() / mask_exp.sum().clamp_min(1.0)

            mse.backward()
            optim_.step()

            total_loss += mse.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        print(f"Epoch {epoch+1}/{num_epochs} - loss={avg_loss:.4e}")

    return model
