import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from pathlib import Path

from seadge.utils.log import log
from seadge.utils.cache import make_pydantic_cache_key
from seadge.utils.psd_dataset import PSDDataset
from seadge import config

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

from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------
# One epoch of training
# ---------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    # tqdm over batches
    pbar = tqdm(loader, desc=f"Train {epoch}/{num_epochs}", unit="batch", leave=False)
    for Y_in, Phi, mask in pbar:
        # Y_in: (B, C, F, L)
        # Phi:  (B, N, F, L)   (target PSD)
        # mask: (B, N, F, L)   (optional speaker mask, or all ones)

        Y_in = Y_in.to(device)
        Phi  = Phi.to(device)
        mask = mask.to(device)

        # Expand mask to (B, N_max, 1, 1) so it broadcasts over F and L
        mask_4d = mask.unsqueeze(-1).unsqueeze(-1).float()

        optimizer.zero_grad()
        Phi_hat = model(Y_in)              # (B, N, F, L)


        loss = criterion(Phi_hat * mask_4d, Phi * mask_4d)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * Y_in.size(0)
        n_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4e}"})

    return running_loss / max(1, n_batches)


# ---------------------------
# Validation
# ---------------------------

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> float:
    model.eval()
    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Valid {epoch}/{num_epochs}", unit="batch", leave=False)
        for Y_in, Phi, mask in pbar:
            Y_in = Y_in.to(device)
            Phi  = Phi.to(device)
            mask = mask.to(device)

            mask_4d = mask.unsqueeze(-1).unsqueeze(-1).float()

            Phi_hat = model(Y_in)
            loss = criterion(Phi_hat * mask_4d, Phi * mask_4d)

            running_loss += loss.item() * Y_in.size(0)
            n_batches += 1
            pbar.set_postfix({"val_loss": f"{loss.item():.4e}"})

    return running_loss / max(1, n_batches)


# ---------------------------
# Full training loop
# ---------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    outdir: Path
):
    model.to(device)

    best_val = float("inf")

    # Optional outer tqdm over epochs
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs
        )
        val_loss = validate(
            model, val_loader, criterion, device, epoch, num_epochs
        )

        log.info(f"[Epoch {epoch:03d}/{num_epochs}] "
              f"train_loss={train_loss:.4e}  val_loss={val_loss:.4e}")

        if val_loss < best_val:
            best_val = val_loss
            # Save best model
            torch.save(model.state_dict(), outdir / "best_psdnet.pt")

def main():
    cfg = config.get()
    log.info("Training PSD Model")

    # Create datasets
    N_max = cfg.roomgen.max_num_source_locations
    L_max = cfg.L_max
    train_ds = PSDDataset(cfg.paths.ml_data_dir, N_max=N_max, L_max=L_max)
    val_ds   = PSDDataset(cfg.paths.ml_data_dir, N_max=N_max, L_max=L_max)

    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.deeplearning.batch_size,
        shuffle=True,
        num_workers=cfg.deeplearning.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.deeplearning.batch_size,
        shuffle=False,
        num_workers=cfg.deeplearning.num_workers,
        pin_memory=True,
    )

    F = cfg.dsp.window_len // 2 + 1
    M = cfg.roomgen.num_mics
    lr = cfg.deeplearning.learning_rate

    model = SimplePSDNet(
        C_in=M * 2,
        N_max=N_max,
        hidden=cfg.deeplearning.hidden_channels,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log.info(
        f"Training setup: "
        f"N_max={N_max}, F={F}, M={M}, hidden={cfg.deeplearning.hidden_channels}, "
        f"batch_size={cfg.deeplearning.batch_size}, epochs={cfg.deeplearning.epochs}, "
        f"lr={lr:.3g}, num_workers={cfg.deeplearning.num_workers}, "
        f"train_samples={len(train_ds)}, val_samples={len(val_ds)}, "
        f"device={device}"
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.deeplearning.epochs,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        outdir=cfg.paths.models_dir,
    )
