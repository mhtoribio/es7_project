import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import os
from scipy import signal
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from seadge.utils.dsp import complex_to_mag_phase
from seadge.utils.files import files_in_path_recursive
from seadge.utils.log import log
from seadge import config

class MicTimeSTFTCNN(nn.Module):
    def __init__(self, num_freqbins, num_mics):
        super().__init__()
        self.num_freqbins = num_freqbins
        self.num_mics = num_mics

        c_in  = 2 * num_freqbins
        c_out = num_freqbins
        c_hidden = c_in

        self.conv1 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_hidden,
            kernel_size=(5, num_mics),   # frames x mic
            padding=(2, 0),
        )
        self.conv2 = nn.Conv1d(
            in_channels=c_hidden,
            out_channels=c_hidden,
            kernel_size=5,
            padding=2,
        )
        self.conv3 = nn.Conv1d(
            in_channels=c_hidden,
            out_channels=c_out,
            kernel_size=1,
        )

    def forward(self, x_ctx):
        # x_ctx: (B, 2K, L, M)
        # output: (B, K, L, 1)
        log.debug(f"input shape: {x_ctx.shape}")
        x = F.relu(self.conv1(x_ctx)).squeeze(3)
        log.debug(f"hidden 1 shape: {x.shape}")
        x = F.relu(self.conv2(x))
        log.debug(f"hidden 2 shape: {x.shape}")
        x = F.softplus(self.conv3(x))
        log.debug(f"output shape: {x.shape}")
        return x

def load_features_and_psd(npz_file: Path, L_max: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads features and PSD from .npz file and zero-pad frames to L_max
    """
    data = np.load(npz_file, allow_pickle=False)

    # Y: (K, L, M), S_early: (N, K, L)
    Y_np = data["Y"]                # (K, L, M)
    S_np = data["S_early"][0, :, :] # (K, L)

    # get dimensions
    mics = Y_np.shape[2]
    freqbins = Y_np.shape[0]
    if freqbins != S_np.shape[0]:
        log.error(f"Malformed npz file. Expected freqbins = ({Y_np.shape[0]=}) == ({S_np.shape[0]=}).")
    frames = Y_np.shape[1]
    if frames != S_np.shape[1]:
        log.error(f"Malformed npz file. Expected frames = ({Y_np.shape[1]=}) == ({S_np.shape[1]=}).")
    if L_max < frames:
        log.error(f"{L_max=} < {frames=}. Should never happen")

    # zero-pad frames
    Y = np.zeros((freqbins, L_max, mics), dtype=np.complex64)
    Y[:, :frames, :] = Y_np
    S = np.zeros((freqbins, L_max), dtype=np.complex64)
    S[:, :frames] = S_np

    # (K, L, M), (K, L)
    return Y, S

def load_tensors_from_dir(npz_dir: Path, L_max: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    npz_files = files_in_path_recursive(npz_dir, "*.npz")
    log.debug(f"Creating tensors from {len(npz_files)} npz files")

    # Compute labels
    X_list = []
    Y_list = []
    for npz_file in npz_files:
        distant, early = load_features_and_psd(npz_file, L_max)

        # compute features
        # distant: (K, L, M)
        # features: (2K, L, M)
        distant_mag, distant_phase = complex_to_mag_phase(distant)
        features = np.concatenate((distant_mag, distant_phase))
        X_list.append(features)
        log.debug(f"feature shape {features.shape}")

        # computes labels
        # psd: (K, L)
        # output: (K, L, 1)
        psd = np.abs(early) ** 2 # ground truth
        Y_list.append(psd)
        log.debug(f"PSD shape {psd.shape}")

    # Create tensors
    X = torch.FloatTensor(np.asarray(X_list))
    Y = torch.FloatTensor(np.asarray(Y_list))
    log.debug(
        "Tensor creation info: "
        f"{X.shape=}, "
        f"{Y.shape=}, "
        f"number of total frames: {X.shape[0]}, "
        f"input features per frame: {X.shape[1]}, "
        f"PSD bins per frame: {Y.shape[1]}"
    )
    return X, Y

def train_psd_model(
        model: nn.Module,
        x_tensor: torch.FloatTensor,
        y_tensor: torch.FloatTensor,
        epochs: int,
        batch_size: int,
        weight_decay: float,
        lr: float,
        device: str,
        test_size: float = 0.2,
        ):
    """Train the PSD estimation model"""

    model.to(device)

    # train-test data split
    x_train, x_test, y_train, y_test = train_test_split(x_tensor, y_tensor, test_size=test_size) 

    # Create dataset and batches
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    criterion = nn.MSELoss() # For regression (MSELoss()), for classification (CrossEntropyLoss())

    train_losses = []
    test_losses = []

    pbar = tqdm(range(epochs), desc=f"Training epochs", unit="batch", leave=False)
    for epoch in pbar:
        # Training phase
        model.train()
        epoch_train_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            log.debug(f"{batch_x.shape=}, {batch_y.shape=}")
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_psd = model(batch_x)
            loss = criterion(pred_psd, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() # Used for plotting

        # Following is used for plotting average loss per epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        pbar.set_postfix({"avg. train loss": f"{avg_train_loss:.6f}"})

    # Testing phase
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test)
        test_loss = criterion(test_pred, y_test).item()

    log.debug(f"Trained model with {test_loss=}")
    return train_losses, test_loss


def main():
    cfg = config.get()
    log.info("Training PSD model")

    # Load tensors
    x_tensor, y_tensor = load_tensors_from_dir(cfg.paths.ml_data_dir, cfg.L_max)

    # Define input/output sizes
    num_freqbins = y_tensor.shape[1]
    num_mics = x_tensor.shape[3]

    # Initialize model
    model = MicTimeSTFTCNN(
        num_freqbins=num_freqbins,
        num_mics=num_mics,
    )

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(
        f"Training setup: "
        f"K={num_freqbins}, M={num_mics}, "
        f"batch_size={cfg.deeplearning.batch_size}, epochs={cfg.deeplearning.epochs}, "
        f"lr={cfg.deeplearning.learning_rate:.3g}, "
        f"device={device}"
    )

    # Train model and get losses
    train_losses, test_loss = train_psd_model(
            model=model,
            x_tensor=x_tensor,
            y_tensor=y_tensor,
            epochs=cfg.deeplearning.epochs,
            batch_size=cfg.deeplearning.batch_size,
            lr=cfg.deeplearning.learning_rate,
            weight_decay=cfg.deeplearning.weight_decay,
            device=device,
            )

    # Plot training losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.title("PSD Estimator Training Loss")
    plt.grid(True)
    plt.legend()
    outfigpath = cfg.paths.debug_dir / "mlp" / "train_loss.png"
    log.debug(f"Writing training loss figure to {outfigpath}")
    plt.savefig(outfigpath)
