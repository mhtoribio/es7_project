import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from seadge.utils.log import log
from seadge.utils.visualization import spectrogram
from seadge.models.psd_cnn import SimplePSDCNN
from seadge.utils.psd_data_loader import load_tensors_from_dir
from seadge import config

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
        x_test = x_test.to(device)
        y_test = y_test.to(device)
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
    model = SimplePSDCNN(
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
    log.info(f"Finished training. Evaluation loss {test_loss:.6f}")

    # Save model
    outpath = cfg.paths.models_dir / "model.pt"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"Writing model to {outpath}")
    torch.save(model.state_dict(), outpath)

    # Plot training losses
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE Loss")
    plt.title("PSD Estimator Training Loss")
    plt.grid(True)
    plt.legend()
    outfigpath = cfg.paths.debug_dir / "cnn" / "train_loss.png"
    outfigpath.parent.mkdir(parents=True, exist_ok=True)
    log.debug(f"Writing training loss figure to {outfigpath}")
    plt.savefig(outfigpath)

    # Use model on a few scenarios and output spectogram
    n_show = 1
    idx = torch.randperm(len(x_tensor))[:n_show]   # random unique indices
    x_sample = x_tensor[idx].to(device)            # (n_show, input_dim, ...)
    y_sample = y_tensor[idx].to(device)            # matching targets
    with torch.no_grad():
        y_pred = model(x_sample)

    spectrogram(y_pred.squeeze(0).numpy(), cfg.paths.debug_dir / "cnn" / "psd_pred.png", title="PSD pred")
    spectrogram(y_sample.squeeze(0).numpy(), cfg.paths.debug_dir / "cnn" / "psd_truth.png", title="PSD ground truth")
