import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

from seadge.utils.log import log
from seadge.utils.psd_data_loader import load_tensors_cache
from seadge.utils.visualization import spectrogram
from seadge.models.psd_cnn import SimplePSDCNN
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
        checkpoint_dir: Path,
        test_size: float = 0.2,
        ):
    """Train the PSD estimation model"""

    model.to(device)

    # train-test data split (still on CPU here)
    x_train, x_test, y_train, y_test = train_test_split(
        x_tensor, y_tensor, test_size=test_size
    )

    # Datasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset  = torch.utils.data.TensorDataset(x_test,  y_test)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    criterion = nn.MSELoss()  # For regression; use CrossEntropyLoss for classification

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Check if there are existing checkpoints
    if os.listdir(checkpoint_dir):
        # If there are existing checkpoints, load the latest one
        latest_checkpoint = max([int(file.split('.')[0]) for file in os.listdir(checkpoint_dir)])
        checkpoint = torch.load(os.path.join(checkpoint_dir, f'{latest_checkpoint}.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = latest_checkpoint + 1
        log.info(f"Found checkpoint. Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0
        log.info(f"Found no checkpoints. Starting from epoch {start_epoch}")

    train_losses = []

    pbar = tqdm(range(start_epoch, epochs), desc="Training epochs", unit="epoch", leave=False)
    for epoch in pbar:
        # -----------------
        # Training phase
        # -----------------
        model.train()
        epoch_train_loss = 0.0
        n_train_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            pred_psd = model(batch_x)
            loss = criterion(pred_psd, batch_y)
            loss.backward()
            optimizer.step()

            # accumulate loss weighted by batch size
            bs = batch_x.size(0)
            epoch_train_loss += loss.item() * bs
            n_train_samples += bs

        # average train loss per sample
        avg_train_loss = epoch_train_loss / n_train_samples
        train_losses.append(avg_train_loss)

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss
            }, checkpoint_dir / f'{epoch}.pt')

        pbar.set_postfix({"avg. train loss": f"{avg_train_loss:.6f}"})

    # -----------------
    # Testing phase
    # -----------------
    model.eval()
    test_loss_sum = 0.0
    n_test_samples = 0
    if start_epoch < (epochs - 1):
        log.info(f"Finished training model. avg. train loss for last epoch: {train_losses[-1]}. Starting evaluation")

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            test_pred = model(batch_x)
            loss = criterion(test_pred, batch_y)

            bs = batch_x.size(0)
            test_loss_sum += loss.item() * bs
            n_test_samples += bs

    test_loss = test_loss_sum / n_test_samples

    log.debug(f"Trained model with {test_loss=}")
    return train_losses, test_loss


def main():
    cfg = config.get()
    log.info("Training PSD model")

    # Load tensors
    x_tensor, y_tensor, _ = load_tensors_cache(cfg.paths.ml_data_dir)

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
            checkpoint_dir=cfg.paths.checkpoint_dir,
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
        y_pred_log = model(x_sample)
        y_pred = torch.expm1(y_pred_log)
        #y_true = torch.expm1(y_sample)
        y_true = y_sample

    spectrogram(y_pred.squeeze(0).cpu().numpy(), cfg.paths.debug_dir / "cnn" / "psd_pred.png", title="PSD pred", scale="mag")
    spectrogram(y_true.squeeze(0).cpu().numpy(), cfg.paths.debug_dir / "cnn" / "psd_truth.png", title="PSD ground truth", scale="mag")
