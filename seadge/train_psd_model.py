import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from seadge.utils.log import log
from seadge.utils.psd_data_loader import load_tensors_cache
from seadge.utils.visualization import spectrogram
from seadge.models.psd_cnn import SimplePSDCNN
from seadge import config
from seadge.utils.log import setup_logger
from seadge.utils.torch_ddp import setup_distributed, cleanup_distributed, launch_ddp
from seadge.models import loss_functions

# ugly hack but it works
import argparse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    return ap.parse_args()


def _is_main(rank: int) -> bool:
    return (rank == 0)


def _ddp_avg_scalar(value: float, device: torch.device) -> float:
    """Average a Python float across ranks (returns the global mean)."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _latest_checkpoint_epoch(checkpoint_dir: Path) -> int | None:
    """Return latest epoch number if any checkpoints exist, else None."""
    if not checkpoint_dir.exists():
        return None
    nums = []
    for f in checkpoint_dir.iterdir():
        if f.suffix == ".pt":
            try:
                nums.append(int(f.stem))
            except ValueError:
                pass
    return max(nums) if nums else None


def train_psd_model(
    model: nn.Module,
    x_tensor: torch.FloatTensor,
    y_tensor: torch.FloatTensor,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    lr: float,
    device: torch.device,
    checkpoint_dir: Path,
    use_ddp: bool,
    rank: int,
    world_size: int,
    test_size: float = 0.2,
):
    """DDP-aware training loop. Expects model already moved to device (and wrapped with DDP if use_ddp)."""

    # -----------------
    # Efficient split by indices (no tensor copies, works with memmap-backed tensors)
    # -----------------
    N = x_tensor.shape[0]
    idx = torch.randperm(N)  # or a fixed permutation if you want reproducibility
    cut = int(N * (1.0 - test_size))
    idx_train = idx[:cut].tolist()
    idx_test  = idx[cut:].tolist()

    full_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_dataset = torch.utils.data.Subset(full_dataset, idx_train)
    test_dataset  = torch.utils.data.Subset(full_dataset,  idx_test)

    # -----------------
    # Samplers & DataLoaders
    # -----------------
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        train_shuffle = False
        test_shuffle = False
    else:
        train_sampler = None
        test_sampler = None
        train_shuffle = True
        test_shuffle = False

    pin = (device.type == "cuda")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    criterion = lambda yhat, y: (loss_functions.lsd_from_logpower(yhat, y), None)

    # -----------------
    # Checkpoint resume (rank 0 decides epoch, all ranks load the same file)
    # -----------------
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    latest_epoch = None

    if _is_main(rank):
        latest_epoch = _latest_checkpoint_epoch(checkpoint_dir)
    if use_ddp:
        # Broadcast "latest_epoch" from rank 0 to everyone
        obj_list = [latest_epoch]
        dist.broadcast_object_list(obj_list, src=0)
        latest_epoch = obj_list[0]

    if latest_epoch is not None:
        ckpt_path = checkpoint_dir / f"{latest_epoch}.pt"
        map_location = {"cuda:%d" % 0: str(device)}  # robust mapping
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Model may be DDP-wrapped; both .load_state_dict() variants work
        (model.module if isinstance(model, DDP) else model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", latest_epoch)) + 1
        if _is_main(rank):
            log.info(f"Found checkpoint {ckpt_path.name}. Resuming from epoch {start_epoch}")
    else:
        if _is_main(rank):
            log.info(f"Found no checkpoints. Starting from epoch {start_epoch}")

    # -----------------
    # Training
    # -----------------
    train_losses = []
    epoch_iter = range(start_epoch, epochs)
    epoch_iter = tqdm(epoch_iter, desc="Training epochs", unit="epoch", leave=False) if _is_main(rank) else epoch_iter

    for epoch in epoch_iter:
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        # We'll accumulate *sum of loss over samples* locally, then average globally
        train_loss_sum_local = 0.0
        n_train_samples_local = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_psd = model(batch_x)
            loss, per_k = criterion(pred_psd, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_x.size(0)
            train_loss_sum_local += float(loss.item()) * bs
            n_train_samples_local += int(bs)

        # Average across ranks
        avg_train_loss_local = train_loss_sum_local / max(1, n_train_samples_local)
        avg_train_loss = _ddp_avg_scalar(avg_train_loss_local, device)

        if _is_main(rank):
            train_losses.append(avg_train_loss)
            if isinstance(epoch_iter, tqdm):
                epoch_iter.set_postfix({"avg. train loss": f"{avg_train_loss:.6f}"})

            # Save checkpoint every epoch (rank 0 only)
            raw_model = model.module if isinstance(model, DDP) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                },
                checkpoint_dir / f"{epoch}.pt",
            )

        if use_ddp:
            dist.barrier()  # keep epochs aligned

    # -----------------
    # Evaluation (distributed, then average loss)
    # -----------------
    model.eval()
    test_loss_sum_local = 0.0
    n_test_samples_local = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            test_pred = model(batch_x)
            loss, per_k = criterion(test_pred, batch_y)

            bs = batch_x.size(0)
            test_loss_sum_local += float(loss.item()) * bs
            n_test_samples_local += int(bs)

    # Per-rank average → global average
    avg_test_loss_local = test_loss_sum_local / max(1, n_test_samples_local)
    test_loss = _ddp_avg_scalar(avg_test_loss_local, device)

    if _is_main(rank):
        log.debug(f"Trained model with test_loss={test_loss:.6f}")

    return train_losses, test_loss


def main():
    args = parse_args()
    config.load(path=args.config, create_dirs=True)
    cfg = config.get()
    setup_logger(cfg.logging)

    # Initialize DDP (or single-process fallback)
    use_ddp, rank, world_size, device = setup_distributed()
    is_main = _is_main(rank)

    if is_main:
        log.info("Training PSD model")

    # Load tensors (each rank loads; OK unless memory-bound. For very large sets, shard here.)
    x_tensor, y_tensor, _ = load_tensors_cache(cfg.paths.ml_data_dir)

    # Define input/output sizes
    num_freqbins = y_tensor.shape[1]
    num_mics = x_tensor.shape[3]

    # Initialize model
    model = SimplePSDCNN(num_freqbins=num_freqbins, num_mics=num_mics).to(device)

    # Wrap with DDP if distributed
    if use_ddp:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    if is_main:
        log.info(
            "Training setup: "
            f"K={num_freqbins}, M={num_mics}, "
            f"batch_size={cfg.deeplearning.batch_size}, epochs={cfg.deeplearning.epochs}, "
            f"lr={cfg.deeplearning.learning_rate:.3g}, "
            f"device={device}, world_size={world_size}"
        )

    # Train & evaluate
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
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
    )

    # -----------------
    # Side effects (rank 0 only)
    # -----------------
    if is_main:
        log.info(f"Finished training. Evaluation loss {test_loss:.6f}")

        raw_model = model.module if isinstance(model, DDP) else model

        outpath = cfg.paths.models_dir / "model.pt"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        log.debug(f"Writing model to {outpath}")
        torch.save(raw_model.state_dict(), outpath)

        # Plot training losses
        if train_losses:  # can be empty if resumed late
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Average L1 Loss")
            plt.title("PSD Estimator Training Loss")
            plt.grid(True)
            plt.legend()
            outfigpath = cfg.paths.debug_dir / "cnn" / "train_loss.png"
            outfigpath.parent.mkdir(parents=True, exist_ok=True)
            log.debug(f"Writing training loss figure to {outfigpath}")
            plt.savefig(outfigpath)

        # Demo prediction images
        n_show = 1
        idx = torch.randperm(len(x_tensor))[:n_show]
        x_sample = x_tensor[idx].to(device)
        y_sample = y_tensor[idx].to(device)
        with torch.no_grad():
            y_pred_log = raw_model(x_sample)
            y_pred = torch.expm1(y_pred_log)
            y_true = torch.expm1(y_sample)

        spectrogram(
            y_pred.squeeze(0).cpu().numpy(),
            cfg.paths.debug_dir / "cnn" / "psd_pred.png",
            title="PSD pred",
            scale="mag",
        )
        spectrogram(
            y_true.squeeze(0).cpu().numpy(),
            cfg.paths.debug_dir / "cnn" / "psd_truth.png",
            title="PSD ground truth",
            scale="mag",
        )

    cleanup_distributed()

def cmd_train(args):
    nproc = args.gpus or torch.cuda.device_count()
    if nproc == 0:
        log.error(f"No GPUs specified/found.")
        exit()
    log.info(f"Launching training module on {nproc} GPUs")
    extra = []
    if args.config:
        extra += ["--", "--config", str(args.config)]
    launch_ddp(
        module="seadge.train_psd_model",
        nproc=nproc,
        extra_args=extra,
    )

if __name__ == "__main__":
    main()
