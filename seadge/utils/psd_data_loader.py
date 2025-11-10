import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from seadge.utils.log import log
from seadge.utils.files import files_in_path_recursive
from seadge.utils.dsp import complex_to_mag_phase

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

from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from tqdm.contrib.concurrent import process_map
import os
import logging

log = logging.getLogger(__name__)


def _get_n_workers() -> int:
    # Prefer SLURM hints if present
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.getenv(var)
        if v:
            return int(v)

    # Try CPU affinity (respects cgroups on many systems)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass

    # Fallback
    return os.cpu_count() or 1


def _load_one_npz_for_training(args) -> Tuple[np.ndarray, np.ndarray]:
    """Worker: load one npz and return (features, psd)."""
    npz_file, L_max = args

    distant, early = load_features_and_psd(npz_file, L_max)

    # distant: (K, L, M)
    distant_mag, distant_phase = complex_to_mag_phase(distant)
    # features: (2K, L, M)
    features = np.concatenate((distant_mag, distant_phase))

    # psd: (K, L)
    psd = np.abs(early) ** 2  # ground truth

    return features, psd


def load_tensors_from_dir(npz_dir: Path, L_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    npz_files = list(files_in_path_recursive(npz_dir, "*.npz"))
    log.info(f"Creating tensors from {len(npz_files)} npz files")

    if not npz_files:
        raise RuntimeError(f"No npz files found in {npz_dir}")

    n_workers = _get_n_workers()
    log.info(f"Loading npz files with {n_workers} workers")

    # process_map gives you tqdm for free
    results: List[Tuple[np.ndarray, np.ndarray]] = process_map(
        _load_one_npz_for_training,
        [(f, L_max) for f in npz_files],
        max_workers=n_workers,
        chunksize=1,
        desc="loading npz files",
    )

    X_list, Y_list = zip(*results)  # tuples of np.ndarrays

    # Stack instead of asarray (clearer intent) + convert dtype once
    X_np = np.stack(X_list).astype(np.float32, copy=False)
    Y_np = np.stack(Y_list).astype(np.float32, copy=False)

    # torch.from_numpy avoids an extra copy
    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)

    log.info(
        "Tensor creation info: "
        f"{X.shape=}, "
        f"{Y.shape=}, "
        f"number of total examples: {X.shape[0]}, "
        f"input features per example (freq axis): {X.shape[1]}, "
        f"PSD bins per example (freq axis): {Y.shape[1]}"
    )

    return X, Y
