import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

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

def load_tensors_from_dir(npz_dir: Path, L_max: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    npz_files = files_in_path_recursive(npz_dir, "*.npz")
    log.debug(f"Creating tensors from {len(npz_files)} npz files")

    # Compute labels
    X_list = []
    Y_list = []
    for npz_file in tqdm(npz_files, desc="loading npz files"):
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

