import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from typing import Tuple, List
from tqdm.contrib.concurrent import process_map
import os
import json

from seadge.utils.log import log
from seadge.utils.files import files_in_path_recursive
from seadge.utils.dsp import complex_to_mag_phase, complex_to_re_im

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
    #re, im = complex_to_re_im(distant)
    mag = complex_to_mag_phase(distant)
    logmag = np.log1p(mag)
    # features: (2K, L, M)
    #features = np.concatenate((re, im))
    features = logmag

    # psd: (K, L)
    psd = np.abs(early) ** 2  # ground truth

    return features.astype(np.float32, copy=False), psd.astype(np.float32, copy=False)


def build_tensors_from_dir(npz_dir: Path, L_max: int, num_max_npz: int) -> tuple[torch.Tensor, torch.Tensor]:
    npz_files = list(files_in_path_recursive(npz_dir, "*.npz"))
    if num_max_npz > len(npz_files):
        log.warning(f"Desired number of npz files (scenarios) too large ({num_max_npz}). Only {len(npz_files)} found.")
    if num_max_npz == 0:
        log.info(f"No desired number of npz files set. Using all available files on disk.")
        num_max_npz = len(npz_files)
    num_npz = min(len(npz_files), num_max_npz)
    log.info(f"Creating tensors from {num_npz} npz files")

    if not npz_files:
        raise RuntimeError(f"No npz files found in {npz_dir}")

    n_workers = _get_n_workers()
    log.info(f"Loading npz files with {n_workers} workers")

    t = time.time()
    # process_map gives you tqdm for free
    results: List[Tuple[np.ndarray, np.ndarray]] = process_map(
        _load_one_npz_for_training,
        [(f, L_max) for f in npz_files[:num_npz]],
        max_workers=n_workers,
        chunksize=1,
        desc="loading npz files",
    )

    log.debug(f"Extracting X_list and Y_list. Previous step took {time.time()-t} s")
    t = time.time()
    X_list, Y_list = zip(*results)  # tuples of np.ndarrays
    del results

    # Convert lists to numpy
    log.debug(f"Converting lists to numpy. Previous step took {time.time()-t} s")
    t = time.time()
    n = len(X_list)
    X_shape = (n,) + X_list[0].shape
    Y_shape = (n,) + Y_list[0].shape
    X_np = np.empty(X_shape, dtype=np.float32)
    Y_np = np.empty(Y_shape, dtype=np.float32)
    for i, arr in tqdm(enumerate(X_list), desc="Converting X_list to X_np", total=len(X_list)):
        X_np[i] = arr
    del X_list
    for i, arr in tqdm(enumerate(Y_list), desc="Converting Y_list to Y_np", total=len(Y_list)):
        Y_np[i] = arr
    del Y_list

    # torch.from_numpy avoids an extra copy
    log.debug(f"Converting numpy to torch. Previous step took {time.time()-t} s")
    t = time.time()
    X = torch.from_numpy(X_np)
    Y = torch.from_numpy(Y_np)
    Y = torch.log1p(Y)

    log.info(
        "Tensor creation info: "
        f"{X.shape=}, "
        f"{Y.shape=}, "
        f"number of total examples: {X.shape[0]}, "
        f"input features per example (freq axis): {X.shape[1]}, "
        f"PSD bins per example (freq axis): {Y.shape[1]}"
    )

    return X, Y

def load_tensors_cache(cache_dir: Path) -> tuple[torch.Tensor, torch.Tensor, int]:
    d = Path(cache_dir)
    X_mm = np.load(d / "X.npy", mmap_mode="c")  # (N, 2K, L, M), float32
    Y_mm = np.load(d / "Y.npy", mmap_mode="c")  # (N, K, L[,...]), float32
    L_max = int(json.loads((d / "meta.json").read_text())["L_max"])
    return torch.from_numpy(X_mm), torch.from_numpy(Y_mm), L_max

def save_tensors_cache(cache_dir: Path, X: torch.Tensor, Y: torch.Tensor, *, L_max: int):
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)

    X_cpu = X.detach().to("cpu", dtype=torch.float32, copy=False).contiguous()
    Y_cpu = Y.detach().to("cpu", dtype=torch.float32, copy=False).contiguous()

    np.save(d / "X.npy", X_cpu.numpy(), allow_pickle=False)
    np.save(d / "Y.npy", Y_cpu.numpy(), allow_pickle=False)

    (d / "meta.json").write_text(json.dumps({"L_max": int(L_max)}))
