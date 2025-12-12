import numpy as np
import pyroomacoustics as pra
from typing import Iterator, Tuple, Any
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import os

from seadge import config
from seadge.utils.visualization import plot_rir
from seadge.utils.files import files_in_path_recursive
from seadge.utils.log import log
from seadge.utils.cache import make_rir_cache_key, update_manifest, save_rir, try_load_cached_rir

def build_room(room_cfg: config.RoomCfg, fs: int) -> pra.ShoeBox:
    Lx, Ly, Lz = room_cfg.dimensions_m

    # Derive uniform absorption from desired RT60 (Sabine)
    absorption = pra.inverse_sabine(room_cfg.rt60, [Lx, Ly, Lz])[0]
    room = pra.ShoeBox(p=np.array([Lx, Ly, Lz], dtype=float),
                       fs=fs,
                       absorption=absorption,
                       max_order=room_cfg.max_image_order)

    R = np.array(room_cfg.mic_pos).T  # shape (3, M)
    room.add_microphone_array(pra.MicrophoneArray(R, fs=fs))
    return room

def make_directivity(pattern: str, az_deg: float, col_deg: float):
    orient = pra.directivities.DirectionVector(
        azimuth=az_deg, colatitude=col_deg, degrees=True
    )
    if pattern == "omni":
        directivity = pra.directivities.Omnidirectional(gain=1.0)
    elif pattern == "cardioid":
        directivity = pra.directivities.Cardioid(orientation=orient, gain=1.0)
    elif pattern == "hypercardioid":
        directivity = pra.directivities.HyperCardioid(orientation=orient, gain=1.0)
    else:
        raise ValueError(f"Invalid directivity pattern '{pattern}'")

    return directivity

def _stack_rirs_rightpad(room, *, fs: int, early_ms: float | None = None) -> np.ndarray:
    """
    Returns a (R, M) array by right-padding each mic's RIR with zeros to the max length.
    If early_ms is given, trims all RIRs to that duration before stacking.
    """
    M = room.mic_array.M
    rirs = [np.asarray(room.rir[m][0], dtype=float) for m in range(M)]
    if early_ms is not None:
        L = int(round(early_ms * 1e-3 * fs))
        rirs = [h[:L] for h in rirs]

    Rmax = max(len(h) for h in rirs) if rirs else 0
    H = np.zeros((Rmax, M), dtype=float)
    for m, h in enumerate(rirs):
        H[:len(h), m] = h
    return H

def rir_for_pose(room_cfg: config.RoomCfg, src_pos: tuple[float,float,float],
                  pattern: str, az_deg: float, col_deg: float):
    cfg = config.get()
    log.debug(f"Computing RIR for {src_pos = }, {pattern = }, {az_deg = }, {col_deg = }")
    # rebuild room each time to ensure correct state reset between computations
    room = build_room(room_cfg, cfg.dsp.datagen_samplerate)
    # add source (with directivity)
    directivity = make_directivity(pattern, az_deg, col_deg)
    room.add_source(src_pos, directivity=directivity)
    room.compute_rir()
    # stack rirs into (R, M) array and return
    rir = _stack_rirs_rightpad(room, fs=cfg.dsp.datagen_samplerate, early_ms=None)  # shape (R, M)
    return rir

class SourcePoses:
    """Iterate all (src_idx, pose_idx, src, loc) across the config."""
    def __init__(self, room_cfg):
        self._room_cfg = room_cfg
        self._num_noise = 1 if room_cfg.noise_source else 0
        self._total = sum(len(_history(s)) for s in room_cfg.sources) + self._num_noise

    def __len__(self) -> int:
        return self._total

    def __iter__(self) -> Iterator[Tuple[int, int, Any, Any]]:
        if self._room_cfg.noise_source:
            for i, src in enumerate(self._room_cfg.sources + [self._room_cfg.noise_source]):
                for j, loc in enumerate(_history(src)):
                    yield i, j, src, loc
        else:
            for i, src in enumerate(self._room_cfg.sources):
                for j, loc in enumerate(_history(src)):
                    yield i, j, src, loc

def _history(src):
    # Support either field name
    return getattr(src, "direction_history",
           getattr(src, "location_history", []))

def get_all_rirs(room_cfg: config.RoomCfg, save_rir_plots=False):
    cfg = config.get()

    cache_root = cfg.paths.rir_cache_dir

    for i, j, _, loc in SourcePoses(room_cfg):
        # ---- cache key & load attempt ----
        key = make_rir_cache_key(room_cfg, cfg.dsp.datagen_samplerate, loc)
        rir = try_load_cached_rir(cache_root, key)

        # ---- compute if cache miss ----
        if rir is None:
            rir = rir_for_pose(room_cfg, loc.location_m, loc.pattern, loc.azimuth_deg, loc.colatitude_deg)

            # ---- persist to disk (.npy + manifest) ----
            meta = {
                "fs": cfg.dsp.datagen_samplerate,
                "room": {
                    "dimensions_m": tuple(room_cfg.dimensions_m),
                    "rt60": float(room_cfg.rt60),
                    "max_order": int(room_cfg.max_image_order),
                    "mics": [tuple(p) for p in room_cfg.mic_pos],
                },
                "pose": {
                    "loc": tuple(loc.location_m),
                    "pattern": loc.pattern,
                    "az_deg": float(loc.azimuth_deg),
                    "col_deg": float(loc.colatitude_deg),
                },
                "src_index": int(i),
                "pose_index": int(j),
                "start_sample": int(loc.start_sample),
                "shape": tuple(rir.shape),
                "dtype": str(rir.dtype),
            }
            rir_path = save_rir(cache_root, key, rir, meta)
            update_manifest(
                cache_root,
                key,
                rir_path,
                src_idx=i,
                pose_idx=j,
                start_sample=loc.start_sample,
                loc=loc,
                rir=rir,
                fs=cfg.dsp.datagen_samplerate,
            )

        # ---- optional plots ----
        if save_rir_plots:
            # full RIR
            plot_rir(
                rir=rir,
                fs=cfg.dsp.datagen_samplerate,
                outputfile=cfg.paths.debug_dir / "rir" / f"src_{i}_pos_{j}.png",
                title=f"RIR src {i}, pos {j} (at sample {loc.start_sample})",
            )
            # first early_ms ms
            early_ms = 80
            plot_rir(
                rir=rir,
                fs=cfg.dsp.datagen_samplerate,
                outputfile=cfg.paths.debug_dir / "rir" / f"early_{early_ms}ms_src_{i}_pos_{j}.png",
                title=f"Early {early_ms} ms, RIR src {i}, pos {j} (at sample {loc.start_sample})",
                early_ms=early_ms,
            )

def _compute_rirs_for_room(room_file: Path, debug: bool):
    room = config.load_room(room_file)
    get_all_rirs(room, save_rir_plots=debug)

def main():
    cfg = config.get()
    room_files = list(files_in_path_recursive(cfg.paths.room_dir, "*.room.json"))

    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    num_processes = int(slurm_cpus) if slurm_cpus else os.cpu_count()
    log.info(f"Computing RIRs for {len(room_files)} rooms with {num_processes} workers")

    worker = partial(_compute_rirs_for_room, debug=cfg.debug)

    with Pool(processes=num_processes) as pool:
        for _ in tqdm(
            pool.imap_unordered(worker, room_files),
            total=len(room_files),
            desc="Computing RIRs for rooms",
        ):
            pass
