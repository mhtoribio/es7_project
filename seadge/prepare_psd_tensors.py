from pathlib import Path

from seadge.utils.psd_data_loader import build_tensors_from_dir
from seadge.utils.log import log
from seadge.utils.psd_data_loader import save_tensors_cache
from seadge import config

def precompute_psd_tensors_from_dir(npz_dir: Path, L_max: int, max_npz_files: int):
    log.info(f"Computing tensors from {npz_dir}")
    x_tensor, y_tensor = build_tensors_from_dir(npz_dir, L_max, max_npz_files)
    save_tensors_cache(npz_dir, x_tensor, y_tensor, L_max=L_max)

def main():
    cfg = config.get()
    precompute_psd_tensors_from_dir(cfg.paths.ml_data_dir, cfg.L_max, cfg.deeplearning.num_max_npz_files)
