# --- hashing & paths ---
import json, hashlib, time
from pathlib import Path
import numpy as np
from functools import lru_cache

def _q(x, q=1e-6):
    if isinstance(x, float): return round(x / q) * q
    if isinstance(x, (list, tuple)): return tuple(_q(t, q) for t in x)
    if isinstance(x, dict): return {k: _q(v, q) for k, v in x.items()}
    return x

def make_room_cache_key(room_cfg) -> str:
    """
    Deterministically hash a RoomCfg into a 40-hex SHA-1.
    Includes: room dims/rt60/max_image_order, expanded mic positions,
    and all source location histories (pattern, start_sample, az/col, location).
    Floats are quantized to reduce spurious diffs.
    """
    sources_payload = []
    for s in room_cfg.sources:
        locs = []
        for loc in s.location_history:
            locs.append({
                "pattern": loc.pattern,
                "start_sample": int(loc.start_sample),
                "az_deg": _q(loc.azimuth_deg),
                "col_deg": _q(loc.colatitude_deg),
                "location_m": _q(loc.location_m),
            })
        sources_payload.append({"location_history": locs})

    payload = {
        "room": {
            "dimensions_m": _q(room_cfg.dimensions_m),
            "rt60": _q(room_cfg.rt60),
            "max_image_order": int(room_cfg.max_image_order),
        },
        "mics": [_q(p) for p in room_cfg.mic_pos], # expanded positions
        "sources": sources_payload,                # full source specs
    }

    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def make_rir_cache_key(room_cfg, fs: int, loc) -> str:
    payload = {
        "fs": fs,
        "room": {
            "dimensions_m": _q(room_cfg.dimensions_m),
            "rt60": _q(room_cfg.rt60),
            "max_order": room_cfg.max_image_order,
        },
        "mics": [_q(p) for p in room_cfg.mic_pos],  # expanded positions
        "pose": {
            "loc": _q(loc.location_m),
            "pattern": loc.pattern,
            "az_deg": _q(loc.azimuth_deg),
            "col_deg": _q(loc.colatitude_deg),
        },
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()  # 40 hex

def _shard_dir(root: Path, key: str) -> Path:
    return root / key[:2] / key

# --- IO ---
@lru_cache(maxsize=256)
def load_rir_mem_or_die(key: str, cache_root_str: str) -> np.ndarray | None:
    """LRU wrapper around disk cache. Returns a *regular* ndarray (not memmap)."""
    arr = try_load_cached_rir(Path(cache_root_str), key)
    if arr is None:
        raise FileNotFoundError(f"The RIR is missing, {key = }")
    arr.setflags(write=False)  # keep it read-only
    return arr

def try_load_cached_rir(cache_root: Path, key: str) -> np.ndarray | None:
    f = _shard_dir(cache_root, key) / "rir.npy"
    if f.is_file():
        return np.load(f)  # use mmap_mode="r" if needed
    return None

def save_rir(cache_root: Path, key: str, rir: np.ndarray, meta: dict) -> Path:
    d = _shard_dir(cache_root, key)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "rir.npy", rir, allow_pickle=False)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d / "rir.npy"

def update_manifest(cache_root: Path, key: str, rir_path: Path, *, src_idx: int, pose_idx: int,
                    start_sample: int, loc, rir: np.ndarray, fs: int) -> None:
    mpath = cache_root / "manifest.json"
    if mpath.is_file():
        manifest = json.loads(mpath.read_text())
    else:
        manifest = {"version": 1, "entries": []}
    entry = {
        "key": key,
        "file": str(rir_path.relative_to(cache_root)),
        "shape": list(rir.shape),
        "dtype": str(rir.dtype),
        "src_index": int(src_idx),
        "pose_index": int(pose_idx),
        "start_sample": int(start_sample),
        "pose": {
            "loc": list(map(float, loc.location_m)),
            "pattern": loc.pattern,
            "az_deg": float(loc.azimuth_deg),
            "col_deg": float(loc.colatitude_deg),
        },
        "fs": int(fs),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # upsert by key
    entries = manifest["entries"]
    for i, e in enumerate(entries):
        if e["key"] == key:
            entries[i] = entry
            break
    else:
        entries.append(entry)
    tmp = mpath.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(mpath)
