# --- hashing & paths ---
import json, hashlib, time
from pathlib import Path
import numpy as np
from functools import lru_cache
import os
import time
import fcntl
import json, hashlib
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel

def _q(x, q=1e-6):
    if isinstance(x, float): return round(x / q) * q
    if isinstance(x, (list, tuple)): return tuple(_q(t, q) for t in x)
    if isinstance(x, dict): return {k: _q(v, q) for k, v in x.items()}
    return x

def _q_float(x: float, ndigits: int = 6) -> float:
    # stable, banker-proof rounding via Decimal
    d = Decimal(str(x)).quantize(Decimal(10) ** -ndigits, rounding=ROUND_HALF_UP)
    return float(d)

def _canonicalize(obj, *, float_ndigits=6):
    # Recursively make a deterministic, JSON-friendly payload
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, float):
        return _q_float(obj, float_ndigits)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        # sort keys for determinism
        return {k: _canonicalize(v, float_ndigits=float_ndigits) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v, float_ndigits=float_ndigits) for v in obj]
    if isinstance(obj, set):
        # sets are unordered → sort after canonicalizing
        return sorted((_canonicalize(v, float_ndigits=float_ndigits) for v in obj), key=lambda x: json.dumps(x, sort_keys=True))
    return obj

def make_pydantic_cache_key(
    model: BaseModel,
    *,
    exclude: set | dict | None = None,
    include: set | dict | None = None,
    float_ndigits: int = 6,
    algo: str = "sha1",
) -> str:
    """
    Deterministic content hash for any Pydantic model.
    - Uses model_dump(mode="json", exclude_none=True) + optional include/exclude
    - Quantizes floats
    - Sorts dict keys
    - Canonicalizes Paths, sets, tuples, etc.
    """
    payload = model.model_dump(mode="json", exclude_none=True, exclude=exclude, include=include)
    canon = _canonicalize(payload, float_ndigits=float_ndigits)
    s = json.dumps(canon, separators=(",", ":"), sort_keys=True)
    h = hashlib.new(algo)
    h.update(s.encode("utf-8"))
    return h.hexdigest()

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
def load_rir_mem_or_die(key: str, cache_root_str: str) -> np.ndarray:
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

def update_manifest(
    cache_root: Path,
    key: str,
    rir_path: Path,
    *,
    src_idx: int,
    pose_idx: int,
    start_sample: int,
    loc,
    rir: np.ndarray,
    fs: int,
) -> None:
    """
    Safe, concurrent update of cache_root/manifest.json.

    Uses:
      - .manifest.lock as an advisory file lock (fcntl.flock, exclusive)
      - atomic write via temp file + rename
    """
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    mpath = cache_root / "manifest.json"
    lock_path = cache_root / ".manifest.lock"

    # Open a dedicated lock file and take an exclusive lock
    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            # --- BEGIN critical section: read → modify → write manifest ---

            if mpath.is_file():
                try:
                    manifest = json.loads(mpath.read_text())
                except json.JSONDecodeError:
                    # Corrupt / partial file -> start fresh (or log)
                    manifest = {"version": 1, "entries": []}
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
            # Write to temp file, fsync, then rename (atomic)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(mpath)

            # --- END critical section ---
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
