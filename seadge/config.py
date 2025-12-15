from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union, Literal, Annotated, List
import os
import threading
import tomllib  # Python 3.11+. For 3.10 use "tomli"
from math import cos, sin, radians

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import contextmanager

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class LoggingCfg(BaseModel):
    level: str = "INFO"     # INFO|DEBUG|WARNING|ERROR
    format: str = "[%(asctime)s %(levelname)s] %(message)s"


class PathsCfg(BaseModel):
    clean_dir: Path = Path("/tmp/seadge_clean")
    output_dir: Path = Path("/tmp/seadge_output")
    download_cache_dir: Path = Path("/tmp/seadge_download_cache")
    matfile_dir: Path = Path("/tmp/seadge_matfiles") # matfiles for ISCLP manually added here (TODO download in download subcommand)
    noise_dir: Path = Path("/tmp/seadge_noise") # Noise wav files manually added here
    psd_model: Path = Path("/tmp/seadge_model/psdmodel.pt") # Trained PSD model saved here

    @field_validator("output_dir", "clean_dir", "download_cache_dir", "matfile_dir", "noise_dir", "psd_model", mode="before")
    @classmethod
    def _normalize_output_dir(cls, v: Any) -> Path:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("output_dir clean_dir, download_cache_dir, matfile_dir, noise_dir and psd_model must be set (non-empty).")
        p = Path(v) if isinstance(v, (str, Path)) else v
        return p.expanduser().resolve()

    # Derived subfolders under output_dir
    @property
    def scenario_dir(self) -> Path: return self.output_dir / "scenario"

    @property
    def distant_dir(self) -> Path: return self.output_dir / "distant"

    @property
    def room_dir(self) -> Path: return self.output_dir / "rooms"

    @property
    def enhanced_dir(self) -> Path: return self.output_dir / "enhanced"

    @property
    def metrics_dir(self) -> Path: return self.output_dir / "metrics"

    @property
    def stats_dir(self) -> Path: return self.output_dir / "stats"

    @property
    def debug_dir(self) -> Path: return self.output_dir / "debug"

    @property
    def rir_cache_dir(self) -> Path: return self.output_dir / "rir_cache"

    @property
    def ml_data_dir(self) -> Path: return self.output_dir / "ml_data"

    @property
    def models_dir(self) -> Path: return self.output_dir / "trained_models"

    @property
    def checkpoint_dir(self) -> Path: return self.output_dir / "checkpoints"

class IsclpCfg(BaseModel):
    L: int = 6                                        # Linear Prediction Length
    alpha_ISCLP_exp_db: float = -25                   # Forgetting factor alpha exponent (1 - power)
    psi_wLP_db: float = -4                            # LP filter variance (eq. 54)
    psi_wSC_db_range: Tuple[float, float] = (0, -15)  # SC filter variance (eq. 53)
    beta_ISCLP_db: float = -2                         # smoothing
    retf_thres_db : float = -2                        # RETF update threshold

class DspCfg(BaseModel):
    window_len: int = 512
    hop_size: int = 256
    window_type: str = "sqrt_hann"
    datagen_samplerate:     int = 48000
    enhancement_samplerate: int = 16000
    early_tmax_ms: float = 32.0
    early_offset_ms: float = 0.0

    # spectogram figure settings
    # TODO make this fit with both samplerates
    x_tick_prop : Tuple[float, float, float] = (0, hop_size/enhancement_samplerate, 10*enhancement_samplerate/hop_size);
    y_tick_prop : Tuple[float, float, float] = (0, enhancement_samplerate/(2000*hop_size), hop_size/2);
    c_range     : Tuple[int, int]            = (-55, 5);

    # RIR convolution settings
    rirconv_method    : str = "oaconv" # "oaconv" or "fft"
    rirconv_normalize : str = "none" # "none"|"direct"|"energy"
    rirconv_xfade_ms  : float = 64.0

    # ISCLP (enhancement algorithm) config
    isclpconf: IsclpCfg = Field(default_factory=IsclpCfg)


class MicDesc(BaseModel):
    array_type: Literal["linear"] = "linear"
    num_mics: int = 5
    spacing: float = 0.08          # meters between adjacent mics
    yaw_deg: float = 0.0           # 0° = +x axis, CCW around +z
    origin: Tuple[float, float, float] = (2.5, 0.05, 1.2)  # center of array (x, y, z)

    def expand_positions(self) -> List[Tuple[float, float, float]]:
        """Return concrete mic positions for a linear array centered at origin."""
        assert self.array_type == "linear"
        # Direction unit vector in XY plane
        ux, uy = cos(radians(self.yaw_deg)), sin(radians(self.yaw_deg))
        # Centered offsets: e.g., for 5 mics -> [-2, -1, 0, 1, 2]*spacing
        c = (self.num_mics - 1) / 2.0
        x0, y0, z = self.origin
        return [
            (x0 + (i - c) * self.spacing * ux,
             y0 + (i - c) * self.spacing * uy,
             z)
            for i in range(self.num_mics)
        ]


class SourceSpec(BaseModel):
    class LocationSpec(BaseModel):
        pattern: Literal["cardioid", "omni", "dipole", "hypercardioid"] = "cardioid"
        start_sample: int = 0
        azimuth_deg: float = 0.0
        colatitude_deg: float = 90.0
        location_m: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    location_history: List[LocationSpec] = Field(min_length=1)


class RoomCfg(BaseModel):
    rt60: float = 0.45
    dimensions_m: Tuple[float, float, float] = (5.0, 6.0, 3.0)
    max_image_order: int = 10

    # Accept either a MicDesc (expanded automatically) or explicit positions
    mic_pos: Union[MicDesc, List[Tuple[float, float, float]]] = Field(default_factory=MicDesc)
    sources: List[SourceSpec] = Field(default_factory=list)
    noise_source: Optional[SourceSpec] = Field(default=None) # currently only 1 fixed noise source/location supported

    @model_validator(mode="after")
    def _expand_and_validate(self):
        Lx, Ly, Lz = self.dimensions_m

        # If mic_pos is a MicDesc, expand to concrete positions
        if isinstance(self.mic_pos, MicDesc):
            self.mic_pos = self.mic_pos.expand_positions()

        # Validate all mic positions are inside the room
        for i, (x, y, z) in enumerate(self.mic_pos):
            if not (0.0 <= x <= Lx and 0.0 <= y <= Ly and 0.0 <= z <= Lz):
                raise ValueError(
                    f"mic_pos[{i}]={(x,y,z)} outside room bounds (0..{Lx}, 0..{Ly}, 0..{Lz})"
                )

        # Validate source locations are inside the room
        for sidx, src in enumerate(self.sources):
            for tidx, loc in enumerate(src.location_history):
                x, y, z = loc.location_m
                if not (0.0 <= x <= Lx and 0.0 <= y <= Ly and 0.0 <= z <= Lz):
                    raise ValueError(
                        f"source[{sidx}].direction_history[{tidx}].location_m {loc.location_m} "
                        f"outside room bounds (0..{Lx}, 0..{Ly}, 0..{Lz})"
                    )
        return self


class RoomGenCfg(BaseModel):
    rt60_min: float = 0.3
    rt60_max: float = 1.0
    min_dimensions_m: Tuple [float, float, float] = (3.0, 4.0, 2.0)
    max_dimensions_m: Tuple [float, float, float] = (8.0, 10.0, 4.0)
    max_image_order: int = 10

    num_generated_rooms: int = 3

    enable_noise: bool = False

    min_num_source_locations: int = 2
    max_num_source_locations: int = 5
    # Minimum distances from sources to wall and mic
    # Be careful choosing these values since they (along with room dimensions) also determine "spread" of sources in the room
    min_source_distance_to_wall_m: float = 1.0
    min_source_distance_to_mic_m: float = 2.0
    min_source_inter_spacing: float = 0.4
    max_source_movement_m: float = 1.0
    max_azimuth_rotation_deg: float = 90.0
    max_movement_steps: int = 3
    min_movement_step_duration: int = 16000 # value in samples

    # mic array
    num_mics: int = 5
    mic_spacing: float = 0.05
    mic_wall_offset: float = 0.05
    mic_height: float = 1.2

class ScenarioGenCfg(BaseModel):
    # required
    scenario_duration_s: Annotated[float,  Field(gt=0)] = 5.0
    min_interference_volume: Annotated[float, Field(ge=0.0, le=1.0)] = 0.1
    max_interference_volume: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    min_snr_db: float = 10 # source-level SNR (based on clean wav files)
    max_snr_db: float = 20 # source-level SNR (based on clean wav files)
    scenarios_per_room: Annotated[int, Field(gt=0)] = 5

    # optional
    num_speakers: Annotated[int | None, Field(gt=0)] = None
    # None => treat as scenario_duration at *use* time
    min_wavsource_duration_s: Annotated[float | None, Field(gt=0)] = None

    @model_validator(mode="after")
    def _cross_checks(self):
        if self.min_interference_volume > self.max_interference_volume:
            raise ValueError("min_speaker_volume must be <= max_speaker_volume")
        if self.min_wavsource_duration_s is not None:
            if self.min_wavsource_duration_s > self.scenario_duration_s:
                raise ValueError("min_wavsource_duration cannot exceed scenario_duration")
        return self

    # Convenience: compute the effective value without mutating the stored one
    @property
    def effective_min_wavsource_duration_s(self) -> float:
        return self.scenario_duration_s if self.min_wavsource_duration_s is None else self.min_wavsource_duration_s

class LearningCfg(BaseModel):
    epochs: int = 50
    hidden_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 8
    num_workers: int = 4
    num_max_npz_files: int = 0

class Config(BaseSettings):
    """
    App settings pulled from:
      - The REQUIRED config file passed to load()
      - Environment variables / .env (overrides file)
      - Explicit programmatic overrides (overrides env)
      - Defaults in this class

    Env examples:
      SEADGE_ENV=prod
      SEADGE_LOGGING__LEVEL=DEBUG
      SEADGE_PATHS__OUTPUT_DIR=/srv/run-001
      SEADGE_PATHS__CLEAN_DIR=/data/clean
    """
    model_config = SettingsConfigDict(
        env_prefix="SEADGE_",
        env_file=".env",                    # optional
        env_nested_delimiter="__",          # nested keys via double underscore
        extra="forbid",                     # catch typos early
    )

    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    # PathsCfg has required fields -> Config.paths is required too
    paths: PathsCfg = Field(default_factory=PathsCfg)
    dsp: DspCfg = Field(default_factory=DspCfg)
    roomgen: RoomGenCfg = Field(default_factory=RoomGenCfg)
    scenariogen: ScenarioGenCfg = Field(default_factory=ScenarioGenCfg)
    debug: bool = Field(default=False)
    clean_zip_files: int = Field(default=1)
    deeplearning: LearningCfg = Field(default_factory=LearningCfg)
    scenarios: Optional[int] = Field(default=None) # optional max number of scenarios to use in a given step (for example enhancement)

    # ---- Derived property: L_max for STFT frames ----
    @property
    def L_max(self) -> int:
        """
        Maximum number of STFT frames for training, derived from:
          - scenario_duration_s (clean length)
          - datagen_samplerate
          - STFT window_len & hop_size
          - an assumed max RIR length (here: dsp.early_tmax_ms)
        """

        fs = self.dsp.enhancement_samplerate
        win = self.dsp.window_len
        hop = self.dsp.hop_size

        # Clean duration in samples
        N_clean = int(round(self.scenariogen.scenario_duration_s * fs))

        max_rir_s = 1.2 # hardcoded max RIR length
        N_rir = int(round(max_rir_s * fs))

        N_td = N_clean + N_rir  # total time-domain length of mic signals

        if N_td <= win:
            return 1

        return 1 + (N_td - win) // hop


# -----------------------------------------------------------------------------
# Module-level state
# -----------------------------------------------------------------------------

_current: Optional[Config] = None
_lock = threading.RLock()


class ConfigError(RuntimeError):
    pass


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def dumpconfig(args):
    from pathlib import PurePath
    def _replace_paths(obj: Any) -> Any:
        """Recursively replace pathlib Paths with strings inside common containers."""
        if isinstance(obj, dict):
            # Also convert keys if they are Paths
            return {_replace_paths(k): _replace_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_replace_paths(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_replace_paths(x) for x in obj)
        if isinstance(obj, set):
            return {_replace_paths(x) for x in obj}
        if isinstance(obj, PurePath):  # catches PosixPath, WindowsPath, etc.
            return str(obj)
        return obj

    import json
    obj = as_dict()
    obj = _replace_paths(obj)
    print(json.dumps(obj, indent=2, ensure_ascii=False))

def load_default() -> Config:
    """Load the default config. ONLY RECOMMENDED FOR TESTING."""
    cfg = Config()
    _set(cfg)
    return cfg

def load(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    create_dirs: bool = False,
) -> Config:
    """
    Load config from a REQUIRED file path, then apply env and explicit overrides.
    Precedence: overrides > env (.env / process) > file > defaults.
    """
    if path is None:
        raise ConfigError("A config file path is required (no auto paths supported).")

    with _lock:
        try:
            file_data = _read_config_file(Path(path))
            merged = _apply_env_overrides(file_data, prefix="SEADGE_")
            if overrides:
                merged = _deep_merge(merged, _flatten_overrides(overrides))

            cfg = Config.model_validate(merged)

            if create_dirs:
                _ensure_output_dirs(cfg)

            _set(cfg)
            return cfg

        except ValidationError as e:
            raise ConfigError(
                "Invalid configuration. Ensure required keys exist and types are correct.\n"
                f"Details:\n{e}"
            ) from e

def load_room(path: str | Path) -> RoomCfg:
    """
    Load Room Config from file
    """
    try:
        file_data = _read_config_file(Path(path))
        room_cfg = RoomCfg.model_validate(file_data)
        return room_cfg

    except ValidationError as e:
        raise ConfigError(
            "Invalid configuration. Ensure required keys exist and types are correct.\n"
            f"Details:\n{e}"
        ) from e

def save_json(cfg: BaseModel, path: Path, *, indent: int = 2, exclude_none: bool = True) -> None:
    """
    Serialize a Pydantic model to JSON on disk.
    Uses an atomic write (temp file + rename).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    json_text = cfg.model_dump_json(indent=indent, exclude_none=exclude_none)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_text, encoding="utf-8", newline="\n")
    tmp.replace(path)

def get() -> Config:
    """Return the active config. Raises if you forgot to call load()."""
    with _lock:
        if _current is None:
            raise ConfigError("Config not loaded. Call load('/path/to/config.toml') first.")
        return _current


def as_dict(include_computed: bool = True) -> dict[str, Any]:
    """
    Dump config to a dict. Derived properties (scenario_dir, etc.) are not model fields,
    so we optionally add them for convenience.
    """
    cfg = get()
    d = cfg.model_dump()
    if include_computed:
        p = cfg.paths
        d.setdefault("paths", {})
        d["paths"] |= {
            "scenario_dir": str(p.scenario_dir),
            "distant_dir": str(p.distant_dir),
            "room_dir": str(p.room_dir),
            "enhanced_dir": str(p.enhanced_dir),
            "metrics_dir": str(p.metrics_dir),
            "stats_dir": str(p.stats_dir),
            "debug_dir": str(p.debug_dir),
            "rir_cache_dir": str(p.rir_cache_dir),
            "ml_data_dir": str(p.ml_data_dir),
            "models_dir": str(p.models_dir),
            "checkpoint_dir": str(p.checkpoint_dir),
        }
        d.setdefault("deeplearning", {})
        d["deeplearning"]["L_max"] = cfg.L_max
    return d


@contextmanager
def temporary(overrides: Mapping[str, Any] | None = None, create_dirs: bool = False):
    """
    Temporarily override the active config inside a 'with' block (useful in tests).
    """
    with _lock:
        base = get()
        tmp_data = base.model_dump()
        if overrides:
            tmp_data = _deep_merge(tmp_data, _flatten_overrides(overrides))
        tmp_cfg = Config.model_validate(tmp_data)
        if create_dirs:
            _ensure_output_dirs(tmp_cfg)
        old = _current
        try:
            _set(tmp_cfg)
            yield
        finally:
            _set(old)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _set(cfg: Optional[Config]) -> None:
    global _current
    _current = cfg


def _read_config_file(p: Path) -> dict[str, Any]:
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".toml":
        with p.open("rb") as f:
            return tomllib.load(f)
    if suffix == ".json":
        import json
        return json.loads(p.read_text())
    if suffix in {".yaml", ".yml"}:
        import yaml  # requires "pyyaml"
        return yaml.safe_load(p.read_text())
    raise ConfigError(f"Unsupported config format: {suffix}")


def _apply_env_overrides(d: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """
    Overlay env vars on top of the given dict.
    Nested keys via double underscores, e.g.:
      SEADGE_LOGGING__LEVEL=DEBUG
      SEADGE_PATHS__OUTPUT_DIR=/tmp/out
      SEADGE_PATHS__CLEAN_DIR=/data/clean
    """
    out = dict(d)

    def parse_val(v: str) -> Any:
        vl = v.lower()
        if vl in {"true", "false"}:
            return vl == "true"
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    for k, v in os.environ.items():
        if not k.startswith(prefix):
            continue
        keypath = k[len(prefix):]  # strip prefix
        parts = keypath.split("__")
        # Keep field names in lower-case for safety
        cur = out
        for part in parts[:-1]:
            cur = cur.setdefault(part.lower(), {}) if isinstance(cur, dict) else cur
        cur[parts[-1].lower()] = parse_val(v)
    return out


def _flatten_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """
    Accept {'paths.output_dir': '/x', 'logging.level': 'DEBUG'} or nested dicts.
    """
    out: dict[str, Any] = {}

    def set_in(d: dict[str, Any], parts: List[str], value: Any) -> None:
        cur = d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    for k, v in overrides.items():
        if isinstance(v, dict):
            out[k] = v
        elif "." in k:
            set_in(out, k.split("."), v)
        else:
            out[k] = v
    return out


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Nested merge: values from b override a."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _ensure_output_dirs(cfg: Config) -> None:
    for p in (
        cfg.paths.output_dir,
        cfg.paths.scenario_dir,
        cfg.paths.distant_dir,
        cfg.paths.enhanced_dir,
        cfg.paths.metrics_dir,
        cfg.paths.stats_dir,
        cfg.paths.debug_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)
