from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Optional
import os
import threading
import tomllib  # Python 3.11+. For 3.10 use "tomli"

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import contextmanager

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class LoggingCfg(BaseModel):
    level: str = "INFO"     # INFO|DEBUG|WARNING|ERROR


class PathsCfg(BaseModel):
    clean_dir: Path
    output_dir: Path

    @field_validator("output_dir", "clean_dir", mode="before")
    @classmethod
    def _normalize_output_dir(cls, v: Any) -> Path:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            raise ValueError("output_dir and clean_dir must be set (non-empty).")
        p = Path(v) if isinstance(v, (str, Path)) else v
        return p.expanduser().resolve()

    # Derived subfolders under output_dir
    @property
    def scenario_dir(self) -> Path: return self.output_dir / "scenario"

    @property
    def distant_dir(self) -> Path: return self.output_dir / "distant"

    @property
    def enhanced_dir(self) -> Path: return self.output_dir / "enhanced"

    @property
    def metrics_dir(self) -> Path: return self.output_dir / "metrics"

    @property
    def stats_dir(self) -> Path: return self.output_dir / "stats"

    @property
    def debug_dir(self) -> Path: return self.output_dir / "debug"


class SimulationCfg(BaseModel):
    samplerate: int = 16000
    clean_duration: int = 5*samplerate


class StftCfg(BaseModel):
    window_len: int = 512
    hop_size: int = 256
    window_type: str = "sqrt_hann"


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
    paths: PathsCfg
    simulation: SimulationCfg = Field(default_factory=SimulationCfg)
    stft: StftCfg = Field(default_factory=StftCfg)


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
            "enhanced_dir": str(p.enhanced_dir),
            "metrics_dir": str(p.metrics_dir),
            "stats_dir": str(p.stats_dir),
            "debug_dir": str(p.debug_dir),
        }
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

    def set_in(d: dict[str, Any], parts: list[str], value: Any) -> None:
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
