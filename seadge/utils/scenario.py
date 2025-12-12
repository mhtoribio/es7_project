from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Tuple
import re
import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from seadge import config
from seadge.utils.wavfiles import wavfile_frames, wavfile_samplerate, load_wav
from seadge.utils.log import log


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class WavSource(BaseModel):
    """
    Single WAV source spec (target or interferer).
    - sourceloc: integer index identifying where to place/mix this source (replaces `seat`)
                 Validated to be >= 0. If you want to check against a specific RoomCfg's
                 number of sources later, do it where you have the room available.
    """
    wav_path: str
    volume: float
    sourceloc: int
    delay_samples: int
    duration_samples: int
    decimation: int
    interpolation: int

    @model_validator(mode="after")
    def _basic_sanity(self) -> "WavSource":
        if not (0.0 <= float(self.volume) <= 1.0):
            raise ValueError(f"volume out of range [0,1]: {self.volume}")
        if self.sourceloc < 0:
            raise ValueError(f"sourceloc must be >= 0 (got {self.sourceloc})")
        if self.delay_samples < 0:
            raise ValueError(f"negative delay: {self.delay_samples}")
        if self.duration_samples <= 0:
            raise ValueError(f"non-positive duration: {self.duration_samples}")
        if self.decimation <= 0 or self.interpolation <= 0:
            raise ValueError("decimation and interpolation must be positive integers")
        return self


_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")

class Scenario(BaseModel):
    """
    Whole scenario definition; validated on load.
    - room_id:     40 hex chars (SHA-1)
    """
    room_id: str
    duration_samples: int
    scenario_type: Optional[str] = None
    target_speaker: WavSource
    interferent_speakers: List[WavSource] = Field(default_factory=list)
    noise_sources: List[WavSource] = Field(default_factory=list)

    @field_validator("room_id")
    @classmethod
    def _check_room_id(cls, v: str) -> str:
        if not _SHA1_RE.match(v):
            raise ValueError("room_id must be a 40-char lowercase SHA-1 hex string")
        return v

    @field_validator("interferent_speakers", mode="before")
    @classmethod
    def _none_to_empty(cls, v):
        return [] if v is None else v

    @model_validator(mode="after")
    def _full_validation(self) -> "Scenario":
        # top-level duration
        if not (self.duration_samples > 0):
            raise ValueError(
                f"duration_samples must be in positive "
                f"(got {self.duration_samples})"
            )

        # timing consistency vs top-level duration
        self._check_timing(self.target_speaker, "target_speaker")
        for i, s in enumerate(self.interferent_speakers or []):
            self._check_timing(s, f"interferent_speakers[{i}]")

        # file existence + length + effective samplerate match
        self._check_file_ok(self.target_speaker, "target_speaker")
        for i, s in enumerate(self.interferent_speakers or []):
            self._check_file_ok(s, f"interferent_speakers[{i}]")

        return self

    # ---- helpers used inside model validation ----

    def _check_timing(self, src: WavSource, label: str) -> None:
        top = self.duration_samples
        if src.delay_samples + src.duration_samples > top:
            raise ValueError(
                f"{label}: delay+duration ({src.delay_samples + src.duration_samples}) "
                f"exceeds scenario duration ({top})"
            )

    def _file_has_enough_audio(
        self,
        abs_wav_path: Path,
        needed_samples: int,
        decimation: int,
        interpolation: int,
        expected_fs: int,
    ) -> bool:
        try:
            frames = wavfile_frames(abs_wav_path)
            fs = wavfile_samplerate(abs_wav_path)
        except Exception as e:
            log.debug("Failed to read WAV header %s: %s", abs_wav_path, e)
            return False

        # Keep your exact integer resampling check
        resampled_fs = fs * interpolation // decimation
        if resampled_fs != expected_fs:
            log.debug(
                "Sample rate mismatch for %s (clean %d, want %d; decim=%d, interp=%d -> %d)",
                abs_wav_path, fs, expected_fs, decimation, interpolation, resampled_fs
            )
            return False

        resampled_frames = frames * interpolation // decimation
        if resampled_frames < needed_samples:
            log.debug(
                "WAV too short: need %d samples, file has %d (%d after resampling)",
                needed_samples, frames, resampled_frames
            )
            return False

        return True

    def _check_file_ok(self, src: WavSource, label: str) -> None:
        cfg = config.get()
        abs_wav = (cfg.paths.clean_dir / src.wav_path).expanduser().resolve()
        if not abs_wav.is_file():
            raise ValueError(f"{label}: wav_path invalid or missing: {src.wav_path}")

        ok = self._file_has_enough_audio(
            abs_wav_path=abs_wav,
            needed_samples=src.duration_samples,
            decimation=src.decimation,
            interpolation=src.interpolation,
            expected_fs=cfg.dsp.datagen_samplerate,
        )
        if not ok:
            raise ValueError(f"{label}: file shorter than requested duration or fs mismatch")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def delay_and_scale_source(output_len: int, x: np.ndarray, delay_samples: int, duration_samples: int, volume: float) -> np.ndarray:
    y = np.zeros(output_len, dtype=float)
    log.debug("Delaying source with %d samples and duration %d, scaling by %g",
              delay_samples, duration_samples, volume)
    y[delay_samples: delay_samples + duration_samples] = x[:duration_samples] * volume
    return y


def load_and_resample_source(source_spec: WavSource) -> np.ndarray:
    """
    Loads and resamples source file according to spec.
    Assumes the Scenario has already been validated.
    """
    cfg = config.get()
    abs_wav = (cfg.paths.clean_dir / source_spec.wav_path).expanduser().resolve()
    fs, x = load_wav(abs_wav, dtype=np.float64)

    # Convert to mono if needed
    if x.ndim == 2:
        x = x[:, 0]

    decim = source_spec.decimation
    interp = source_spec.interpolation
    if decim != interp:
        x_resampled = resample_poly(x, interp, decim)
        log.debug(
            "Resampled with decimation=%d interpolation=%d from %d to %d (shape=%s)",
            decim, interp, fs, fs * interp // decim, x_resampled.shape
        )
        return x_resampled
    else:
        return x


def prepare_source(source_spec: WavSource, output_len: int) -> tuple[np.ndarray, int]:
    """
    Loads WAV source and processes it according to the spec.
    Returns: (signal to mix, sourceloc index)
    """
    log.debug("Preparing source %s", source_spec.wav_path)
    x = load_and_resample_source(source_spec)
    x_to_mix = delay_and_scale_source(
        output_len, x,
        source_spec.delay_samples,
        source_spec.duration_samples,
        source_spec.volume,
    )
    return x_to_mix, source_spec.sourceloc


# -----------------------------------------------------------------------------
# Load API
# -----------------------------------------------------------------------------

def load_scenario(path: Path) -> Scenario:
    """
    Load a scenario JSON file and validate into a Scenario model.
    Raises ValidationError on invalid input.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except ValueError as e:
        raise ValueError("Error parsing scenario file") from e
    scen = Scenario.model_validate(data)
    return scen

# -----------------------------------------------------------------------------
# Commandline funcs
# -----------------------------------------------------------------------------

def dumpscenario(args):
    cfg = config.get()
    log.info(f"Dumping values for scenario {args.id}")
    scen = load_scenario(cfg.paths.scenario_dir / f"{args.id}.scenario.json")

    if args.room:
        log.info(f"Room id {scen.room_id}")
        with open(cfg.paths.room_dir / f"{scen.room_id}.room.json", "r") as f:
            print(f.read())

    if args.wav:
        log.info("Distant wav file path")
        print(cfg.paths.distant_dir / f"{args.id}.wav")
