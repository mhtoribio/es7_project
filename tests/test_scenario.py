import os
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

MOD = "seadge.utils.scenario"
from seadge.utils.scenario import Scenario  # Pydantic model under test


class TestScenarioValidation(unittest.TestCase):
    def setUp(self):
        # temp clean_dir with a "speech" subfolder and a dummy WAV file
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clean_dir = Path(self.tmp.name) / "clean"
        (self.clean_dir / "speech").mkdir(parents=True, exist_ok=True)

        self.rel_wav = Path("speech/book_09703_chp_0036_reader_08986_26_seg_1.wav")
        (self.clean_dir / self.rel_wav).touch()

        # Stub config: target samplerate 16k, clean_dir = temp
        self.cfg = SimpleNamespace(
            dsp=SimpleNamespace(datagen_samplerate=16_000),
            # num_seats is no longer validated here; keep a minimal room stub if needed elsewhere
            room=SimpleNamespace(num_seats=8),
            paths=SimpleNamespace(clean_dir=self.clean_dir),
        )

        # Default WAV header values:
        # fs=48k so fs*1//3 == 16k matches cfg.dsp.samplerate
        self.default_fs = 48_000
        # Enough frames so frames*1//3 >= 160_000 (needed samples)
        # Need >= 480_000 frames at 48k to cover 160k after //3.
        self.default_frames = 600_000

        # make SEADGE_* env vars inert for tests
        self._saved_env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith("SEADGE_"):
                del os.environ[k]
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        os.environ.clear()
        os.environ.update(self._saved_env)

    def _base_scenario(self) -> dict:
        """Valid baseline scenario dict for Scenario.model_validate()."""
        return {
            "scenario_type": "speaker_and_noise_and_interference",
            "room_id": "0123456789abcdef0123456789abcdef01234567",  # 40 hex (SHA-1)
            "duration_samples": 160_000,  # < 16k * 20 = 320_000
            "target_speaker": {
                "wav_path": str(self.rel_wav),
                "volume": 1.0,
                "sourceloc": 5,
                "delay_samples": 0,
                "duration_samples": 160_000,
                "decimation": 3,
                "interpolation": 1,
            },
            "other_sources": [
                {
                    "wav_path": str(self.rel_wav),
                    "volume": 0.8,
                    "sourceloc": 1,
                    "delay_samples": 16_000,
                    "duration_samples": 48_000,
                    "decimation": 3,
                    "interpolation": 1,
                },
                {
                    "wav_path": str(self.rel_wav),
                    "volume": 0.3,
                    "sourceloc": 7,
                    "delay_samples": 0,
                    "duration_samples": 160_000,
                    "decimation": 3,
                    "interpolation": 1,
                },
            ],
        }

    def _enter_patches(self, stack: ExitStack, *, frames=None, fs=None):
        stack.enter_context(patch(f"{MOD}.config.get", return_value=self.cfg))
        stack.enter_context(
            patch(f"{MOD}.wavfile_frames", return_value=self.default_frames if frames is None else frames)
        )
        stack.enter_context(
            patch(f"{MOD}.wavfile_samplerate", return_value=self.default_fs if fs is None else fs)
        )

    # -------- tests ----------

    def test_valid_scenario_ok(self):
        scen = self._base_scenario()
        with ExitStack() as stack:
            self._enter_patches(stack)
            model = Scenario.model_validate(scen)
            self.assertIsInstance(model, Scenario)

    def test_missing_target_key_raises(self):
        scen = self._base_scenario()
        del scen["target_speaker"]["wav_path"]
        with ExitStack() as stack:
            self._enter_patches(stack)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_negative_delay_raises(self):
        scen = self._base_scenario()
        scen["target_speaker"]["delay_samples"] = -1
        with ExitStack() as stack:
            self._enter_patches(stack)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_volume_out_of_range_raises(self):
        scen = self._base_scenario()
        scen["target_speaker"]["volume"] = 1.5
        with ExitStack() as stack:
            self._enter_patches(stack)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_negative_sourceloc_raises(self):
        scen = self._base_scenario()
        scen["target_speaker"]["sourceloc"] = -1
        with ExitStack() as stack:
            self._enter_patches(stack)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_missing_file_raises(self):
        scen = self._base_scenario()
        scen["target_speaker"]["wav_path"] = "speech/does_not_exist.wav"
        with ExitStack() as stack:
            self._enter_patches(stack)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_samplerate_mismatch_raises(self):
        scen = self._base_scenario()
        # Make resampled_fs != expected 16k (e.g., fs=44.1k -> 44_100//3 = 14_700)
        with ExitStack() as stack:
            self._enter_patches(stack, fs=44_100)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_too_short_after_resampling_raises(self):
        scen = self._base_scenario()
        # After //3 must be < needed (160_000), so give < 480_000 frames
        too_small_frames = 300_000  # -> 100_000 after //3
        with ExitStack() as stack:
            self._enter_patches(stack, frames=too_small_frames)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_other_sources_not_list_raises(self):
        scen = self._base_scenario()
        scen["other_sources"] = {"oops": "not a list"}
        with ExitStack() as stack:
            self._enter_patches(stack)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_room_id_wrong_length_raises(self):
        scen = self._base_scenario()
        scen["room_id"] = "deadbeef"  # not 40 hex chars
        with ExitStack() as stack:
            self._enter_patches(stack)
            with self.assertRaises(Exception):
                Scenario.model_validate(scen)

    def test_other_sources_none_ok(self):
        scen = self._base_scenario()
        scen["other_sources"] = None
        with ExitStack() as stack:
            self._enter_patches(stack)
            model = Scenario.model_validate(scen)
            self.assertIsInstance(model, Scenario)


if __name__ == "__main__":
    unittest.main()
