import os
import unittest
import tempfile
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

MOD = "seadge.utils.distant_sim"
from seadge.utils.distant_sim import sim_distant_src, convolve_time_varying_exact_via_hn


class TestTimeVaryingConvolutionEquivalence(unittest.TestCase):
    def setUp(self):
        # Deterministic RNG
        self.rs = np.random.RandomState(0)

        # Fake config-like bits
        self.fs = 16_000
        self.M  = 3             # number of mics
        self.R  = 256           # RIR length
        self.N  = 24_000        # signal length (samples)
        self.xfade_ms = 30.0    # crossfade duration (ms)
        self.xfade_len = int(round(self.xfade_ms * 1e-3 * self.fs))

        # One source with two poses: at 0 and at 8k samples
        self.boundary = 8_000

        # Build two simple but distinct test RIRs (R, M)
        t = np.arange(self.R)
        H0 = np.stack([(0.9 ** t),
                       (0.88 ** t),
                       (0.86 ** t)], axis=1).astype(np.float64)
        H1 = np.stack([(0.85 ** t),
                       (0.83 ** t),
                       (0.81 ** t)], axis=1).astype(np.float64)
        H0[0, :] += np.array([0.5, 0.3, 0.2])
        H1[0, :] += np.array([0.2, 0.4, 0.6])
        self.H_map = {
            "key_0": self._clip_len(H0, self.R),
            f"key_{self.boundary}": self._clip_len(H1, self.R),
        }

        # Minimal room cfg: only mic_pos length is used
        self.room_cfg = SimpleNamespace(mic_pos=[(0.0, 0.0, 0.0)] * self.M)

        # Minimal SourceSpec-like: just needs location_history with start_sample
        Loc = lambda start: SimpleNamespace(
            start_sample=start,
            pattern="cardioid", azimuth_deg=0.0, colatitude_deg=90.0,
            location_m=(1.0, 1.0, 1.0)
        )
        self.src = SimpleNamespace(location_history=[Loc(0), Loc(self.boundary)])

        # Clean signal
        n = np.arange(self.N)
        self.clean = (0.5 * self.rs.randn(self.N) + 0.1 * np.sin(2*np.pi*440*n/self.fs)).astype(np.float64)

        # temp cache root
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_root = Path(self.tmp.name)

        # Neutralize SEADGE_* env
        self._saved_env = dict(os.environ)
        for k in list(os.environ):
            if k.startswith("SEADGE_"):
                del os.environ[k]
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        os.environ.clear()
        os.environ.update(self._saved_env)

    def _clip_len(self, H, R):
        return H[:R, :]

    def _enter_patches(self, stack: ExitStack):
        # Patch cache key & loader
        def fake_make_key(room_cfg, fs, loc):
            return f"key_{int(loc.start_sample)}"
        stack.enter_context(patch(f"{MOD}.make_rir_cache_key", side_effect=fake_make_key))

        def fake_load(key, cache_root_str):
            return self.H_map[key]
        stack.enter_context(patch(f"{MOD}.load_rir_mem_or_die", side_effect=fake_load))

    # ---------- FFT path tests ----------

    def test_equivalence_fftconvolve(self):
        """sim_distant_src(FFT) ≈ exact instantaneous renderer."""
        with ExitStack() as stack:
            self._enter_patches(stack)
            y_fast = sim_distant_src(
                self.clean, self.src,
                fs=self.fs, room_cfg=self.room_cfg, cache_root=self.cache_root,
                xfade_ms=self.xfade_ms, method="fft", normalize=None,
            )
            y_exact = convolve_time_varying_exact_via_hn(
                self.clean, self.src,
                fs=self.fs, room_cfg=self.room_cfg, cache_root=self.cache_root,
                xfade_ms=self.xfade_ms, normalize=None,
            )
        self.assertEqual(y_fast.shape, y_exact.shape)
        mse = float(np.mean((y_fast - y_exact) ** 2))
        self.assertLess(mse, 1e-12, msg=f"MSE too large: {mse}")

    def test_equivalence_fft_longer_xfade(self):
        """FFT path with longer crossfade."""
        long_ms = 120.0
        with ExitStack() as stack:
            self._enter_patches(stack)
            y_fast = sim_distant_src(
                self.clean, self.src,
                fs=self.fs, room_cfg=self.room_cfg, cache_root=self.cache_root,
                xfade_ms=long_ms, method="fft", normalize=None,
            )
            y_exact = convolve_time_varying_exact_via_hn(
                self.clean, self.src,
                fs=self.fs, room_cfg=self.room_cfg, cache_root=self.cache_root,
                xfade_ms=long_ms, normalize=None,
            )
        self.assertEqual(y_fast.shape, y_exact.shape)
        mse = float(np.mean((y_fast - y_exact) ** 2))
        self.assertLess(mse, 1e-12, msg=f"MSE too large (long xfade): {mse}")

    # ---------- OLA (oaconvolve) path tests ----------

    def test_equivalence_oaconvolve(self):
        """oaconvolve path should match the exact renderer closely."""
        with ExitStack() as stack:
            self._enter_patches(stack)
            y_fast = sim_distant_src(
                self.clean, self.src,
                fs=self.fs, room_cfg=self.room_cfg, cache_root=self.cache_root,
                xfade_ms=self.xfade_ms, method="oaconv", normalize=None,
            )
            y_exact = convolve_time_varying_exact_via_hn(
                self.clean, self.src,
                fs=self.fs, room_cfg=self.room_cfg, cache_root=self.cache_root,
                xfade_ms=self.xfade_ms, normalize=None,
            )
        self.assertEqual(y_fast.shape, y_exact.shape)
        mse = float(np.mean((y_fast - y_exact) ** 2))
        # OLA can differ by tiny numeric noise vs exact TD; use slightly looser tol
        self.assertLess(mse, 1e-10, msg=f"MSE too large (oaconv): {mse}")

    def test_equivalence_oaconvolve_longer_xfade(self):
        """oaconvolve path with longer crossfade."""
        long_ms = 120.0
        with ExitStack() as stack:
            self._enter_patches(stack)
            y_fast = sim_distant_src(
                self.clean, self.src,
                fs=self.fs, room_cfg=self.room_cfg, cache_root=self.cache_root,
                xfade_ms=long_ms, method="oaconv", normalize=None,
            )
            y_exact = convolve_time_varying_exact_via_hn(
                self.clean, self.src,
                fs=self.fs, room_cfg=self.room_cfg, cache_root=self.cache_root,
                xfade_ms=long_ms, normalize=None,
            )
        self.assertEqual(y_fast.shape, y_exact.shape)
        mse = float(np.mean((y_fast - y_exact) ** 2))
        self.assertLess(mse, 1e-10, msg=f"MSE too large (oaconv long xfade): {mse}")


if __name__ == "__main__":
    unittest.main()
