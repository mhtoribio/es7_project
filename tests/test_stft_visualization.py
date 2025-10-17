import numpy as np

import unittest

from .common import TEST_OUTPUT_DIR
from seadge.utils import stft
from seadge.utils import visualization
from seadge import config

class StftVisualizationTestSuite(unittest.TestCase):
    """Test STFT and visualization"""

    def test_simple_stft(self):
        # Load default config
        config.load_default()

        # Create random signal, transform and transform back
        x = np.random.rand(16384)
        S = stft.stft(x)
        x_hat = stft.istft(S)
        mse = np.mean((x_hat[:len(x)] - x)**2)

        assert mse < 1e-20

    '''STFT Spectrogram'''
    def test_stft_spectogram(self):
        # load default config
        config.load_default()
        cfg = config.get()
        t = np.arange(0, 16384, 1)
        x = np.sin(2000*2*np.pi*t/cfg.dsp.samplerate)
        S = stft.stft(x)

        visualization.spectrogram(S, TEST_OUTPUT_DIR/"spectrogram_test.png", title="Test Spectrogram")

        assert True

if __name__ == '__main__':
    unittest.main()
