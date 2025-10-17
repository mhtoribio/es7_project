import numpy as np

import unittest

from seadge.utils import stft
from seadge import config

class StftTestSuite(unittest.TestCase):
    """Test STFT"""

    def test_simple_stft(self):
        # Load default config
        config.load_default()

        # Create random signal, transform and transform back
        x = np.random.rand(16384)
        S = stft.stft(x)
        x_hat = stft.istft(S)
        mse = np.mean((x_hat[:len(x)] - x)**2)

        assert mse < 1e-20

if __name__ == '__main__':
    unittest.main()
