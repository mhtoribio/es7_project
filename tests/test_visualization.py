# import numpy as np
# import unittest
# from seadge.utils import visualization
# from seadge import config
# from .common import TEST_OUTPUT_DIR

# class VisualizationTestSuite(unittest.TestCase):
#     """Make sure the test framework runs as it is intended."""

#     def test_visualization(self):
#         # load default config
#         config.load_default()
        
#         spec = np.random.rand(256, 512)

#         visualization.spectrogram(spec, TEST_OUTPUT_DIR/"spectrogram_test.png")

#         assert True


# if __name__ == '__main__':
#     unittest.main()