import unittest


class BasicTestSuite(unittest.TestCase):
    """Make sure the test framework runs as it is intended."""

    def test_absolute_truth_and_meaning(self):
        assert True


if __name__ == '__main__':
    unittest.main()