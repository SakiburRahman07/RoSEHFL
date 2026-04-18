import unittest

from shapefl.utils.drift import PageHinkleyBank


class DriftTests(unittest.TestCase):
    def test_page_hinkley_triggers_after_shift(self):
        detector = PageHinkleyBank(delta=0.01, threshold=0.2, initial_edges=[0])
        for value in [0.01, 0.02, 0.03, 0.02, 0.01]:
            _, triggered = detector.update(0, value)
            self.assertFalse(triggered)

        statistic, triggered = detector.update(0, 0.8)
        self.assertTrue(triggered)
        self.assertGreater(statistic, 0.2)


if __name__ == "__main__":
    unittest.main()

