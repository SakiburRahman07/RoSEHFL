import unittest

import numpy as np

from shapefl.utils.shapley import add_gaussian_noise, deserialize_probe_logits, serialize_probe_logits


class ProbeSerializationTests(unittest.TestCase):
    def test_round_trip_serialization(self):
        logits = np.arange(30, dtype=np.float32).reshape(3, 10)
        payload = serialize_probe_logits(logits)
        restored = deserialize_probe_logits(payload)
        self.assertIsNotNone(restored)
        np.testing.assert_allclose(restored, logits)

    def test_gaussian_noise_is_deterministic_for_seed(self):
        logits = np.zeros((2, 3), dtype=np.float32)
        noisy_a = add_gaussian_noise(logits, epsilon=4.0, seed=42)
        noisy_b = add_gaussian_noise(logits, epsilon=4.0, seed=42)
        np.testing.assert_allclose(noisy_a, noisy_b)


if __name__ == "__main__":
    unittest.main()

