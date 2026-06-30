import unittest

import numpy as np

from rosehfl.utils.compression import compress_weight_update, decompress_layers


class CompressionTests(unittest.TestCase):
    def test_error_feedback_recovers_residual_signal(self):
        reference = [np.zeros((4,), dtype=np.float32)]
        target = [np.array([4.0, 0.0, -3.0, 0.0], dtype=np.float32)]

        first = compress_weight_update(
            reference_weights=reference,
            target_weights=target,
            keep_ratio=0.25,
        )
        second = compress_weight_update(
            reference_weights=reference,
            target_weights=reference,
            keep_ratio=0.25,
            residuals=first.residuals,
        )

        recovered = first.reconstructed_delta[0] + second.reconstructed_delta[0]
        np.testing.assert_allclose(recovered, target[0], atol=5e-3)
        self.assertLess(
            float(np.linalg.norm(second.residuals[0])),
            float(np.linalg.norm(first.residuals[0])),
        )

    def test_dense_head_is_exempt_and_payload_is_accounted(self):
        reference = [
            np.zeros((4,), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
        ]
        target = [
            np.array([4.0, 0.0, -2.0, 0.0], dtype=np.float32),
            np.array([0.25, -0.25], dtype=np.float32),
        ]

        result = compress_weight_update(
            reference_weights=reference,
            target_weights=target,
            keep_ratio=0.25,
            dense_layer_indices={1},
        )

        sparse_layer = result.encoded_layers[0]
        dense_layer = result.encoded_layers[1]
        self.assertEqual(sparse_layer.kind, "sparse")
        self.assertEqual(dense_layer.kind, "dense")
        np.testing.assert_allclose(result.reconstructed_weights[1], target[1], atol=1e-6)

        expected_sparse_bytes = (
            sparse_layer.indices.nbytes
            + sparse_layer.values.nbytes
            + np.asarray(sparse_layer.shape, dtype=np.int32).nbytes
        )
        expected_total = expected_sparse_bytes + target[1].nbytes
        self.assertEqual(result.payload_bytes, expected_total)

    def test_sparse_round_trip_matches_reconstructed_weights(self):
        reference = [
            np.zeros((6,), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
        ]
        target = [
            np.array([3.0, 0.0, -2.0, 0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.5, -0.5], dtype=np.float32),
        ]

        result = compress_weight_update(
            reference_weights=reference,
            target_weights=target,
            keep_ratio=0.34,
            dense_layer_indices={1},
        )
        restored = decompress_layers(result.encoded_layers, reference)

        np.testing.assert_allclose(restored[0], result.reconstructed_weights[0], atol=1e-6)
        np.testing.assert_allclose(restored[1], target[1], atol=1e-6)
        self.assertLess(result.payload_bytes, result.dense_payload_bytes)


if __name__ == "__main__":
    unittest.main()
