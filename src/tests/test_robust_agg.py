from __future__ import annotations

import numpy as np

from ._module_loader import load_utils_module


trust_edge_aggregate = load_utils_module("robust_agg").trust_edge_aggregate


def _weights(value: float):
    return [np.array([value], dtype=np.float32)]


def test_shrinkage_trust_caps_alpha_and_downweights_outlier() -> None:
    node_ids = [0, 1, 2, 3, 4]
    weights_list = [_weights(0.0), _weights(0.1), _weights(-0.1), _weights(0.05), _weights(8.0)]
    sizes = [10, 10, 10, 10, 10]
    phi = {node_id: 1.0 for node_id in node_ids}

    _, shrink_info = trust_edge_aggregate(
        node_ids=node_ids,
        weights_list=weights_list,
        sizes=sizes,
        phi=phi,
        use_shrinkage=True,
        alpha_cap_multiplier=2.0,
        prior_a=2.0,
        nu=1.0,
        dev_clip_q=0.9,
    )
    _, legacy_info = trust_edge_aggregate(
        node_ids=node_ids,
        weights_list=weights_list,
        sizes=sizes,
        phi=phi,
        use_shrinkage=False,
        alpha_cap_multiplier=10.0,
        beta=2.0,
        zeta=2.0,
    )

    alpha_cap = 2.0 / len(node_ids)
    assert float(shrink_info["alpha"].max()) <= alpha_cap + 1e-6
    assert float(shrink_info["alpha"][-1]) < float(np.median(shrink_info["alpha"][:-1]))
    assert float(np.std(shrink_info["alpha"])) < float(np.std(legacy_info["alpha"]))
    assert np.isfinite(shrink_info["trust_scores"]).all()
