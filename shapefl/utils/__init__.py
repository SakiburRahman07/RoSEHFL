from .similarity import (
    compute_cosine_similarity,
    compute_data_distribution_diversity,
    compute_similarity_matrix,
    compute_similarity_from_updates,
)
from .shapley import (
    compute_hybrid_phi,
    build_probe_set,
    compute_smc_shapley,
    compute_exact_shapley,
    serialize_probe_logits,
    deserialize_probe_logits,
)
from .drift import PageHinkleyBank, weights_l2_distance
from .robust_agg import aggregate_with_rule
from .fairness import summarise_fairness, per_client_accuracy_from_weights
from .compression import (
    CompressionResult,
    CompressedLayer,
    compress_weight_update,
    decompress_layers,
    dense_payload_num_bytes,
    scaled_cost_from_payload,
    zero_residuals_like,
)

from .seed import set_seed

from .json_utils import NumpyEncoder, save_json

from .network_topology import generate_topology, TopologyInfo

from .visualization import (
    visualize_simulation,
    visualize_comparison,
)
