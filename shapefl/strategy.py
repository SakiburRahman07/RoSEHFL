"""
ShapeFL Flower Strategy — Complete Paper Architecture
=====================================================
Custom Flower Strategy implementing the full ShapeFL three-tier HFL
pipeline (Algorithm 3).

Architecture
    Cloud  ⟵  Strategy state + aggregate_fit
    Edge   ⟵  Internal grouping + per-edge FedAvg inside aggregate_fit
    Node   ⟵  Flower Clients (ShapeFlClient)

Phase machine (Flower server rounds)
    Round 1:              Pre-training (κ_p epochs) + LoS/GoA planning
    Rounds 2..1+κ·κ_c:   Training (edge epochs + cloud aggregation)

Planning modes (paper Section V-A):
    "shapefl"    – full LoS + GoA (γ=gamma)
    "cost_first" – exact PCCM baseline
    "data_first" – label-aware uniformity baseline
    "share"      – preliminary KL-to-uniform baseline
    "random"     – random edge & node assignment

Also provides FedAvgFlatStrategy / FedProxFlatStrategy as no-hierarchy baselines.
"""

import math
import os
import pickle
from collections import defaultdict
import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import flwr as fl
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from .models.factory import get_model
from .data.data_loader import DATASET_INFO
from .algorithms.los import run_los
from .algorithms.los_rose import run_los_rose
from .algorithms.label_planning import run_label_planning
from .algorithms.cost_first_exact import run_cost_first_exact
from .utils.similarity import compute_similarity_matrix
from .utils.shapley import (
    accuracy_from_logits,
    compute_hybrid_phi,
    compute_smc_shapley,
    deserialize_probe_logits,
    extract_targets,
    mean_softmax_distribution,
    normalise_shapley,
    predict_probe_logits,
    probe_payload_num_bytes,
)
from .utils.robust_agg import aggregate_with_rule
from .utils.compression import (
    compress_weight_update,
    dense_payload_num_bytes,
    scaled_cost_from_payload,
    zero_residuals_like,
)
from .utils.drift import PageHinkleyBank, weights_l2_distance
from .utils.json_utils import save_json
from .utils.model_state import batch_norm_state_keys, head_state_keys, state_key_indices
from .utils.network_topology import generate_topology


# ═══════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def generate_communication_costs(
    num_nodes: int, model_size_bytes: int, topology: str = "geant2010",
):
    """
    Generate communication costs using real/simulated network topology.

    Returns:
        (c_ne, c_ec) — node-edge and edge-cloud cost dicts.
    """
    topo = generate_topology(
        topology=topology,
        num_clients=num_nodes,
        num_edges=num_nodes,          # all nodes are candidate edges
        model_size_bytes=model_size_bytes,
        seed=123,
    )
    # Remap: the topology returns costs keyed by (client, edge_idx).
    # For ShapeFL, every node is a candidate edge, so edge_idx == node_idx.
    c_ne = topo.node_edge_costs
    c_ec = topo.edge_cloud_costs
    return c_ne, c_ec


def _weighted_average(
    weights_list: List[List[np.ndarray]], sizes: List[int],
) -> List[np.ndarray]:
    """Weighted FedAvg across a list of model weight arrays."""
    total = sum(sizes)
    num_layers = len(weights_list[0])
    avg = [np.zeros_like(weights_list[0][i]) for i in range(num_layers)]
    for w, s in zip(weights_list, sizes):
        for i in range(num_layers):
            avg[i] += w[i] * (s / total)
    return avg


# ═══════════════════════════════════════════════════════════════════════════
#  ShapeFlStrategy — full paper implementation
# ═══════════════════════════════════════════════════════════════════════════

class ShapeFlStrategy(fl.server.strategy.Strategy):
    """Complete ShapeFL (Algorithm 3) as a Flower Strategy."""

    def __init__(
        self,
        model_name: str = "lenet5",
        dataset_name: str = "fmnist",
        num_nodes: int = 30,
        kappa_p: int = 30,
        kappa_e: int = 1,
        kappa_c: int = 10,
        kappa: int = 50,
        gamma: float = 2800.0,
        B_e: Optional[int] = None,
        T_max: int = 30,
        lr: float = 0.001,
        momentum: float = 0.0,
        initial_parameters: Optional[Parameters] = None,
        planning_mode: str = "shapefl",
        evaluate_fn: Optional[Callable] = None,
        topology: str = "geant2010",
        node_label_counts: Optional[Dict[int, np.ndarray]] = None,
        total_local_epochs: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_nodes = num_nodes
        self.kappa_p = kappa_p
        self.kappa_e = kappa_e
        self.kappa_c = kappa_c
        self.kappa = kappa
        self.gamma = gamma
        self.B_e = B_e or max(3, math.ceil(num_nodes / 3))
        self.T_max = T_max
        self.topology = topology
        self.lr = lr
        self.momentum = momentum
        self.planning_mode = planning_mode
        self.evaluate_fn = evaluate_fn
        self.node_label_counts = node_label_counts
        self.total_local_epochs = total_local_epochs

        self.initial_parameters = initial_parameters
        self.global_parameters = initial_parameters
        self.edge_parameters: Dict[int, Parameters] = {}

        self.phase = "pretrain"
        self.cloud_round = 0
        self.edge_epoch = 0
        self.completed_local_epochs = 0
        self._current_round_local_epochs = self.kappa_e

        # CID mapping: Flower simulation uses large arbitrary CIDs,
        # not 0..N-1.  Built on first configure_fit call.
        self._cid_to_partition: Dict[int, int] = {}

        self.selected_edges: Set[int] = set()
        self.edge_nodes: Dict[int, List[int]] = {}
        self.node_edge: Dict[int, int] = {}
        self.edge_data_sizes: Dict[int, int] = {}

        self.c_ne: Dict = {}
        self.c_ec: Dict = {}
        self.per_round_cost_gb = 0.0
        self.cumulative_cost_gb = 0.0

        self.metrics_history = {
            "cloud_round": [],
            "accuracy": [],
            "loss": [],
            "per_round_cost_gb": [],
            "cumulative_cost_gb": [],
        }

    @property
    def total_flower_rounds(self) -> int:
        if self.total_local_epochs is not None:
            return 1 + math.ceil(self.total_local_epochs / self.kappa_e)
        return 1 + self.kappa * self.kappa_c

    # ════════════════════════════════════════════════════════════════════
    #  Strategy interface
    # ════════════════════════════════════════════════════════════════════

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return self.initial_parameters

    def _resolve_node_id(self, cid: str) -> int:
        """Map Flower's arbitrary CID to partition index 0..N-1."""
        raw = int(cid)
        if raw in self._cid_to_partition:
            return self._cid_to_partition[raw]
        return raw % self.num_nodes  # fallback

    def _build_cid_map(self, clients):
        """Build CID→partition mapping on first call (once)."""
        if self._cid_to_partition:
            return
        sorted_cids = sorted(int(c.cid) for c in clients)
        self._cid_to_partition = {cid: idx for idx, cid in enumerate(sorted_cids)}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(
            num_clients=self.num_nodes, min_num_clients=self.num_nodes,
        )
        self._build_cid_map(clients)

        if self.phase == "pretrain":
            pretrain_epochs = self.kappa_p if self.planning_mode == "shapefl" else 0
            config = {"phase": "pretrain", "epochs": pretrain_epochs, "lr": self.lr, "momentum": self.momentum}
            return [(c, FitIns(self.initial_parameters, config)) for c in clients]

        epochs_this_round = self.kappa_e
        if self.total_local_epochs is not None:
            remaining = self.total_local_epochs - self.completed_local_epochs
            epochs_this_round = min(self.kappa_e, max(remaining, 0))
        self._current_round_local_epochs = epochs_this_round
        config = {"phase": "train", "epochs": epochs_this_round, "lr": self.lr, "momentum": self.momentum}
        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
        for client in clients:
            node_id = self._resolve_node_id(client.cid)
            edge_id = self.node_edge.get(node_id)
            if self.edge_epoch == 0:
                params = self.global_parameters
            else:
                params = self.edge_parameters.get(edge_id, self.global_parameters)
            fit_ins_list.append((client, FitIns(params, config)))
        return fit_ins_list

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.phase == "pretrain":
            return self._aggregate_pretrain(results)
        return self._aggregate_train(results)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Skip distributed eval when using centralized evaluate_fn
        if self.evaluate_fn is not None:
            return []
        if self.phase == "pretrain":
            return []
        if self.edge_epoch != 0:
            return []
        if self.cloud_round == 0:
            return []
        clients = client_manager.sample(
            num_clients=self.num_nodes, min_num_clients=self.num_nodes,
        )
        return [(c, EvaluateIns(self.global_parameters, {})) for c in clients]

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        total_examples = sum(r.num_examples for _, r in results)
        weighted_loss = sum(r.loss * r.num_examples for _, r in results) / total_examples
        weighted_acc = (
            sum(r.metrics.get("accuracy", 0.0) * r.num_examples for _, r in results)
            / total_examples
        )
        self.metrics_history["cloud_round"].append(self.cloud_round)
        self.metrics_history["accuracy"].append(weighted_acc)
        self.metrics_history["loss"].append(weighted_loss)
        self.metrics_history["per_round_cost_gb"].append(self.per_round_cost_gb)
        self.metrics_history["cumulative_cost_gb"].append(self.cumulative_cost_gb)
        print(
            f"  Cloud Round {self.cloud_round}/{self.kappa} | "
            f"Acc: {weighted_acc:.4f} | Loss: {weighted_loss:.4f} | "
            f"CumCost: {self.cumulative_cost_gb:.4f} GB"
        )
        return weighted_loss, {"accuracy": weighted_acc}

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        return self.num_nodes, self.num_nodes

    def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
        return self.num_nodes, self.num_nodes

    def evaluate(
        self, server_round: int, parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized server-side evaluation (more reliable than aggregated client eval).

        Only runs after cloud aggregation rounds (edge_epoch == 0).
        Falls back to *None* if no evaluate_fn was provided, in which case
        Flower uses distributed evaluation via configure_evaluate / aggregate_evaluate.
        """
        if self.evaluate_fn is None:
            return None
        if self.phase == "pretrain" or self.edge_epoch != 0:
            return None
        params = parameters_to_ndarrays(parameters)
        loss, metrics = self.evaluate_fn(server_round, params, {})
        accuracy = metrics.get("accuracy", 0.0)
        # Feed into metrics_history so results stay consistent
        self.metrics_history["cloud_round"].append(self.cloud_round)
        self.metrics_history["accuracy"].append(accuracy)
        self.metrics_history["loss"].append(loss)
        self.metrics_history["per_round_cost_gb"].append(self.per_round_cost_gb)
        self.metrics_history["cumulative_cost_gb"].append(self.cumulative_cost_gb)
        print(
            f"  Cloud Round {self.cloud_round}/{self.kappa} | "
            f"Acc: {accuracy:.4f} | Loss: {loss:.4f} | "
            f"CumCost: {self.cumulative_cost_gb:.4f} GB"
        )
        return loss, metrics

    # ════════════════════════════════════════════════════════════════════
    #  Private — phase handlers
    # ════════════════════════════════════════════════════════════════════

    def _aggregate_pretrain(self, results):
        initial_ndarrays = parameters_to_ndarrays(self.initial_parameters)
        model_size_bytes = sum(w.nbytes for w in initial_ndarrays)
        self.c_ne, self.c_ec = generate_communication_costs(
            self.num_nodes, model_size_bytes, topology=self.topology,
        )

        if self.planning_mode == "random":
            data_sizes = {
                self._resolve_node_id(client_proxy.cid): fit_res.num_examples
                for client_proxy, fit_res in results
            }
            self._plan_random(data_sizes)
        elif self.planning_mode == "cost_first":
            data_sizes = self._label_based_data_sizes()
            self._plan_cost_first(data_sizes)
        elif self.planning_mode in {"data_first", "share"}:
            data_sizes = self._label_based_data_sizes()
            self._plan_label_based(data_sizes, objective_mode=self.planning_mode)
        else:
            ds_info = DATASET_INFO[self.dataset_name]
            model = get_model(self.model_name, ds_info["num_classes"], ds_info["input_channels"], "cpu")
            keys = list(model.state_dict().keys())
            linear_name = model.linear_layer_name
            linear_indices = [i for i, k in enumerate(keys) if k.startswith(linear_name)]

            linear_updates = {}
            data_sizes = {}
            for client_proxy, fit_res in results:
                node_id = self._resolve_node_id(client_proxy.cid)
                trained = parameters_to_ndarrays(fit_res.parameters)
                deltas = [trained[i] - initial_ndarrays[i] for i in linear_indices]
                update = torch.tensor(np.concatenate([d.flatten() for d in deltas]))
                linear_updates[node_id] = update
                data_sizes[node_id] = fit_res.num_examples
                print(f"  Node {node_id} pre-trained (norm {update.norm():.4f})")

            node_ids_present = sorted(linear_updates.keys())
            S_partial = compute_similarity_matrix(linear_updates)
            S = np.zeros((self.num_nodes, self.num_nodes))
            for _i, _ni in enumerate(node_ids_present):
                for _j, _nj in enumerate(node_ids_present):
                    S[_ni, _nj] = S_partial[_i, _j]
            print(
                f"  Similarity: shape {S_partial.shape} "
                f"({len(node_ids_present)} nodes present), mean {S_partial.mean():.4f}"
            )
            self._plan_with_los(S, data_sizes)

        self._compute_per_round_cost()

        print(f"\n  Planning mode: {self.planning_mode}")
        print(f"  Selected edges: {sorted(self.selected_edges)}")
        for e in sorted(self.selected_edges):
            print(f"    Edge {e}: {sorted(self.edge_nodes[e])}")
        print(f"  Per-round cost: {self.per_round_cost_gb:.6f} GB")

        self.phase = "train"
        self.cloud_round = 0
        self.edge_epoch = 0
        self.completed_local_epochs = 0
        return self.initial_parameters, {"phase": "pretrain_done"}

    def _aggregate_train(self, results):
        edge_groups: Dict[int, Tuple[List[List[np.ndarray]], List[int]]] = {
            e: ([], []) for e in self.selected_edges
        }
        for client_proxy, fit_res in results:
            node_id = self._resolve_node_id(client_proxy.cid)
            edge_id = self.node_edge.get(node_id)
            if edge_id is None:
                # Node was absent during pre-training; skip its update.
                continue
            weights = parameters_to_ndarrays(fit_res.parameters)
            edge_groups[edge_id][0].append(weights)
            edge_groups[edge_id][1].append(fit_res.num_examples)

        for edge_id, (ws, ss) in edge_groups.items():
            if not ws:
                continue
            avg = _weighted_average(ws, ss)
            self.edge_parameters[edge_id] = ndarrays_to_parameters(avg)

        self.edge_epoch += 1
        self.completed_local_epochs += self._current_round_local_epochs
        reached_local_budget = (
            self.total_local_epochs is not None
            and self.completed_local_epochs >= self.total_local_epochs
        )

        if self.edge_epoch >= self.kappa_c or reached_local_budget:
            edge_weights, edge_sizes = [], []
            for e in sorted(self.selected_edges):
                edge_weights.append(parameters_to_ndarrays(self.edge_parameters[e]))
                edge_sizes.append(self.edge_data_sizes[e])

            global_avg = _weighted_average(edge_weights, edge_sizes)
            self.global_parameters = ndarrays_to_parameters(global_avg)
            if self.edge_epoch >= self.kappa_c:
                self.cumulative_cost_gb += self.per_round_cost_gb
            self.edge_epoch = 0
            self.cloud_round += 1
            return self.global_parameters, {"cloud_round": self.cloud_round}

        return self.global_parameters, {"edge_epoch": self.edge_epoch}

    # ════════════════════════════════════════════════════════════════════
    #  Private — planning helpers
    # ════════════════════════════════════════════════════════════════════

    def _plan_with_los(self, S, data_sizes):
        # Only plan with nodes that actually sent updates (some may have
        # failed during pre-training due to transient errors).
        present_nodes = sorted(data_sizes.keys())

        los_result = run_los(
            candidate_edges=present_nodes,
            all_nodes=present_nodes,
            communication_costs_ne=self.c_ne,
            communication_costs_ec=self.c_ec,
            similarity_matrix=S,
            data_sizes=data_sizes,
            kappa_c=self.kappa_c,
            gamma=self.gamma,
            B_e=self.B_e,
            T_max=self.T_max,
        )

        if los_result is None or los_result.node_associations is None:
            raise RuntimeError(
                "LOS planning returned no result. This usually means no clients "
                "responded in the pre-train round. Check that client_fn works correctly."
            )
        self.selected_edges = {int(e) for e in los_result.selected_edges}
        self.edge_nodes = {
            int(e): [int(n) for n in ns]
            for e, ns in los_result.node_associations.edge_nodes.items()
        }
        self.edge_data_sizes = {
            int(e): v for e, v in los_result.node_associations.edge_data_sizes.items()
        }
        self.node_edge = {}
        for e, nodes in self.edge_nodes.items():
            for n in nodes:
                self.node_edge[n] = e

    def _label_based_data_sizes(self) -> Dict[int, int]:
        if self.node_label_counts is None:
            raise RuntimeError(
                f"Planning mode '{self.planning_mode}' requires simulation-side "
                "label counts. Pass node_label_counts into ShapeFlStrategy."
            )
        return {
            int(node_id): int(counts.sum())
            for node_id, counts in self.node_label_counts.items()
        }

    def _plan_cost_first(self, data_sizes: Dict[int, int]) -> None:
        present_nodes = sorted(data_sizes.keys())
        result = run_cost_first_exact(
            candidate_edges=present_nodes,
            all_nodes=present_nodes,
            communication_costs_ne=self.c_ne,
            communication_costs_ec=self.c_ec,
            data_sizes=data_sizes,
            kappa_c=self.kappa_c,
            B_e=self.B_e,
        )
        self.selected_edges = {int(e) for e in result.selected_edges}
        self.edge_nodes = {int(e): [int(n) for n in sorted(nodes)] for e, nodes in result.edge_nodes.items()}
        self.edge_data_sizes = {int(e): int(v) for e, v in result.edge_data_sizes.items()}
        self.node_edge = {int(n): int(e) for n, e in result.associations.items()}

    def _plan_label_based(self, data_sizes: Dict[int, int], objective_mode: str) -> None:
        if self.node_label_counts is None:
            raise RuntimeError(
                f"Planning mode '{objective_mode}' requires simulation-side label counts."
            )

        present_nodes = sorted(data_sizes.keys())
        label_counts = {int(n): self.node_label_counts[int(n)] for n in present_nodes}
        result = run_label_planning(
            candidate_edges=present_nodes,
            all_nodes=present_nodes,
            node_label_counts=label_counts,
            data_sizes=data_sizes,
            communication_costs_ne=self.c_ne,
            communication_costs_ec=self.c_ec,
            kappa_c=self.kappa_c,
            gamma=self.gamma,
            B_e=self.B_e,
            T_max=self.T_max,
            objective_mode=objective_mode,
        )
        self.selected_edges = {int(e) for e in result.selected_edges}
        self.edge_nodes = {
            int(e): [int(n) for n in sorted(nodes)]
            for e, nodes in result.node_associations.edge_nodes.items()
        }
        self.edge_data_sizes = {
            int(e): int(v)
            for e, v in result.node_associations.edge_data_sizes.items()
        }
        self.node_edge = {
            int(n): int(e)
            for n, e in result.node_associations.associations.items()
        }

    def _plan_random(self, data_sizes):
        # Only plan with nodes that actually sent updates.
        present_nodes = sorted(data_sizes.keys())
        N = len(present_nodes)
        num_edges = max(2, math.ceil(N / self.B_e))
        np.random.seed(999)
        edge_list = sorted(
            np.array(present_nodes)[
                np.random.choice(N, min(num_edges, N), replace=False)
            ].tolist()
        )
        self.selected_edges = set(edge_list)
        self.edge_nodes = {e: [] for e in self.selected_edges}
        self.edge_data_sizes = {e: 0 for e in self.selected_edges}

        nodes_shuffled = list(present_nodes)
        np.random.shuffle(nodes_shuffled)
        edge_cycle = list(self.selected_edges)
        idx = 0
        for n in nodes_shuffled:
            for attempt in range(len(edge_cycle)):
                e = edge_cycle[(idx + attempt) % len(edge_cycle)]
                if len(self.edge_nodes[e]) < self.B_e:
                    self.edge_nodes[e].append(n)
                    self.edge_data_sizes[e] += data_sizes[n]
                    idx = (idx + attempt + 1) % len(edge_cycle)
                    break
        self.node_edge = {}
        for e, nodes in self.edge_nodes.items():
            for n in nodes:
                self.node_edge[n] = e

    def _compute_per_round_cost(self):
        """Per-round communication cost (Eq. 11, one-direction to match paper)."""
        ne = sum(
            self.c_ne[(n, e)]
            for e in self.selected_edges
            for n in self.edge_nodes[e]
        )
        ec = sum(self.c_ec[e] for e in self.selected_edges)
        self.per_round_cost_gb = self.kappa_c * ne + ec


# ═══════════════════════════════════════════════════════════════════════════
#  FedAvgFlatStrategy — no-hierarchy baseline
# ═══════════════════════════════════════════════════════════════════════════

class FedAvgFlatStrategy(fl.server.strategy.Strategy):
    """
    Flat FedAvg baseline — all nodes communicate directly with the cloud.
    Each Flower round = one FedAvg cloud round with (κ_c × κ_e) local epochs.
    """

    def __init__(
        self,
        num_nodes: int,
        kappa: int,
        local_epochs: int,
        lr: float,
        momentum: float = 0.0,
        prox_mu: float = 0.0,
        total_local_epochs: Optional[int] = None,
        initial_parameters: Parameters = None,
        evaluate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.kappa = kappa
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.prox_mu = prox_mu
        self.total_local_epochs = total_local_epochs
        self.initial_parameters = initial_parameters
        self.global_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.completed_local_epochs = 0
        self._current_round_local_epochs = local_epochs

        self.per_round_cost_gb = 0.0
        self.cumulative_cost_gb = 0.0
        self.metrics_history = {
            "cloud_round": [],
            "accuracy": [],
            "loss": [],
            "per_round_cost_gb": [],
            "cumulative_cost_gb": [],
        }

    @property
    def total_flower_rounds(self) -> int:
        if self.total_local_epochs is not None:
            return math.ceil(self.total_local_epochs / self.local_epochs)
        return self.kappa

    def set_comm_costs(self, c_ec: Dict[int, float]):
        """Per-round cost for flat FedAvg (one-direction to match paper)."""
        self.per_round_cost_gb = sum(c_ec[n] for n in range(self.num_nodes))

    def initialize_parameters(self, client_manager):
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(
            num_clients=self.num_nodes, min_num_clients=self.num_nodes,
        )
        epochs_this_round = self.local_epochs
        if self.total_local_epochs is not None:
            remaining = self.total_local_epochs - self.completed_local_epochs
            epochs_this_round = min(self.local_epochs, max(remaining, 0))
        self._current_round_local_epochs = epochs_this_round
        config = {
            "phase": "train",
            "epochs": epochs_this_round,
            "lr": self.lr,
            "momentum": self.momentum,
        }
        if self.prox_mu > 0.0:
            config["prox_mu"] = self.prox_mu
        return [(c, FitIns(self.global_parameters, config)) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        weights_list, sizes = [], []
        for _, fit_res in results:
            weights_list.append(parameters_to_ndarrays(fit_res.parameters))
            sizes.append(fit_res.num_examples)
        avg = _weighted_average(weights_list, sizes)
        self.global_parameters = ndarrays_to_parameters(avg)
        self.cumulative_cost_gb += self.per_round_cost_gb
        self.completed_local_epochs += self._current_round_local_epochs
        return self.global_parameters, {"round": server_round}

    def evaluate(
        self, server_round: int, parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Centralized server-side evaluation (preferred when evaluate_fn is set)."""
        if self.evaluate_fn is None:
            return None
        params = parameters_to_ndarrays(parameters)
        loss, metrics = self.evaluate_fn(server_round, params, {})
        accuracy = metrics.get("accuracy", 0.0)
        self.metrics_history["cloud_round"].append(server_round)
        self.metrics_history["accuracy"].append(accuracy)
        self.metrics_history["loss"].append(loss)
        self.metrics_history["per_round_cost_gb"].append(self.per_round_cost_gb)
        self.metrics_history["cumulative_cost_gb"].append(self.cumulative_cost_gb)
        print(
            f"  Round {server_round}/{self.kappa} | "
            f"Acc: {accuracy:.4f} | Loss: {loss:.4f} | "
            f"CumCost: {self.cumulative_cost_gb:.4f} GB"
        )
        return loss, metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.evaluate_fn is not None:
            return []
        clients = client_manager.sample(
            num_clients=self.num_nodes, min_num_clients=self.num_nodes,
        )
        return [(c, EvaluateIns(self.global_parameters, {})) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        total_examples = sum(r.num_examples for _, r in results)
        weighted_loss = sum(r.loss * r.num_examples for _, r in results) / total_examples
        weighted_acc = (
            sum(r.metrics.get("accuracy", 0.0) * r.num_examples for _, r in results)
            / total_examples
        )
        self.metrics_history["cloud_round"].append(server_round)
        self.metrics_history["accuracy"].append(weighted_acc)
        self.metrics_history["loss"].append(weighted_loss)
        self.metrics_history["per_round_cost_gb"].append(self.per_round_cost_gb)
        self.metrics_history["cumulative_cost_gb"].append(self.cumulative_cost_gb)
        print(
            f"  Round {server_round}/{self.kappa} | "
            f"Acc: {weighted_acc:.4f} | Loss: {weighted_loss:.4f} | "
            f"CumCost: {self.cumulative_cost_gb:.4f} GB"
        )
        return weighted_loss, {"accuracy": weighted_acc}

    def num_fit_clients(self, num_available_clients):
        return self.num_nodes, self.num_nodes

    def num_evaluate_clients(self, num_available_clients):
        return self.num_nodes, self.num_nodes


class FedProxFlatStrategy(FedAvgFlatStrategy):
    """
    Flat FedProx baseline.

    Same topology/communication model as flat FedAvg, but clients optimize
    the FedProx local objective with proximal coefficient ``prox_mu``.
    """

    def __init__(
        self,
        num_nodes: int,
        kappa: int,
        local_epochs: int,
        lr: float,
        momentum: float = 0.0,
        prox_mu: float = 0.01,
        total_local_epochs: Optional[int] = None,
        initial_parameters: Parameters = None,
        evaluate_fn: Optional[Callable] = None,
    ):
        super().__init__(
            num_nodes=num_nodes,
            kappa=kappa,
            local_epochs=local_epochs,
            lr=lr,
            momentum=momentum,
            prox_mu=prox_mu,
            total_local_epochs=total_local_epochs,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
        )


class RoSEHFLStrategy(ShapeFlStrategy):
    """
    RoSE-HFL strategy with warm-start planning, drift-triggered replanning,
    trust-aware edge aggregation, and checkpointable run state.
    """

    def __init__(
        self,
        model_name: str = "lenet5",
        dataset_name: str = "fmnist",
        num_nodes: int = 30,
        warmup_epochs: int = 1,
        kappa_e: int = 1,
        kappa_c: int = 10,
        kappa: int = 50,
        gamma_max: float = 2800.0,
        gamma_min: float = 1400.0,
        gamma_anneal: str = "cosine",
        B_e: Optional[int] = None,
        T_max: int = 30,
        lr: float = 0.001,
        momentum: float = 0.0,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable] = None,
        topology: str = "geant2010",
        node_label_counts: Optional[Dict[int, np.ndarray]] = None,
        total_local_epochs: Optional[int] = None,
        probe_loader=None,
        model_factory: Optional[Callable[[], torch.nn.Module]] = None,
        server_device: str = "cpu",
        output_dir: Optional[str] = None,
        seed: int = 42,
        shapley_T: int = 4,
        shapley_K: int = 6,
        planning_signal: str = "shapley",
        emit_probe_logits: bool = True,
        hybrid_lambda_floor: float = 0.1,
        hybrid_lambda_ceiling: float = 0.9,
        dp_epsilon: float = 0.0,
        dp_delta: float = 1e-5,
        probe_size: int = 1000,
        agg_rule: str = "trust",
        agg_trim_ratio: float = 0.2,
        beta: float = 2.0,
        eta: float = 0.5,
        xi: float = 1.0,
        zeta: float = 2.0,
        alpha_cap_multiplier: float = 2.0,
        krum_f: int = 1,
        drift_enabled: bool = True,
        drift_delta: float = 1e-3,
        drift_lambda: float = 0.5,
        max_replans: int = 8,
        trust_use_shrinkage: bool = True,
        trust_prior_a: float = 2.0,
        trust_prior_b: Optional[float] = None,
        trust_nu: float = 1.0,
        trust_dev_clip_q: float = 0.9,
        adaptive_gamma_eta: float = 0.5,
        adaptive_gamma_target: float = 0.25,
        warm_start_replan: bool = False,
        warm_start_threshold: float = 0.0,
        replan_cost_increase_tolerance: float = 0.1,
        compression_enabled: bool = False,
        compression_keep_ratio_min: float = 0.05,
        compression_keep_ratio_max: float = 0.25,
        compression_eta: float = 1.0,
        compression_target_deficit: float = 0.25,
        compress_edge_to_cloud: bool = True,
        edge_min_members: int = 0,
        edge_underfill_penalty: float = 0.0,
        local_objective_prox_mu: float = 0.0,
        logit_adjustment_tau: float = 0.0,
        local_bn: bool = False,
        edge_swa_k: int = 1,
    ):
        super().__init__(
            model_name=model_name,
            dataset_name=dataset_name,
            num_nodes=num_nodes,
            kappa_p=0,
            kappa_e=kappa_e,
            kappa_c=kappa_c,
            kappa=kappa,
            gamma=gamma_max,
            B_e=B_e,
            T_max=T_max,
            lr=lr,
            momentum=momentum,
            initial_parameters=initial_parameters,
            planning_mode="shapefl",
            evaluate_fn=evaluate_fn,
            topology=topology,
            node_label_counts=node_label_counts,
            total_local_epochs=total_local_epochs,
        )
        self.phase = "warmup"
        self.warmup_epochs = int(warmup_epochs)
        self.gamma_max = float(gamma_max)
        self.gamma_min = float(gamma_min)
        self.gamma_anneal = gamma_anneal
        self.seed = int(seed)
        self.shapley_T = int(shapley_T)
        self.shapley_K = int(shapley_K)
        self.planning_signal = planning_signal
        self.emit_probe_logits = bool(emit_probe_logits)
        self.hybrid_lambda_floor = float(hybrid_lambda_floor)
        self.hybrid_lambda_ceiling = float(hybrid_lambda_ceiling)
        self.dp_epsilon = float(dp_epsilon)
        self.dp_delta = float(dp_delta)
        self.probe_size = int(probe_size)
        self.probe_loader = probe_loader
        self.model_factory = model_factory
        self.server_device = torch.device(server_device)
        self.output_dir = output_dir
        self.agg_rule = agg_rule
        self.agg_trim_ratio = float(agg_trim_ratio)
        self.beta = float(beta)
        self.eta = float(eta)
        self.xi = float(xi)
        self.zeta = float(zeta)
        self.alpha_cap_multiplier = float(alpha_cap_multiplier)
        self.krum_f = int(krum_f)
        self.drift_enabled = bool(drift_enabled)
        self.drift_delta = float(drift_delta)
        self.drift_lambda = float(drift_lambda)
        self.max_replans = int(max_replans)
        self.trust_use_shrinkage = bool(trust_use_shrinkage)
        self.trust_prior_a = float(trust_prior_a)
        self.trust_prior_b = None if trust_prior_b is None else float(trust_prior_b)
        self.trust_nu = float(trust_nu)
        self.trust_dev_clip_q = float(trust_dev_clip_q)
        self.adaptive_gamma_eta = float(adaptive_gamma_eta)
        self.adaptive_gamma_target = float(adaptive_gamma_target)
        self.warm_start_replan = bool(warm_start_replan)
        self.warm_start_threshold = float(max(warm_start_threshold, 0.0))
        self.replan_cost_increase_tolerance = float(max(replan_cost_increase_tolerance, 0.0))
        self.compression_enabled = bool(compression_enabled)
        self.compression_keep_ratio_min = float(np.clip(compression_keep_ratio_min, 0.0, 1.0))
        self.compression_keep_ratio_max = float(np.clip(compression_keep_ratio_max, 0.0, 1.0))
        self.compression_eta = float(max(compression_eta, 0.0))
        self.compression_target_deficit = float(max(compression_target_deficit, 0.0))
        self.compress_edge_to_cloud = bool(compress_edge_to_cloud)
        self.edge_min_members = int(max(edge_min_members, 0))
        self.edge_underfill_penalty = float(edge_underfill_penalty)
        self.local_objective_prox_mu = float(local_objective_prox_mu)
        self.logit_adjustment_tau = float(logit_adjustment_tau)
        self.local_bn = bool(local_bn)
        self.edge_swa_k = int(max(edge_swa_k, 1))

        self.completed_flower_rounds = 0
        self.replan_count = 0
        self.replan_rounds: List[int] = []
        self.plan_history: List[Dict[str, object]] = []
        self.shapley_history: List[Dict[str, object]] = []
        self.drift_history: List[Dict[str, object]] = []
        self.round_probe_payload_bytes: List[int] = []
        self.total_probe_payload_bytes = 0
        self.edge_aggregation_history: List[Dict[str, object]] = []
        self.current_phi_raw: Dict[int, float] = {}
        self.current_phi: Dict[int, float] = {}
        self.current_gamma = self.gamma_max
        self.current_gamma_used = self.gamma_max
        self.current_edge_balance_deficit = 0.0
        self.current_client_distributions: Dict[int, np.ndarray] = {}
        self.current_hybrid_info: Dict[str, object] = {}
        self._latest_client_weights: Dict[int, List[np.ndarray]] = {}
        self._latest_client_sizes: Dict[int, int] = {}
        self._latest_probe_logits: Dict[int, np.ndarray] = {}
        self._current_cycle_reference_weights = parameters_to_ndarrays(self.initial_parameters)
        self.edge_anchor_weights: Dict[int, List[np.ndarray]] = {}
        self.edge_swa_buffers: Dict[int, List[List[np.ndarray]]] = {}
        self.model_size_bytes = dense_payload_num_bytes(self._current_cycle_reference_weights)
        self.client_compression_residuals: Dict[int, List[np.ndarray]] = {}
        self.edge_compression_residuals: Dict[int, List[np.ndarray]] = {}
        self.current_cycle_model_payload_bytes = 0
        self.current_cycle_probe_payload_bytes = 0
        self.current_cycle_effective_cost_gb = 0.0
        self.effective_cumulative_cost_gb = 0.0
        self._reported_paper_per_round_cost_gb = 0.0
        self._reported_effective_per_round_cost_gb = 0.0
        self._reported_model_payload_bytes = 0
        self._reported_probe_payload_bytes = 0
        self.probe_targets = (
            extract_targets(self.probe_loader.dataset)
            if self.probe_loader is not None
            else np.zeros((0,), dtype=np.int64)
        )
        self.drift_bank = PageHinkleyBank(
            delta=self.drift_delta,
            threshold=self.drift_lambda,
        )
        metadata_model = self.model_factory() if self.model_factory is not None else get_model(
            self.model_name,
            DATASET_INFO[self.dataset_name]["num_classes"],
            DATASET_INFO[self.dataset_name]["input_channels"],
            "cpu",
        )
        self.parameter_keys = list(metadata_model.state_dict().keys())
        self.bn_parameter_indices = state_key_indices(
            self.parameter_keys,
            batch_norm_state_keys(metadata_model),
        )
        self.head_parameter_indices = state_key_indices(
            self.parameter_keys,
            head_state_keys(metadata_model),
        )
        self.dense_compression_indices = set(self.bn_parameter_indices) | set(self.head_parameter_indices)
        self.metrics_history.update({
            "paper_per_round_cost_gb": [],
            "paper_cumulative_cost_gb": [],
            "effective_per_round_cost_gb": [],
            "effective_cumulative_cost_gb": [],
            "model_payload_bytes": [],
            "probe_payload_bytes": [],
        })

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    @property
    def total_flower_rounds(self) -> int:
        if self.total_local_epochs is not None:
            remaining_training_epochs = max(self.total_local_epochs - self.warmup_epochs, 0)
            return 1 + math.ceil(remaining_training_epochs / self.kappa_e)
        return 1 + self.kappa * self.kappa_c

    @property
    def remaining_flower_rounds(self) -> int:
        return max(0, self.total_flower_rounds - self.completed_flower_rounds)

    def _weights_copy(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return [layer.copy() for layer in weights]

    def _average_weights(self, weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        if not weights_list:
            raise ValueError("_average_weights: empty input")
        num_layers = len(weights_list[0])
        aggregate = [
            np.zeros_like(weights_list[0][layer_idx], dtype=np.float64)
            for layer_idx in range(num_layers)
        ]
        for weights in weights_list:
            for layer_idx in range(num_layers):
                aggregate[layer_idx] += weights[layer_idx].astype(np.float64)
        scale = 1.0 / max(len(weights_list), 1)
        return [
            (layer * scale).astype(weights_list[0][layer_idx].dtype)
            for layer_idx, layer in enumerate(aggregate)
        ]

    def _mask_batch_norm_weights(
        self,
        weights: List[np.ndarray],
        reference_weights: List[np.ndarray],
    ) -> List[np.ndarray]:
        if not self.local_bn or not self.bn_parameter_indices:
            return weights
        masked = self._weights_copy(weights)
        for index in self.bn_parameter_indices:
            masked[index] = reference_weights[index].copy()
        return masked

    def _update_edge_swa_buffer(
        self,
        edge_id: int,
        weights: List[np.ndarray],
    ) -> None:
        buffers = self.edge_swa_buffers.setdefault(int(edge_id), [])
        buffers.append(self._weights_copy(weights))
        if len(buffers) > self.edge_swa_k:
            del buffers[0]

    def _edge_swa_average(
        self,
        edge_id: int,
    ) -> Optional[List[np.ndarray]]:
        buffers = self.edge_swa_buffers.get(int(edge_id), [])
        if not buffers:
            return None
        return self._average_weights(buffers)

    def _estimate_plan_cost_gb(self, edge_nodes: Dict[int, List[int]]) -> float:
        ne = sum(
            self.c_ne[(node_id, edge_id)]
            for edge_id, nodes in edge_nodes.items()
            for node_id in nodes
        )
        ec = sum(self.c_ec[edge_id] for edge_id in edge_nodes)
        return float(self.kappa_c * ne + ec)

    def _compute_edge_balance_deficit(
        self,
        edge_nodes: Dict[int, List[int]],
        client_distributions: Dict[int, np.ndarray],
        client_sizes: Dict[int, int],
    ) -> float:
        if not edge_nodes or not client_distributions:
            return 0.0
        first_distribution = next(iter(client_distributions.values()))
        uniform = np.full(len(first_distribution), 1.0 / max(len(first_distribution), 1), dtype=np.float64)
        deficits: List[float] = []
        for nodes in edge_nodes.values():
            if not nodes:
                continue
            class_mass = np.zeros_like(uniform)
            for node_id in nodes:
                distribution = client_distributions.get(int(node_id))
                if distribution is None:
                    continue
                class_mass += float(client_sizes.get(int(node_id), 1)) * distribution.astype(np.float64)
            total = float(class_mass.sum())
            if total <= 0.0:
                continue
            probabilities = np.clip(class_mass / total, 1e-12, None)
            probabilities /= probabilities.sum()
            deficits.append(float(np.sum(probabilities * np.log(probabilities / uniform))))
        return float(np.mean(deficits)) if deficits else 0.0

    def _client_prior(self, node_id: int) -> Optional[List[float]]:
        if node_id in self.current_client_distributions:
            return self.current_client_distributions[node_id].astype(np.float64).tolist()
        if self.node_label_counts is None or node_id not in self.node_label_counts:
            return None
        counts = np.asarray(self.node_label_counts[node_id], dtype=np.float64)
        total = float(counts.sum())
        if total <= 0.0:
            return None
        return (counts / total).tolist()

    @staticmethod
    def _serialise_class_prior(class_prior: Optional[List[float]]) -> Optional[str]:
        if class_prior is None:
            return None
        return ",".join(f"{float(value):.8g}" for value in class_prior)

    def _serialise_plan(self) -> Dict[str, object]:
        return {
            "selected_edges": sorted(int(edge_id) for edge_id in self.selected_edges),
            "edge_nodes": {
                str(edge_id): sorted(int(node_id) for node_id in nodes)
                for edge_id, nodes in self.edge_nodes.items()
            },
            "node_edge": {
                str(node_id): int(edge_id)
                for node_id, edge_id in self.node_edge.items()
            },
            "edge_data_sizes": {
                str(edge_id): int(size)
                for edge_id, size in self.edge_data_sizes.items()
            },
            "plan_history": self.plan_history,
        }

    def _serialise_metrics(self) -> Dict[str, object]:
        return {
            "cloud_round": [int(value) for value in self.metrics_history["cloud_round"]],
            "accuracy": [float(value) for value in self.metrics_history["accuracy"]],
            "loss": [float(value) for value in self.metrics_history["loss"]],
            "per_round_cost_gb": [float(value) for value in self.metrics_history["per_round_cost_gb"]],
            "cumulative_cost_gb": [float(value) for value in self.metrics_history["cumulative_cost_gb"]],
            "paper_per_round_cost_gb": [
                float(value) for value in self.metrics_history.get("paper_per_round_cost_gb", [])
            ],
            "paper_cumulative_cost_gb": [
                float(value) for value in self.metrics_history.get("paper_cumulative_cost_gb", [])
            ],
            "effective_per_round_cost_gb": [
                float(value) for value in self.metrics_history.get("effective_per_round_cost_gb", [])
            ],
            "effective_cumulative_cost_gb": [
                float(value) for value in self.metrics_history.get("effective_cumulative_cost_gb", [])
            ],
            "model_payload_bytes": [
                int(value) for value in self.metrics_history.get("model_payload_bytes", [])
            ],
            "probe_payload_bytes": [
                int(value) for value in self.metrics_history.get("probe_payload_bytes", [])
            ],
            "replan_rounds": [int(value) for value in self.replan_rounds],
            "drift_history": self.drift_history,
            "edge_aggregation_history": self.edge_aggregation_history,
        }

    def _serialise_privacy(self) -> Dict[str, object]:
        avg_bytes = (
            float(np.mean(self.round_probe_payload_bytes))
            if self.round_probe_payload_bytes
            else 0.0
        )
        return {
            "planning_signal": self.planning_signal,
            "emit_probe_logits": self.emit_probe_logits,
            "probe_size": self.probe_size,
            "dp_epsilon": self.dp_epsilon,
            "dp_delta": self.dp_delta,
            "local_bn": self.local_bn,
            "compression_enabled": self.compression_enabled,
            "compression_keep_ratio_min": self.compression_keep_ratio_min,
            "compression_keep_ratio_max": self.compression_keep_ratio_max,
            "compression_eta": self.compression_eta,
            "compression_target_deficit": self.compression_target_deficit,
            "compress_edge_to_cloud": self.compress_edge_to_cloud,
            "edge_min_members": self.edge_min_members,
            "edge_underfill_penalty": self.edge_underfill_penalty,
            "total_probe_payload_bytes": int(self.total_probe_payload_bytes),
            "average_probe_payload_bytes_per_round": avg_bytes,
            "per_round_probe_payload_bytes": [int(value) for value in self.round_probe_payload_bytes],
            "total_model_payload_bytes": int(sum(self.metrics_history.get("model_payload_bytes", []))),
        }

    def _status_payload(self, completed: bool) -> Dict[str, object]:
        return {
            "completed": bool(completed),
            "phase": self.phase,
            "cloud_round": int(self.cloud_round),
            "edge_epoch": int(self.edge_epoch),
            "completed_flower_rounds": int(self.completed_flower_rounds),
            "completed_local_epochs": int(self.completed_local_epochs),
            "remaining_flower_rounds": int(self.remaining_flower_rounds),
            "replan_count": int(self.replan_count),
        }

    def get_checkpoint_state(self) -> Dict[str, object]:
        return {
            "phase": self.phase,
            "cloud_round": self.cloud_round,
            "edge_epoch": self.edge_epoch,
            "completed_flower_rounds": self.completed_flower_rounds,
            "completed_local_epochs": self.completed_local_epochs,
            "global_parameters": parameters_to_ndarrays(self.global_parameters),
            "edge_parameters": {
                int(edge_id): parameters_to_ndarrays(parameters)
                for edge_id, parameters in self.edge_parameters.items()
            },
            "selected_edges": sorted(self.selected_edges),
            "edge_nodes": {int(edge_id): sorted(nodes) for edge_id, nodes in self.edge_nodes.items()},
            "node_edge": {int(node_id): int(edge_id) for node_id, edge_id in self.node_edge.items()},
            "edge_data_sizes": {int(edge_id): int(size) for edge_id, size in self.edge_data_sizes.items()},
            "c_ne": self.c_ne,
            "c_ec": self.c_ec,
            "per_round_cost_gb": self.per_round_cost_gb,
            "cumulative_cost_gb": self.cumulative_cost_gb,
            "metrics_history": self.metrics_history,
            "current_phi_raw": self.current_phi_raw,
            "current_phi": self.current_phi,
            "current_gamma": self.current_gamma,
            "current_gamma_used": self.current_gamma_used,
            "current_edge_balance_deficit": self.current_edge_balance_deficit,
            "plan_history": self.plan_history,
            "shapley_history": self.shapley_history,
            "drift_history": self.drift_history,
            "replan_rounds": self.replan_rounds,
            "replan_count": self.replan_count,
            "round_probe_payload_bytes": self.round_probe_payload_bytes,
            "total_probe_payload_bytes": self.total_probe_payload_bytes,
            "edge_aggregation_history": self.edge_aggregation_history,
            "current_hybrid_info": self.current_hybrid_info,
            "edge_anchor_weights": {
                int(edge_id): self._weights_copy(weights)
                for edge_id, weights in self.edge_anchor_weights.items()
            },
            "edge_swa_buffers": {
                int(edge_id): [self._weights_copy(weights) for weights in buffers]
                for edge_id, buffers in self.edge_swa_buffers.items()
            },
            "drift_bank": self.drift_bank.snapshot(),
            "current_cycle_reference_weights": self._weights_copy(self._current_cycle_reference_weights),
            "model_size_bytes": int(self.model_size_bytes),
            "effective_cumulative_cost_gb": float(self.effective_cumulative_cost_gb),
            "reported_paper_per_round_cost_gb": float(self._reported_paper_per_round_cost_gb),
            "reported_effective_per_round_cost_gb": float(self._reported_effective_per_round_cost_gb),
            "reported_model_payload_bytes": int(self._reported_model_payload_bytes),
            "reported_probe_payload_bytes": int(self._reported_probe_payload_bytes),
            "current_cycle_model_payload_bytes": int(self.current_cycle_model_payload_bytes),
            "current_cycle_probe_payload_bytes": int(self.current_cycle_probe_payload_bytes),
            "current_cycle_effective_cost_gb": float(self.current_cycle_effective_cost_gb),
            "client_compression_residuals": {
                int(node_id): self._weights_copy(weights)
                for node_id, weights in self.client_compression_residuals.items()
            },
            "edge_compression_residuals": {
                int(edge_id): self._weights_copy(weights)
                for edge_id, weights in self.edge_compression_residuals.items()
            },
            "numpy_rng_state": np.random.get_state(),
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        self.phase = str(state["phase"])
        self.cloud_round = int(state["cloud_round"])
        self.edge_epoch = int(state["edge_epoch"])
        self.completed_flower_rounds = int(state["completed_flower_rounds"])
        self.completed_local_epochs = int(state["completed_local_epochs"])
        self.global_parameters = ndarrays_to_parameters(state["global_parameters"])
        self.edge_parameters = {
            int(edge_id): ndarrays_to_parameters(weights)
            for edge_id, weights in state["edge_parameters"].items()
        }
        self.selected_edges = {int(edge_id) for edge_id in state["selected_edges"]}
        self.edge_nodes = {
            int(edge_id): [int(node_id) for node_id in nodes]
            for edge_id, nodes in state["edge_nodes"].items()
        }
        self.node_edge = {
            int(node_id): int(edge_id)
            for node_id, edge_id in state["node_edge"].items()
        }
        self.edge_data_sizes = {
            int(edge_id): int(size)
            for edge_id, size in state["edge_data_sizes"].items()
        }
        self.c_ne = state["c_ne"]
        self.c_ec = state["c_ec"]
        self.per_round_cost_gb = float(state["per_round_cost_gb"])
        self.cumulative_cost_gb = float(state["cumulative_cost_gb"])
        self.metrics_history = state["metrics_history"]
        self.metrics_history.setdefault("paper_per_round_cost_gb", list(self.metrics_history["per_round_cost_gb"]))
        self.metrics_history.setdefault(
            "paper_cumulative_cost_gb",
            list(self.metrics_history["cumulative_cost_gb"]),
        )
        self.metrics_history.setdefault(
            "effective_per_round_cost_gb",
            list(self.metrics_history["per_round_cost_gb"]),
        )
        self.metrics_history.setdefault(
            "effective_cumulative_cost_gb",
            list(self.metrics_history["cumulative_cost_gb"]),
        )
        self.metrics_history.setdefault("model_payload_bytes", [])
        self.metrics_history.setdefault("probe_payload_bytes", [])
        self.current_phi_raw = {int(node_id): float(value) for node_id, value in state["current_phi_raw"].items()}
        self.current_phi = {int(node_id): float(value) for node_id, value in state["current_phi"].items()}
        self.current_gamma = float(state["current_gamma"])
        self.current_gamma_used = float(state.get("current_gamma_used", self.current_gamma))
        self.current_edge_balance_deficit = float(state.get("current_edge_balance_deficit", 0.0))
        self.plan_history = list(state["plan_history"])
        self.shapley_history = list(state["shapley_history"])
        self.drift_history = list(state["drift_history"])
        self.replan_rounds = [int(value) for value in state["replan_rounds"]]
        self.replan_count = int(state["replan_count"])
        self.round_probe_payload_bytes = [int(value) for value in state["round_probe_payload_bytes"]]
        self.total_probe_payload_bytes = int(state["total_probe_payload_bytes"])
        self.edge_aggregation_history = list(state["edge_aggregation_history"])
        self.current_hybrid_info = dict(state.get("current_hybrid_info", {}))
        self.edge_anchor_weights = {
            int(edge_id): self._weights_copy(weights)
            for edge_id, weights in state["edge_anchor_weights"].items()
        }
        self.edge_swa_buffers = {
            int(edge_id): [self._weights_copy(weights) for weights in buffers]
            for edge_id, buffers in state.get("edge_swa_buffers", {}).items()
        }
        self.drift_bank.load_snapshot(state["drift_bank"])
        self._current_cycle_reference_weights = self._weights_copy(
            state["current_cycle_reference_weights"]
        )
        self.model_size_bytes = int(state.get("model_size_bytes", self.model_size_bytes))
        self.effective_cumulative_cost_gb = float(
            state.get("effective_cumulative_cost_gb", self.cumulative_cost_gb)
        )
        self._reported_paper_per_round_cost_gb = float(
            state.get("reported_paper_per_round_cost_gb", self.per_round_cost_gb)
        )
        self._reported_effective_per_round_cost_gb = float(
            state.get("reported_effective_per_round_cost_gb", self.per_round_cost_gb)
        )
        self._reported_model_payload_bytes = int(state.get("reported_model_payload_bytes", 0))
        self._reported_probe_payload_bytes = int(state.get("reported_probe_payload_bytes", 0))
        self.current_cycle_model_payload_bytes = int(state.get("current_cycle_model_payload_bytes", 0))
        self.current_cycle_probe_payload_bytes = int(state.get("current_cycle_probe_payload_bytes", 0))
        self.current_cycle_effective_cost_gb = float(state.get("current_cycle_effective_cost_gb", 0.0))
        self.client_compression_residuals = {
            int(node_id): self._weights_copy(weights)
            for node_id, weights in state.get("client_compression_residuals", {}).items()
        }
        self.edge_compression_residuals = {
            int(edge_id): self._weights_copy(weights)
            for edge_id, weights in state.get("edge_compression_residuals", {}).items()
        }
        if "numpy_rng_state" in state:
            np.random.set_state(state["numpy_rng_state"])

    def _persist_artifacts(self, completed: bool = False) -> None:
        if not self.output_dir:
            return
        save_json(self._serialise_metrics(), os.path.join(self.output_dir, "metrics.json"))
        save_json(self._serialise_plan(), os.path.join(self.output_dir, "plan.json"))
        save_json({"events": self.shapley_history}, os.path.join(self.output_dir, "shapley_history.json"))
        save_json(self._serialise_privacy(), os.path.join(self.output_dir, "privacy.json"))
        save_json(self._status_payload(completed), os.path.join(self.output_dir, "status.json"))
        with open(os.path.join(self.output_dir, "checkpoint.pkl"), "wb") as handle:
            pickle.dump(self.get_checkpoint_state(), handle)

    def _is_complete(self) -> bool:
        if self.total_local_epochs is not None and self.completed_local_epochs >= self.total_local_epochs:
            return self.edge_epoch == 0
        return self.cloud_round >= self.kappa

    def _gamma_at_cloud_round(self, cloud_round: int) -> float:
        if self.gamma_anneal == "adaptive":
            return float(np.clip(self.current_gamma, self.gamma_min, self.gamma_max))
        if self.gamma_anneal == "fixed":
            return self.gamma_max
        if self.gamma_anneal == "linear":
            return self.gamma_max * max(0.0, 1.0 - cloud_round / max(self.kappa, 1))
        angle = math.pi * cloud_round / (2.0 * max(self.kappa, 1))
        return self.gamma_max * (math.cos(angle) ** 2)

    def _effective_edge_underfill_penalty(self) -> float:
        if self.edge_underfill_penalty > 0.0:
            return self.edge_underfill_penalty
        if self.edge_underfill_penalty < 0.0 and self.c_ec:
            return float(np.median(list(self.c_ec.values())))
        return 0.0

    def _compression_keep_ratio(self, *, edge_to_cloud: bool = False) -> float:
        if not self.compression_enabled or self.cloud_round == 0:
            return 1.0
        ratio = self.compression_keep_ratio_min * math.exp(
            self.compression_eta
            * (self.current_edge_balance_deficit - self.compression_target_deficit)
        )
        ratio = float(np.clip(
            ratio,
            min(self.compression_keep_ratio_min, self.compression_keep_ratio_max),
            max(self.compression_keep_ratio_min, self.compression_keep_ratio_max),
        ))
        if edge_to_cloud:
            return max(0.10, ratio)
        return ratio

    def _reset_cycle_accounting(self) -> None:
        self.current_cycle_model_payload_bytes = 0
        self.current_cycle_probe_payload_bytes = 0
        self.current_cycle_effective_cost_gb = 0.0

    def _finalise_completed_cycle(
        self,
        *,
        paper_cost_gb: float,
    ) -> None:
        self.cumulative_cost_gb += float(paper_cost_gb)
        self.effective_cumulative_cost_gb += float(self.current_cycle_effective_cost_gb)
        self._reported_paper_per_round_cost_gb = float(paper_cost_gb)
        self._reported_effective_per_round_cost_gb = float(self.current_cycle_effective_cost_gb)
        self._reported_model_payload_bytes = int(self.current_cycle_model_payload_bytes)
        self._reported_probe_payload_bytes = int(self.current_cycle_probe_payload_bytes)
        self._reset_cycle_accounting()

    def _record_completed_cloud_metrics(
        self,
        *,
        accuracy: float,
        loss: float,
    ) -> None:
        self.metrics_history["cloud_round"].append(int(self.cloud_round))
        self.metrics_history["accuracy"].append(float(accuracy))
        self.metrics_history["loss"].append(float(loss))
        self.metrics_history["per_round_cost_gb"].append(float(self._reported_paper_per_round_cost_gb))
        self.metrics_history["cumulative_cost_gb"].append(float(self.cumulative_cost_gb))
        self.metrics_history["paper_per_round_cost_gb"].append(float(self._reported_paper_per_round_cost_gb))
        self.metrics_history["paper_cumulative_cost_gb"].append(float(self.cumulative_cost_gb))
        self.metrics_history["effective_per_round_cost_gb"].append(
            float(self._reported_effective_per_round_cost_gb)
        )
        self.metrics_history["effective_cumulative_cost_gb"].append(
            float(self.effective_cumulative_cost_gb)
        )
        self.metrics_history["model_payload_bytes"].append(int(self._reported_model_payload_bytes))
        self.metrics_history["probe_payload_bytes"].append(int(self._reported_probe_payload_bytes))

    def _linear_updates_from_weights(
        self,
        client_weights: Dict[int, List[np.ndarray]],
        reference_weights: List[np.ndarray],
    ) -> Dict[int, torch.Tensor]:
        ds_info = DATASET_INFO[self.dataset_name]
        model = get_model(
            self.model_name,
            ds_info["num_classes"],
            ds_info["input_channels"],
            "cpu",
        )
        keys = list(model.state_dict().keys())
        linear_name = model.linear_layer_name
        linear_indices = [
            index for index, key in enumerate(keys)
            if key.startswith(linear_name)
        ]
        updates: Dict[int, torch.Tensor] = {}
        for node_id, weights in client_weights.items():
            deltas = [weights[index] - reference_weights[index] for index in linear_indices]
            updates[node_id] = torch.tensor(np.concatenate([delta.ravel() for delta in deltas]))
        return updates

    def _build_full_similarity_matrix(
        self,
        linear_updates: Dict[int, torch.Tensor],
    ) -> np.ndarray:
        present_nodes = sorted(linear_updates.keys())
        partial = compute_similarity_matrix(linear_updates)
        full = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        for i, node_i in enumerate(present_nodes):
            for j, node_j in enumerate(present_nodes):
                full[node_i, node_j] = partial[i, j]
        return full

    def _collect_client_state(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[
        Dict[int, List[np.ndarray]],
        Dict[int, int],
        Dict[int, np.ndarray],
        int,
        Dict[int, int],
    ]:
        client_weights: Dict[int, List[np.ndarray]] = {}
        client_sizes: Dict[int, int] = {}
        probe_logits: Dict[int, np.ndarray] = {}
        round_probe_bytes = 0
        probe_payload_bytes: Dict[int, int] = {}

        for client_proxy, fit_res in results:
            node_id = self._resolve_node_id(client_proxy.cid)
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_weights[node_id] = weights
            client_sizes[node_id] = int(fit_res.num_examples)
            logits = deserialize_probe_logits(fit_res.metrics)
            if logits is not None:
                probe_logits[node_id] = logits
            payload_bytes = int(probe_payload_num_bytes(fit_res.metrics))
            probe_payload_bytes[node_id] = payload_bytes
            round_probe_bytes += payload_bytes

        self._latest_client_weights = client_weights
        self._latest_client_sizes = client_sizes
        self._latest_probe_logits = probe_logits
        if round_probe_bytes > 0:
            self.total_probe_payload_bytes += round_probe_bytes
        self.round_probe_payload_bytes.append(int(round_probe_bytes))
        return client_weights, client_sizes, probe_logits, round_probe_bytes, probe_payload_bytes

    def _extract_client_distributions(
        self,
        client_weights: Dict[int, List[np.ndarray]],
        probe_logits: Dict[int, np.ndarray],
    ) -> Dict[int, np.ndarray]:
        distributions: Dict[int, np.ndarray] = {}
        for node_id, weights in client_weights.items():
            logits = probe_logits.get(node_id)
            if logits is None:
                if self.model_factory is None or self.probe_loader is None:
                    raise RuntimeError(
                        f"Missing probe logits for node {node_id} and no server-side fallback configured"
                    )
                logits = predict_probe_logits(
                    weights=weights,
                    model_factory=self.model_factory,
                    probe_loader=self.probe_loader,
                    device=self.server_device,
                )
            distributions[node_id] = mean_softmax_distribution(logits)
        return distributions

    def _apply_los_result(self, los_result) -> None:
        if los_result is None or los_result.node_associations is None:
            raise RuntimeError("RoSE planning produced no valid node associations")
        self.selected_edges = {int(edge_id) for edge_id in los_result.selected_edges}
        self.edge_nodes = {
            int(edge_id): [int(node_id) for node_id in sorted(nodes)]
            for edge_id, nodes in los_result.node_associations.edge_nodes.items()
        }
        self.edge_data_sizes = {
            int(edge_id): int(size)
            for edge_id, size in los_result.node_associations.edge_data_sizes.items()
        }
        self.node_edge = {}
        for edge_id, nodes in self.edge_nodes.items():
            for node_id in nodes:
                self.node_edge[node_id] = edge_id
        self._compute_per_round_cost()
        self.edge_parameters = {
            int(edge_id): ndarrays_to_parameters(
                self._weights_copy(parameters_to_ndarrays(self.global_parameters))
            )
            for edge_id in self.selected_edges
        }
        self.edge_anchor_weights = {
            int(edge_id): self._weights_copy(parameters_to_ndarrays(self.global_parameters))
            for edge_id in self.selected_edges
        }
        self.edge_swa_buffers = {int(edge_id): [] for edge_id in self.selected_edges}
        self.drift_bank.reset(self.selected_edges)

    def _plan_with_signal(
        self,
        client_weights: Dict[int, List[np.ndarray]],
        client_sizes: Dict[int, int],
        probe_logits: Dict[int, np.ndarray],
        reason: str,
        planning_round: int,
    ) -> None:
        present_nodes = sorted(client_sizes.keys())
        gamma_t = self._gamma_at_cloud_round(planning_round)
        self.current_gamma_used = gamma_t

        warm_start_edges = (
            set(self.selected_edges)
            if self.warm_start_replan and self.selected_edges and reason != "warmup"
            else None
        )
        warm_start_associations = (
            dict(self.node_edge)
            if self.warm_start_replan and self.node_edge and reason != "warmup"
            else None
        )

        class_distributions: Dict[int, np.ndarray] = {}
        self.current_hybrid_info = {}
        fill_penalty = self._effective_edge_underfill_penalty()
        if self.planning_signal in {"shapley", "hybrid"}:
            if self.probe_loader is None or self.model_factory is None:
                raise RuntimeError("RoSE planning requires probe_loader and model_factory")
            class_distributions = self._extract_client_distributions(client_weights, probe_logits)

        if self.planning_signal == "shapley":
            phi_raw = compute_smc_shapley(
                client_weights=client_weights,
                client_sizes=client_sizes,
                probe_loader=self.probe_loader,
                model_factory=self.model_factory,
                device=self.server_device,
                T=self.shapley_T,
                K=self.shapley_K,
                seed=self.seed + planning_round,
            )
            phi = normalise_shapley(phi_raw)
            los_result = run_los_rose(
                candidate_edges=present_nodes,
                all_nodes=present_nodes,
                communication_costs_ne=self.c_ne,
                communication_costs_ec=self.c_ec,
                phi=phi,
                client_class_distributions=class_distributions,
                data_sizes=client_sizes,
                kappa_c=self.kappa_c,
                gamma=gamma_t,
                B_e=self.B_e,
                T_max=self.T_max,
                initial_edges=warm_start_edges,
                initial_associations=warm_start_associations,
                warm_start_threshold=self.warm_start_threshold,
                edge_min_members=self.edge_min_members,
                edge_underfill_penalty=fill_penalty,
                verbose=False,
            )
            self.current_phi_raw = {int(node_id): float(value) for node_id, value in phi_raw.items()}
            self.current_phi = {int(node_id): float(value) for node_id, value in phi.items()}
            self.current_client_distributions = class_distributions
            event = {
                "cloud_round": int(planning_round),
                "reason": reason,
                "planning_signal": "shapley",
                "gamma_t": float(gamma_t),
                "phi_raw": self.current_phi_raw,
                "phi": self.current_phi,
                "objective_value": float(los_result.objective_value),
            }
        elif self.planning_signal == "hybrid":
            phi, hybrid_info = compute_hybrid_phi(
                client_weights=client_weights,
                client_sizes=client_sizes,
                probe_loader=self.probe_loader,
                model_factory=self.model_factory,
                device=self.server_device,
                reference_weights=self._current_cycle_reference_weights,
                probe_logits=probe_logits,
                probe_targets=self.probe_targets,
                T=self.shapley_T,
                K=self.shapley_K,
                seed=self.seed + planning_round,
                lambda_floor=self.hybrid_lambda_floor,
                lambda_ceiling=self.hybrid_lambda_ceiling,
            )
            self.current_phi_raw = {int(node_id): float(value) for node_id, value in phi.items()}
            self.current_phi = {int(node_id): float(value) for node_id, value in phi.items()}
            self.current_client_distributions = class_distributions
            self.current_hybrid_info = {
                key: (
                    {int(node_id): float(value) for node_id, value in value.items()}
                    if isinstance(value, dict)
                    else float(value)
                )
                for key, value in hybrid_info.items()
            }
            los_result = run_los_rose(
                candidate_edges=present_nodes,
                all_nodes=present_nodes,
                communication_costs_ne=self.c_ne,
                communication_costs_ec=self.c_ec,
                phi=self.current_phi,
                client_class_distributions=class_distributions,
                data_sizes=client_sizes,
                kappa_c=self.kappa_c,
                gamma=gamma_t,
                B_e=self.B_e,
                T_max=self.T_max,
                initial_edges=warm_start_edges,
                initial_associations=warm_start_associations,
                warm_start_threshold=self.warm_start_threshold,
                edge_min_members=self.edge_min_members,
                edge_underfill_penalty=fill_penalty,
                verbose=False,
            )
            event = {
                "cloud_round": int(planning_round),
                "reason": reason,
                "planning_signal": "hybrid",
                "gamma_t": float(gamma_t),
                "phi_raw": self.current_phi_raw,
                "phi": self.current_phi,
                "hybrid_info": self.current_hybrid_info,
                "objective_value": float(los_result.objective_value),
            }
        else:
            linear_updates = self._linear_updates_from_weights(
                client_weights=client_weights,
                reference_weights=self._current_cycle_reference_weights,
            )
            similarity = self._build_full_similarity_matrix(linear_updates)
            los_result = run_los(
                candidate_edges=present_nodes,
                all_nodes=present_nodes,
                communication_costs_ne=self.c_ne,
                communication_costs_ec=self.c_ec,
                similarity_matrix=similarity,
                data_sizes=client_sizes,
                kappa_c=self.kappa_c,
                gamma=gamma_t,
                B_e=self.B_e,
                T_max=self.T_max,
                initial_edges=warm_start_edges,
            )
            self.current_phi_raw = {int(node_id): 1.0 for node_id in present_nodes}
            self.current_phi = {int(node_id): 1.0 for node_id in present_nodes}
            self.current_client_distributions = {}
            event = {
                "cloud_round": int(planning_round),
                "reason": reason,
                "planning_signal": "cosine",
                "gamma_t": float(gamma_t),
                "objective_value": float(los_result.objective_value),
            }

        candidate_edge_nodes = {
            int(edge_id): [int(node_id) for node_id in sorted(nodes)]
            for edge_id, nodes in los_result.node_associations.edge_nodes.items()
        }
        candidate_cost = self._estimate_plan_cost_gb(candidate_edge_nodes)
        cost_cap_triggered = (
            reason.startswith("drift@")
            and self.per_round_cost_gb > 0.0
            and candidate_cost > self.per_round_cost_gb * (1.0 + self.replan_cost_increase_tolerance)
        )
        if not cost_cap_triggered or not self.selected_edges:
            self._apply_los_result(los_result)
        event["cost_cap_triggered"] = bool(cost_cap_triggered)
        event["candidate_per_round_cost_gb"] = float(candidate_cost)

        self.current_edge_balance_deficit = self._compute_edge_balance_deficit(
            self.edge_nodes,
            self.current_client_distributions,
            client_sizes,
        )
        if self.gamma_anneal == "adaptive":
            self.current_gamma = float(np.clip(
                gamma_t * math.exp(
                    self.adaptive_gamma_eta * (
                        self.current_edge_balance_deficit - self.adaptive_gamma_target
                    )
                ),
                self.gamma_min,
                self.gamma_max,
            ))
        else:
            self.current_gamma = gamma_t
        event["edge_balance_deficit"] = float(self.current_edge_balance_deficit)
        event["gamma_next"] = float(self.current_gamma)
        event["edge_min_members"] = int(self.edge_min_members)
        event["edge_underfill_penalty"] = float(fill_penalty)
        event["selected_edges"] = sorted(int(edge_id) for edge_id in self.selected_edges)
        event["edge_nodes"] = {
            str(edge_id): sorted(int(node_id) for node_id in nodes)
            for edge_id, nodes in self.edge_nodes.items()
        }
        self.shapley_history.append(event)
        self.plan_history.append(event)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(
            num_clients=self.num_nodes,
            min_num_clients=self.num_nodes,
        )
        self._build_cid_map(clients)

        if self.phase == "warmup":
            self._current_round_local_epochs = self.warmup_epochs
            self._current_cycle_reference_weights = self._weights_copy(
                parameters_to_ndarrays(self.global_parameters)
            )
            emit_probe_logits = bool(self.emit_probe_logits and self.planning_signal in {"shapley", "hybrid"})
            config = {
                "phase": "warmup",
                "epochs": self.warmup_epochs,
                "lr": self.lr,
                "momentum": self.momentum,
                "emit_probe_logits": emit_probe_logits,
                "dp_epsilon": self.dp_epsilon,
                "dp_delta": self.dp_delta,
                "probe_noise_seed": self.seed + self.completed_flower_rounds,
                "prox_mu": self.local_objective_prox_mu,
                "logit_adjustment_tau": self.logit_adjustment_tau,
                "local_bn": self.local_bn,
            }
            fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
            for client in clients:
                node_id = self._resolve_node_id(client.cid)
                node_config = dict(config)
                class_prior = self._client_prior(node_id)
                if class_prior is not None:
                    node_config["class_prior"] = self._serialise_class_prior(class_prior)
                fit_ins_list.append((client, FitIns(self.global_parameters, node_config)))
            return fit_ins_list

        epochs_this_round = self.kappa_e
        if self.total_local_epochs is not None:
            remaining = self.total_local_epochs - self.completed_local_epochs
            epochs_this_round = min(self.kappa_e, max(remaining, 0))
        self._current_round_local_epochs = epochs_this_round

        if self.edge_epoch == 0:
            self._current_cycle_reference_weights = self._weights_copy(
                parameters_to_ndarrays(self.global_parameters)
            )

        config = {
            "phase": "train",
            "epochs": epochs_this_round,
            "lr": self.lr,
            "momentum": self.momentum,
            "emit_probe_logits": bool(self.emit_probe_logits and self.planning_signal in {"shapley", "hybrid"}),
            "dp_epsilon": self.dp_epsilon,
            "dp_delta": self.dp_delta,
            "probe_noise_seed": self.seed + self.completed_flower_rounds,
            "prox_mu": self.local_objective_prox_mu,
            "logit_adjustment_tau": self.logit_adjustment_tau,
            "local_bn": self.local_bn,
        }
        fit_ins_list: List[Tuple[ClientProxy, FitIns]] = []
        for client in clients:
            node_id = self._resolve_node_id(client.cid)
            edge_id = self.node_edge.get(node_id)
            if self.edge_epoch == 0 or edge_id is None:
                params = self.global_parameters
            else:
                params = self.edge_parameters.get(edge_id, self.global_parameters)
            node_config = dict(config)
            class_prior = self._client_prior(node_id)
            if class_prior is not None:
                node_config["class_prior"] = self._serialise_class_prior(class_prior)
            fit_ins_list.append((client, FitIns(params, node_config)))
        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if self.phase == "warmup":
            return self._aggregate_warmup(results)
        return self._aggregate_train_rose(results)

    def _aggregate_warmup(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            raise RuntimeError("RoSE warmup failed: no client updates received")

        initial_weights = parameters_to_ndarrays(self.initial_parameters)
        model_size_bytes = sum(weights.nbytes for weights in initial_weights)
        self.c_ne, self.c_ec = generate_communication_costs(
            self.num_nodes,
            model_size_bytes,
            topology=self.topology,
        )

        client_weights, client_sizes, probe_logits, _, _ = self._collect_client_state(results)
        warmup_global = _weighted_average(
            list(client_weights.values()),
            list(client_sizes.values()),
        )
        warmup_global = self._mask_batch_norm_weights(
            warmup_global,
            parameters_to_ndarrays(self.global_parameters),
        )
        self.global_parameters = ndarrays_to_parameters(warmup_global)
        self.completed_local_epochs += self.warmup_epochs
        self._plan_with_signal(
            client_weights=client_weights,
            client_sizes=client_sizes,
            probe_logits=probe_logits,
            reason="warmup",
            planning_round=0,
        )
        self.phase = "train"
        self.cloud_round = 0
        self.edge_epoch = 0
        self.completed_flower_rounds += 1
        self._reset_cycle_accounting()
        self._persist_artifacts(completed=False)
        return self.global_parameters, {"phase": "warmup_done"}

    def _aggregate_train_rose(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return self.global_parameters, {}

        (
            client_weights,
            client_sizes,
            probe_logits,
            _,
            probe_payload_bytes,
        ) = self._collect_client_state(results)
        for node_id, payload_bytes in probe_payload_bytes.items():
            if payload_bytes <= 0:
                continue
            self.current_cycle_probe_payload_bytes += int(payload_bytes)
            self.current_cycle_effective_cost_gb += scaled_cost_from_payload(
                self.c_ec.get(int(node_id), 0.0),
                int(payload_bytes),
                self.model_size_bytes,
            )

        compress_this_cycle = self.compression_enabled and self.cloud_round >= 1
        node_keep_ratio = self._compression_keep_ratio(edge_to_cloud=False)
        edge_keep_ratio = self._compression_keep_ratio(edge_to_cloud=True)

        edge_groups: Dict[int, Dict[str, object]] = defaultdict(
            lambda: {"weights": [], "sizes": [], "nodes": [], "payload_bytes": []}
        )
        for node_id, weights in client_weights.items():
            edge_id = self.node_edge.get(node_id)
            if edge_id is None:
                continue
            if self.edge_epoch == 0:
                reference_weights = parameters_to_ndarrays(self.global_parameters)
            else:
                reference_weights = parameters_to_ndarrays(
                    self.edge_parameters.get(edge_id, self.global_parameters)
                )

            transmitted_weights = self._weights_copy(weights)
            payload_bytes = self.model_size_bytes
            if compress_this_cycle:
                compression = compress_weight_update(
                    reference_weights=reference_weights,
                    target_weights=weights,
                    keep_ratio=node_keep_ratio,
                    residuals=self.client_compression_residuals.get(int(node_id)),
                    dense_layer_indices=self.dense_compression_indices,
                )
                transmitted_weights = compression.reconstructed_weights
                payload_bytes = int(compression.payload_bytes)
                self.client_compression_residuals[int(node_id)] = compression.residuals
            else:
                self.client_compression_residuals.setdefault(
                    int(node_id),
                    zero_residuals_like(weights),
                )

            self.current_cycle_model_payload_bytes += int(payload_bytes)
            self.current_cycle_effective_cost_gb += scaled_cost_from_payload(
                self.c_ne.get((int(node_id), int(edge_id)), 0.0),
                int(payload_bytes),
                self.model_size_bytes,
            )
            edge_groups[edge_id]["weights"].append(transmitted_weights)
            edge_groups[edge_id]["sizes"].append(client_sizes[node_id])
            edge_groups[edge_id]["nodes"].append(node_id)
            edge_groups[edge_id]["payload_bytes"].append(int(payload_bytes))

        edge_summary: Dict[str, object] = {"cloud_round": int(self.cloud_round), "edge_epoch": int(self.edge_epoch)}
        for edge_id, group in edge_groups.items():
            weights_list = group["weights"]
            if not weights_list:
                continue
            sizes = group["sizes"]
            nodes = group["nodes"]
            aggregate, info = aggregate_with_rule(
                rule=self.agg_rule,
                node_ids=nodes,
                weights_list=weights_list,
                sizes=sizes,
                phi=self.current_phi if self.current_phi else None,
                trim_ratio=self.agg_trim_ratio,
                krum_f=self.krum_f,
                beta=self.beta,
                eta=self.eta,
                xi=self.xi,
                zeta=self.zeta,
                alpha_cap_multiplier=self.alpha_cap_multiplier,
                use_shrinkage=self.trust_use_shrinkage,
                prior_a=self.trust_prior_a,
                prior_b=self.trust_prior_b,
                nu=self.trust_nu,
                dev_clip_q=self.trust_dev_clip_q,
            )
            reference_weights = parameters_to_ndarrays(
                self.edge_parameters.get(int(edge_id), self.global_parameters)
            )
            aggregate = self._mask_batch_norm_weights(aggregate, reference_weights)
            self._update_edge_swa_buffer(int(edge_id), aggregate)
            self.edge_parameters[int(edge_id)] = ndarrays_to_parameters(aggregate)
            serialisable_info = {"rule": info["rule"], "nodes": [int(node_id) for node_id in nodes]}
            if "alpha" in info:
                serialisable_info["alpha"] = [float(value) for value in info["alpha"]]
            if "trust_scores" in info:
                serialisable_info["trust_scores"] = [float(value) for value in info["trust_scores"]]
            serialisable_info["swa_buffer_size"] = int(len(self.edge_swa_buffers.get(int(edge_id), [])))
            serialisable_info["model_payload_bytes"] = int(sum(group["payload_bytes"]))
            serialisable_info["compression_keep_ratio"] = float(node_keep_ratio if compress_this_cycle else 1.0)
            edge_summary[str(edge_id)] = serialisable_info

        self.edge_aggregation_history.append(edge_summary)
        self.edge_epoch += 1
        self.completed_local_epochs += self._current_round_local_epochs
        self.completed_flower_rounds += 1

        reached_local_budget = (
            self.total_local_epochs is not None
            and self.completed_local_epochs >= self.total_local_epochs
        )

        if self.edge_epoch < self.kappa_c and not reached_local_budget:
            return self.global_parameters, {"edge_epoch": self.edge_epoch}

        edge_weights: List[List[np.ndarray]] = []
        edge_sizes: List[int] = []
        global_reference_weights = parameters_to_ndarrays(self.global_parameters)
        for edge_id in sorted(self.selected_edges):
            if edge_id not in self.edge_parameters:
                continue
            edge_weights_current = self._edge_swa_average(int(edge_id))
            if edge_weights_current is None:
                edge_weights_current = parameters_to_ndarrays(self.edge_parameters[edge_id])
            edge_weights_current = self._mask_batch_norm_weights(
                edge_weights_current,
                parameters_to_ndarrays(self.global_parameters),
            )
            transmitted_edge_weights = self._weights_copy(edge_weights_current)
            payload_bytes = self.model_size_bytes
            if compress_this_cycle and self.compress_edge_to_cloud:
                compression = compress_weight_update(
                    reference_weights=global_reference_weights,
                    target_weights=edge_weights_current,
                    keep_ratio=edge_keep_ratio,
                    residuals=self.edge_compression_residuals.get(int(edge_id)),
                    dense_layer_indices=self.dense_compression_indices,
                )
                transmitted_edge_weights = compression.reconstructed_weights
                payload_bytes = int(compression.payload_bytes)
                self.edge_compression_residuals[int(edge_id)] = compression.residuals
            else:
                self.edge_compression_residuals.setdefault(
                    int(edge_id),
                    zero_residuals_like(edge_weights_current),
                )
            self.current_cycle_model_payload_bytes += int(payload_bytes)
            self.current_cycle_effective_cost_gb += scaled_cost_from_payload(
                self.c_ec.get(int(edge_id), 0.0),
                int(payload_bytes),
                self.model_size_bytes,
            )
            edge_weights.append(transmitted_edge_weights)
            edge_sizes.append(int(self.edge_data_sizes.get(edge_id, 1)))
            edge_summary.setdefault(str(edge_id), {})
            edge_summary[str(edge_id)]["edge_to_cloud_payload_bytes"] = int(payload_bytes)
            edge_summary[str(edge_id)]["edge_to_cloud_keep_ratio"] = float(
                edge_keep_ratio if (compress_this_cycle and self.compress_edge_to_cloud) else 1.0
            )

        if edge_weights:
            global_average = _weighted_average(edge_weights, edge_sizes)
            global_average = self._mask_batch_norm_weights(
                global_average,
                parameters_to_ndarrays(self.global_parameters),
            )
            self.global_parameters = ndarrays_to_parameters(global_average)

        self._finalise_completed_cycle(paper_cost_gb=self.per_round_cost_gb)
        self.cloud_round += 1
        self.edge_epoch = 0

        triggered_edges: List[int] = []
        if self.drift_enabled and self.selected_edges:
            distances = {
                int(edge_id): weights_l2_distance(
                    parameters_to_ndarrays(self.edge_parameters[edge_id]),
                    self.edge_anchor_weights.get(
                        int(edge_id),
                        parameters_to_ndarrays(self.global_parameters),
                    ),
                )
                for edge_id in self.selected_edges
                if edge_id in self.edge_parameters
            }
            statistics, triggered_edges = self.drift_bank.update_many(distances)
            self.drift_history.append(
                {
                    "cloud_round": int(self.cloud_round),
                    "distances": {str(edge_id): float(value) for edge_id, value in distances.items()},
                    "statistics": {str(edge_id): float(value) for edge_id, value in statistics.items()},
                    "triggered_edges": [int(edge_id) for edge_id in triggered_edges],
                }
            )

        replanned = False
        if triggered_edges and self.replan_count < self.max_replans:
            previous_assignments = dict(self.node_edge)
            self.replan_count += 1
            self.replan_rounds.append(self.cloud_round)
            self._plan_with_signal(
                client_weights=client_weights,
                client_sizes=client_sizes,
                probe_logits=probe_logits,
                reason=f"drift@{self.cloud_round}",
                planning_round=self.cloud_round,
            )
            replanned = previous_assignments != self.node_edge

        self._persist_artifacts(completed=self._is_complete())
        return self.global_parameters, {
            "cloud_round": self.cloud_round,
            "replanned": int(replanned),
        }

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.evaluate_fn is not None:
            return []
        if self.phase == "warmup" or self.edge_epoch != 0 or self.cloud_round == 0:
            return []
        clients = client_manager.sample(
            num_clients=self.num_nodes,
            min_num_clients=self.num_nodes,
        )
        return [(client, EvaluateIns(self.global_parameters, {})) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        total_examples = sum(result.num_examples for _, result in results)
        weighted_loss = sum(result.loss * result.num_examples for _, result in results) / total_examples
        weighted_accuracy = sum(
            result.metrics.get("accuracy", 0.0) * result.num_examples
            for _, result in results
        ) / total_examples
        self._record_completed_cloud_metrics(
            accuracy=weighted_accuracy,
            loss=weighted_loss,
        )
        self._persist_artifacts(completed=self._is_complete())
        print(
            f"  RoSE Cloud Round {self.cloud_round}/{self.kappa} | "
            f"Acc: {weighted_accuracy:.4f} | Loss: {weighted_loss:.4f} | "
            f"PaperCost: {self.cumulative_cost_gb:.4f} GB | "
            f"EffectiveCost: {self.effective_cumulative_cost_gb:.4f} GB"
        )
        return weighted_loss, {"accuracy": weighted_accuracy}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None
        if self.phase == "warmup" or self.edge_epoch != 0 or self.cloud_round == 0:
            return None
        params = parameters_to_ndarrays(parameters)
        loss, metrics = self.evaluate_fn(server_round, params, {})
        accuracy = metrics.get("accuracy", 0.0)
        self._record_completed_cloud_metrics(
            accuracy=accuracy,
            loss=loss,
        )
        self._persist_artifacts(completed=self._is_complete())
        print(
            f"  RoSE Cloud Round {self.cloud_round}/{self.kappa} | "
            f"Acc: {accuracy:.4f} | Loss: {loss:.4f} | "
            f"PaperCost: {self.cumulative_cost_gb:.4f} GB | "
            f"EffectiveCost: {self.effective_cumulative_cost_gb:.4f} GB"
        )
        return loss, metrics
