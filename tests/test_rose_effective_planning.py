import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from flwr.common import ndarrays_to_parameters

from shapefl.strategy import RoSEHFLStrategy


def _strategy(**overrides) -> RoSEHFLStrategy:
    initial_parameters = ndarrays_to_parameters([np.zeros((1,), dtype=np.float32)])
    config = {
        "model_name": "lenet5",
        "dataset_name": "fmnist",
        "num_nodes": 6,
        "kappa_e": 1,
        "kappa_c": 1,
        "kappa": 10,
        "warmup_epochs": 1,
        "gamma_max": 10.0,
        "B_e": 3,
        "T_max": 3,
        "lr": 0.01,
        "momentum": 0.0,
        "initial_parameters": initial_parameters,
        "planning_objective": "effective",
        "effective_planning_start_cloud_round": 0,
        "accuracy_guard_tolerance": 0.02,
    }
    config.update(overrides)
    return RoSEHFLStrategy(**config)


def _edge_nodes_to_node_edge(edge_nodes):
    node_edge = {}
    for edge_id, nodes in edge_nodes.items():
        for node_id in nodes:
            node_edge[int(node_id)] = int(edge_id)
    return node_edge


def _candidate(
    *,
    source: str,
    edge_nodes,
    planning_score_gb: float,
    paper_per_round_cost_gb: float,
    effective_per_round_cost_gb: float,
    simulated_probe_accuracy: float,
):
    normalised_edge_nodes = {
        int(edge_id): [int(node_id) for node_id in nodes]
        for edge_id, nodes in edge_nodes.items()
    }
    return {
        "source": source,
        "selected_edges": set(int(edge_id) for edge_id in normalised_edge_nodes),
        "edge_nodes": normalised_edge_nodes,
        "node_edge": _edge_nodes_to_node_edge(normalised_edge_nodes),
        "objective_value": paper_per_round_cost_gb,
        "paper_per_round_cost_gb": paper_per_round_cost_gb,
        "effective_per_round_cost_gb": effective_per_round_cost_gb,
        "change_cost_gb": 0.0,
        "projected_remaining_rounds": 1,
        "planning_score_gb": planning_score_gb,
        "simulated_probe_accuracy": simulated_probe_accuracy,
    }


class RoSEQ1SPlanningTests(unittest.TestCase):
    @staticmethod
    def _los_result(edge_nodes, objective_value: float):
        normalised_edge_nodes = {
            int(edge_id): [int(node_id) for node_id in nodes]
            for edge_id, nodes in edge_nodes.items()
        }
        return SimpleNamespace(
            selected_edges=set(normalised_edge_nodes),
            node_associations=SimpleNamespace(
                edge_nodes=normalised_edge_nodes,
                edge_data_sizes={
                    int(edge_id): len(nodes)
                    for edge_id, nodes in normalised_edge_nodes.items()
                },
            ),
            objective_value=float(objective_value),
        )

    def test_selection_prefers_lower_effective_total_cost_over_lower_paper_cost(self):
        strategy = _strategy()
        candidates = [
            _candidate(
                source="current",
                edge_nodes={0: [0, 1, 2], 1: [3, 4, 5]},
                planning_score_gb=2.40,
                paper_per_round_cost_gb=0.20,
                effective_per_round_cost_gb=0.80,
                simulated_probe_accuracy=0.80,
            ),
            _candidate(
                source="planner_0",
                edge_nodes={0: [0, 1, 2], 2: [3, 4, 5]},
                planning_score_gb=1.80,
                paper_per_round_cost_gb=0.35,
                effective_per_round_cost_gb=0.40,
                simulated_probe_accuracy=0.79,
            ),
        ]

        chosen, annotated = strategy._select_effective_candidate(
            candidates,
            planning_round=0,
        )

        self.assertEqual(chosen["source"], "planner_0")
        self.assertGreater(
            float(annotated[1]["paper_per_round_cost_gb"]),
            float(annotated[0]["paper_per_round_cost_gb"]),
        )
        self.assertLess(
            float(annotated[1]["planning_score_gb"]),
            float(annotated[0]["planning_score_gb"]),
        )

    def test_accuracy_guard_rejects_cheaper_candidate_below_tolerance(self):
        strategy = _strategy()
        candidates = [
            _candidate(
                source="current",
                edge_nodes={0: [0, 1, 2], 1: [3, 4, 5]},
                planning_score_gb=3.00,
                paper_per_round_cost_gb=0.30,
                effective_per_round_cost_gb=0.60,
                simulated_probe_accuracy=0.82,
            ),
            _candidate(
                source="planner_0",
                edge_nodes={0: [0, 1, 2], 2: [3, 4, 5]},
                planning_score_gb=1.20,
                paper_per_round_cost_gb=0.25,
                effective_per_round_cost_gb=0.45,
                simulated_probe_accuracy=0.77,
            ),
        ]

        chosen, annotated = strategy._select_effective_candidate(
            candidates,
            planning_round=0,
        )

        rejected = next(candidate for candidate in annotated if candidate["source"] == "planner_0")
        self.assertEqual(chosen["source"], "current")
        self.assertFalse(bool(rejected["accuracy_guard_passed"]))
        self.assertEqual(rejected["selection_reason"], "rejected_accuracy_guard")

    def test_mid_phase_uses_best_probe_delta_baseline(self):
        strategy = _strategy(
            effective_planning_start_cloud_round=3,
            late_phase_start_fraction=0.8,
            effective_accuracy_delta=0.01,
            kappa=10,
        )
        candidates = [
            _candidate(
                source="current",
                edge_nodes={0: [0, 1, 2], 1: [3, 4, 5]},
                planning_score_gb=3.00,
                paper_per_round_cost_gb=0.30,
                effective_per_round_cost_gb=0.60,
                simulated_probe_accuracy=0.81,
            ),
            _candidate(
                source="planner_0",
                edge_nodes={0: [0, 1, 2], 2: [3, 4, 5]},
                planning_score_gb=2.50,
                paper_per_round_cost_gb=0.28,
                effective_per_round_cost_gb=0.40,
                simulated_probe_accuracy=0.83,
            ),
            _candidate(
                source="planner_1",
                edge_nodes={1: [0, 1, 2], 2: [3, 4, 5]},
                planning_score_gb=1.10,
                paper_per_round_cost_gb=0.26,
                effective_per_round_cost_gb=0.35,
                simulated_probe_accuracy=0.821,
            ),
        ]

        chosen, annotated = strategy._select_effective_candidate(
            candidates,
            planning_round=2,
        )

        baseline = next(candidate for candidate in annotated if candidate["source"] == "planner_0")
        self.assertEqual(chosen["source"], "planner_1")
        self.assertEqual(baseline["baseline_reason"], "best_probe_accuracy")
        self.assertEqual(baseline["planning_phase"], "mid")

    def test_late_phase_returns_to_current_topology_baseline(self):
        strategy = _strategy(
            effective_planning_start_cloud_round=3,
            late_phase_start_fraction=0.8,
            effective_accuracy_delta=0.01,
            kappa=10,
        )
        candidates = [
            _candidate(
                source="current",
                edge_nodes={0: [0, 1, 2], 1: [3, 4, 5]},
                planning_score_gb=3.00,
                paper_per_round_cost_gb=0.30,
                effective_per_round_cost_gb=0.60,
                simulated_probe_accuracy=0.81,
            ),
            _candidate(
                source="planner_0",
                edge_nodes={0: [0, 1, 2], 2: [3, 4, 5]},
                planning_score_gb=2.50,
                paper_per_round_cost_gb=0.28,
                effective_per_round_cost_gb=0.40,
                simulated_probe_accuracy=0.83,
            ),
        ]

        _, annotated = strategy._select_effective_candidate(
            candidates,
            planning_round=7,
        )

        baseline = next(candidate for candidate in annotated if candidate["source"] == "current")
        self.assertEqual(baseline["baseline_reason"], "current_topology")
        self.assertEqual(baseline["planning_phase"], "late")

    def test_hard_edge_minimum_falls_back_to_current_topology(self):
        strategy = _strategy(hard_edge_min_members=3)
        candidates = [
            _candidate(
                source="current",
                edge_nodes={0: [0, 1], 1: [2, 3], 2: [4, 5]},
                planning_score_gb=3.00,
                paper_per_round_cost_gb=0.30,
                effective_per_round_cost_gb=0.60,
                simulated_probe_accuracy=0.80,
            ),
            _candidate(
                source="planner_0",
                edge_nodes={0: [0], 1: [1, 2], 2: [3, 4, 5]},
                planning_score_gb=1.00,
                paper_per_round_cost_gb=0.20,
                effective_per_round_cost_gb=0.30,
                simulated_probe_accuracy=0.79,
            ),
        ]

        chosen, annotated = strategy._select_effective_candidate(
            candidates,
            planning_round=0,
        )

        planner = next(candidate for candidate in annotated if candidate["source"] == "planner_0")
        self.assertEqual(chosen["source"], "current")
        self.assertEqual(chosen["selection_reason"], "selected_current_hard_filter_fallback")
        self.assertFalse(bool(planner["hard_member_feasible"]))
        self.assertEqual(planner["selection_reason"], "rejected_hard_edge_min_members")

    def test_planning_stage_changes_across_rounds(self):
        strategy = _strategy(
            effective_planning_start_cloud_round=3,
            late_phase_start_fraction=0.8,
            kappa=10,
        )

        self.assertEqual(strategy._planning_phase(0), "paper")
        self.assertEqual(strategy._planning_phase(1), "paper")
        self.assertEqual(strategy._planning_phase(2), "mid")
        self.assertEqual(strategy._planning_phase(7), "late")

    def test_probe_emit_mode_cycle_start_only_emits_once_per_cycle(self):
        strategy = _strategy(probe_emit_mode="cycle_start")

        strategy.phase = "warmup"
        self.assertTrue(strategy._should_emit_probe_logits())

        strategy.phase = "train"
        strategy.edge_epoch = 0
        self.assertTrue(strategy._should_emit_probe_logits())

        strategy.edge_epoch = 1
        self.assertFalse(strategy._should_emit_probe_logits())

    def test_stage_replan_keeps_current_topology_in_effective_candidates(self):
        strategy = _strategy(
            planning_signal="hybrid",
            effective_planning_start_cloud_round=3,
            late_phase_start_fraction=0.8,
            effective_accuracy_delta=0.01,
            kappa=10,
        )
        strategy.selected_edges = {0, 1}
        strategy.edge_nodes = {0: [0, 1, 2], 1: [3, 4, 5]}
        strategy.edge_data_sizes = {0: 30, 1: 30}
        strategy.node_edge = _edge_nodes_to_node_edge(strategy.edge_nodes)
        strategy.per_round_cost_gb = 1.5
        strategy.c_ne = {}
        strategy.c_ec = {}
        strategy.probe_loader = object()
        strategy.model_factory = object()

        client_weights = {
            node_id: [np.array([float(node_id)], dtype=np.float32)]
            for node_id in range(6)
        }
        client_sizes = {node_id: 10 for node_id in range(6)}
        probe_logits = {
            node_id: np.zeros((1, 1), dtype=np.float32)
            for node_id in range(6)
        }
        probe_payload_bytes = {node_id: 4 for node_id in range(6)}
        planner_result = self._los_result(
            {0: [0, 1, 2], 2: [3, 4, 5]},
            objective_value=0.9,
        )

        def fake_simulate(*, candidate, **_kwargs):
            if candidate["source"] == "current":
                return {
                    "paper_per_round_cost_gb": 1.5,
                    "effective_per_round_cost_gb": 1.5,
                    "simulated_probe_accuracy": 0.81,
                    "model_payload_bytes": 0,
                    "probe_payload_bytes": 0,
                }
            return {
                "paper_per_round_cost_gb": 1.0,
                "effective_per_round_cost_gb": 1.0,
                "simulated_probe_accuracy": 0.83,
                "model_payload_bytes": 0,
                "probe_payload_bytes": 0,
            }

        with patch(
            "shapefl.strategy.compute_hybrid_phi",
            return_value=(
                {node_id: 1.0 for node_id in range(6)},
                {"lambda_dynamic": 0.5},
            ),
        ), patch(
            "shapefl.strategy.run_los_rose_candidates",
            return_value=(planner_result, []),
        ), patch(
            "shapefl.strategy.evaluate_on_probe",
            return_value=0.80,
        ), patch.object(
            strategy,
            "_simulate_candidate_cycle",
            side_effect=fake_simulate,
        ):
            strategy._plan_with_signal(
                client_weights=client_weights,
                client_sizes=client_sizes,
                probe_logits=probe_logits,
                probe_payload_bytes=probe_payload_bytes,
                reason="stage@7:late",
                planning_round=7,
            )

        event = strategy.plan_history[-1]
        candidates = {
            candidate["source"]: candidate
            for candidate in event["candidate_evaluations"]
        }
        self.assertIn("current", candidates)
        self.assertEqual(candidates["current"]["baseline_reason"], "current_topology")
        self.assertEqual(candidates["current"]["planning_phase"], "late")

    def test_delayed_compression_schedule(self):
        strategy = _strategy(
            compression_enabled=True,
            client_compression_start_cloud_round=3,
            edge_compression_start_cloud_round=4,
        )

        self.assertFalse(strategy._compression_enabled_for_cycle(cycle_cloud_round=2))
        self.assertTrue(strategy._compression_enabled_for_cycle(cycle_cloud_round=3))
        self.assertFalse(
            strategy._compression_enabled_for_cycle(
                edge_to_cloud=True,
                cycle_cloud_round=3,
            )
        )
        self.assertTrue(
            strategy._compression_enabled_for_cycle(
                edge_to_cloud=True,
                cycle_cloud_round=4,
            )
        )

    def test_fedadam_checkpoint_resume_preserves_update_trajectory(self):
        strategy = _strategy(
            server_optimizer="fedadam",
            server_lr=0.03,
            server_beta1=0.9,
            server_beta2=0.99,
            server_tau=1e-3,
        )
        reference = [np.zeros((2,), dtype=np.float32)]
        first_aggregate = [np.array([1.0, -1.0], dtype=np.float32)]
        second_aggregate = [np.array([0.5, 0.25], dtype=np.float32)]

        first_update = strategy._apply_server_optimizer(
            reference_weights=reference,
            aggregated_weights=first_aggregate,
            update_state=True,
        )
        checkpoint = strategy.get_checkpoint_state()

        resumed = _strategy(
            server_optimizer="fedadam",
            server_lr=0.03,
            server_beta1=0.9,
            server_beta2=0.99,
            server_tau=1e-3,
        )
        resumed.load_checkpoint_state(checkpoint)

        continued = strategy._apply_server_optimizer(
            reference_weights=first_update,
            aggregated_weights=second_aggregate,
            update_state=True,
        )
        resumed_continued = resumed._apply_server_optimizer(
            reference_weights=first_update,
            aggregated_weights=second_aggregate,
            update_state=True,
        )

        self.assertEqual(strategy.server_optimizer_step, resumed.server_optimizer_step)
        self.assertTrue(np.allclose(continued[0], resumed_continued[0]))


if __name__ == "__main__":
    unittest.main()
