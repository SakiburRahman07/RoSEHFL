import json
import os
import tempfile
import unittest

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters
from torch.utils.data import DataLoader, Dataset

from rosehfl.client import client_fn_factory
from rosehfl.data.data_loader import get_partition_label_counts
from rosehfl.strategy import RoSEHFLStrategy
from rosehfl.utils.seed import set_seed
from rosehfl.utils.shapley import build_probe_set


class ToyDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.targets = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class RoSESmokeTests(unittest.TestCase):
    def test_resume_after_warmup(self):
        set_seed(7)
        train_x = torch.rand(24, 1, 28, 28)
        train_y = torch.tensor([0, 1, 2, 3] * 6, dtype=torch.long)
        test_x = torch.rand(12, 1, 28, 28)
        test_y = torch.tensor([0, 1, 2, 3] * 3, dtype=torch.long)
        train_dataset = ToyDataset(train_x, train_y)
        test_dataset = ToyDataset(test_x, test_y)
        partitions = {0: list(range(0, 8)), 1: list(range(8, 16)), 2: list(range(16, 24))}
        node_label_counts = get_partition_label_counts(train_dataset, partitions, num_classes=10)
        probe_subset = build_probe_set(test_dataset, probe_size=8, num_classes=10, seed=7)
        probe_indices = list(probe_subset.indices)
        probe_loader = DataLoader(probe_subset, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        def model_factory():
            from rosehfl.models.lenet5 import LeNet5

            return LeNet5(num_classes=10, input_channels=1)

        def evaluate_fn(server_round, parameters_ndarrays, config):
            model = model_factory()
            state_dict = {
                key: torch.tensor(value)
                for key, value in zip(model.state_dict().keys(), parameters_ndarrays)
            }
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            criterion = torch.nn.CrossEntropyLoss()
            total_loss, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for features, labels in test_loader:
                    outputs = model(features)
                    total_loss += criterion(outputs, labels).item() * labels.size(0)
                    correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total += labels.size(0)
            return total_loss / max(total, 1), {"accuracy": correct / max(total, 1)}

        initial_model = model_factory()
        initial_parameters = ndarrays_to_parameters(
            [value.detach().cpu().numpy() for _, value in initial_model.state_dict().items()]
        )
        client_fn = client_fn_factory(
            "lenet5",
            "fmnist",
            train_dataset,
            test_dataset,
            partitions,
            batch_size=4,
            device="cpu",
            probe_indices=probe_indices,
            seed=7,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            strategy = RoSEHFLStrategy(
                model_name="lenet5",
                dataset_name="fmnist",
                num_nodes=3,
                warmup_epochs=3,
                kappa_e=1,
                kappa_c=1,
                kappa=4,
                gamma_max=10.0,
                B_e=3,
                T_max=3,
                lr=0.01,
                momentum=0.0,
                initial_parameters=initial_parameters,
                evaluate_fn=evaluate_fn,
                topology="geant2010",
                node_label_counts=node_label_counts,
                probe_loader=probe_loader,
                model_factory=model_factory,
                server_device="cpu",
                output_dir=tmpdir,
                seed=7,
                shapley_T=2,
                shapley_K=2,
                probe_size=8,
                compression_enabled=True,
                planning_objective="effective",
                target_accuracy=0.4,
                accuracy_guard_tolerance=0.02,
                effective_planning_start_cloud_round=3,
                late_phase_start_fraction=0.8,
                effective_accuracy_delta=0.01,
                probe_emit_mode="cycle_start",
                client_compression_start_cloud_round=3,
                edge_compression_start_cloud_round=4,
                server_optimizer="fedadam",
                server_lr=0.03,
                server_beta1=0.9,
                server_beta2=0.99,
                server_tau=1e-3,
                hard_edge_min_members=3,
            )

            fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=3,
                config=fl.server.ServerConfig(num_rounds=1),
                strategy=strategy,
                client_resources={"num_cpus": 1},
            )

            checkpoint_path = os.path.join(tmpdir, "checkpoint.pkl")
            self.assertTrue(os.path.isfile(checkpoint_path))

            import pickle

            with open(checkpoint_path, "rb") as handle:
                checkpoint = pickle.load(handle)

            resumed = RoSEHFLStrategy(
                model_name="lenet5",
                dataset_name="fmnist",
                num_nodes=3,
                warmup_epochs=3,
                kappa_e=1,
                kappa_c=1,
                kappa=4,
                gamma_max=10.0,
                B_e=3,
                T_max=3,
                lr=0.01,
                momentum=0.0,
                initial_parameters=initial_parameters,
                evaluate_fn=evaluate_fn,
                topology="geant2010",
                node_label_counts=node_label_counts,
                probe_loader=probe_loader,
                model_factory=model_factory,
                server_device="cpu",
                output_dir=tmpdir,
                seed=7,
                shapley_T=2,
                shapley_K=2,
                probe_size=8,
                compression_enabled=True,
                planning_objective="effective",
                target_accuracy=0.4,
                accuracy_guard_tolerance=0.02,
                effective_planning_start_cloud_round=3,
                late_phase_start_fraction=0.8,
                effective_accuracy_delta=0.01,
                probe_emit_mode="cycle_start",
                client_compression_start_cloud_round=3,
                edge_compression_start_cloud_round=4,
                server_optimizer="fedadam",
                server_lr=0.03,
                server_beta1=0.9,
                server_beta2=0.99,
                server_tau=1e-3,
                hard_edge_min_members=3,
            )
            resumed.load_checkpoint_state(checkpoint)
            self.assertEqual(resumed.remaining_flower_rounds, 4)

            fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=3,
                config=fl.server.ServerConfig(num_rounds=resumed.remaining_flower_rounds),
                strategy=resumed,
                client_resources={"num_cpus": 1},
            )

            for filename in [
                "metrics.json",
                "plan.json",
                "shapley_history.json",
                "privacy.json",
                "status.json",
                "checkpoint.pkl",
            ]:
                self.assertTrue(os.path.isfile(os.path.join(tmpdir, filename)))

            with open(os.path.join(tmpdir, "status.json"), "r", encoding="utf-8") as handle:
                status = json.load(handle)
            self.assertTrue(status["completed"])

            with open(os.path.join(tmpdir, "metrics.json"), "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.assertTrue(metrics["effective_per_round_cost_gb"])
            self.assertTrue(metrics["model_payload_bytes"])
            self.assertGreater(sum(metrics["model_payload_bytes"]), 0)
            self.assertLess(
                metrics["effective_per_round_cost_gb"][-1],
                metrics["paper_per_round_cost_gb"][-1],
            )

            with open(os.path.join(tmpdir, "plan.json"), "r", encoding="utf-8") as handle:
                plan = json.load(handle)
            self.assertEqual(plan["planning_objective"], "effective")
            self.assertAlmostEqual(float(plan["accuracy_guard_tolerance"]), 0.02, places=6)
            self.assertEqual(int(plan["effective_planning_start_cloud_round"]), 3)
            self.assertEqual(plan["probe_emit_mode"], "cycle_start")
            self.assertEqual(plan["server_optimizer"], "fedadam")
            self.assertEqual(int(plan["hard_edge_min_members"]), 3)
            self.assertTrue(plan["plan_history"])
            first_event = plan["plan_history"][0]
            self.assertEqual(first_event["planning_objective"], "effective")
            self.assertEqual(first_event["planning_objective_active"], "paper")
            effective_events = [
                event for event in plan["plan_history"]
                if event.get("planning_objective_active") == "effective"
            ]
            self.assertTrue(effective_events)
            self.assertTrue(any(event.get("candidate_evaluations") for event in effective_events))
            self.assertTrue(any("selected_plan_source" in event for event in effective_events))

            with open(os.path.join(tmpdir, "privacy.json"), "r", encoding="utf-8") as handle:
                privacy = json.load(handle)
            self.assertEqual(privacy["probe_emit_mode"], "cycle_start")
            self.assertEqual(int(privacy["client_compression_start_cloud_round"]), 3)
            self.assertEqual(int(privacy["edge_compression_start_cloud_round"]), 4)
            self.assertEqual(privacy["server_optimizer"], "fedadam")


if __name__ == "__main__":
    unittest.main()
