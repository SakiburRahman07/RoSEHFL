"""
Shared strategy construction helpers for research and deployment scripts.
"""

from __future__ import annotations

from rosehfl.strategy import (
    FedAvgFlatStrategy,
    FedProxFlatStrategy,
    RoSEHFLStrategy,
    ShapeFlStrategy,
    generate_communication_costs,
)

from .baselines.gtg_shapley import GTGShapleyFlatStrategy
from .baselines.q_fedavg import QFedAvgFlatStrategy


DEFAULT_TARGET_ACCURACY = {
    "fmnist": 0.70,
    "cifar10": 0.40,
    "cifar100": 0.20,
}


def default_target_accuracy(dataset_name: str) -> float:
    return float(DEFAULT_TARGET_ACCURACY[dataset_name])


def _arg(args, name: str, default):
    return getattr(args, name, default)


def _edge_underfill_penalty(args) -> float:
    value = getattr(args, "edge_underfill_penalty", None)
    return -1.0 if value is None else float(value)


def _flat_strategy_comm_costs(args, shared):
    model_size_bytes = sum(weights.nbytes for weights in shared["initial_ndarrays"])
    _, c_ec = generate_communication_costs(
        args.num_nodes,
        model_size_bytes,
        topology=args.topology,
    )
    return c_ec


def build_strategy(name: str, args, shared, output_dir: str):
    target_accuracy = _arg(args, "target_accuracy", None)
    if target_accuracy is None:
        target_accuracy = default_target_accuracy(args.dataset)

    if name == "rose":
        return RoSEHFLStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            warmup_epochs=_arg(args, "warmup_epochs", 1),
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma_max=args.gamma_max,
            gamma_anneal=_arg(args, "gamma_anneal", "cosine"),
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            topology=args.topology,
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=_arg(args, "total_local_epochs", None),
            probe_loader=shared["probe_loader"],
            model_factory=shared["model_factory"],
            server_device=shared["server_device"],
            output_dir=output_dir,
            seed=args.seed,
            shapley_T=_arg(args, "shapley_T", 4),
            shapley_K=_arg(args, "shapley_K", 6),
            planning_signal=_arg(args, "planning_signal", "shapley"),
            probe_size=_arg(args, "probe_size", 1000),
            dp_epsilon=_arg(args, "dp_epsilon", 0.0),
            dp_delta=_arg(args, "dp_delta", 1e-5),
            agg_rule=_arg(args, "agg_rule", "trust"),
            agg_trim_ratio=_arg(args, "agg_trim_ratio", 0.2),
            krum_f=_arg(args, "krum_f", 1),
            drift_enabled=not _arg(args, "disable_drift", False),
            drift_delta=_arg(args, "drift_delta", 1e-3),
            drift_lambda=_arg(args, "drift_lambda", 0.5),
            max_replans=_arg(args, "max_replans", 8),
        )
    if name == "roseplusplus":
        return RoSEHFLStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            warmup_epochs=_arg(args, "warmup_epochs", 1),
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma_max=args.gamma_max,
            gamma_min=1400.0,
            gamma_anneal="adaptive",
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            topology=args.topology,
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=_arg(args, "total_local_epochs", None),
            probe_loader=shared["probe_loader"],
            model_factory=shared["model_factory"],
            server_device=shared["server_device"],
            output_dir=output_dir,
            seed=args.seed,
            shapley_T=1,
            shapley_K=64,
            planning_signal="hybrid",
            probe_size=_arg(args, "probe_size", 1000),
            agg_rule="trust",
            trust_use_shrinkage=True,
            adaptive_gamma_eta=0.5,
            adaptive_gamma_target=0.25,
            warm_start_replan=True,
            warm_start_threshold=0.05,
            replan_cost_increase_tolerance=0.1,
            local_objective_prox_mu=_arg(args, "fedprox_mu", 0.01),
            logit_adjustment_tau=1.0,
            local_bn=True,
            edge_swa_k=3,
        )
    if name == "rose_q1":
        return RoSEHFLStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            warmup_epochs=_arg(args, "warmup_epochs", 1),
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma_max=args.gamma_max,
            gamma_min=1400.0,
            gamma_anneal="adaptive",
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            topology=args.topology,
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=_arg(args, "total_local_epochs", None),
            probe_loader=shared["probe_loader"],
            model_factory=shared["model_factory"],
            server_device=shared["server_device"],
            output_dir=output_dir,
            seed=args.seed,
            shapley_T=1,
            shapley_K=64,
            planning_signal="hybrid",
            probe_size=_arg(args, "probe_size", 1000),
            agg_rule="trust",
            trust_use_shrinkage=True,
            adaptive_gamma_eta=0.5,
            adaptive_gamma_target=0.25,
            warm_start_replan=True,
            warm_start_threshold=0.05,
            replan_cost_increase_tolerance=0.1,
            compression_enabled=True,
            compression_keep_ratio_min=_arg(args, "compression_keep_ratio_min", 0.05),
            compression_keep_ratio_max=_arg(args, "compression_keep_ratio_max", 0.25),
            compression_eta=_arg(args, "compression_eta", 1.0),
            compression_target_deficit=_arg(args, "compression_target_deficit", 0.25),
            compress_edge_to_cloud=not _arg(args, "disable_edge_to_cloud_compression", False),
            edge_min_members=_arg(args, "edge_min_members", 2),
            edge_underfill_penalty=_edge_underfill_penalty(args),
            local_objective_prox_mu=_arg(args, "fedprox_mu", 0.01),
            logit_adjustment_tau=1.0,
            local_bn=True,
            edge_swa_k=3,
        )
    if name == "rose_q1s":
        return RoSEHFLStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            warmup_epochs=3,
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma_max=args.gamma_max,
            gamma_min=1400.0,
            gamma_anneal="adaptive",
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            topology=args.topology,
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=_arg(args, "total_local_epochs", None),
            probe_loader=shared["probe_loader"],
            model_factory=shared["model_factory"],
            server_device=shared["server_device"],
            output_dir=output_dir,
            seed=args.seed,
            shapley_T=2,
            shapley_K=64,
            planning_signal="hybrid",
            probe_size=_arg(args, "probe_size", 1000),
            agg_rule="trust",
            trust_use_shrinkage=True,
            adaptive_gamma_eta=0.5,
            adaptive_gamma_target=0.25,
            warm_start_replan=True,
            warm_start_threshold=0.05,
            replan_cost_increase_tolerance=0.1,
            compression_enabled=True,
            compression_keep_ratio_min=0.15,
            compression_keep_ratio_max=0.30,
            compression_eta=_arg(args, "compression_eta", 1.0),
            compression_target_deficit=_arg(args, "compression_target_deficit", 0.25),
            compress_edge_to_cloud=not _arg(args, "disable_edge_to_cloud_compression", False),
            edge_min_members=max(_arg(args, "edge_min_members", 2), 3),
            edge_underfill_penalty=_edge_underfill_penalty(args),
            local_objective_prox_mu=_arg(args, "fedprox_mu", 0.01),
            logit_adjustment_tau=1.0,
            local_bn=True,
            edge_swa_k=3,
            planning_objective="effective",
            target_accuracy=target_accuracy,
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
    if name == "rose_effective":
        return RoSEHFLStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            warmup_epochs=_arg(args, "warmup_epochs", 1),
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma_max=args.gamma_max,
            gamma_min=1400.0,
            gamma_anneal="adaptive",
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            topology=args.topology,
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=_arg(args, "total_local_epochs", None),
            probe_loader=shared["probe_loader"],
            model_factory=shared["model_factory"],
            server_device=shared["server_device"],
            output_dir=output_dir,
            seed=args.seed,
            shapley_T=1,
            shapley_K=64,
            planning_signal="hybrid",
            probe_size=_arg(args, "probe_size", 1000),
            agg_rule="trust",
            trust_use_shrinkage=True,
            adaptive_gamma_eta=0.5,
            adaptive_gamma_target=0.25,
            warm_start_replan=True,
            warm_start_threshold=0.05,
            replan_cost_increase_tolerance=0.1,
            compression_enabled=True,
            compression_keep_ratio_min=_arg(args, "compression_keep_ratio_min", 0.05),
            compression_keep_ratio_max=_arg(args, "compression_keep_ratio_max", 0.25),
            compression_eta=_arg(args, "compression_eta", 1.0),
            compression_target_deficit=_arg(args, "compression_target_deficit", 0.25),
            compress_edge_to_cloud=not _arg(args, "disable_edge_to_cloud_compression", False),
            edge_min_members=_arg(args, "edge_min_members", 2),
            edge_underfill_penalty=_edge_underfill_penalty(args),
            local_objective_prox_mu=_arg(args, "fedprox_mu", 0.01),
            logit_adjustment_tau=1.0,
            local_bn=True,
            edge_swa_k=3,
            planning_objective="effective",
            target_accuracy=target_accuracy,
            accuracy_guard_tolerance=0.02,
        )
    if name == "rose_median":
        strategy = build_strategy("rose", args, shared, output_dir)
        strategy.agg_rule = "median"
        return strategy
    if name == "rose_trimmed_mean":
        strategy = build_strategy("rose", args, shared, output_dir)
        strategy.agg_rule = "trimmed_mean"
        return strategy
    if name == "rose_krum":
        strategy = build_strategy("rose", args, shared, output_dir)
        strategy.agg_rule = "krum"
        return strategy
    if name == "shapefl":
        return ShapeFlStrategy(
            model_name=args.model,
            dataset_name=args.dataset,
            num_nodes=args.num_nodes,
            kappa_p=_arg(args, "kappa_p", 30),
            kappa_e=args.kappa_e,
            kappa_c=args.kappa_c,
            kappa=args.kappa,
            gamma=args.gamma_max,
            B_e=args.B_e,
            T_max=args.T_max,
            lr=args.lr,
            momentum=args.momentum,
            initial_parameters=shared["initial_parameters"],
            planning_mode="shapefl",
            topology=args.topology,
            evaluate_fn=shared["evaluate_fn"],
            node_label_counts=shared["node_label_counts"],
            total_local_epochs=_arg(args, "total_local_epochs", None),
        )
    if name == "fedavg":
        strategy = FedAvgFlatStrategy(
            num_nodes=args.num_nodes,
            kappa=args.kappa,
            local_epochs=args.kappa_c * args.kappa_e,
            lr=args.lr,
            momentum=args.momentum,
            total_local_epochs=_arg(args, "total_local_epochs", None),
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
        )
        strategy.set_comm_costs(_flat_strategy_comm_costs(args, shared))
        return strategy
    if name == "fedprox":
        strategy = FedProxFlatStrategy(
            num_nodes=args.num_nodes,
            kappa=args.kappa,
            local_epochs=args.kappa_c * args.kappa_e,
            lr=args.lr,
            momentum=args.momentum,
            prox_mu=_arg(args, "fedprox_mu", 0.01),
            total_local_epochs=_arg(args, "total_local_epochs", None),
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
        )
        strategy.set_comm_costs(_flat_strategy_comm_costs(args, shared))
        return strategy
    if name == "gtg_shapley":
        strategy = GTGShapleyFlatStrategy(
            num_nodes=args.num_nodes,
            kappa=args.kappa,
            local_epochs=args.kappa_c * args.kappa_e,
            lr=args.lr,
            momentum=args.momentum,
            total_local_epochs=_arg(args, "total_local_epochs", None),
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            probe_loader=shared["probe_loader"],
            model_factory=shared["model_factory"],
            server_device=shared["server_device"],
            shapley_T=_arg(args, "shapley_T", 4),
            shapley_K=_arg(args, "shapley_K", 6),
            seed=args.seed,
        )
        strategy.set_comm_costs(_flat_strategy_comm_costs(args, shared))
        return strategy
    if name == "q_fedavg":
        strategy = QFedAvgFlatStrategy(
            num_nodes=args.num_nodes,
            kappa=args.kappa,
            local_epochs=args.kappa_c * args.kappa_e,
            lr=args.lr,
            momentum=args.momentum,
            total_local_epochs=_arg(args, "total_local_epochs", None),
            initial_parameters=shared["initial_parameters"],
            evaluate_fn=shared["evaluate_fn"],
            q=_arg(args, "q_fedavg_q", 2.0),
        )
        strategy.set_comm_costs(_flat_strategy_comm_costs(args, shared))
        return strategy
    raise ValueError(f"Unknown strategy: {name}")
