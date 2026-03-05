from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# 复用项目已有实现
from config import Config
from core.registry import MODELS
import core.attacks
import models.resnet
from data.distribution import dirichlet_split
from federated.client import create_clients
from federated.server import Server, federated_training
from main import (
    setup_seed,
    load_dataset,
    create_test_loaders,
    build_backdoor_factory,
)


class TargetSubset(Subset):
    @property
    def targets(self):
        base_targets = self.dataset.targets
        return [base_targets[i] for i in self.indices]


def _subset_dataset(dataset, size: int | None, seed: int):
    if size is None or size <= 0 or size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=size, replace=False)
    return TargetSubset(dataset, [int(i) for i in indices])


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean_v = statistics.mean(values)
    std_v = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean_v, std_v


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.2%}"


def _fmt_mean_std(mean_v: float | None, std_v: float | None) -> str:
    if mean_v is None or std_v is None:
        return "-"
    return f"{mean_v:.2%} ± {std_v:.2%}"


def _resolve_output_dir(requested_output_dir: str) -> Path:
    """
    Ensure output directory is writable.

    On Kaggle, paths under `/kaggle/input` are read-only. If user passes a
    read-only output path, we automatically fall back to:
      /kaggle/working/results/minimum_verification
    """
    requested = Path(requested_output_dir).expanduser()

    try:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    except OSError as exc:
        requested_text = str(requested).replace("\\", "/")
        readonly_like = requested_text.startswith("/kaggle/input/") or requested_text == "/kaggle/input"

        if readonly_like or exc.errno in (13, 30):
            fallback = Path("/kaggle/working/results/minimum_verification")
            fallback.mkdir(parents=True, exist_ok=True)
            print(
                "[Warning] Output directory is read-only: "
                f"{requested}. Fallback to writable path: {fallback}"
            )
            return fallback
        raise


def _build_cfg(args, seed: int, condition: str) -> Config:
    cfg = Config()

    cfg.dataset = args.dataset
    cfg.data_dir = args.data_dir
    cfg.results_dir = args.output_dir
    cfg.weights_dir = str(Path(args.output_dir) / "weights")

    cfg.num_clients = args.num_clients
    cfg.poison_ratio = args.poison_ratio
    cfg.target_label = args.target_label
    cfg.epsilon = args.epsilon
    cfg.alpha = args.alpha
    cfg.poison_rate = args.poison_rate
    cfg.scaling_factor = args.scaling_factor

    cfg.num_rounds = args.num_rounds
    cfg.local_epochs = args.local_epochs
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.client_fraction = args.client_fraction
    cfg.num_workers = args.num_workers
    cfg.pin_memory = args.pin_memory
    cfg.seed = seed

    cfg.defense_method = args.defense_method
    cfg.model_name = args.model_name

    if condition == "anb_freqfed":
        cfg.freq_strategy = "ANB"
        cfg.defense_enabled = True
        cfg.use_phased_chaos = True
        cfg.use_spectral_smoothing = True
        cfg.use_freq_sharding = True
        cfg.use_dual_routing = True
    elif condition == "fixed_freqfed":
        cfg.freq_strategy = "FIXED"
        cfg.defense_enabled = True
        cfg.use_phased_chaos = False
        cfg.use_spectral_smoothing = False
        cfg.use_freq_sharding = False
        cfg.use_dual_routing = False
    elif condition == "anb_no_defense":
        cfg.freq_strategy = "ANB"
        cfg.defense_enabled = False
        cfg.use_phased_chaos = True
        cfg.use_spectral_smoothing = True
        cfg.use_freq_sharding = True
        cfg.use_dual_routing = True
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return cfg


def run_one_condition(args, condition: str, seed: int) -> dict[str, Any]:
    cfg = _build_cfg(args, seed=seed, condition=condition)
    setup_seed(cfg.seed)

    train_dataset, test_dataset, num_classes = load_dataset(cfg.dataset, cfg.data_dir)
    train_dataset = _subset_dataset(train_dataset, args.train_subset, seed=cfg.seed)
    test_dataset = _subset_dataset(test_dataset, args.test_subset, seed=cfg.seed + 1)

    num_malicious = max(1, int(cfg.num_clients * cfg.poison_ratio))
    malicious_indices = list(range(num_malicious))

    client_indices = dirichlet_split(train_dataset, num_clients=cfg.num_clients, alpha=cfg.alpha)
    backdoor_factory = build_backdoor_factory(cfg)

    clients = create_clients(
        dataset=train_dataset,
        num_clients=cfg.num_clients,
        malicious_indices=malicious_indices,
        client_indices=client_indices,
        target_label=cfg.target_label,
        epsilon=cfg.epsilon,
        freq_strategy=cfg.freq_strategy,
        batch_size=cfg.batch_size,
        local_epochs=cfg.local_epochs,
        lr=cfg.learning_rate,
        dataset_name=cfg.dataset,
        backdoor_factory=backdoor_factory,
        poison_rate=cfg.poison_rate,
        scaling_factor=cfg.scaling_factor,
    )

    model_builder = MODELS.get(cfg.model_name)
    global_model = model_builder(num_classes=num_classes)

    clean_loader, poisoned_loader, multi_loader, per_client_loaders = create_test_loaders(
        test_dataset,
        cfg.target_label,
        cfg.epsilon,
        cfg.freq_strategy,
        malicious_indices,
        cfg.batch_size,
        dataset_name=cfg.dataset,
        backdoor_factory=backdoor_factory,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    server = Server(
        model=global_model,
        device=device,
        defense_enabled=cfg.defense_enabled,
        defense_method=cfg.defense_method,
        target_label=cfg.target_label,
    )

    federated_training(
        server=server,
        clients=clients,
        test_loader=clean_loader,
        poisoned_test_loader=poisoned_loader,
        multi_trigger_loader=multi_loader,
        per_client_loaders=per_client_loaders,
        num_rounds=cfg.num_rounds,
        malicious_indices=malicious_indices,
        client_fraction=cfg.client_fraction,
        save_weights_at_rounds=[],
    )

    h = server.history
    result = {
        "condition": condition,
        "seed": seed,
        "acc_final": h["test_acc"][-1] if h["test_acc"] else None,
        "asr_single_final": h["test_asr"][-1] if h["test_asr"] else None,
        "asr_multi_final": h["test_asr_multi"][-1] if h["test_asr_multi"] else None,
        "bypass_final": h["defense_bypass_rate"][-1] if h["defense_bypass_rate"] else None,
        "recall_final": h["defense_recall"][-1] if h["defense_recall"] else None,
        "precision_final": h["defense_precision"][-1] if h["defense_precision"] else None,
        "f1_final": h["defense_f1"][-1] if h["defense_f1"] else None,
        "rounds_recorded": len(h["test_acc"]),
    }

    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_condition.setdefault(item["condition"], []).append(item)

    summary_rows = []
    for condition, rows in sorted(by_condition.items()):
        acc_vals = [r["acc_final"] for r in rows if r["acc_final"] is not None]
        asr_vals = [r["asr_single_final"] for r in rows if r["asr_single_final"] is not None]
        bypass_vals = [r["bypass_final"] for r in rows if r["bypass_final"] is not None]

        acc_mean, acc_std = _mean_std(acc_vals)
        asr_mean, asr_std = _mean_std(asr_vals)
        bypass_mean, bypass_std = _mean_std(bypass_vals)

        summary_rows.append(
            {
                "condition": condition,
                "num_runs": len(rows),
                "acc_mean": acc_mean,
                "acc_std": acc_std,
                "asr_mean": asr_mean,
                "asr_std": asr_std,
                "bypass_mean": bypass_mean,
                "bypass_std": bypass_std,
                "seeds": sorted([r["seed"] for r in rows]),
            }
        )

    lookup = {row["condition"]: row for row in summary_rows}
    fixed = lookup.get("fixed_freqfed")
    anb = lookup.get("anb_freqfed")

    hypothesis = {
        "description": "ANB+FreqFed 相比 FIXED+FreqFed 在 ASR 与 Bypass 上更高",
        "asr_gain": None,
        "bypass_gain": None,
        "supported": None,
    }
    if fixed and anb and fixed["asr_mean"] is not None and anb["asr_mean"] is not None:
        asr_gain = anb["asr_mean"] - fixed["asr_mean"]
        bypass_gain = None
        if fixed["bypass_mean"] is not None and anb["bypass_mean"] is not None:
            bypass_gain = anb["bypass_mean"] - fixed["bypass_mean"]

        hypothesis["asr_gain"] = asr_gain
        hypothesis["bypass_gain"] = bypass_gain
        hypothesis["supported"] = (
            asr_gain > 0.0 and (bypass_gain is None or bypass_gain >= 0.0)
        )

    return {
        "summary_rows": summary_rows,
        "hypothesis_check": hypothesis,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print("Minimum Verification Summary")
    print("=" * 88)
    print(f"{'Condition':<20} {'Runs':>4} {'ACC(mean±std)':>22} {'ASR(mean±std)':>22} {'Bypass(mean±std)':>22}")
    print("-" * 88)
    for row in summary["summary_rows"]:
        print(
            f"{row['condition']:<20} {row['num_runs']:>4} "
            f"{_fmt_mean_std(row['acc_mean'], row['acc_std']):>22} "
            f"{_fmt_mean_std(row['asr_mean'], row['asr_std']):>22} "
            f"{_fmt_mean_std(row['bypass_mean'], row['bypass_std']):>22}"
        )

    h = summary["hypothesis_check"]
    print("-" * 88)
    print(f"Hypothesis: {h['description']}")
    print(f"ASR gain (ANB - FIXED): {_fmt_pct(h['asr_gain']) if h['asr_gain'] is not None else '-'}")
    print(f"Bypass gain (ANB - FIXED): {_fmt_pct(h['bypass_gain']) if h['bypass_gain'] is not None else '-'}")
    print(f"Supported: {h['supported']}")
    print("=" * 88)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimum verification for ANB method (Kaggle-friendly)",
    )
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/minimum_verification")

    parser.add_argument("--num-rounds", type=int, default=30)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--client-fraction", type=float, default=1.0)

    parser.add_argument("--poison-ratio", type=float, default=0.2)
    parser.add_argument("--poison-rate", type=float, default=0.9)
    parser.add_argument("--scaling-factor", type=float, default=4.5)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--model-name", type=str, default="resnet18")

    parser.add_argument("--train-subset", type=int, default=6000)
    parser.add_argument("--test-subset", type=int, default=1200)

    parser.add_argument("--defense-method", type=str, default="hdbscan")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["fixed_freqfed", "anb_freqfed"],
        choices=["fixed_freqfed", "anb_freqfed", "anb_no_defense"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    args.pin_memory = bool(args.pin_memory)

    output_dir = _resolve_output_dir(args.output_dir)
    args.output_dir = str(output_dir)

    print("\nPlanned runs:")
    for condition in args.conditions:
        for seed in args.seeds:
            print(
                f"  - condition={condition}, seed={seed}, rounds={args.num_rounds}, "
                f"poison_rate={args.poison_rate}, scaling={args.scaling_factor}, epochs={args.local_epochs}"
            )

    if args.dry_run:
        print("\nDry-run enabled. No training executed.")
        return

    all_results: list[dict[str, Any]] = []

    for condition in args.conditions:
        for seed in args.seeds:
            print("\n" + "#" * 88)
            print(f"Running condition={condition}, seed={seed}")
            print("#" * 88)
            result = run_one_condition(args, condition=condition, seed=seed)
            all_results.append(result)

            print(
                f"Result -> ACC={_fmt_pct(result['acc_final'])}, "
                f"ASR={_fmt_pct(result['asr_single_final'])}, "
                f"Bypass={_fmt_pct(result['bypass_final'])}"
            )

    summary = summarize(all_results)
    print_summary(summary)

    payload = {
        "args": vars(args),
        "config_template": asdict(Config()),
        "runs": all_results,
        "summary": summary,
    }

    summary_path = output_dir / "minimum_verification_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
