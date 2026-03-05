"""
Cross-Dataset Generalization Experiment  [P3-4]

Runs the three core conditions on CIFAR-100 to verify that ANB's defense-evasion
capability generalises beyond CIFAR-10.

Conditions (matching Table 1 structure in the thesis):
    1. ANB  — no defense          (upper bound ASR)
    2. FIXED — FreqFed defense    (baseline, should be detected)
    3. ANB   — FreqFed defense    (ours, should evade detection)

Results are saved to:
    ./results/cifar100/cifar100_results.json
    ./results/cifar100/cifar100_summary.png

Usage (GPU recommended; CPU run is slow but correct):
    python analysis/cifar100_experiment.py
    python analysis/cifar100_experiment.py --num-rounds 30   # quick smoke test
    python analysis/cifar100_experiment.py --plot-only       # re-plot saved JSON
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.registry import ATTACKS, MODELS
import core.attacks   # register
import models.resnet  # register
from data.distribution import dirichlet_split
from main import (
    setup_seed, load_dataset, create_test_loaders,
    build_backdoor_factory, print_experiment_config
)
from federated.client import create_clients
from federated.server import Server, federated_training

OUT_DIR = './results/cifar100'
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment conditions
# ---------------------------------------------------------------------------

CONDITIONS = [
    {
        'name':            'ANB_NoDefense',
        'label':           'ANB (no defense)',
        'freq_strategy':   'ANB',
        'defense_enabled': False,
        'defense_method':  'hdbscan',
    },
    {
        'name':            'FIXED_FreqFed',
        'label':           'FIXED + FreqFed',
        'freq_strategy':   'FIXED',
        'defense_enabled': True,
        'defense_method':  'hdbscan',
    },
    {
        'name':            'ANB_FreqFed',
        'label':           'ANB + FreqFed (ours)',
        'freq_strategy':   'ANB',
        'defense_enabled': True,
        'defense_method':  'hdbscan',
    },
]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_condition(cfg: Config) -> dict:
    """
    Run one complete federated experiment and return final metrics.

    Returns:
        dict with keys: asr, acc, defense_recall, defense_bypass_rate
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(cfg.seed)

    train_dataset, test_dataset, num_classes = load_dataset(cfg.dataset, cfg.data_dir)

    num_malicious    = max(1, int(cfg.num_clients * cfg.poison_ratio))
    malicious_indices = list(range(num_malicious))

    client_indices   = dirichlet_split(train_dataset,
                                       num_clients=cfg.num_clients,
                                       alpha=cfg.alpha)
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
        scaling_factor=cfg.scaling_factor,
    )

    model_builder = MODELS.get(cfg.model_name)
    global_model  = model_builder(num_classes=num_classes)

    clean_loader, poisoned_loader, multi_loader, per_client_loaders = create_test_loaders(
        test_dataset, cfg.target_label, cfg.epsilon, cfg.freq_strategy,
        malicious_indices, cfg.batch_size, dataset_name=cfg.dataset,
        backdoor_factory=backdoor_factory,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
    )

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
    return {
        'asr':            h['test_asr'][-1]          if h['test_asr']          else 0.0,
        'acc':            h['test_acc'][-1]           if h['test_acc']          else 0.0,
        'asr_multi':      h['test_asr_multi'][-1]     if h['test_asr_multi']    else None,
        'defense_recall': h['defense_recall'][-1]     if h['defense_recall']    else None,
        'defense_bypass': h['defense_bypass_rate'][-1]
                          if h['defense_bypass_rate'] else None,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary(results: list, save_path: str):
    """
    Bar chart comparing ASR and ACC across the three conditions.
    Also shows defense bypass rate where applicable.
    """
    labels  = [r['label'] for r in results]
    asrs    = [r['metrics'].get('asr', 0) * 100 for r in results]
    accs    = [r['metrics'].get('acc', 0) * 100 for r in results]
    bypasses = [
        r['metrics'].get('defense_bypass', float('nan')) or float('nan')
        for r in results
    ]
    bypasses_pct = [b * 100 if not np.isnan(b) else float('nan') for b in bypasses]

    x     = np.arange(len(labels))
    width = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: ASR
    bars = axes[0].bar(x, asrs, width * 2, color=['#e74c3c', '#3498db', '#2ecc71'],
                       alpha=0.85, edgecolor='black', linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10, rotation=12, ha='right')
    axes[0].set_ylabel('Attack Success Rate (%)', fontsize=12)
    axes[0].set_title('ASR Comparison (CIFAR-100)', fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 110)
    axes[0].axhline(85, color='gray', linestyle='--', alpha=0.6, label='85% target')
    axes[0].grid(axis='y', alpha=0.35)
    axes[0].legend(fontsize=9)
    for bar, v in zip(bars, asrs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                     f'{v:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Right: ACC + bypass
    bars_acc = axes[1].bar(x - width / 2, accs, width, color=['#e74c3c', '#3498db', '#2ecc71'],
                           alpha=0.85, edgecolor='black', linewidth=0.8, label='Clean ACC')
    # Bypass rate (only for defense conditions)
    bypass_vals = [b if not np.isnan(b) else 0 for b in bypasses_pct]
    bars_byp = axes[1].bar(x + width / 2, bypass_vals, width, color='#f39c12',
                           alpha=0.75, edgecolor='black', linewidth=0.8, label='Defense Bypass %')

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10, rotation=12, ha='right')
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_title('Clean ACC & Defense Bypass (CIFAR-100)', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis='y', alpha=0.35)
    axes[1].legend(fontsize=9)

    plt.suptitle('Cross-Dataset Generalization: CIFAR-100\n'
                 '(ANB backdoor attack vs FreqFed defense)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[P3-4] Summary figure saved → {save_path}')


def print_table(results: list):
    """Print a LaTeX-ready text table of results."""
    print('\n' + '=' * 72)
    print(f'{"Method":<30} {"ASR (↑)":>10} {"ACC":>10} {"DR (↓)":>10} {"Bypass (↑)":>12}')
    print('-' * 72)
    for r in results:
        m = r['metrics']
        asr     = f"{m.get('asr', 0):.1%}"         if m.get('asr')             is not None else '-'
        acc     = f"{m.get('acc', 0):.1%}"         if m.get('acc')             is not None else '-'
        dr      = f"{m.get('defense_recall', 0):.1%}" if m.get('defense_recall') is not None else '-'
        bypass  = f"{m.get('defense_bypass', 0):.1%}" if m.get('defense_bypass') is not None else '-'
        print(f'{r["label"]:<30} {asr:>10} {acc:>10} {dr:>10} {bypass:>12}')
    print('=' * 72)
    print('DR = Defense Recall (fraction of malicious clients detected)')
    print('Bypass = fraction of malicious clients that evaded detection\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='P3-4: Cross-Dataset Generalization (CIFAR-100)'
    )
    parser.add_argument('--num-rounds', type=int, default=None,
                        help='Override number of training rounds (default: Config default)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training; re-plot from saved JSON')
    parser.add_argument('--results-dir', type=str, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    global OUT_DIR
    OUT_DIR = args.results_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    json_path = os.path.join(OUT_DIR, 'cifar100_results.json')
    png_path  = os.path.join(OUT_DIR, 'cifar100_summary.png')

    if args.plot_only:
        if not os.path.exists(json_path):
            print(f'[P3-4] No saved results found at {json_path}. Run without --plot-only first.')
            return
        with open(json_path) as f:
            all_results = json.load(f)
        print_table(all_results)
        plot_summary(all_results, png_path)
        return

    # Build base config for CIFAR-100
    base_cfg            = Config()
    base_cfg.dataset    = 'CIFAR100'
    base_cfg.num_classes = 100
    if args.num_rounds is not None:
        base_cfg.num_rounds = args.num_rounds

    all_results = []

    for cond in CONDITIONS:
        print(f"\n{'='*65}")
        print(f"[P3-4] Condition: {cond['label']}")
        print(f"{'='*65}")

        cfg = copy.copy(base_cfg)
        cfg.freq_strategy   = cond['freq_strategy']
        cfg.defense_enabled = cond['defense_enabled']
        cfg.defense_method  = cond['defense_method']
        # For FIXED baseline, fall back to standard (non-adaptive) attack
        if cond['freq_strategy'] == 'FIXED':
            cfg.use_phased_chaos       = False
            cfg.use_spectral_smoothing = False
            cfg.use_freq_sharding      = False
            cfg.use_dual_routing       = False

        print_experiment_config(cfg)

        try:
            metrics = run_condition(cfg)
        except Exception as e:
            print(f'[WARNING] Condition "{cond["label"]}" failed: {e}')
            metrics = {'asr': None, 'acc': None, 'asr_multi': None,
                       'defense_recall': None, 'defense_bypass': None,
                       'error': str(e)}

        all_results.append({
            'name':    cond['name'],
            'label':   cond['label'],
            'config':  {
                'dataset':        cfg.dataset,
                'freq_strategy':  cfg.freq_strategy,
                'defense_enabled': cfg.defense_enabled,
                'defense_method': cfg.defense_method,
                'num_rounds':     cfg.num_rounds,
            },
            'metrics': metrics,
        })

        print(f"  → ASR={metrics.get('asr')}, ACC={metrics.get('acc')}, "
              f"Bypass={metrics.get('defense_bypass')}")

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[P3-4] Results saved → {json_path}')

    print_table(all_results)
    plot_summary(all_results, png_path)

    print(f"\n{'='*65}")
    print(f'[P3-4] CIFAR-100 experiment complete. Results in: {OUT_DIR}')
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
