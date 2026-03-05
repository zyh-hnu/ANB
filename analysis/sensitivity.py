"""
Parameter Sensitivity Analysis Script  [P3-3]

Sweeps one hyperparameter at a time while keeping all others at their
default values (Config defaults).  Records final ASR and ACC for each
value, then produces publication-quality sensitivity curves.

Parameters studied (from plan.md §5.3):
    --param epsilon       values: 0.05 0.10 0.15 0.20
    --param poison_ratio  values: 0.10 0.20 0.30 0.40
    --param alpha         values: 0.10 0.30 0.50 1.00
    --param num_rounds    values: 30   50   100

Usage (from project root, GPU recommended):
    python analysis/sensitivity.py --param epsilon
    python analysis/sensitivity.py --param poison_ratio
    python analysis/sensitivity.py --param alpha
    python analysis/sensitivity.py --param num_rounds
    python analysis/sensitivity.py --all          # run all four sweeps
    python analysis/sensitivity.py --plot-only    # re-plot from saved JSON

Results are saved to:
    ./results/sensitivity/<param>_sweep.json
    ./results/sensitivity/<param>_curve.png
    ./results/sensitivity/sensitivity_summary.png  (all params in one figure)
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, load_config
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

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

SWEEPS = {
    'epsilon': {
        'values': [0.05, 0.10, 0.15, 0.20],
        'xlabel': 'Trigger Strength ε',
        'fmt':    '{:.2f}',
        'scale':  'linear',
    },
    'poison_ratio': {
        'values': [0.10, 0.20, 0.30, 0.40],
        'xlabel': 'Malicious Client Ratio',
        'fmt':    '{:.0%}',
        'scale':  'linear',
    },
    'alpha': {
        'values': [0.10, 0.30, 0.50, 1.00],
        'xlabel': 'Dirichlet α (Non-IID degree, ← more skewed)',
        'fmt':    '{:.2f}',
        'scale':  'log',
    },
    'num_rounds': {
        'values': [30, 50, 100],
        'xlabel': 'Training Rounds',
        'fmt':    '{:d}',
        'scale':  'linear',
    },
}

OUT_DIR = './results/sensitivity'
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Core: run one federated experiment and return final metrics
# ---------------------------------------------------------------------------

def run_one(cfg: Config) -> dict:
    """
    Run a complete federated training experiment with the given Config and
    return a dict of final-round metrics:
        {asr, acc, asr_multi, defense_recall, defense_bypass_rate}
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seed(cfg.seed)

    train_dataset, test_dataset, num_classes = load_dataset(cfg.dataset, cfg.data_dir)

    # Recompute num_malicious from current poison_ratio (may have changed in sweep)
    num_malicious   = max(1, int(cfg.num_clients * cfg.poison_ratio))
    malicious_indices = list(range(num_malicious))

    client_indices  = dirichlet_split(train_dataset,
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
        save_weights_at_rounds=[],   # skip weight saving during sensitivity sweep
    )

    h = server.history
    result = {
        'asr':               h['test_asr'][-1]       if h['test_asr']       else 0.0,
        'acc':               h['test_acc'][-1]        if h['test_acc']       else 0.0,
        'asr_multi':         h['test_asr_multi'][-1]  if h['test_asr_multi'] else None,
        'defense_recall':    h['defense_recall'][-1]  if h['defense_recall'] else None,
        'defense_bypass':    h['defense_bypass_rate'][-1]
                             if h['defense_bypass_rate'] else None,
        'train_loss_final':  h['train_loss'][-1]      if h['train_loss']     else None,
    }
    return result


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(param_name: str, base_cfg: Config) -> dict:
    """
    Sweep `param_name` over its predefined values, running one full experiment
    per value.  Returns a dict ready to be JSON-serialised.
    """
    if param_name not in SWEEPS:
        raise ValueError(f"Unknown parameter '{param_name}'. "
                         f"Choose from: {list(SWEEPS.keys())}")

    spec   = SWEEPS[param_name]
    values = spec['values']
    results = []

    print(f"\n{'='*65}")
    print(f"[P3-3] Sensitivity sweep: {param_name}  values={values}")
    print(f"{'='*65}")

    for val in values:
        cfg = copy.copy(base_cfg)
        setattr(cfg, param_name, val)

        # num_rounds sweep: also shorten early-stop threshold to avoid
        # irrelevant early exits confounding short-round experiments
        label = spec['fmt'].format(val)
        print(f"\n--- {param_name}={label} ---")
        print_experiment_config(cfg)

        try:
            metrics = run_one(cfg)
        except Exception as e:
            print(f"[WARNING] Experiment failed for {param_name}={val}: {e}")
            metrics = {'asr': None, 'acc': None, 'asr_multi': None,
                       'defense_recall': None, 'defense_bypass': None,
                       'train_loss_final': None, 'error': str(e)}

        metrics['param_value'] = float(val) if not isinstance(val, int) else int(val)
        results.append(metrics)
        print(f"  → ASR={metrics['asr']}, ACC={metrics['acc']}")

    sweep_result = {
        'param':   param_name,
        'values':  values,
        'results': results,
        'xlabel':  spec['xlabel'],
        'scale':   spec['scale'],
    }

    # Save to JSON
    out_path = os.path.join(OUT_DIR, f'{param_name}_sweep.json')
    with open(out_path, 'w') as f:
        json.dump(sweep_result, f, indent=2)
    print(f"\n[P3-3] Saved sweep results → {out_path}")
    return sweep_result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _safe(lst, key):
    """Extract values for a metric key, replacing None with np.nan."""
    return [r.get(key) if r.get(key) is not None else float('nan') for r in lst]


def plot_sweep(sweep_result: dict, save_path: str = None):
    """Plot ASR and ACC curves for a single parameter sweep."""
    param   = sweep_result['param']
    results = sweep_result['results']
    spec    = SWEEPS.get(param, {})
    xlabel  = sweep_result.get('xlabel', param)
    scale   = sweep_result.get('scale', 'linear')

    if save_path is None:
        save_path = os.path.join(OUT_DIR, f'{param}_curve.png')

    xs  = [r['param_value'] for r in results]
    asr = _safe(results, 'asr')
    acc = _safe(results, 'acc')
    bypass = _safe(results, 'defense_bypass')

    has_bypass = any(not np.isnan(b) for b in bypass)

    n_axes = 3 if has_bypass else 2
    fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 5))
    if n_axes == 1:
        axes = [axes]

    # --- ASR ---
    axes[0].plot(xs, [a * 100 for a in asr], 'o-', color='#e74c3c',
                 linewidth=2, markersize=8, label='ASR')
    axes[0].set_xlabel(xlabel, fontsize=12)
    axes[0].set_ylabel('Attack Success Rate (%)', fontsize=12)
    axes[0].set_title(f'ASR vs {param}', fontsize=13, fontweight='bold')
    axes[0].set_xscale(scale)
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.35)
    axes[0].axhline(y=85, color='gray', linestyle='--', alpha=0.6, label='85% target')
    axes[0].legend(fontsize=10)
    _annotate(axes[0], xs, [a * 100 for a in asr])

    # --- ACC ---
    axes[1].plot(xs, [a * 100 for a in acc], 's-', color='#2980b9',
                 linewidth=2, markersize=8, label='Clean ACC')
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].set_ylabel('Clean Accuracy (%)', fontsize=12)
    axes[1].set_title(f'Clean ACC vs {param}', fontsize=13, fontweight='bold')
    axes[1].set_xscale(scale)
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(fontsize=10)
    _annotate(axes[1], xs, [a * 100 for a in acc])

    # --- Bypass Rate (defense-on experiments) ---
    if has_bypass:
        axes[2].plot(xs, [b * 100 for b in bypass], '^-', color='#27ae60',
                     linewidth=2, markersize=8, label='Defense Bypass Rate')
        axes[2].set_xlabel(xlabel, fontsize=12)
        axes[2].set_ylabel('Defense Bypass Rate (%)', fontsize=12)
        axes[2].set_title(f'Bypass Rate vs {param}', fontsize=13, fontweight='bold')
        axes[2].set_xscale(scale)
        axes[2].set_ylim(0, 105)
        axes[2].grid(True, alpha=0.35)
        axes[2].axhline(y=50, color='gray', linestyle='--', alpha=0.6,
                        label='50% bypass target')
        axes[2].legend(fontsize=10)
        _annotate(axes[2], xs, [b * 100 for b in bypass])

    plt.suptitle(f'Parameter Sensitivity: {param}\n'
                 f'(ANB attack, FreqFed defense, CIFAR-10)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[P3-3] Curve saved → {save_path}")
    return save_path


def _annotate(ax, xs, ys):
    """Add value labels above each data point."""
    for x, y in zip(xs, ys):
        if not np.isnan(y):
            ax.annotate(f'{y:.1f}', (x, y),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=9)


def plot_summary(param_names=None):
    """
    Load all sweep JSONs and plot a 2×len(params) summary grid:
      top row = ASR curves, bottom row = ACC curves.
    Suitable for a full-page figure in the appendix.
    """
    if param_names is None:
        param_names = list(SWEEPS.keys())

    available = []
    sweep_data = {}
    for p in param_names:
        fp = os.path.join(OUT_DIR, f'{p}_sweep.json')
        if os.path.exists(fp):
            with open(fp) as f:
                sweep_data[p] = json.load(f)
            available.append(p)
        else:
            print(f"[P3-3] No data for '{p}', skipping in summary.")

    if not available:
        print("[P3-3] No sweep data found. Run sweeps first.")
        return

    n = len(available)
    fig = plt.figure(figsize=(6 * n, 10))
    gs  = gridspec.GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.35)

    for col, p in enumerate(available):
        sd      = sweep_data[p]
        results = sd['results']
        xs      = [r['param_value'] for r in results]
        asr     = [r.get('asr', float('nan')) or float('nan') for r in results]
        acc     = [r.get('acc', float('nan')) or float('nan') for r in results]
        scale   = sd.get('scale', 'linear')
        xlabel  = sd.get('xlabel', p)

        # ASR (top row)
        ax_asr = fig.add_subplot(gs[0, col])
        ax_asr.plot(xs, [v * 100 for v in asr], 'o-', color='#e74c3c',
                    linewidth=2, markersize=7)
        ax_asr.axhline(85, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax_asr.set_xscale(scale)
        ax_asr.set_ylim(0, 105)
        ax_asr.set_title(p, fontsize=12, fontweight='bold')
        ax_asr.set_ylabel('ASR (%)' if col == 0 else '', fontsize=11)
        ax_asr.set_xlabel(xlabel, fontsize=9)
        ax_asr.grid(True, alpha=0.3)
        _annotate(ax_asr, xs, [v * 100 for v in asr])

        # ACC (bottom row)
        ax_acc = fig.add_subplot(gs[1, col])
        ax_acc.plot(xs, [v * 100 for v in acc], 's-', color='#2980b9',
                    linewidth=2, markersize=7)
        ax_acc.set_xscale(scale)
        ax_acc.set_ylim(0, 105)
        ax_acc.set_ylabel('Clean ACC (%)' if col == 0 else '', fontsize=11)
        ax_acc.set_xlabel(xlabel, fontsize=9)
        ax_acc.grid(True, alpha=0.3)
        _annotate(ax_acc, xs, [v * 100 for v in acc])

    fig.text(0.5, 0.98, 'Parameter Sensitivity Analysis (ANB, CIFAR-10, FreqFed defense)',
             ha='center', fontsize=14, fontweight='bold')
    fig.text(0.01, 0.75, 'ASR (%)',      va='center', rotation='vertical', fontsize=12)
    fig.text(0.01, 0.27, 'Clean ACC (%)', va='center', rotation='vertical', fontsize=12)

    out = os.path.join(OUT_DIR, 'sensitivity_summary.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[P3-3] Summary figure saved → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='P3-3: Parameter Sensitivity Analysis for ANB'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--param', choices=list(SWEEPS.keys()),
                       help='Single parameter to sweep')
    group.add_argument('--all',  action='store_true',
                       help='Run all parameter sweeps sequentially')
    group.add_argument('--plot-only', action='store_true',
                       help='Skip training; re-plot from existing JSON files')

    # Allow overriding the base config for the sweep
    parser.add_argument('--defense-enabled', type=int, default=1,
                        help='1=defense on (default), 0=no defense')
    parser.add_argument('--defense-method', type=str, default='hdbscan',
                        help='Defense method (default: hdbscan)')
    parser.add_argument('--num-rounds', type=int, default=None,
                        help='Override default number of rounds (for quick testing)')
    parser.add_argument('--results-dir', type=str, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    global OUT_DIR
    OUT_DIR = args.results_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.plot_only:
        print("[P3-3] --plot-only: regenerating figures from saved JSON files.")
        for p in SWEEPS:
            fp = os.path.join(OUT_DIR, f'{p}_sweep.json')
            if os.path.exists(fp):
                with open(fp) as f:
                    sd = json.load(f)
                plot_sweep(sd)
        plot_summary()
        return

    # Build base config from defaults, then apply CLI overrides
    base_cfg             = Config()
    base_cfg.defense_enabled = bool(args.defense_enabled)
    base_cfg.defense_method  = args.defense_method
    if args.num_rounds is not None:
        base_cfg.num_rounds = args.num_rounds

    params_to_run = list(SWEEPS.keys()) if args.all else [args.param]

    all_results = {}
    for p in params_to_run:
        sweep_result   = run_sweep(p, base_cfg)
        all_results[p] = sweep_result
        plot_sweep(sweep_result)

    # Summary figure (only meaningful when multiple sweeps are available)
    plot_summary()

    print(f"\n{'='*65}")
    print(f"[P3-3] All done. Results in: {OUT_DIR}")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
