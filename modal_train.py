"""
Modal Cloud Training Script for ANB Experiments  (Modal >= 1.x)
================================================

Runs all thesis experiments on a remote T4 GPU via Modal.
Results are persisted to a Modal Volume and can be pulled back locally.

Usage:
    # Smoke test: one condition, 30 rounds
    modal run modal_train.py::run_single --condition anb_freqfed --num-rounds 30

    # Full thesis suite (all conditions + sweeps + CIFAR-100)
    modal run modal_train.py

    # Pull results to local machine after runs finish
    modal volume get safb-results /results ./results_from_modal

Available conditions:
    anb_no_defense     ANB attack, no defense (upper-bound ASR)
    fixed_no_defense   FIXED attack, no defense (baseline)
    anb_freqfed        ANB + FreqFed defense  (our main claim)
    fixed_freqfed      FIXED + FreqFed defense (detected baseline)
    anb_fltrust        ANB + FLTrust defense  (extra baseline)
"""

import os
import modal

# ---------------------------------------------------------------------------
# Volumes  — persisted across runs, survive container restarts / interruptions
# ---------------------------------------------------------------------------

data_volume   = modal.Volume.from_name("safb-data",   create_if_missing=True)
result_volume = modal.Volume.from_name("safb-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Container image
# .add_local_dir() replaces the old modal.Mount API in Modal >= 0.60
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install([
        "torch==2.1.2",
        "torchvision==0.16.2",
        "opencv-python-headless>=4.7.0",
        "Pillow>=9.4.0",
        "scikit-image>=0.20.0",
        "numpy>=1.24.0,<2.0",   # hdbscan wheels compiled against NumPy 1.x
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "hdbscan>=0.8.29",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "lpips>=0.1.4",
        "tqdm>=4.65.0",
    ])
    # Upload the entire project source tree into /safb inside the container.
    # This replaces the deprecated modal.Mount approach.
    .add_local_dir(
        local_path=PROJECT_ROOT,
        remote_path="/safb",
        ignore=[
            ".git",
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            ".pytest_cache",
            "results_from_modal",
            ".ipynb_checkpoints",
        ],
    )
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App(name="safb-anb-experiments", image=image)

# ---------------------------------------------------------------------------
# Experiment condition registry
# ---------------------------------------------------------------------------

CONDITIONS = {
    "anb_no_defense": dict(
        freq_strategy="ANB",
        defense_enabled=False,
        defense_method="hdbscan",
        label="ANB (no defense)",
    ),
    "fixed_no_defense": dict(
        freq_strategy="FIXED",
        defense_enabled=False,
        defense_method="hdbscan",
        label="FIXED (no defense)",
    ),
    "anb_freqfed": dict(
        freq_strategy="ANB",
        defense_enabled=True,
        defense_method="hdbscan",
        label="ANB + FreqFed",
    ),
    "fixed_freqfed": dict(
        freq_strategy="FIXED",
        defense_enabled=True,
        defense_method="hdbscan",
        label="FIXED + FreqFed",
    ),
    "anb_fltrust": dict(
        freq_strategy="ANB",
        defense_enabled=True,
        defense_method="fltrust",
        label="ANB + FLTrust",
    ),
}

# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=7200,
    volumes={
        "/data":    data_volume,
        "/results": result_volume,
    },
)
def train_condition(condition_name: str,
                    num_rounds: int = 50,
                    poison_rate: float = 1.0,
                    scaling_factor: float = 5.0,
                    local_epochs: int = 5,
                    learning_rate: float = 0.01,
                    seed: int = 42,
                    backdoor_boost_weight: float = 0.3):
    """Run one complete federated experiment and save results to /results/<condition>/."""
    import sys, os, json
    import torch

    sys.path.insert(0, "/safb")

    from config import Config
    from core.registry import MODELS
    import core.attacks          # register
    import models.resnet         # register
    from data.distribution import dirichlet_split
    from main import (
        setup_seed, load_dataset, create_test_loaders,
        build_backdoor_factory, print_experiment_config,
    )
    from federated.client import create_clients
    from federated.server import Server, federated_training

    cond = CONDITIONS[condition_name]
    print(f"\n{'='*65}")
    print(f"[Modal] Condition : {cond['label']}")
    print(f"[Modal] GPU       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*65}\n")

    # ── Config ────────────────────────────────────────────────────────────
    cfg = Config()
    cfg.data_dir        = "/data"
    cfg.results_dir     = f"/results/{condition_name}"
    cfg.weights_dir     = f"/results/{condition_name}/weights"
    cfg.num_rounds      = num_rounds
    cfg.poison_rate     = poison_rate
    cfg.scaling_factor  = scaling_factor
    cfg.local_epochs    = local_epochs
    cfg.learning_rate   = learning_rate
    cfg.seed            = seed
    cfg.backdoor_boost_weight = backdoor_boost_weight
    cfg.freq_strategy   = cond["freq_strategy"]
    cfg.defense_enabled = cond["defense_enabled"]
    cfg.defense_method  = cond["defense_method"]

    # FIXED baseline: all adaptive features OFF
    if cfg.freq_strategy == "FIXED":
        cfg.use_phased_chaos       = False
        cfg.use_spectral_smoothing = False
        cfg.use_freq_sharding      = False
        cfg.use_dual_routing       = False

    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.weights_dir, exist_ok=True)
    print_experiment_config(cfg)

    # ── Data & clients ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(cfg.seed)

    train_dataset, test_dataset, num_classes = load_dataset(cfg.dataset, cfg.data_dir)
    num_malicious     = max(1, int(cfg.num_clients * cfg.poison_ratio))
    malicious_indices = list(range(num_malicious))
    client_indices    = dirichlet_split(train_dataset,
                                        num_clients=cfg.num_clients,
                                        alpha=cfg.alpha)
    backdoor_factory  = build_backdoor_factory(cfg)

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
        backdoor_boost_weight=cfg.backdoor_boost_weight,
    )

    model_builder = MODELS.get(cfg.model_name)
    global_model  = model_builder(num_classes=num_classes)

    clean_loader, poisoned_loader, multi_loader, per_client_loaders = create_test_loaders(
        test_dataset, cfg.target_label, cfg.epsilon, cfg.freq_strategy,
        malicious_indices, cfg.batch_size, dataset_name=cfg.dataset,
        backdoor_factory=backdoor_factory,
        num_workers=2, pin_memory=True,
    )

    # ── Training ──────────────────────────────────────────────────────────
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
        save_weights_at_rounds=[cfg.num_rounds // 2, cfg.num_rounds],
    )

    # ── Persist results ───────────────────────────────────────────────────
    history_path = os.path.join(cfg.results_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(server.history, f, indent=2)

    model_path = os.path.join(cfg.results_dir, "model_final.pth")
    server.save_model(model_path)
    print(f"[Modal] Saved → {history_path}")
    print(f"[Modal] Saved → {model_path}")

    # ── Return summary ────────────────────────────────────────────────────
    final = {
        "condition":  condition_name,
        "label":      cond["label"],
        "final_asr":  server.history["test_asr"][-1],
        "final_acc":  server.history["test_acc"][-1],
        "num_rounds": len(server.history["test_asr"]),
        "defense_bypass_last": (
            server.history["defense_bypass_rate"][-1]
            if server.history["defense_bypass_rate"] else None
        ),
    }
    print(f"\n[Modal] === RESULT: {cond['label']} ===")
    print(f"  ASR    : {final['final_asr']:.2%}")
    print(f"  ACC    : {final['final_acc']:.2%}")
    if final["defense_bypass_last"] is not None:
        print(f"  Bypass : {final['defense_bypass_last']:.2%}")
    return final


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=10800,
    volumes={
        "/data":    data_volume,
        "/results": result_volume,
    },
)
def run_sensitivity_sweep(param_name: str, num_rounds: int = 50):
    """Sweep one hyperparameter over its predefined grid."""
    import sys, os
    sys.path.insert(0, "/safb")

    from config import Config
    import analysis.sensitivity as sens_mod
    from analysis.sensitivity import run_sweep, plot_sweep

    out_dir = "/results/sensitivity"
    os.makedirs(out_dir, exist_ok=True)
    sens_mod.OUT_DIR = out_dir          # redirect JSON / PNG output

    base_cfg = Config()
    base_cfg.data_dir        = "/data"
    base_cfg.num_rounds      = num_rounds
    base_cfg.defense_enabled = True
    base_cfg.defense_method  = "hdbscan"

    sweep_result = run_sweep(param_name, base_cfg)
    plot_sweep(sweep_result, save_path=os.path.join(out_dir, f"{param_name}_curve.png"))

    return {"param": param_name, "done": True}


# ---------------------------------------------------------------------------
# CIFAR-100 generalization
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=10800,
    volumes={
        "/data":    data_volume,
        "/results": result_volume,
    },
)
def run_cifar100(num_rounds: int = 50):
    """Run the three-condition CIFAR-100 cross-dataset experiment."""
    import sys, os
    sys.path.insert(0, "/safb")

    import analysis.cifar100_experiment as c100

    c100.OUT_DIR = "/results/cifar100"
    os.makedirs(c100.OUT_DIR, exist_ok=True)

    # Redirect data_dir inside each condition run
    _orig = c100.run_condition
    def _patched(cfg):
        cfg.data_dir = "/data"
        return _orig(cfg)
    c100.run_condition = _patched

    # Drive main() without touching sys.argv (parse_args has defaults)
    import sys as _sys
    _sys.argv = ["cifar100_experiment.py", "--num-rounds", str(num_rounds)]
    c100.main()

    return {"cifar100": "done"}


# ---------------------------------------------------------------------------
# Summary table (CPU only, reads saved JSONs)
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/results": result_volume},
)
def generate_summary_table():
    """Print a formatted results table from saved history JSONs."""
    import json, os

    order = ["anb_no_defense", "fixed_no_defense",
             "anb_freqfed", "fixed_freqfed", "anb_fltrust"]

    print("\n" + "="*75)
    print(f"{'Method':<28} {'ASR (↑)':>9} {'ACC':>9} {'Bypass (↑)':>12} {'Rounds':>8}")
    print("-"*75)

    for name in order:
        path = f"/results/{name}/history.json"
        if not os.path.exists(path):
            print(f"  {CONDITIONS[name]['label']:<26}  (no data yet)")
            continue
        with open(path) as f:
            h = json.load(f)
        asr    = h["test_asr"][-1]  if h["test_asr"] else 0
        acc    = h["test_acc"][-1]  if h["test_acc"] else 0
        bypass = (h["defense_bypass_rate"][-1]
                  if h.get("defense_bypass_rate") else None)
        rounds = len(h["test_asr"])
        bypass_str = f"{bypass:.1%}" if bypass is not None else "-"
        label = CONDITIONS[name]["label"]
        print(f"  {label:<26} {asr:>8.1%} {acc:>9.1%} {bypass_str:>12} {rounds:>8}")

    print("="*75)


# ---------------------------------------------------------------------------
# Full suite entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(num_rounds: int = 50,
         poison_rate: float = 1.0,
         scaling_factor: float = 5.0,
         local_epochs: int = 5,
         learning_rate: float = 0.01,
         seed: int = 42):
    import json

    print("\n" + "="*65)
    print("ANB Thesis Experiments — Modal Cloud Runner")
    print("="*65)

    # Step 1: 5 main conditions in parallel
    print("\n[Step 1] Launching 5 conditions in parallel...")
    all_results = []
    for result in train_condition.starmap(
        [(name, num_rounds, poison_rate, scaling_factor, local_epochs, learning_rate, seed)
         for name in CONDITIONS]
    ):
        all_results.append(result)
        bypass = result["defense_bypass_last"]
        bypass_str = f"  Bypass={bypass:.1%}" if bypass is not None else ""
        print(f"  ✓ {result['label']:<28} ASR={result['final_asr']:.1%}  ACC={result['final_acc']:.1%}{bypass_str}")

    with open("./modal_results_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("[Step 1] Done → ./modal_results_summary.json")

    # Step 2: Sensitivity sweeps in parallel
    print("\n[Step 2] Launching sensitivity sweeps in parallel...")
    for result in run_sensitivity_sweep.starmap(
        [(p, 50) for p in ["epsilon", "poison_ratio", "alpha", "num_rounds"]]
    ):
        print(f"  ✓ sensitivity: {result['param']}")
    print("[Step 2] Done.")

    # Step 3: CIFAR-100
    print("\n[Step 3] Launching CIFAR-100 experiment...")
    run_cifar100.remote(num_rounds=50)
    print("[Step 3] Done.")

    # Step 4: Summary
    print("\n[Step 4] Results table:")
    generate_summary_table.remote()

    print("\n" + "="*65)
    print("All experiments complete!")
    print("Pull results:  modal volume get safb-results /results ./results_from_modal")
    print("="*65)


# ---------------------------------------------------------------------------
# Single-condition entrypoint (smoke test)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_single(condition: str = "anb_freqfed",
               num_rounds: int = 30,
               poison_rate: float = 1.0,
               scaling_factor: float = 5.0,
               local_epochs: int = 5,
               learning_rate: float = 0.01,
               seed: int = 42,
               backdoor_boost_weight: float = 0.3):
    """
    Quick smoke test — one condition, fewer rounds.

    modal run modal_train.py::run_single --condition anb_freqfed --num-rounds 30
    modal run modal_train.py::run_single --condition anb_freqfed --num-rounds 50 --poison-rate 0.9 --scaling-factor 4.5 --local-epochs 5 --learning-rate 0.01 --seed 123
    """
    print(f"\n[Modal] Smoke test: {condition}, {num_rounds} rounds")
    print(f"[Modal] Params: poison_rate={poison_rate}, scaling_factor={scaling_factor}, local_epochs={local_epochs}, lr={learning_rate}, seed={seed}, boost_weight={backdoor_boost_weight}")
    result = train_condition.remote(condition, num_rounds, poison_rate, scaling_factor, local_epochs, learning_rate, seed, backdoor_boost_weight)
    print(f"\n[Modal] ASR={result['final_asr']:.2%}  ACC={result['final_acc']:.2%}")
