"""
Main Entry Point for Adaptive Nebula Backdoor (ANB) Experiments

This script orchestrates the complete experimental pipeline:
- Stage 1: Atomic Verification (visual inspection)
- Stage 2: Attack Effectiveness (ASR without defense)
- Stage 3: Defense Evasion (ASR with FreqFed defense)
- Stage 4: Quantitative Stealth & Automated Visualization
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import json
import glob

# Import project modules
from config import load_config
from core.registry import ATTACKS, MODELS
import core.attacks  # register attacks
import models.resnet  # register models
from data.dataset import PoisonedTestDataset, CleanTestDataset, MultiTriggerTestDataset, get_transforms
from data.distribution import dirichlet_split
from federated.client import create_clients
from federated.server import Server, federated_training

# Import Visualization Modules
from analysis.create_visualizations import (
    visualize_trigger_generation_pipeline, 
    visualize_multi_client_triggers, 
    visualize_frequency_comparison, 
    create_defense_evasion_illustration
)
from analysis.visualize_clusters import visualize_cluster_results, load_real_weights
from core.defenses import cluster_clients


def setup_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(dataset_name='CIFAR10', data_dir='./data'):
    """Load base dataset."""
    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_dataset, test_dataset, num_classes


def create_test_loaders(test_dataset, target_label, epsilon, freq_strategy,
                        malicious_client_ids, batch_size=128, dataset_name='CIFAR10',
                        backdoor_factory=None, num_workers=0, pin_memory=True):
    """Create test data loaders for evaluation."""
    test_transform = get_transforms(train=False, dataset=dataset_name)

    # Clean test set for accuracy
    clean_test = CleanTestDataset(test_dataset, transform=test_transform)
    clean_loader = DataLoader(
        clean_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # Single-trigger poisoned test set
    poisoned_test = PoisonedTestDataset(
        test_dataset, target_label=target_label, epsilon=epsilon,
        freq_strategy=freq_strategy, client_id=malicious_client_ids[0] if malicious_client_ids else 0,
        transform=test_transform, backdoor_factory=backdoor_factory
    )
    poisoned_loader = DataLoader(
        poisoned_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # Multi-trigger test set (comprehensive evaluation)
    multi_trigger_test = MultiTriggerTestDataset(
        test_dataset, malicious_client_ids=malicious_client_ids,
        target_label=target_label, epsilon=epsilon,
        freq_strategy=freq_strategy, transform=test_transform, backdoor_factory=backdoor_factory
    )
    multi_trigger_loader = DataLoader(
        multi_trigger_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    # Per-client test sets
    per_client_loaders = {}
    for client_id in malicious_client_ids:
        client_test = PoisonedTestDataset(
            test_dataset, target_label=target_label, epsilon=epsilon,
            freq_strategy=freq_strategy, client_id=client_id, transform=test_transform,
            backdoor_factory=backdoor_factory
        )
        per_client_loaders[client_id] = DataLoader(
            client_test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

    return clean_loader, poisoned_loader, multi_trigger_loader, per_client_loaders


def print_experiment_config(cfg):
    """Print experiment configuration."""
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Attack Mode: {cfg.attack_mode}")
    print(f"Frequency Strategy: {cfg.freq_strategy} (Check: ANB enables Spatial Tint)")
    print(f"Defense Enabled: {cfg.defense_enabled} ({cfg.defense_method})")
    print(f"Dataset: {cfg.dataset}, Clients: {cfg.num_clients}, Poison Ratio: {cfg.poison_ratio:.0%}")
    print(f"Target Label: {cfg.target_label}, Epsilon: {cfg.epsilon}, Poison Rate: {cfg.poison_rate:.2f}")
    print(f"Training: {cfg.num_rounds} Rounds, Local Epochs: {cfg.local_epochs}, LR: {cfg.learning_rate}")
    print(f"Attack Scaling Factor: {cfg.scaling_factor:.2f}")
    print("="*70 + "\n")


def generate_experiment_visualizations(results_dir='./results', weights_dir='./results/weights',
                                       defense_method='hdbscan', freq_strategy='ANB'):
    """Generate all verification and analysis plots after training."""
    print("\n" + "="*70)
    print("STEP 10: Generating Visualizations")
    print("="*70)
    
    os.makedirs(results_dir, exist_ok=True)

    # 1. Trigger Mechanism Visualization
    print("[Viz 1/4] Generating Trigger Pipeline & Multi-client Shards...")
    visualize_trigger_generation_pipeline(save_path=os.path.join(results_dir, 'anb_trigger_pipeline.png'))
    visualize_multi_client_triggers(num_clients=8, save_path=os.path.join(results_dir, 'anb_sharding_grid.png'))
    
    # 2. Frequency Analysis
    print("[Viz 2/4] Generating Frequency Domain Comparison...")
    visualize_frequency_comparison(save_path=os.path.join(results_dir, 'anb_freq_comparison.png'))
    
    # 3. Defense Evasion Concept
    print("[Viz 3/4] Generating Defense Evasion Concept...")
    create_defense_evasion_illustration(save_path=os.path.join(results_dir, 'defense_evasion_concept.png'))
    
    # 4. Actual Clustering Results (using saved weights if available)
    print("[Viz 4/4] Generating Real-data Clustering Visualization...")
    try:
        if not os.path.exists(weights_dir):
            print(f"  Warning: Weights directory {weights_dir} not found. Using synthetic fallback.")
            # Fallback handled implicitly if we don't call real weight loader
        
        # Find the latest weight file
        weight_files = sorted(glob.glob(os.path.join(weights_dir, '*.pkl')))
        if weight_files:
            latest_weights = weight_files[-1]
            print(f"  Loading real weights from: {latest_weights}")
            
            client_weights, malicious_indices, metadata = load_real_weights(latest_weights)
            labels, _ = cluster_clients(client_weights, method=defense_method)
            
            title = f"FreqFed Defense Result (Round {metadata.get('round', 'Unknown')}) - {freq_strategy}"
            save_path = os.path.join(results_dir, f'clustering_result_real.png')
            
            visualize_cluster_results(client_weights, labels, malicious_indices, title, save_path)
            print(f"  ✓ Real clustering result saved to {save_path}")
        else:
            print("  No weight files found. Skipping real-data clustering visualization.")
            
    except Exception as e:
        print(f"  Error visualizing clustering: {e}")
        print("  Skipping clustering visualization.")

    print("\n✓ All Visualizations generated in ./results/")


def build_backdoor_factory(cfg):
    attack_cls = ATTACKS.get(cfg.backdoor_name)
    if attack_cls.__name__ == "AdaptiveNebulaBackdoor":
        return lambda cid: attack_cls(
            client_id=cid,
            target_label=cfg.target_label,
            epsilon=cfg.epsilon,
            max_rounds=cfg.num_rounds,
            strategy=cfg.freq_strategy,
            # [P2-1] Ablation switches from config
            use_phased_chaos=cfg.use_phased_chaos,
            use_spectral_smoothing=cfg.use_spectral_smoothing,
            use_freq_sharding=cfg.use_freq_sharding,
            use_dual_routing=cfg.use_dual_routing
        )
    return lambda cid: attack_cls(
        client_id=cid,
        target_label=cfg.target_label,
        epsilon=cfg.epsilon,
        freq_strategy=cfg.freq_strategy,
        # [P2-1] Ablation switches (FrequencyBackdoor wraps ANB)
        use_phased_chaos=cfg.use_phased_chaos,
        use_spectral_smoothing=cfg.use_spectral_smoothing,
        use_freq_sharding=cfg.use_freq_sharding,
        use_dual_routing=cfg.use_dual_routing
    )


def main():
    """Main experimental pipeline."""
    cfg = load_config()
    setup_seed(cfg.seed)
    print_experiment_config(cfg)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # ===== Step 1-6: Setup =====
    print("Initializing Data and Clients...")
    train_dataset, test_dataset, num_classes = load_dataset(cfg.dataset, cfg.data_dir)
    client_indices = dirichlet_split(train_dataset, num_clients=cfg.num_clients, alpha=cfg.alpha)
    num_malicious = int(cfg.num_clients * cfg.poison_ratio)
    malicious_indices = list(range(num_malicious))
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
        scaling_factor=cfg.scaling_factor  # [P1-2] Model replacement scaling
    )

    model_builder = MODELS.get(cfg.model_name)
    global_model = model_builder(num_classes=num_classes)
    
    clean_test_loader, poisoned_test_loader, multi_trigger_loader, per_client_loaders = create_test_loaders(
        test_dataset, cfg.target_label, cfg.epsilon, cfg.freq_strategy,
        malicious_indices, cfg.batch_size, dataset_name=cfg.dataset,
        backdoor_factory=backdoor_factory,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )

    server = Server(
        model=global_model,
        device=device,
        defense_enabled=cfg.defense_enabled,
        defense_method=cfg.defense_method,
        target_label=cfg.target_label
    )

    # ===== Step 7: Training =====
    print("Starting Federated Learning...")
    # Save weights at mid and final round for visualization
    save_rounds = [cfg.num_rounds // 2, cfg.num_rounds]
    
    server = federated_training(
        server=server,
        clients=clients,
        test_loader=clean_test_loader,
        poisoned_test_loader=poisoned_test_loader,
        multi_trigger_loader=multi_trigger_loader,
        per_client_loaders=per_client_loaders,
        num_rounds=cfg.num_rounds,
        malicious_indices=malicious_indices,
        client_fraction=cfg.client_fraction,
        save_weights_at_rounds=save_rounds
    )

    # ===== Step 8: Final Results & Saving =====
    print("\nFINAL RESULTS")
    final_acc = server.history['test_acc'][-1]
    final_asr = server.history['test_asr'][-1]
    print(f"Final Clean ACC: {final_acc:.2%}")
    print(f"Final Attack ASR: {final_asr:.2%}")

    os.makedirs(cfg.results_dir, exist_ok=True)
    experiment_name = f"{cfg.attack_mode}_{cfg.freq_strategy}_defense_{cfg.defense_enabled}"
    server.save_model(os.path.join(cfg.results_dir, f"model_{experiment_name}.pth"))
    
    with open(os.path.join(cfg.results_dir, f"history_{experiment_name}.json"), 'w') as f:
        json.dump(server.history, f, indent=2)

    # ===== Step 9 & 10: Visualization =====
    # Calls the generator to produce all plots based on the trained model/logic
    generate_experiment_visualizations(
        results_dir=cfg.results_dir,
        weights_dir=cfg.weights_dir,
        defense_method=cfg.defense_method,
        freq_strategy=cfg.freq_strategy
    )

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
