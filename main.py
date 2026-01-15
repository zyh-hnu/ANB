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
from config import *
from models.resnet import ResNet18
from data.dataset import BackdoorDataset, PoisonedTestDataset, CleanTestDataset, MultiTriggerTestDataset, get_transforms
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
                        malicious_client_ids, batch_size=128):
    """Create test data loaders for evaluation."""
    test_transform = get_transforms(train=False, dataset=DATASET)

    # Clean test set for accuracy
    clean_test = CleanTestDataset(test_dataset, transform=test_transform)
    clean_loader = DataLoader(clean_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Single-trigger poisoned test set
    poisoned_test = PoisonedTestDataset(
        test_dataset, target_label=target_label, epsilon=epsilon,
        freq_strategy=freq_strategy, client_id=malicious_client_ids[0] if malicious_client_ids else 0,
        transform=test_transform
    )
    poisoned_loader = DataLoader(poisoned_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Multi-trigger test set (comprehensive evaluation)
    multi_trigger_test = MultiTriggerTestDataset(
        test_dataset, malicious_client_ids=malicious_client_ids,
        target_label=target_label, epsilon=epsilon,
        freq_strategy=freq_strategy, transform=test_transform
    )
    multi_trigger_loader = DataLoader(multi_trigger_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Per-client test sets
    per_client_loaders = {}
    for client_id in malicious_client_ids:
        client_test = PoisonedTestDataset(
            test_dataset, target_label=target_label, epsilon=epsilon,
            freq_strategy=freq_strategy, client_id=client_id, transform=test_transform
        )
        per_client_loaders[client_id] = DataLoader(client_test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return clean_loader, poisoned_loader, multi_trigger_loader, per_client_loaders


def print_experiment_config():
    """Print experiment configuration."""
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Attack Mode: {ATTACK_MODE}")
    print(f"Frequency Strategy: {FREQ_STRATEGY} (Check: ANB enables Spatial Tint)")
    print(f"Defense Enabled: {DEFENSE_ENABLED} ({DEFENSE_METHOD})")
    print(f"Dataset: {DATASET}, Clients: {NUM_CLIENTS}, Poison Ratio: {POISON_RATIO:.0%}")
    print(f"Target Label: {TARGET_LABEL}, Epsilon: {EPSILON}")
    print(f"Training: {NUM_ROUNDS} Rounds")
    print("="*70 + "\n")


def generate_experiment_visualizations(results_dir='./results', weights_dir='./results/weights'):
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
            labels, _ = cluster_clients(client_weights, method=DEFENSE_METHOD)
            
            title = f"FreqFed Defense Result (Round {metadata.get('round', 'Unknown')}) - {FREQ_STRATEGY}"
            save_path = os.path.join(results_dir, f'clustering_result_real.png')
            
            visualize_cluster_results(client_weights, labels, malicious_indices, title, save_path)
            print(f"  ✓ Real clustering result saved to {save_path}")
        else:
            print("  No weight files found. Skipping real-data clustering visualization.")
            
    except Exception as e:
        print(f"  Error visualizing clustering: {e}")
        print("  Skipping clustering visualization.")

    print("\n✓ All Visualizations generated in ./results/")


def main():
    """Main experimental pipeline."""
    setup_seed(42)
    print_experiment_config()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # ===== Step 1-6: Setup =====
    print("Initializing Data and Clients...")
    train_dataset, test_dataset, num_classes = load_dataset(DATASET)
    client_indices = dirichlet_split(train_dataset, num_clients=NUM_CLIENTS, alpha=ALPHA)
    num_malicious = int(NUM_CLIENTS * POISON_RATIO)
    malicious_indices = list(range(num_malicious))

    clients = create_clients(
        dataset=train_dataset,
        num_clients=NUM_CLIENTS,
        malicious_indices=malicious_indices,
        client_indices=client_indices,
        target_label=TARGET_LABEL,
        epsilon=EPSILON,
        freq_strategy=FREQ_STRATEGY,
        batch_size=BATCH_SIZE,
        local_epochs=LOCAL_EPOCHS,
        lr=LEARNING_RATE
    )

    global_model = ResNet18(num_classes=num_classes)
    
    clean_test_loader, poisoned_test_loader, multi_trigger_loader, per_client_loaders = create_test_loaders(
        test_dataset, TARGET_LABEL, EPSILON, FREQ_STRATEGY, malicious_indices, BATCH_SIZE
    )

    server = Server(
        model=global_model,
        device=device,
        defense_enabled=DEFENSE_ENABLED,
        defense_method=DEFENSE_METHOD,
        target_label=TARGET_LABEL
    )

    # ===== Step 7: Training =====
    print("Starting Federated Learning...")
    # Save weights at mid and final round for visualization
    save_rounds = [NUM_ROUNDS // 2, NUM_ROUNDS]
    
    server = federated_training(
        server=server,
        clients=clients,
        test_loader=clean_test_loader,
        poisoned_test_loader=poisoned_test_loader,
        multi_trigger_loader=multi_trigger_loader,
        per_client_loaders=per_client_loaders,
        num_rounds=NUM_ROUNDS,
        malicious_indices=malicious_indices,
        client_fraction=1.0,
        save_weights_at_rounds=save_rounds
    )

    # ===== Step 8: Final Results & Saving =====
    print("\nFINAL RESULTS")
    final_acc = server.history['test_acc'][-1]
    final_asr = server.history['test_asr'][-1]
    print(f"Final Clean ACC: {final_acc:.2%}")
    print(f"Final Attack ASR: {final_asr:.2%}")

    os.makedirs('./results', exist_ok=True)
    experiment_name = f"{ATTACK_MODE}_{FREQ_STRATEGY}_defense_{DEFENSE_ENABLED}"
    server.save_model(f"./results/model_{experiment_name}.pth")
    
    with open(f"./results/history_{experiment_name}.json", 'w') as f:
        json.dump(server.history, f, indent=2)

    # ===== Step 9 & 10: Visualization =====
    # Calls the generator to produce all plots based on the trained model/logic
    generate_experiment_visualizations(results_dir='./results', weights_dir='./results/weights')

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()