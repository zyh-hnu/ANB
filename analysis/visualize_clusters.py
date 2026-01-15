"""
Defense Clustering Visualization Tool

Demonstrates how FreqFed defense detects attacks and how ANB evades detection.
Adapted for ANB's frequency sharding logic.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys

# Removed Chinese font configuration to ensure English environment compatibility.

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.defenses import cluster_clients, apply_dct_to_weights, extract_frequency_features
from core.attacks import FrequencyBackdoor


def load_real_weights(weights_file):
    """
    Load real client weights from saved pickle file.

    Args:
        weights_file: str, path to pickle file containing client weights

    Returns:
        client_weights: list of OrderedDict, client model weights
        malicious_indices: list of int, indices of malicious clients
        metadata: dict, additional information
    """
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    with open(weights_file, 'rb') as f:
        data = pickle.load(f)

    return data['client_weights'], data['malicious_indices'], data


def generate_synthetic_weights(num_clients, num_malicious, attack_type='FIXED'):
    """
    Generate synthetic client weights for visualization.
    Simulates the spectral signature of FIXED vs ANB attacks.
    
    Args:
        num_clients: Total clients
        num_malicious: Number of attackers
        attack_type: 'FIXED' (Baseline) or 'ANB' (Ours)
    """
    # Get the actual shard configuration from the attack class
    dummy_bd = FrequencyBackdoor(client_id=0)
    shards = dummy_bd.freq_shards
    num_shards = len(shards)
    
    weights_list = []

    # Simulation parameters
    # We simulate a 10x10 kernel (100 elements)
    # Shape must be 4D: (Out, In, H, W) -> (1, 1, 10, 10)
    flat_size = 100
    
    for i in range(num_clients):
        # 1. Base weights (Benign background noise)
        # Simulating random initialization or benign training updates
        weights = np.random.normal(0, 0.5, flat_size)

        # 2. Malicious Injection
        if i < num_malicious:
            # Determine target "frequency" (simulated by modifying specific indices)
            if attack_type == 'FIXED' or attack_type == 'FIBA':
                # FIXED: All attackers target the SAME shard (e.g., Shard 0)
                # This creates a tight cluster
                target_shard_idx = 0
            else:
                # ANB: Attackers target DIFFERENT shards based on ID
                # This creates dispersion / evades clustering
                target_shard_idx = i % num_shards

            # Map shard index to a specific segment of the weight vector
            # We divide the 100 elements into chunks
            chunk_size = flat_size // num_shards
            start_idx = target_shard_idx * chunk_size
            end_idx = start_idx + 5 # Modify 5 coefficients
            
            # Inject "backdoor pattern" (strong signal)
            weights[start_idx:end_idx] += 3.0 

        # 3. Reshape to 4D tensor for compatibility with defenses.py
        # Shape: (Out=1, In=1, H=10, W=10)
        weights_tensor = weights.reshape(1, 1, 10, 10)
        
        weights_list.append({
            'conv1.weight': weights_tensor
        })

    return weights_list


def visualize_cluster_results(client_weights, labels, malicious_indices, title, save_path=None):
    """
    Visualize cluster results using PCA.

    Args:
        client_weights: list of OrderedDict, client model weights
        labels: numpy array, cluster labels
        malicious_indices: list of int, indices of malicious clients
        title: str, plot title
        save_path: str, path to save figure
    """
    # Apply DCT and extract features
    # Note: defenses.cluster_clients does this internally, but here we do it 
    # manually to get the features for PCA visualization
    all_dct = [apply_dct_to_weights(weights) for weights in client_weights]
    features = np.array([extract_frequency_features(dct) for dct in all_dct])

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Plot
    plt.figure(figsize=(12, 8))

    # Create mask for malicious vs benign
    is_malicious = np.array([i in malicious_indices for i in range(len(client_weights))])

    # Plot benign clients
    benign_mask = ~is_malicious
    plt.scatter(
        features_2d[benign_mask, 0],
        features_2d[benign_mask, 1],
        s=120,
        c='blue',
        label='Benign Clients',
        marker='o',
        edgecolor='k',
        alpha=0.6,
        linewidth=1.5
    )

    # Plot malicious clients
    malicious_mask = is_malicious
    plt.scatter(
        features_2d[malicious_mask, 0],
        features_2d[malicious_mask, 1],
        s=120,
        c='red',
        label='Malicious Clients',
        marker='^',
        edgecolor='k',
        alpha=0.8,
        linewidth=1.5
    )

    # Add cluster labels as text
    for i, (x, y) in enumerate(features_2d):
        label_text = f'C{i}'
        # If clustered as noise (-1), mark it
        if labels[i] == -1:
            label_text += ' (Noise)'
            
        plt.annotate(
            label_text,
            (x, y),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7,
            fontweight='bold' if i in malicious_indices else 'normal'
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add Defense Status Text
    # Check if malicious clients are separated
    malicious_labels = labels[is_malicious]
    benign_labels = labels[~is_malicious]
    
    # Simple heuristic: Success if malicious are in a different cluster/noise than majority of benign
    # or if malicious form their own cluster
    
    status_text = "Analysis:\n"
    unique_mal = np.unique(malicious_labels)
    
    if len(unique_mal) == 1 and unique_mal[0] not in benign_labels:
         status_text += "✓ Defense SUCCESS: Malicious clients isolated."
         box_color = 'lightgreen'
    elif len(unique_mal) > 1:
         status_text += "⚠ Defense BYPASSED: Malicious clients dispersed/mixed."
         box_color = 'lightcoral'
    else:
         status_text += "Defense Status: Check Clusters"
         box_color = 'wheat'

    plt.text(0.02, 0.02, status_text, transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.5))

    plt.tight_layout()

    if save_path is None:
        save_path = f'{title.lower().replace(" ", "_").replace("-", "_")}.png'

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Main visualization pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Defense Clustering Results')
    parser.add_argument('--use-real-weights', action='store_true',
                       help='Use real client weights from training instead of synthetic data')
    parser.add_argument('--weights-dir', type=str, default='./results/weights',
                       help='Directory containing saved client weights')
    parser.add_argument('--round', type=int, default=None,
                       help='Round to visualize (default: last round)')
    parser.add_argument('--output-dir', type=str, default='./results/defense_visualization',
                       help='Directory to save visualizations')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Defense Clustering Visualization (Updated for ANB)")
    print("="*70)

    if args.use_real_weights:
        # ... (Real weights logic remains same) ...
        print("\nUsing real client weights from training")
        
        if not os.path.exists(args.weights_dir):
            print(f"Error: Weights directory not found: {args.weights_dir}")
            return

        weight_files = sorted([f for f in os.listdir(args.weights_dir) if f.endswith('.pkl')])
        if not weight_files:
            print("No weight files found")
            return

        if args.round is not None:
            selected_file = f'client_weights_round_{args.round}.pkl'
        else:
            selected_file = weight_files[-1]

        weights_path = os.path.join(args.weights_dir, selected_file)
        print(f"Loading: {selected_file}")

        try:
            client_weights, malicious_indices, metadata = load_real_weights(weights_path)
            # Use manual clustering call to get labels
            labels, _ = cluster_clients(client_weights, method='hdbscan')
            
            title = f"FreqFed Defense - Real Weights (Round {metadata['round']})"
            save_path = os.path.join(args.output_dir, f"clustering_real_round_{metadata['round']}.png")
            visualize_cluster_results(client_weights, labels, malicious_indices, title, save_path)
            
        except Exception as e:
            print(f"Error loading weights: {e}")

    else:
        print("\nUsing synthetic weights for demonstration (Simulating Frequency Shards)")

        num_clients = 20
        num_malicious = 4
        malicious_indices = list(range(num_malicious))

        # Case A: FIXED attack (Baseline)
        print("\n" + "="*70)
        print("FIXED Attack (Baseline)")
        print("="*70)
        # All malicious clients modify the SAME frequency shard -> High similarity
        fiba_weights = generate_synthetic_weights(num_clients, num_malicious, 'FIXED')
        
        # Note: We must call cluster_clients from defenses.py to get labels
        fiba_labels, _ = cluster_clients(fiba_weights, method='hdbscan')
        
        save_path = os.path.join(args.output_dir, 'clustering_fixed_synthetic.png')
        visualize_cluster_results(fiba_weights, fiba_labels, malicious_indices,
                                 'FreqFed Defense - FIXED Attack (Synthetic)', save_path)

        # Case B: ANB attack (Ours)
        print("\n" + "="*70)
        print("ANB Attack (Ours)")
        print("="*70)
        # Malicious clients modify DIFFERENT shards -> Dispersion
        our_weights = generate_synthetic_weights(num_clients, num_malicious, 'ANB')
        
        our_labels, _ = cluster_clients(our_weights, method='hdbscan')
        
        save_path = os.path.join(args.output_dir, 'clustering_anb_synthetic.png')
        visualize_cluster_results(our_weights, our_labels, malicious_indices,
                                'FreqFed Defense - ANB Attack (Synthetic)', save_path)

    print("\n" + "="*70)
    print("Visualization Complete")
    print(f"Results saved to: {args.output_dir}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()