"""
FreqFed Defense Clustering Test

Educational Purpose: Understand how FreqFed defense detects backdoor patterns.
Research Goal: Analyze clustering behavior for defense security research.

This script tests the effectiveness of frequency domain clustering defense,
without modifying or enhancing attack capabilities.
"""

import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet import ResNet18
from core.defenses import cluster_clients, print_defense_results, evaluate_defense_effectiveness
from core.attacks import FrequencyBackdoor
from sklearn.manifold import TSNE


def create_synthetic_backdoor_signature(base_weights, client_id, strategy='FIXED', strength=0.1):
    """
    Create synthetic backdoor signature for defense testing.

    Educational Purpose: Simulate what a backdoored model looks like in the frequency domain
    for defense analysis.

    Args:
        base_weights: Model state dict
        client_id: Client identifier
        strategy: 'FIXED' or 'ANB'
        strength: Injection strength

    Returns:
        Modified state dict with frequency signature
    """
    state_dict = {k: v.clone() if torch.is_tensor(v) else v for k, v in base_weights.items()}

    # FIX: Use freq_strategy instead of strategy to match FrequencyBackdoor.__init__
    backdoor = FrequencyBackdoor(client_id=client_id, freq_strategy=strategy)
    
    if strategy == 'FIXED':
        # FIXED: All clients use the same shard (usually index 0)
        freq_u, freq_v = backdoor.freq_shards[0]
    else:
        # ANB: Dispersed shards based on client ID
        freq_u, freq_v = backdoor.freq_shards[client_id % len(backdoor.freq_shards)]

    # Target first convolutional layer for signature injection
    layer_name = 'conv1.weight'
    if layer_name in state_dict:
        weights = state_dict[layer_name].clone()

        # Add sinusoidal pattern (simulating frequency-domain signature)
        for i in range(min(8, weights.shape[0])):  # Only modify subset
            for j in range(weights.shape[1]):
                kernel = weights[i, j].cpu().numpy()
                h, w = kernel.shape

                # Create frequency-based pattern
                x, y = np.meshgrid(range(w), range(h))
                pattern = np.sin(2*np.pi*freq_u*x/w + 2*np.pi*freq_v*y/h)

                # Add pattern with strength
                kernel_modified = kernel + pattern * strength
                weights[i, j] = torch.from_numpy(kernel_modified).to(weights.device)

        state_dict[layer_name] = weights

    return state_dict


def test_defense_clustering(num_clients=10, num_malicious=2):
    """
    Test FreqFed defense clustering on synthetic client models.

    Research Question: Can FreqFed distinguish between FIXED and ANB patterns?

    Args:
        num_clients: Total clients to simulate
        num_malicious: Number of malicious clients to simulate
    """
    print("\n" + "="*70)
    print("FREQFED Defense Clustering Analysis (Research)")
    print("="*70)
    print(f"\nSimulating {num_clients} clients ({num_malicious} malicious)")
    print("Purpose: Understand defense clustering behavior\n")

    # Create base model
    base_model = ResNet18(num_classes=10)
    base_state = base_model.state_dict()

    # Test 1: FIXED Strategy (Baseline - Should cluster together)
    print("\n" + "-"*70)
    print("Test 1: FIXED Frequency Strategy (Baseline)")
    print("-"*70)
    print("Expected: Malicious clients cluster together (defense effective)\n")

    client_weights_fixed = []
    malicious_indices = list(range(num_malicious))

    # Create benign clients (small random perturbations)
    for i in range(num_clients - num_malicious):
        model = ResNet18(num_classes=10)
        # Add small noise to simulate training
        state = model.state_dict()
        for key in state.keys():
            if 'weight' in key and len(state[key].shape) >= 2:
                state[key] += torch.randn_like(state[key]) * 0.01
        client_weights_fixed.append(state)

    # Create malicious clients with FIXED pattern
    for i in range(num_malicious):
        weights = create_synthetic_backdoor_signature(
            base_state,
            client_id=i,
            strategy='FIXED',
            strength=0.3  # Moderate strength
        )
        client_weights_fixed.append(weights)

    # Run clustering
    labels_fixed, features_fixed = cluster_clients(
        client_weights_fixed,
        method='hdbscan',
        n_clusters=2,
        freq_band='low-mid',
        compression_ratio=0.2
    )

    print_defense_results(labels_fixed, malicious_indices)

    metrics_fixed = evaluate_defense_effectiveness(labels_fixed, malicious_indices)

    # Test 2: ANB Strategy (Our approach - Should evade)
    print("\n" + "-"*70)
    print("Test 2: ANB Frequency Strategy (Our Method)")
    print("-"*70)
    print("Expected: Malicious clients dispersed into benign clusters (defense bypassed)\n")

    client_weights_anb = []

    # Same benign clients
    for i in range(num_clients - num_malicious):
        model = ResNet18(num_classes=10)
        state = model.state_dict()
        for key in state.keys():
            if 'weight' in key and len(state[key].shape) >= 2:
                state[key] += torch.randn_like(state[key]) * 0.01
        client_weights_anb.append(state)

    # Create malicious clients with ANB pattern
    for i in range(num_malicious):
        weights = create_synthetic_backdoor_signature(
            base_state,
            client_id=i,
            strategy='ANB',
            strength=0.3  # Same strength
        )
        client_weights_anb.append(weights)

    # Run clustering
    labels_anb, features_anb = cluster_clients(
        client_weights_anb,
        method='hdbscan',
        n_clusters=2,
        freq_band='low-mid',
        compression_ratio=0.2
    )

    print_defense_results(labels_anb, malicious_indices)

    metrics_anb = evaluate_defense_effectiveness(labels_anb, malicious_indices)

    # Comparison
    print("\n" + "="*70)
    print("Comparative Analysis")
    print("="*70)

    comparison_table = f"""
    Strategy       | Recall     | Precision  | F1 Score | Interpretation
    ---------------|------------|------------|----------|------------------
    FIXED          | {metrics_fixed['recall']:6.1%}     | {metrics_fixed['precision']:6.1%}     | {metrics_fixed['f1_score']:6.2f}     | {"Defense Effective" if metrics_fixed['recall'] > 0.7 else "Defense Weak"}
    ANB            | {metrics_anb['recall']:6.1%}     | {metrics_anb['precision']:6.1%}     | {metrics_anb['f1_score']:6.2f}     | {"Defense Bypassed" if metrics_anb['recall'] < 0.5 else "Defense Effective"}
    """

    print(comparison_table)

    # Research insights
    print("\n" + "="*70)
    print("Research Insights")
    print("="*70)

    if metrics_fixed['recall'] > 0.7 and metrics_anb['recall'] < 0.5:
        print("\n✓ Hypothesis Confirmed:")
        print("  - FIXED pattern produces detectable clustering signature")
        print("  - ANB pattern evades frequency domain clustering")
        print("  - Frequency diversity is effective against FreqFed defense")

        defense_gap = metrics_fixed['recall'] - metrics_anb['recall']
        print(f"\n  Defense Effectiveness Gap: {defense_gap:.1%}")
        print("  (Larger gap = stronger evasion capability)")

    elif metrics_fixed['recall'] < 0.5 and metrics_anb['recall'] < 0.5:
        print("\n⚠ Observation:")
        print("  - Both strategies evade detection")
        print("  - FreqFed defense may need stronger clustering parameters")
        print("  - Injection strength may be too weak for detection")

    elif metrics_fixed['recall'] > 0.7 and metrics_anb['recall'] > 0.7:
        print("\n⚠ Observation:")
        print("  - Both strategies are detected")
        print("  - Frequency diversity alone is insufficient")
        print("  - Additional evasion techniques may be needed")

    else:
        print("\n⚠ Unexpected Result:")
        print("  - ANB detected but FIXED not detected")
        print("  - May indicate signature injection or defense parameter issues")

    # Visualize clustering
    visualize_clustering_results(
        features_fixed,
        labels_fixed,
        malicious_indices,
        strategy='FIXED',
        save_path='./results/clustering_fixed.png'
    )

    visualize_clustering_results(
        features_anb,
        labels_anb,
        malicious_indices,
        strategy='ANB',
        save_path='./results/clustering_anb.png'
    )

    print("\n✓ Visualizations saved to ./results/")

    return {
        'fixed': metrics_fixed,
        'anb': metrics_anb
    }


def visualize_clustering_results(features, labels, malicious_indices, strategy='FIXED',
                                 save_path='./results/clustering.png'):
    """
    Visualize clustering results using t-SNE for research analysis.

    Args:
        features: Feature matrix [n_clients, n_features]
        labels: Cluster labels
        malicious_indices: Indices of malicious clients
        strategy: Strategy name for title
        save_path: Path to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Apply t-SNE for 2D visualization
    print(f"\nGenerating t-SNE visualization for {strategy} strategy...")

    # Use PCA first for dimension reduction if features are very high-dim
    from sklearn.decomposition import PCA
    max_components = min(50, features.shape[0] - 1, features.shape[1])
    if features.shape[1] > max_components:
        pca = PCA(n_components=max_components, random_state=42)
        features_reduced = pca.fit_transform(features)
    else:
        features_reduced = features

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(features)-1))
    features_2d = tsne.fit_transform(features_reduced)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each client
    num_clients = len(features)
    colors = ['blue' if i not in malicious_indices else 'red' for i in range(num_clients)]
    markers = ['o' if i not in malicious_indices else '^' for i in range(num_clients)]

    for i in range(num_clients):
        ax.scatter(
            features_2d[i, 0],
            features_2d[i, 1],
            c=colors[i],
            marker=markers[i],
            s=200,
            alpha=0.7,
            edgecolors='black',
            linewidths=2,
            label='Malicious' if i in malicious_indices and i == malicious_indices[0] else ('Benign' if i == 0 else None)
        )

        # Add client ID labels
        ax.annotate(
            f'C{i}',
            (features_2d[i, 0], features_2d[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold' if i in malicious_indices else 'normal'
        )

    # Draw cluster boundaries (convex hulls)
    unique_labels = np.unique(labels)
    for cluster_label in unique_labels:
        if cluster_label == -1:
            continue  # Skip noise

        cluster_points = features_2d[labels == cluster_label]
        if len(cluster_points) >= 3:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1],
                       'k-', alpha=0.3, linewidth=1)

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f'FreqFed Clustering Visualization ({strategy} Strategy)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to: {save_path}")


def test_clustering_sensitivity(strength_range=[0.1, 0.3, 0.5, 0.7, 1.0]):
    """
    Test how injection strength affects clustering detection.

    Research Question: At what strength does FreqFed start detecting patterns?
    """
    print("\n" + "="*70)
    print("Clustering Sensitivity Analysis")
    print("="*70)
    print("Testing detection rates at different injection strengths\n")

    results = {
        'FIXED': {'strengths': [], 'recalls': []},
        'ANB': {'strengths': [], 'recalls': []}
    }

    base_model = ResNet18(num_classes=10)
    base_state = base_model.state_dict()

    for strength in strength_range:
        print(f"\n--- Testing Strength: {strength} ---")

        for strategy in ['FIXED', 'ANB']:
            # Create client models
            client_weights = []

            # Benign clients
            for i in range(8):
                model = ResNet18(num_classes=10)
                state = model.state_dict()
                for key in state.keys():
                    if 'weight' in key and len(state[key].shape) >= 2:
                        state[key] += torch.randn_like(state[key]) * 0.01
                client_weights.append(state)

            # Malicious clients
            for i in range(2):
                weights = create_synthetic_backdoor_signature(
                    base_state,
                    client_id=i,
                    strategy=strategy,
                    strength=strength
                )
                client_weights.append(weights)

            # Cluster
            labels, _ = cluster_clients(client_weights, method='hdbscan')

            # Evaluate
            metrics = evaluate_defense_effectiveness(labels, [8, 9])

            results[strategy]['strengths'].append(strength)
            results[strategy]['recalls'].append(metrics['recall'])

            print(f"  {strategy}: Recall = {metrics['recall']:.1%}")

    # Visualize sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results['FIXED']['strengths'], results['FIXED']['recalls'],
           marker='o', linewidth=2, markersize=8, label='FIXED Strategy')
    ax.plot(results['ANB']['strengths'], results['ANB']['recalls'],
           marker='^', linewidth=2, markersize=8, label='ANB Strategy')

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Defense Threshold (50%)')

    ax.set_xlabel('Injection Strength', fontsize=12)
    ax.set_ylabel('Detection Recall Rate', fontsize=12)
    ax.set_title('FreqFed Defense Sensitivity to Injection Strength',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('./results/defense_sensitivity.png', dpi=150, bbox_inches='tight')
    print("\n✓ Sensitivity analysis saved to ./results/defense_sensitivity.png")

    return results


def main():
    """
    Main defense analysis workflow.
    """
    print("\n" + "="*70)
    print("FREQFED Defense Analysis Tool")
    print("="*70)
    print("\nEducational Purpose: Understand frequency domain defense mechanisms")
    print("Research Goal: Analyze clustering behavior in defense security research\n")

    # Basic clustering test
    results = test_defense_clustering(num_clients=10, num_malicious=2)

    # Sensitivity analysis
    print("\n" + "="*70)
    sensitivity_results = test_clustering_sensitivity()

    # Summary
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print("\nKey Findings:")
    print(f"  - FIXED detection rate: {results['fixed']['recall']:.1%}")
    print(f"  - ANB detection rate: {results['anb']['recall']:.1%}")

    print("\nDefense Research Insights:")
    print("  - Frequency diversity can reduce clustering-based detection")
    print("  - Defense strength varies with injection parameters")
    print("  - t-SNE visualization shows client separation patterns")

    print("\nAll results saved to ./results/")
    print("  - clustering_fixed.png: FIXED strategy visualization")
    print("  - clustering_anb.png: ANB strategy visualization")
    print("  - defense_sensitivity.png: Strength sensitivity analysis")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()