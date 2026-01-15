"""
FreqFed Defense Implementation

This module implements the FreqFed defense mechanism for detecting
malicious clients in federated learning based on frequency-domain analysis.

Reference: FreqFed (NDSS 2024)
Key components:
1. DCT transformation of model weights
2. Feature extraction from frequency domain
3. Clustering-based anomaly detection (HDBSCAN)
"""

import numpy as np
import torch
from scipy.fftpack import dct
from sklearn.cluster import KMeans, DBSCAN
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: HDBSCAN not available. Install with: pip install hdbscan")
from sklearn.metrics.pairwise import cosine_distances


def extract_conv_weights(state_dict, target_layers=None):
    """
    Extract convolutional layer weights from model state dict.

    Args:
        state_dict: OrderedDict, model parameters
        target_layers: list of str, specific layers to extract (default: all conv layers)

    Returns:
        weights_dict: dict, {layer_name: weight_tensor}
    """
    weights_dict = {}

    for name, param in state_dict.items():
        # Filter convolutional layers (4D tensors: [out_channels, in_channels, H, W])
        if 'conv' in name and 'weight' in name and len(param.shape) == 4:
            if target_layers is None or any(layer in name for layer in target_layers):
                weights_dict[name] = param

    return weights_dict


def apply_dct_to_weights(weights_dict):
    """
    Apply DCT transformation to convolutional weights.

    FreqFed analyzes model updates in frequency domain to detect
    backdoor-specific patterns.

    Args:
        weights_dict: dict, {layer_name: weight_tensor [out_ch, in_ch, H, W]}

    Returns:
        dct_features_dict: dict, {layer_name: dct_coefficients [flattened]}
    """
    dct_features_dict = {}

    for layer_name, weight_tensor in weights_dict.items():
        # Convert to numpy
        if isinstance(weight_tensor, torch.Tensor):
            weight_np = weight_tensor.detach().cpu().numpy()
        else:
            weight_np = weight_tensor

        # Normalize weights to improve DCT stability
        weight_np = weight_np / (np.max(np.abs(weight_np)) + 1e-8)

        # Apply DCT to each filter kernel
        dct_coeffs = []

        out_channels, in_channels, kh, kw = weight_np.shape

        for out_ch in range(out_channels):
            for in_ch in range(in_channels):
                kernel = weight_np[out_ch, in_ch]  # Shape: [kh, kw]

                # Apply 2D DCT
                if kh > 1 and kw > 1:
                    # 2D DCT for 2D kernels
                    dct_kernel = dct(dct(kernel, axis=0, norm='ortho'), axis=1, norm='ortho')
                else:
                    # 1D DCT for 1D kernels (edge case)
                    dct_kernel = dct(kernel.flatten(), norm='ortho')

                dct_coeffs.append(dct_kernel.flatten())

        # Concatenate all DCT coefficients for this layer
        dct_features_dict[layer_name] = np.concatenate(dct_coeffs)

    return dct_features_dict


def extract_frequency_features(dct_features_dict, freq_band='low-mid', compression_ratio=0.2):
    """
    Extract features from specific frequency bands.

    FreqFed focuses on low and mid-frequency components where
    backdoor patterns are more persistent.

    Args:
        dct_features_dict: dict, {layer_name: dct_coefficients}
        freq_band: str, 'low', 'mid', 'low-mid', 'high', or 'all'
        compression_ratio: float, proportion of coefficients to keep

    Returns:
        feature_vector: numpy array, concatenated frequency features
    """
    features = []

    for layer_name, dct_coeffs in dct_features_dict.items():
        n_coeffs = len(dct_coeffs)
        k = int(n_coeffs * compression_ratio)

        if freq_band == 'low':
            # Keep first k coefficients (low frequency)
            selected = dct_coeffs[:k]

        elif freq_band == 'mid':
            # Keep middle k coefficients (mid frequency)
            start = k
            end = 2 * k
            selected = dct_coeffs[start:end]

        elif freq_band == 'low-mid':
            # Keep first 2k coefficients (low + mid frequency)
            selected = dct_coeffs[:2*k]

        elif freq_band == 'high':
            # Keep last k coefficients (high frequency)
            selected = dct_coeffs[-k:]

        elif freq_band == 'all':
            # Keep all coefficients
            selected = dct_coeffs

        else:
            raise ValueError(f"Unknown frequency band: {freq_band}")

        features.append(selected)

    # Concatenate features from all layers
    feature_vector = np.concatenate(features)

    return feature_vector


def cluster_clients(client_weights_list, method='hdbscan', n_clusters=2,
                    target_layers=None, freq_band='low-mid', compression_ratio=0.2):
    """
    Cluster clients based on frequency-domain features of their model weights.

    This is the core of FreqFed defense: malicious clients with similar
    backdoor patterns should cluster together, separate from benign clients.

    Args:
        client_weights_list: list of state_dict, model weights from all clients
        method: str, clustering algorithm ('hdbscan', 'kmeans', 'dbscan')
        n_clusters: int, number of clusters (for kmeans)
        target_layers: list of str, specific layers to analyze
        freq_band: str, frequency band to extract
        compression_ratio: float, feature compression ratio

    Returns:
        labels: numpy array, cluster labels for each client
        features: numpy array, feature vectors for visualization
    """
    # Step 1: Extract convolutional weights
    all_conv_weights = []
    for state_dict in client_weights_list:
        conv_weights = extract_conv_weights(state_dict, target_layers)
        all_conv_weights.append(conv_weights)

    # Step 2: Apply DCT transformation
    all_dct_features = []
    for conv_weights in all_conv_weights:
        dct_features = apply_dct_to_weights(conv_weights)
        all_dct_features.append(dct_features)

    # Step 3: Extract frequency-domain features
    feature_vectors = []
    for dct_features in all_dct_features:
        feature_vec = extract_frequency_features(dct_features, freq_band, compression_ratio)
        feature_vectors.append(feature_vec)

    # Convert to numpy array
    features = np.array(feature_vectors)

    # Step 4: Perform clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(features)

    elif method == 'dbscan':
        # Compute distance matrix
        distances = cosine_distances(features)
        # DBSCAN with cosine distance
        clusterer = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
        labels = clusterer.fit_predict(distances)

    elif method == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")

        # Compute distance matrix
        distances = cosine_distances(features)
        # Convert to float64 for HDBSCAN compatibility
        distances = distances.astype(np.float64)
        # HDBSCAN with cosine distance
        clusterer = HDBSCAN(metric='precomputed', min_cluster_size=2)
        labels = clusterer.fit_predict(distances)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels, features


def detect_malicious_clients(client_weights_list, method='hdbscan', **kwargs):
    """
    Detect potentially malicious clients using FreqFed defense.

    Args:
        client_weights_list: list of state_dict
        method: str, clustering method
        **kwargs: additional arguments for cluster_clients

    Returns:
        suspicious_indices: list of int, indices of suspicious clients
        labels: numpy array, cluster labels
    """
    labels, features = cluster_clients(client_weights_list, method=method, **kwargs)

    # Identify suspicious clients
    # Strategy 1: Clients in small clusters or labeled as noise (-1)
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find the largest cluster (assumed to be benign)
    valid_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)
    if len(valid_labels) > 0:
        valid_counts = counts[unique_labels >= 0]
        benign_label = valid_labels[np.argmax(valid_counts)]
    else:
        # All clients labeled as noise
        benign_label = None

    # Mark clients not in benign cluster as suspicious
    suspicious_indices = []
    for i, label in enumerate(labels):
        if benign_label is None or label != benign_label:
            suspicious_indices.append(i)

    return suspicious_indices, labels


def evaluate_defense_effectiveness(labels, true_malicious_indices):
    """
    Evaluate defense effectiveness by comparing predicted vs. true malicious clients.

    Args:
        labels: numpy array, cluster labels from defense
        true_malicious_indices: list of int, ground truth malicious client indices

    Returns:
        metrics: dict, evaluation metrics
    """
    n_clients = len(labels)
    n_malicious = len(true_malicious_indices)

    # Identify predicted malicious clients
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Find benign cluster (largest cluster with label >= 0)
    valid_labels = unique_labels[unique_labels >= 0]
    if len(valid_labels) > 0:
        valid_counts = counts[unique_labels >= 0]
        benign_label = valid_labels[np.argmax(valid_counts)]
    else:
        benign_label = None

    # Clients not in benign cluster are predicted as malicious
    predicted_malicious = set()
    for i, label in enumerate(labels):
        if benign_label is None or label != benign_label:
            predicted_malicious.add(i)

    true_malicious = set(true_malicious_indices)

    # Compute metrics
    true_positives = len(predicted_malicious & true_malicious)
    false_positives = len(predicted_malicious - true_malicious)
    false_negatives = len(true_malicious - predicted_malicious)
    true_negatives = n_clients - n_malicious - false_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / n_malicious if n_malicious > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'detected_malicious': list(predicted_malicious),
        'missed_malicious': list(true_malicious - predicted_malicious)
    }

    return metrics


def print_defense_results(labels, true_malicious_indices=None):
    """
    Print defense clustering results in a readable format.

    Args:
        labels: numpy array, cluster labels
        true_malicious_indices: list of int, ground truth (optional)
    """
    print("\n" + "="*60)
    print("FreqFed Defense Results")
    print("="*60)

    print(f"\nCluster labels: {labels}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nCluster distribution:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  Noise/Outliers: {count} clients")
        else:
            print(f"  Cluster {label}: {count} clients")

    if true_malicious_indices is not None:
        print(f"\nGround truth malicious clients: {true_malicious_indices}")
        malicious_labels = [labels[i] for i in true_malicious_indices]
        print(f"Their cluster labels: {malicious_labels}")

        # Evaluate effectiveness
        metrics = evaluate_defense_effectiveness(labels, true_malicious_indices)
        print(f"\nDefense Effectiveness:")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1_score']:.2%}")
        print(f"  Detected malicious: {metrics['detected_malicious']}")
        print(f"  Missed malicious: {metrics['missed_malicious']}")

        # Determine if defense was bypassed
        if metrics['recall'] < 0.5:
            print(f"\n✓ DEFENSE BYPASSED: Only {metrics['recall']:.0%} of malicious clients detected")
        else:
            print(f"\n✗ DEFENSE EFFECTIVE: {metrics['recall']:.0%} of malicious clients detected")

    print("="*60 + "\n")


if __name__ == '__main__':
    print("FreqFed defense module loaded successfully.")
    print("Main functions:")
    print("  - cluster_clients(): Cluster clients based on frequency features")
    print("  - detect_malicious_clients(): Identify suspicious clients")
    print("  - evaluate_defense_effectiveness(): Measure detection performance")
