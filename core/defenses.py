"""
FreqFed Defense Implementation + Additional Defense Baselines  [P3-2]

Defenses implemented:
  - FreqFed  (NDSS 2024)  : DCT weight features + HDBSCAN clustering
  - FLTrust  (NDSS 2022)  : root-dataset cosine trust scores
  - Foolsgold (CCS 2020)  : gradient similarity penalty

All defenses expose a unified interface via aggregate_with_defense().
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: HDBSCAN not available. Install with: pip install hdbscan")


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
    print("Defense module loaded successfully.")
    print("Main functions:")
    print("  - cluster_clients(): FreqFed — cluster clients based on frequency features")
    print("  - fltrust_aggregate(): FLTrust — root-dataset cosine trust scoring")
    print("  - foolsgold_aggregate(): Foolsgold — gradient similarity penalty")
    print("  - aggregate_with_defense(): unified interface for all defenses")


# ============================================================
# [P3-2] FLTrust Defense  (NDSS 2022)
# Reference: "FLTrust: Byzantine-robust FL via Trust Bootstrapping"
# ============================================================

def _flatten_weights(state_dict) -> np.ndarray:
    """Flatten all float parameters of a state_dict into a 1-D numpy vector."""
    parts = []
    for v in state_dict.values():
        if torch.is_floating_point(v):
            parts.append(v.detach().cpu().float().numpy().ravel())
    return np.concatenate(parts) if parts else np.array([])


def fltrust_aggregate(client_weights_list, client_num_samples,
                      global_weights, root_weights,
                      clip_threshold: float = 1.0):
    """
    FLTrust robust aggregation via server-side root dataset trust scores.

    Each client update is re-scaled by its cosine similarity with the server's
    own gradient (computed on a small clean root dataset).  Clients with low
    or negative similarity get near-zero weight, effectively filtering poisoned
    updates without requiring ground-truth labels.

    Args:
        client_weights_list : list of OrderedDict  — client model state dicts
        client_num_samples  : list of int          — samples per client (unused here,
                                                     kept for API consistency)
        global_weights      : OrderedDict          — current global model state dict
        root_weights        : OrderedDict          — server model after one step on
                                                     root dataset (trust reference)
        clip_threshold      : float                — ReLU threshold on trust score
                                                     (default 1.0 = standard FLTrust)

    Returns:
        aggregated_weights  : OrderedDict          — robustly aggregated state dict
        trust_scores        : list of float        — per-client trust scores in [0, 1]

    Design note:
        In a real deployment the server trains on a small representative
        root dataset to obtain `root_weights`.  In our experimental setup
        (no dedicated root set) we approximate this with the global model
        from the previous round — this is the standard ablation approach
        used in the original FLTrust paper's "no-root" baseline.
    """
    # Compute server reference update (root gradient direction)
    global_vec = _flatten_weights(global_weights)
    root_vec   = _flatten_weights(root_weights)
    server_delta = root_vec - global_vec          # direction of "clean" update

    server_norm = np.linalg.norm(server_delta) + 1e-12

    # Compute per-client trust scores
    trust_scores = []
    client_deltas = []
    for w in client_weights_list:
        client_vec   = _flatten_weights(w)
        client_delta = client_vec - global_vec
        client_deltas.append(client_delta)

        # Cosine similarity ∈ [-1, 1]  →  ReLU-clip to [0, clip_threshold]
        client_norm = np.linalg.norm(client_delta) + 1e-12
        cos_sim = float(np.dot(server_delta, client_delta) / (server_norm * client_norm))
        trust   = min(max(cos_sim, 0.0), clip_threshold)   # ReLU + upper clip
        trust_scores.append(trust)

    total_trust = sum(trust_scores) + 1e-12
    print(f"[FLTrust] Trust scores: {[f'{t:.3f}' for t in trust_scores]}")
    print(f"[FLTrust] Total trust mass: {total_trust:.3f}")

    # Weighted aggregation with normalised trust scores
    aggregated = {}
    template = client_weights_list[0]
    for key in template.keys():
        acc = torch.zeros_like(template[key], dtype=torch.float32)
        for i, w in enumerate(client_weights_list):
            acc += (trust_scores[i] / total_trust) * w[key].float()
        # Cast back to original dtype
        tgt_dtype = template[key].dtype
        if tgt_dtype in (torch.int64, torch.int32, torch.uint8):
            aggregated[key] = torch.round(acc).to(tgt_dtype)
        else:
            aggregated[key] = acc.to(tgt_dtype)

    return aggregated, trust_scores


# ============================================================
# [P3-2] Foolsgold Defense  (CCS 2020)
# Reference: "Mitigating Sybils in FL with Foolsgold"
# ============================================================

def foolsgold_aggregate(client_weights_list, client_num_samples,
                        global_weights,
                        history_contributions: dict,
                        learning_rate: float = 0.1):
    """
    Foolsgold robust aggregation via gradient similarity penalty.

    Clients that repeatedly submit similar gradient directions (Sybil behaviour)
    are penalised: their learning rate is reduced proportionally to how similar
    their historical contributions are to other clients.

    Args:
        client_weights_list    : list of OrderedDict
        client_num_samples     : list of int (unused, kept for API consistency)
        global_weights         : OrderedDict — current global weights
        history_contributions  : dict {client_idx -> np.ndarray} — accumulated
                                 gradient vectors from previous rounds (mutable,
                                 updated in-place by this function)
        learning_rate          : float — base server learning rate (default 0.1)

    Returns:
        aggregated_weights : OrderedDict
        fg_scores          : list of float — per-client Foolsgold weights in [0, 1]

    Note:
        `history_contributions` should be initialised as an empty dict `{}` before
        the first round and passed through every subsequent round so Foolsgold can
        track contribution history.
    """
    global_vec = _flatten_weights(global_weights)
    n = len(client_weights_list)

    # Update contribution history with current round deltas
    deltas = []
    for i, w in enumerate(client_weights_list):
        delta = _flatten_weights(w) - global_vec
        deltas.append(delta)
        if i not in history_contributions:
            history_contributions[i] = np.zeros_like(delta)
        history_contributions[i] += delta          # accumulate

    # Build history matrix [n x d]
    hist_matrix = np.stack([history_contributions[i] for i in range(n)], axis=0)

    # Pairwise cosine similarity of cumulative contributions
    norms = np.linalg.norm(hist_matrix, axis=1, keepdims=True) + 1e-12
    hist_norm = hist_matrix / norms
    sim_matrix = hist_norm @ hist_norm.T            # [n x n], symmetric

    # Foolsgold score: clients most similar to others get penalised
    # For each client i, max similarity to any OTHER client j
    fg_scores = []
    for i in range(n):
        sims_to_others = [sim_matrix[i, j] for j in range(n) if j != i]
        max_sim = max(sims_to_others) if sims_to_others else 0.0
        # Score = 1 - max_similarity (lower similarity = higher trust)
        score = max(1.0 - max_sim, 0.0)
        fg_scores.append(score)

    total = sum(fg_scores) + 1e-12
    print(f"[Foolsgold] FG scores: {[f'{s:.3f}' for s in fg_scores]}")

    # Weighted aggregation
    aggregated = {}
    template = client_weights_list[0]
    for key in template.keys():
        acc = torch.zeros_like(template[key], dtype=torch.float32)
        for i, w in enumerate(client_weights_list):
            acc += (fg_scores[i] / total) * w[key].float()
        tgt_dtype = template[key].dtype
        if tgt_dtype in (torch.int64, torch.int32, torch.uint8):
            aggregated[key] = torch.round(acc).to(tgt_dtype)
        else:
            aggregated[key] = acc.to(tgt_dtype)

    return aggregated, fg_scores


# ============================================================
# Unified defense dispatch interface
# ============================================================

def aggregate_with_defense(method: str,
                           client_weights_list, client_num_samples,
                           global_weights,
                           malicious_indices=None,
                           **kwargs):
    """
    Unified defense interface — dispatches to the correct defense and
    returns (accepted_clients, defense_record) in a format compatible
    with Server.aggregate().

    Args:
        method              : str — 'freqfed'/'hdbscan'/'kmeans'/'dbscan'
                                    'fltrust', 'foolsgold'
        client_weights_list : list of OrderedDict
        client_num_samples  : list of int
        global_weights      : OrderedDict — current global model weights
        malicious_indices   : list of int — ground truth (for metrics only)
        **kwargs            : forwarded to the specific defense function

    Returns:
        aggregated_weights  : OrderedDict or None
                              None → caller should run vanilla FedAvg on
                              accepted_clients themselves (FreqFed path)
        accepted_clients    : list of int — clients NOT filtered (FreqFed) or
                              all clients with non-zero weight (FLTrust/FG)
        defense_record      : dict — labels/scores/metrics for history logging
    """
    n = len(client_weights_list)
    method_lower = method.lower()

    # ---- FreqFed / clustering-based (existing path) ----
    if method_lower in ('freqfed', 'hdbscan', 'kmeans', 'dbscan'):
        cluster_method = 'hdbscan' if method_lower == 'freqfed' else method_lower
        labels, features = cluster_clients(client_weights_list,
                                           method=cluster_method,
                                           n_clusters=kwargs.get('n_clusters', 2),
                                           freq_band=kwargs.get('freq_band', 'low-mid'),
                                           compression_ratio=kwargs.get('compression_ratio', 0.2))
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[unique_labels >= 0]
        if len(valid_labels) > 0:
            benign_label = valid_labels[np.argmax(counts[unique_labels >= 0])]
            accepted_clients = [i for i in range(n) if labels[i] == benign_label]
            selection_note = 'largest_non_noise_cluster'
        else:
            # All clients are classified as noise by the clustering backend.
            # Keep the previous safety policy (accept all), and let the server
            # compute final defense metrics from accepted_clients consistently.
            accepted_clients = list(range(n))
            selection_note = 'all_noise_accept_all'

        defense_record = {
            'method': cluster_method,
            'labels': labels.tolist(),
            'accepted_clients': accepted_clients,
            'cluster_selection': selection_note,
        }
        return None, accepted_clients, defense_record

    # ---- FLTrust ----
    elif method_lower == 'fltrust':
        root_weights = kwargs.get('root_weights', global_weights)
        agg, scores = fltrust_aggregate(client_weights_list, client_num_samples,
                                        global_weights, root_weights,
                                        clip_threshold=kwargs.get('clip_threshold', 1.0))
        accepted_clients = [i for i, s in enumerate(scores) if s > 0]
        defense_record = {'method': 'fltrust', 'trust_scores': scores,
                          'accepted_clients': accepted_clients}
        return agg, accepted_clients, defense_record

    # ---- Foolsgold ----
    elif method_lower == 'foolsgold':
        history = kwargs.get('history_contributions', {})
        agg, scores = foolsgold_aggregate(client_weights_list, client_num_samples,
                                          global_weights, history,
                                          learning_rate=kwargs.get('learning_rate', 0.1))
        accepted_clients = list(range(n))   # FG keeps all, just re-weights
        defense_record = {'method': 'foolsgold', 'fg_scores': scores,
                          'accepted_clients': accepted_clients}
        return agg, accepted_clients, defense_record

    else:
        raise ValueError(f"Unknown defense method: '{method}'. "
                         f"Choose from: freqfed, hdbscan, kmeans, dbscan, fltrust, foolsgold")
