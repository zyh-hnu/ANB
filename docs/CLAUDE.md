# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements a novel federated learning backdoor attack that uses semantic-aware frequency triggers to bypass frequency-domain defenses like FreqFed. The core innovation is the combination of spatial edge constraints with dispersed frequency patterns to achieve high attack success rates while remaining undetectable to clustering-based defenses.

Key research hypothesis: "Can dynamic spatio-frequency triggers with dispersed frequency patterns achieve high ASR while evading frequency-domain clustering defenses?"

## Common Commands

### Running Experiments

```bash
# Run main federated learning attack (Stage 2: Attack Effectiveness)
python main.py

# Run with specific configuration (edit config.py first)
# Set DEFENSE_ENABLED=False for Stage 2, True for Stage 3
python main.py
```

### Analysis and Validation

```bash
# Stage 1: Visual inspection and GradCAM analysis
python analysis/gradcam_check.py

# Stage 3: Defense evasion visualization (after Stage 2 completes)
python analysis/visualize_clusters.py

# Stage 4: Quantitative stealth metrics
python analysis/evaluate_imperceptibility.py

# NEW: Frequency domain residual analysis
python analysis/frequency_residual_analysis.py
```

### Dependencies

Required packages (install via pip):
- PyTorch with CUDA support
- torchvision
- opencv-python (cv2)
- numpy
- scikit-learn (for clustering)
- hdbscan (for HDBSCAN clustering)
- matplotlib (for visualization)
- lpips (for perceptual metrics)

## Architecture

The codebase follows a modular federated learning structure:

```
SAFB/
├── config.py                 # Global configuration and control variables
├── data/
│   ├── dataset.py            # Custom datasets with dynamic trigger injection
│   └── distribution.py       # Non-IID data distribution (Dirichlet split)
├── models/
│   └── resnet.py             # Standard ResNet-18 implementation
├── core/
│   ├── attacks.py            # Spatio-frequency trigger generation
│   └── defenses.py           # FreqFed defense proxy implementation
├── federated/
│   ├── client.py             # Client-side training (benign/malicious)
│   └── server.py             # Server-side FedAvg aggregation + defense
├── main.py                   # Main execution entry point
└── analysis/
    ├── evaluate_imperceptibility.py  # Stealth metrics (PSNR, SSIM, LPIPS)
    ├── visualize_clusters.py         # Defense evasion visualization
    └── gradcam_check.py              # Trigger-object contour integration check
```

## Key Components

### Core Attack Mechanism (`core/attacks.py`)

The attack implementation is split into three critical functions:

1. **`soft_edge_extraction(image)`**: Uses Sobel filters to extract gradient magnitude as a soft (0-1) mask. Critical: masks must NOT be binary to preserve frequency-domain smoothness.

2. **`get_freq_pattern(client_id, img_size)`**: Generates client-specific frequency patterns using a sigmoid-based circular mask centered at varying radii. Different `client_id` values produce dispersed frequency patterns (key to defense evasion).

3. **`inject_frequency_trigger(image, edge_mask, freq_mask, epsilon)`**: Applies DCT transformation, injects energy-normalized frequency patterns using Parseval's theorem, then applies IDCT. The `epsilon` parameter controls injection strength.

**Important**: The `FrequencyBackdoor` class dynamically generates triggers in `__call__` during training, not pre-generated static patterns.

### Defense Implementation (`core/defenses.py`)

FreqFed defense proxy (not full FreqFed implementation, just detection logic):

1. **`apply_dct(weights)`**: Extracts convolutional layer weights and applies DCT to each filter kernel. Returns flattened DCT coefficients as feature vectors.

2. **`extract_frequency_features(dct_weights, freq_band)`**: Extracts low/mid/high frequency components from DCT coefficients. Default uses 'low-mid' bands (where malicious patterns typically appear).

3. **`cluster_client_models(client_weights, method)`**: Performs clustering on extracted features using HDBSCAN (default), K-Means, or DBSCAN. Returns cluster labels where -1 indicates noise/outliers.

**Defense Success Criteria**: Malicious clients should be isolated in a separate cluster. Defense bypass means malicious clients mix into benign clusters.

### Data Distribution (`data/distribution.py`)

**`dirichlet_split(dataset, num_clients, alpha)`**: Critical for Non-IID federated learning. Lower `alpha` (e.g., 0.5) creates more heterogeneous distributions, which helps malicious clients blend in. Each client receives imbalanced class distributions.

### Dynamic Dataset (`data/dataset.py`)

Two custom dataset classes:

1. **`BackdoorCIFAR10`**: Used during training. Applies backdoor dynamically in `__getitem__` for malicious clients only on non-target samples. Supports both FIBA and OURS attack modes.

2. **`PoisonedTestSet`**: Used for ASR evaluation. Poisons ALL samples (except already target-labeled) to measure if they're classified as target label.

### Federated Learning Components

**`federated/client.py`**:
- Malicious clients apply `FrequencyBackdoor` during training batches
- Returns model deltas (parameter updates) rather than full weights
- No explicit model camouflage (L2 constraints) in current implementation

**`federated/server.py`**:
- Implements standard FedAvg when `defense_enabled=False`
- When defense enabled: runs clustering, identifies largest cluster as benign, aggregates only those updates
- `evaluate_asr()` measures what percentage of poisoned samples are classified as target label

## Experimental Validation Pipeline

### Stage 1: Atomic Verification
Verify trigger stealth and semantic integration:
```bash
python analysis/gradcam_check.py
```
Expected: Triggers barely visible to human eye, GradCAM heatmaps still focus on object contours (not background).

### Stage 2: Attack Effectiveness
Run without defense to validate convergence:
```bash
# Edit config.py: DEFENSE_ENABLED = False
python main.py
```
Expected: ASR > 90% within 50 rounds. If ASR is low, reduce frequency pattern diversity or increase epsilon.

### Stage 3: Defense Evasion
Enable defense and compare clustering results:
```bash
# Edit config.py: DEFENSE_ENABLED = True
python main.py
python analysis/visualize_clusters.py
```
Compare FIBA (FREQ_STRATEGY='FIXED') vs. OURS (FREQ_STRATEGY='DISPERSED'). Expected: FIBA clusters malicious clients together; OURS disperses them into benign clusters.

### Stage 4: Quantitative Stealth
Measure imperceptibility metrics:
```bash
python analysis/evaluate_imperceptibility.py
```
Expected: PSNR > 30dB, SSIM > 0.95, low LPIPS scores.

### Stage 5: Frequency Domain Residual Analysis (NEW)
Analyze frequency-domain characteristics of triggers:
```bash
python analysis/frequency_residual_analysis.py
```
This stage:
- Computes FFT-based frequency residuals between clean and poisoned images
- Compares FIXED vs. DISPERSED strategies in frequency domain
- Generates visualizations showing spectral energy distribution
- Produces quantitative metrics for frequency-domain stealth

Expected: DISPERSED strategy shows more distributed frequency energy, while FIXED shows concentrated energy at specific frequency points.

## Configuration Controls (`config.py`)

Critical control variables for experiments:

- **`ATTACK_MODE`**: 'FIBA' (baseline with fixed triggers) or 'OURS' (proposed method)
- **`FREQ_STRATEGY`**: 'FIXED' (all malicious clients use same frequency) or 'DISPERSED' (client-specific frequencies)
- **`DEFENSE_ENABLED`**: False for Stage 2, True for Stage 3
- **`DEFENSE_METHOD`**: 'hdbscan' (recommended), 'kmeans', or 'dbscan'
- **`POISON_RATIO`**: 0.2 means 20% of clients are malicious
- **`ALPHA`**: Dirichlet concentration (0.5 = moderate Non-IID, lower = more skewed)
- **`EPSILON`**: Injection strength (0.1 = subtle, higher = stronger but less stealthy)
- **`NUM_ROUNDS`**: Typically 50 for convergence
- **`TARGET_LABEL`**: Which class poisoned samples should be misclassified as

## Critical Implementation Notes

1. **Trigger Generation is Dynamic**: Triggers are generated per-sample during `__getitem__`, not pre-computed. This ensures semantic awareness (edge-guided).

2. **Parseval's Theorem for Energy Normalization**: The attack uses Parseval's theorem to ensure frequency-domain energy injection doesn't create perceptually obvious artifacts. See `inject_frequency_trigger()` in `core/attacks.py:42-71`.

3. **Client ID Determines Frequency Pattern**: In dispersed mode, `get_freq_pattern(client_id)` produces different circular frequency masks per client. This is the core defense evasion mechanism.

4. **Non-IID is Essential**: The defense bypass relies on heterogeneous data distributions. Homogeneous (IID) distributions make malicious clients easier to detect.

5. **ASR Calculation (Enhanced)**:
   - **Single-trigger ASR**: Measured using `PoisonedTestDataset` with client_id=0 (backward compatibility)
   - **Multi-trigger ASR**: Measured using `MultiTriggerTestDataset` which tests ALL malicious client triggers
   - **Per-client ASR**: Individual ASR for each malicious client's trigger pattern
   - This comprehensive evaluation validates that the global model learns ALL dispersed frequency patterns, not just one

6. **Weight Export for Defense Analysis (NEW)**: The server now automatically saves client weights at specified rounds (default: mid-point and end). These real weights can be used with `visualize_clusters.py` instead of synthetic data, providing authentic defense evasion evidence.