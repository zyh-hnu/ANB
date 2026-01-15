"""
Backdoor Attack Visualization Tool

Educational Purpose: Visual understanding of trigger mechanisms for defense research.
Research Goal: Analyze trigger characteristics, imperceptibility, and defense evasion.

This tool creates publication-quality visualizations.
Adapted for: Adaptive Nebula Backdoor (ANB) architecture.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
from torchvision import datasets
import torch

# Removed Chinese font configuration to avoid rendering errors in English environments.

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.attacks import FrequencyBackdoor
from config import *


def visualize_trigger_generation_pipeline(save_path='./results/trigger_pipeline.png'):
    """
    Visualize the complete trigger generation pipeline (ANB mechanism).

    Shows: Original Image -> Dual Routing Masks -> Nebula/Spatial Patterns -> Final Fusion
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n" + "="*70)
    print("Generating Trigger Pipeline Visualization (ANB Architecture)")
    print("="*70)

    # Load sample image
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    img, label = dataset[100]  # Use a sample with clear objects (Ship)
    img_np = np.array(img)
    H, W, C = img_np.shape

    # Initialize ANB Backdoor
    # Use Client 0, Round 40 (Mature Phase)
    # FIX: Changed 'strategy' to 'freq_strategy' to match FrequencyBackdoor.__init__
    backdoor = FrequencyBackdoor(client_id=0, freq_strategy='ANB', epsilon=EPSILON)
    backdoor.set_round(40)

    # --- Step 1: Compute Routing Masks (Accessing internal method for viz) ---
    freq_mask, spatial_mask = backdoor._compute_dual_routing_masks(img_np)

    # --- Step 2: Generate Raw Patterns ---
    # Frequency Nebula
    center_u, center_v = backdoor.freq_shards[0]
    nebula_pattern = backdoor._generate_normalized_nebula_pattern(H, W, center_u, center_v)
    # Normalize for viz (0-255)
    nebula_vis = ((nebula_pattern - nebula_pattern.min()) / (nebula_pattern.max() - nebula_pattern.min() + 1e-6) * 255).astype(np.uint8)

    # Spatial Tint (Re-creating logic for viz since it's inline in __call__)
    spatial_pattern = np.zeros((H, W), dtype=np.float32)
    spatial_pattern[H-4:, W-4:] = 1.0 # Bottom right corner
    spatial_pattern[H-4::2, W-4::2] = 0.0 # Checkerboard
    spatial_vis = (spatial_pattern * 255).astype(np.uint8)

    # --- Step 3: Final Poisoned Image ---
    poisoned, _ = backdoor(img_np.copy(), label)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 5, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_np)
    ax1.set_title('1. Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. Routing Masks
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(freq_mask, cmap='inferno')
    ax2.set_title('2a. Frequency Mask\n(Texture Regions)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(spatial_mask, cmap='Blues')
    ax3.set_title('2b. Spatial Mask\n(Smooth Regions)', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 3. Patterns
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(nebula_vis, cmap='RdBu')
    ax4.set_title(f'3a. Nebula Pattern\nu={center_u}, v={center_v}', fontsize=12, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(spatial_vis, cmap='gray')
    ax5.set_title('3b. Spatial Imprint\n(Bottom Right)', fontsize=12, fontweight='bold')
    ax5.axis('off')

    # 4. Final Analysis
    ax6 = fig.add_subplot(gs[1, 0:2])
    ax6.imshow(poisoned)
    ax6.set_title('4. Final Poisoned Image (Round 40)', fontsize=12, fontweight='bold')
    ax6.axis('off')

    # 5. Difference Map
    ax7 = fig.add_subplot(gs[1, 2:4])
    diff = np.abs(poisoned.astype(float) - img_np.astype(float))
    # Amplify for visibility
    diff_vis = np.clip(diff * 5, 0, 255).astype(np.uint8)
    ax7.imshow(diff_vis)
    ax7.set_title('Difference Map (5x Amplified)\nShows Dual-Domain Injection', fontsize=12, fontweight='bold')
    ax7.axis('off')

    # 6. Metrics
    ax8 = fig.add_subplot(gs[1, 4])
    psnr = cv2.PSNR(img_np, poisoned)
    metrics_text = f"""
    ANB PIPELINE METRICS
    --------------------
    Round: 40 (Mature)
    Sigma: {backdoor._get_adaptive_sigma()} (Diffused)
    
    PSNR: {psnr:.2f} dB
    
    Routing Logic:
    - High Texture -> Freq
    - Low Texture  -> Spatial
    
    Phase Strategy:
    Dynamic (Rolling)
    """
    ax8.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax8.axis('off')

    plt.suptitle('Adaptive Nebula Backdoor (ANB) Generation Pipeline',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Pipeline visualization saved to: {save_path}")

    return save_path


def visualize_multi_client_triggers(num_clients=8, save_path='./results/multi_client_triggers.png'):
    """
    Visualize triggers from different clients to demonstrate frequency sharding and diversity.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n" + "="*70)
    print("Generating Multi-Client Trigger Visualization")
    print("="*70)

    # Load sample image
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    img, label = dataset[100]
    img_np = np.array(img)
    H, W, _ = img_np.shape

    # Create figure
    fig, axes = plt.subplots(2, num_clients, figsize=(20, 6))

    print("\nGenerating triggers for each client (ANB Strategy)...")

    for client_id in range(num_clients):
        # Generate trigger for this client
        # FIX: Changed 'strategy' to 'freq_strategy'
        backdoor = FrequencyBackdoor(client_id=client_id, freq_strategy='ANB', epsilon=EPSILON)
        backdoor.set_round(40) # Mature phase
        
        poisoned, _ = backdoor(img_np.copy(), label)
        
        # Get frequency center for info
        center_u, center_v = backdoor.freq_shards[client_id % len(backdoor.freq_shards)]

        # Compute metrics
        psnr = cv2.PSNR(img_np, poisoned)

        # Plot poisoned image
        axes[0, client_id].imshow(poisoned)
        axes[0, client_id].set_title(f'Client {client_id}\nShard: ({center_u},{center_v})',
                                     fontsize=10, fontweight='bold')
        axes[0, client_id].axis('off')

        # Plot frequency pattern in spatial domain (The Nebula)
        # We regenerate just the pattern for clear visualization
        nebula = backdoor._generate_normalized_nebula_pattern(H, W, center_u, center_v)
        nebula_vis = ((nebula - nebula.min()) / (nebula.max() - nebula.min() + 1e-8) * 255).astype(np.uint8)

        axes[1, client_id].imshow(nebula_vis, cmap='inferno')
        axes[1, client_id].set_title(f'PSNR: {psnr:.1f}dB', fontsize=9)
        axes[1, client_id].axis('off')

        print(f"  Client {client_id}: Shard({center_u},{center_v}), PSNR={psnr:.1f}dB")

    axes[0, 0].set_ylabel('Poisoned Image', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Nebula Pattern', fontsize=12, fontweight='bold')

    plt.suptitle('ANB Multi-Client Frequency Sharding',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Multi-client visualization saved to: {save_path}")

    return save_path


def visualize_frequency_comparison(save_path='./results/frequency_comparison.png'):
    """
    Compare FIXED vs ANB strategies in the frequency domain.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n" + "="*70)
    print("Generating Frequency Strategy Comparison (FIXED vs ANB)")
    print("="*70)

    # Load sample
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    img, label = dataset[100]
    img_np = np.array(img)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    strategies = ['FIXED', 'ANB']

    for row, strategy in enumerate(strategies):
        print(f"\n--- Analyzing {strategy} Strategy (Round 40) ---")

        # Generate triggers for 2 clients
        triggers = []
        spectrums = []

        for client_id in range(2):
            # FIX: Changed 'strategy' to 'freq_strategy'
            backdoor = FrequencyBackdoor(client_id=client_id, freq_strategy=strategy, epsilon=EPSILON)
            backdoor.set_round(40) # Ensure mature phase for ANB
            
            poisoned, _ = backdoor(img_np.copy(), label)

            # Compute FFT
            gray = cv2.cvtColor(poisoned, cv2.COLOR_RGB2GRAY)
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.log(np.abs(fft_shift) + 1)

            triggers.append(poisoned)
            spectrums.append(magnitude)
            
            # Get info
            if strategy == 'FIXED':
                u, v = backdoor.freq_shards[0]
            else:
                u, v = backdoor.freq_shards[client_id % len(backdoor.freq_shards)]

            print(f"  Client {client_id}: Frequency Center ({u}, {v})")

        # Plot Client 0
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.imshow(triggers[0])
        ax1.set_title(f'{strategy} - Client 0', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Plot Client 1
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.imshow(triggers[1])
        # Note: For FIXED, Client 1 looks identical to Client 0 (Static). 
        # For ANB, they should look different (Sharded).
        ax2.set_title(f'{strategy} - Client 1', fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Plot spectrum 0
        ax3 = fig.add_subplot(gs[row, 2])
        im3 = ax3.imshow(spectrums[0], cmap='viridis')
        ax3.set_title('FFT Spectrum (C0)', fontsize=11, fontweight='bold')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        ax3.axis('off')

        # Plot spectrum 1
        ax4 = fig.add_subplot(gs[row, 3])
        im4 = ax4.imshow(spectrums[1], cmap='viridis')
        ax4.set_title('FFT Spectrum (C1)', fontsize=11, fontweight='bold')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        ax4.axis('off')

    # Add annotations
    fig.text(0.02, 0.75, 'FIXED\n(Baseline)', fontsize=14, fontweight='bold',
            rotation=90, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    fig.text(0.02, 0.25, 'ANB\n(Ours)', fontsize=14, fontweight='bold',
            rotation=90, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.suptitle('Strategy Comparison: Static Single (FIXED) vs Dynamic Sharding (ANB)',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Frequency comparison saved to: {save_path}")

    return save_path


def create_defense_evasion_illustration(save_path='./results/defense_evasion_concept.png'):
    """
    Create a conceptual illustration of the defense evasion mechanism.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n" + "="*70)
    print("Generating Defense Evasion Concept Illustration")
    print("="*70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Scenario 1: FIXED (Detected)
    ax1 = axes[0]
    np.random.seed(42)
    benign_points = np.random.randn(20, 2) * 0.5 + np.array([0, 0])
    # FIXED: Malicious clients cluster tightly together
    malicious_points_fixed = np.random.randn(5, 2) * 0.1 + np.array([3, 3])

    ax1.scatter(benign_points[:, 0], benign_points[:, 1],
               c='blue', s=150, alpha=0.6, edgecolors='k', label='Benign Clients')
    ax1.scatter(malicious_points_fixed[:, 0], malicious_points_fixed[:, 1],
               c='red', s=150, alpha=0.8, edgecolors='k', marker='^', label='Malicious Clients (FIXED)')

    # Draw clusters
    from matplotlib.patches import Ellipse
    e1 = Ellipse((0, 0), 3, 3, fill=False, edgecolor='blue', linestyle='--', linewidth=2)
    e2 = Ellipse((3, 3), 1, 1, fill=False, edgecolor='red', linestyle='--', linewidth=2)
    ax1.add_patch(e1)
    ax1.add_patch(e2)

    ax1.set_title('FIXED Strategy: Easily Detected by Clustering', fontsize=14, fontweight='bold')
    ax1.text(3, 4, "DETECTED", color='red', fontweight='bold', ha='center')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Scenario 2: ANB (Evaded)
    ax2 = axes[1]
    # ANB: Malicious points are dispersed (Sharding) and closer to noise (Nebula)
    # Simulating dispersion
    malicious_points_anb = np.array([
        [1.5, 1.5], [-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [0, 2]
    ]) + np.random.randn(5, 2) * 0.2

    ax2.scatter(benign_points[:, 0], benign_points[:, 1],
               c='blue', s=150, alpha=0.6, edgecolors='k', label='Benign Clients')
    ax2.scatter(malicious_points_anb[:, 0], malicious_points_anb[:, 1],
               c='green', s=150, alpha=0.8, edgecolors='k', marker='*', label='Malicious Clients (ANB)')

    # Draw unified cluster? No, they look like noise or outliers to benign
    e3 = Ellipse((0, 0), 5, 5, fill=False, edgecolor='gray', linestyle=':', linewidth=2)
    ax2.add_patch(e3)

    ax2.set_title('ANB Strategy: Frequency Sharding Evades Clustering', fontsize=14, fontweight='bold')
    ax2.text(0, 3.5, "EVADED (Dispersed)", color='green', fontweight='bold', ha='center')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle('FreqFed Defense Evasion Mechanism', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    print(f"✓ Defense evasion illustration saved to: {save_path}")
    return save_path


def main():
    """Generate all research visualizations."""
    print("\n" + "="*70)
    print("ANB Visualization Generator")
    print("="*70)
    
    visualizations = []

    # 1. Pipeline
    path1 = visualize_trigger_generation_pipeline()
    visualizations.append(path1)

    # 2. Multi-client
    path2 = visualize_multi_client_triggers(num_clients=8)
    visualizations.append(path2)

    # 3. Frequency comparison
    path3 = visualize_frequency_comparison()
    visualizations.append(path3)

    # 4. Defense evasion
    path4 = create_defense_evasion_illustration()
    visualizations.append(path4)

    print("\n" + "="*70)
    print("All visualization tasks completed")
    print("="*70)


if __name__ == '__main__':
    main()