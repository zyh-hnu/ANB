"""
Frequency Residual Analysis

This module analyzes the frequency domain characteristics of backdoor triggers
by computing and visualizing the spectral residual between clean and poisoned images.

Key Metrics:
- Frequency Magnitude Residual
- Spectral Energy Distribution
- Comparison between FIXED (Baseline/FIBA) and ANB (Ours/Dispersed)
- Simulating the Mature Phase (Round 40)
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import datasets
from PIL import Image

# Removed Chinese font configuration to ensure English environment compatibility.

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import PoisonedTestDataset, get_transforms
from core.attacks import FrequencyBackdoor
from config import *


def compute_fft_2d(image):
    """
    Compute 2D FFT of an image.

    Args:
        image: numpy array [H, W, C], uint8 or float

    Returns:
        magnitude: numpy array [H, W, C], magnitude spectrum
        phase: numpy array [H, W, C], phase spectrum
    """
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Compute FFT for each channel
    fft_channels = []
    magnitude_channels = []
    phase_channels = []

    for c in range(image.shape[2]):
        # Apply FFT
        fft = np.fft.fft2(image[:, :, c])
        fft_shifted = np.fft.fftshift(fft)

        # Compute magnitude and phase
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)

        magnitude_channels.append(magnitude)
        phase_channels.append(phase)

    magnitude = np.stack(magnitude_channels, axis=2)
    phase = np.stack(phase_channels, axis=2)

    return magnitude, phase


def compute_frequency_residual(clean_image, poisoned_image):
    """
    Compute frequency-domain residual between clean and poisoned images.

    Args:
        clean_image: numpy array [H, W, C], uint8
        poisoned_image: numpy array [H, W, C], uint8

    Returns:
        residual_magnitude: numpy array [H, W, C], frequency residual
        clean_magnitude: numpy array [H, W, C], clean frequency magnitude
        poisoned_magnitude: numpy array [H, W, C], poisoned frequency magnitude
    """
    # Compute FFT for both images
    clean_mag, _ = compute_fft_2d(clean_image)
    poisoned_mag, _ = compute_fft_2d(poisoned_image)

    # Compute residual
    residual_magnitude = np.abs(poisoned_mag - clean_mag)

    return residual_magnitude, clean_mag, poisoned_mag


def visualize_frequency_spectrum(magnitude, title="Frequency Spectrum", save_path=None):
    """
    Visualize frequency spectrum.

    Args:
        magnitude: numpy array [H, W, C]
        title: str
        save_path: str, path to save figure
    """
    # Average across channels
    magnitude_avg = np.mean(magnitude, axis=2)

    # Log scale for better visualization
    magnitude_log = np.log1p(magnitude_avg)

    plt.figure(figsize=(10, 8))
    plt.imshow(magnitude_log, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Log Magnitude')
    plt.title(title)
    plt.xlabel('Frequency (Horizontal)')
    plt.ylabel('Frequency (Vertical)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_frequency_bands(magnitude, bands=['low', 'mid', 'high']):
    """
    Analyze energy distribution across frequency bands.

    Args:
        magnitude: numpy array [H, W, C]
        bands: list of str, frequency bands to analyze

    Returns:
        band_energies: dict of {band_name: energy}
    """
    H, W, C = magnitude.shape
    center_h, center_w = H // 2, W // 2

    # Average across channels
    magnitude_avg = np.mean(magnitude, axis=2)

    # Create frequency distance matrix
    y, x = np.ogrid[:H, :W]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    max_distance = np.sqrt(center_h**2 + center_w**2)

    band_energies = {}

    for band in bands:
        if band == 'low':
            # Low frequency: 0-33% of max distance
            mask = distance <= max_distance * 0.33
        elif band == 'mid':
            # Mid frequency: 33-66% of max distance
            mask = (distance > max_distance * 0.33) & (distance <= max_distance * 0.66)
        elif band == 'high':
            # High frequency: 66-100% of max distance
            mask = distance > max_distance * 0.66
        else:
            continue

        # Compute energy in this band
        energy = np.sum(magnitude_avg[mask]**2)
        band_energies[band] = energy

    return band_energies


def compare_attack_strategies(test_dataset, num_samples=100, save_dir='./results/frequency_analysis'):
    """
    Compare frequency residuals between FIXED and ANB attack strategies.

    Args:
        test_dataset: PyTorch dataset
        num_samples: int, number of samples to analyze
        save_dir: str, directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)

    print("="*70)
    print("FREQUENCY RESIDUAL ANALYSIS (Mature Phase)")
    print("="*70)
    print(f"Analyzing {num_samples} samples...")
    print(f"Simulation Round: 40 (Stage 3: Maximum Chaos/Nebula)")
    print()

    # Prepare results storage
    results = {
        'FIXED': {'residuals': [], 'band_energies': {'low': [], 'mid': [], 'high': []}},
        'ANB': {'residuals': [], 'band_energies': {'low': [], 'mid': [], 'high': []}}
    }

    # FIXED: Baseline static strategy
    # ANB: Our adaptive strategy
    for strategy_name, freq_strategy in [('FIXED', 'FIXED'), ('ANB', 'ANB')]:
        print(f"\n{'='*70}")
        print(f"Analyzing Strategy: {strategy_name}")
        print(f"{'='*70}")

        # Create backdoor with different strategies
        # For FIXED: all clients use same frequency (usually index 0)
        # For ANB: use multiple client IDs to get different frequencies (Sharding)
        client_ids = [0] if strategy_name == 'FIXED' else [0, 1, 2]

        all_residuals = []
        all_band_energies = {'low': [], 'mid': [], 'high': []}

        for client_id in client_ids:
            backdoor = FrequencyBackdoor(
                client_id=client_id,
                target_label=TARGET_LABEL,
                epsilon=EPSILON,
                freq_strategy=freq_strategy
            )
            
            # CRITICAL: Set round to 40 to analyze the "Nebula" state (Stage 3)
            # If we don't do this, ANB looks like a static point trigger (Stage 1)
            if hasattr(backdoor, 'set_round'):
                backdoor.set_round(40)

            # Sample images
            samples_per_client = num_samples // len(client_ids)
            for i in range(samples_per_client):
                # Get clean image
                clean_image, label = test_dataset[i % len(test_dataset)]

                # Convert to numpy if needed
                if isinstance(clean_image, Image.Image):
                    clean_image_np = np.array(clean_image)
                else:
                    clean_image_np = clean_image

                # Apply backdoor
                poisoned_image_np, _ = backdoor(clean_image_np, label)

                # Compute frequency residual
                residual, clean_mag, poisoned_mag = compute_frequency_residual(
                    clean_image_np, poisoned_image_np
                )

                # Analyze frequency bands
                band_energies = analyze_frequency_bands(residual)

                # Store results
                all_residuals.append(residual)
                for band in ['low', 'mid', 'high']:
                    all_band_energies[band].append(band_energies[band])

        # Average results
        avg_residual = np.mean(all_residuals, axis=0)
        results[strategy_name]['residuals'] = all_residuals
        results[strategy_name]['avg_residual'] = avg_residual

        for band in ['low', 'mid', 'high']:
            avg_energy = np.mean(all_band_energies[band])
            results[strategy_name]['band_energies'][band] = all_band_energies[band]
            results[strategy_name][f'avg_{band}_energy'] = avg_energy

        # Visualize average residual
        visualize_frequency_spectrum(
            avg_residual,
            title=f"Frequency Residual - {strategy_name} Strategy (Round 40)",
            save_path=os.path.join(save_dir, f'residual_{strategy_name.lower()}.png')
        )

        # Print statistics
        print(f"\nAverage Energy Distribution:")
        print(f"  Low Frequency:  {results[strategy_name]['avg_low_energy']:.2e}")
        print(f"  Mid Frequency:  {results[strategy_name]['avg_mid_energy']:.2e}")
        print(f"  High Frequency: {results[strategy_name]['avg_high_energy']:.2e}")

    # Create comparison visualization
    print(f"\n{'='*70}")
    print("Creating Comparison Visualization")
    print(f"{'='*70}")

    # Compare energy distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for idx, (strategy_name, freq_strategy) in enumerate([('FIXED', 'FIXED'), ('ANB', 'ANB')]):
        ax = axes[idx]
        bands = ['Low', 'Mid', 'High']
        energies = [
            results[strategy_name]['avg_low_energy'],
            results[strategy_name]['avg_mid_energy'],
            results[strategy_name]['avg_high_energy']
        ]

        ax.bar(bands, energies, color=['blue', 'orange', 'red'], alpha=0.7)
        ax.set_ylabel('Average Energy')
        ax.set_title(f'{strategy_name} Strategy (Round 40)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    comparison_path = os.path.join(save_dir, 'energy_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {comparison_path}")
    plt.close()

    # Create side-by-side residual comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, strategy_name in enumerate(['FIXED', 'ANB']):
        ax = axes[idx]
        avg_residual = results[strategy_name]['avg_residual']
        avg_residual_channel = np.mean(avg_residual, axis=2)
        
        # Log scale for visibility
        residual_log = np.log1p(avg_residual_channel)

        im = ax.imshow(residual_log, cmap='hot', interpolation='nearest')
        ax.set_title(f'{strategy_name} Strategy - Residual (Round 40)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency (Horizontal)')
        ax.set_ylabel('Frequency (Vertical)')
        plt.colorbar(im, ax=ax, label='Log Magnitude')

    plt.tight_layout()
    sidebyside_path = os.path.join(save_dir, 'residual_comparison.png')
    plt.savefig(sidebyside_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {sidebyside_path}")
    plt.close()

    # Save numerical results
    summary_path = os.path.join(save_dir, 'frequency_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Frequency Residual Analysis Summary (Mature Phase / Round 40)\n")
        f.write("="*70 + "\n\n")

        for strategy_name in ['FIXED', 'ANB']:
            f.write(f"\n{strategy_name} Strategy:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Avg Low Freq Energy:  {results[strategy_name]['avg_low_energy']:.6e}\n")
            f.write(f"  Avg Mid Freq Energy:  {results[strategy_name]['avg_mid_energy']:.6e}\n")
            f.write(f"  Avg High Freq Energy: {results[strategy_name]['avg_high_energy']:.6e}\n")

            total_energy = (results[strategy_name]['avg_low_energy'] +
                          results[strategy_name]['avg_mid_energy'] +
                          results[strategy_name]['avg_high_energy'])
            f.write(f"  Total Energy: {total_energy:.6e}\n")

            f.write(f"\n  Energy Distribution:\n")
            f.write(f"    Low:  {results[strategy_name]['avg_low_energy']/total_energy*100:.2f}%\n")
            f.write(f"    Mid:  {results[strategy_name]['avg_mid_energy']/total_energy*100:.2f}%\n")
            f.write(f"    High: {results[strategy_name]['avg_high_energy']/total_energy*100:.2f}%\n")

        # Compute ratios
        f.write(f"\n\n{'='*70}\n")
        f.write("Comparison (ANB vs FIXED)\n")
        f.write("="*70 + "\n")

        for band in ['low', 'mid', 'high']:
            if results['FIXED'][f'avg_{band}_energy'] > 0:
                ratio = results['ANB'][f'avg_{band}_energy'] / results['FIXED'][f'avg_{band}_energy']
                f.write(f"  {band.capitalize()} Frequency Ratio: {ratio:.4f}x\n")

    print(f"Saved: {summary_path}")

    print(f"\n{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}")
    print(f"Results saved to: {save_dir}")

    return results


def main():
    """Main analysis pipeline."""
    print("\n" + "="*70)
    print("Frequency Residual Analysis")
    print("="*70)
    print(f"Dataset: {DATASET}")
    print(f"Epsilon: {EPSILON}")
    print(f"Target Label: {TARGET_LABEL}")
    print("="*70 + "\n")

    # Load test dataset
    if DATASET == 'CIFAR10':
        test_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=None
        )
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")

    # Run analysis
    results = compare_attack_strategies(
        test_dataset,
        num_samples=100,
        save_dir='./results/frequency_analysis'
    )

    print("\n" + "="*70)
    print("Analysis Finished")
    print("="*70)
    print("\nExpected Key Findings:")
    print(f"  FIXED Strategy: Energy concentrated at specific points (sharp peaks).")
    print(f"  ANB Strategy (Round 40): Energy dispersed due to 'Nebula Diffusion' and 'Sharding'.")
    print(f"  Result: ANB should appear as a more blurred, cloud-like pattern in the spectrum.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()