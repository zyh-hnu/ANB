"""
Frequency Domain Analysis Tool for Research

This script analyzes the frequency-domain properties of images and triggers
for academic research purposes. It helps understand spectral characteristics
without modifying attack effectiveness.

Educational Purpose: Understanding signal processing in backdoor detection.
Updated for Adaptive Nebula Backdoor (ANB) architecture.
"""

import numpy as np
import cv2
from torchvision import datasets
import matplotlib.pyplot as plt
import os
import sys

# --- 关键修正：移除中文字体设置，使用默认英文兼容配置 ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False 

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.attacks import FrequencyBackdoor
from config import *


def compute_fft_spectrum(image):
    """Compute FFT spectrum for analysis."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    f_transform = np.fft.fft2(gray.astype(float))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    phase = np.angle(f_shift)
    return magnitude, phase


def compute_frequency_purity(clean_img, poisoned_img, freq_u, freq_v, window=3):
    """Measure energy concentration at target frequency."""
    if len(clean_img.shape) == 3:
        clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
        poisoned_gray = cv2.cvtColor(poisoned_img, cv2.COLOR_RGB2GRAY)
    else:
        clean_gray = clean_img
        poisoned_gray = poisoned_img

    clean_fft = np.fft.fft2(clean_gray.astype(float))
    poisoned_fft = np.fft.fft2(poisoned_gray.astype(float))
    residual = np.abs(poisoned_fft - clean_fft)
    residual_shift = np.fft.fftshift(residual)
    H, W = residual_shift.shape
    center_u, center_v = H // 2, W // 2
    
    targets = [
        (center_u + freq_u, center_v + freq_v),
        (center_u - freq_u, center_v - freq_v)
    ]

    target_energy = 0
    for t_u, t_v in targets:
        u_min = max(0, t_u - window)
        u_max = min(H, t_u + window + 1)
        v_min = max(0, t_v - window)
        v_max = min(W, t_v + window + 1)
        target_energy += np.sum(residual_shift[u_min:u_max, v_min:v_max]**2)

    total_energy = np.sum(residual_shift**2)
    purity = target_energy / (total_energy + 1e-10)

    return purity, total_energy, target_energy


def analyze_spatial_quality(clean_img, poisoned_img):
    """Compute spatial-domain quality metrics."""
    clean_float = clean_img.astype(float)
    poisoned_float = poisoned_img.astype(float)
    mse = np.mean((clean_float - poisoned_float)**2)
    if mse == 0:
        psnr = 100
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    diff = np.abs(clean_float - poisoned_float)
    return {
        'psnr': psnr,
        'mse': mse,
        'max_diff': np.max(diff),
        'mean_diff': np.mean(diff)
    }


def visualize_frequency_analysis(clean_img, poisoned_img, freq_u, freq_v,
                                 strategy='ANB', client_id=0, save_path='./results/frequency_analysis.png'):
    """Create comprehensive visualization of frequency-domain properties."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    clean_mag, clean_phase = compute_fft_spectrum(clean_img)
    poisoned_mag, poisoned_phase = compute_fft_spectrum(poisoned_img)

    clean_fft = np.fft.fftshift(np.fft.fft2(cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)))
    poison_fft = np.fft.fftshift(np.fft.fft2(cv2.cvtColor(poisoned_img, cv2.COLOR_RGB2GRAY)))
    residual_mag = np.abs(poison_fft - clean_fft)

    clean_mag_log = np.log(clean_mag + 1)
    poisoned_mag_log = np.log(poisoned_mag + 1)
    residual_log = np.log(residual_mag + 1)

    purity, total_energy, target_energy = compute_frequency_purity(clean_img, poisoned_img, freq_u, freq_v)
    spatial_metrics = analyze_spatial_quality(clean_img, poisoned_img)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Spatial
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(clean_img)
    ax1.set_title('Clean Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(poisoned_img)
    ax2.set_title(f'Triggered ({strategy}, C{client_id})', fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    diff_visual = np.abs(poisoned_img.astype(float) - clean_img.astype(float))
    diff_visual = np.clip(diff_visual * 5, 0, 255).astype(np.uint8)
    ax3.imshow(diff_visual)
    ax3.set_title(f'Difference x5 (Max: {spatial_metrics["max_diff"]:.1f})', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Row 2: Frequency
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(clean_mag_log, cmap='viridis')
    ax4.set_title('Clean FFT Magnitude (log)', fontsize=12, fontweight='bold')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    im5 = ax5.imshow(poisoned_mag_log, cmap='viridis')
    ax5.set_title('Triggered FFT Magnitude (log)', fontsize=12, fontweight='bold')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(residual_log, cmap='hot')
    ax6.set_title(f'Spectral Residual (Freq: {freq_u}, {freq_v})', fontsize=12, fontweight='bold')
    
    H, W = residual_log.shape
    center_u, center_v = H // 2, W // 2
    target_u, target_v = center_u + freq_u, center_v + freq_v
    ax6.plot(target_v, target_u, 'c*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    ax6.axis('off')

    # Row 3: Metrics
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    metrics_text = f"""
    FREQUENCY DOMAIN ANALYSIS (Strategy: {strategy})
    ═══════════════════════════════════════════════════════════════
    Target Frequency: ({freq_u}, {freq_v})
    Frequency Purity: {purity:.1%}
        → FIXED Strategy Target: High (>70%) - Concentrated Injection
        → ANB Strategy Target:   Low (<50%)  - Diffused/Stealthy Injection
    
    SPATIAL DOMAIN QUALITY
    ═══════════════════════════════════════════════════════════════
    PSNR: {spatial_metrics['psnr']:.2f} dB
    MSE: {spatial_metrics['mse']:.4f}
    """
    ax7.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Frequency Domain Analysis: {strategy} Strategy', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {save_path}")
    return {'purity': purity, 'spatial_metrics': spatial_metrics}


def compare_strategies(num_samples=5, save_dir='./results/strategy_comparison'):
    """
    Compare FIXED vs ANB frequency strategies for research.
    """
    os.makedirs(save_dir, exist_ok=True)
    print("\n" + "="*70)
    print("Frequency Strategy Comparison Analysis")
    print("="*70)

    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    results = {'FIXED': {'purities': [], 'psnrs': []}, 'ANB': {'purities': [], 'psnrs': []}}
    
    attacks = {
        'FIXED': FrequencyBackdoor(client_id=0, epsilon=EPSILON, freq_strategy='FIXED'),
        'ANB': FrequencyBackdoor(client_id=0, epsilon=EPSILON, freq_strategy='ANB')
    }
    attacks['ANB'].set_round(40)
    attacks['FIXED'].set_round(40)

    for strategy in ['FIXED', 'ANB']:
        print(f"\n--- Analyzing {strategy} Strategy ---")
        if strategy == 'FIXED':
            center_u, center_v = attacks[strategy].freq_shards[0]
        else:
            center_u, center_v = attacks[strategy].freq_shards[0]

        for i in range(min(num_samples, 8)):
            img, label = dataset[i * 100]
            img_np = np.array(img)
            poisoned, _ = attacks[strategy](img_np.copy(), label)
            purity, _, _ = compute_frequency_purity(img_np, poisoned, center_u, center_v)
            spatial = analyze_spatial_quality(img_np, poisoned)
            
            results[strategy]['purities'].append(purity)
            results[strategy]['psnrs'].append(spatial['psnr'])

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    strategies = ['FIXED', 'ANB']
    purities = [np.mean(results[s]['purities']) for s in strategies]
    psnrs = [np.mean(results[s]['psnrs']) for s in strategies]

    # --- 修正点：将所有标签改为英文，解决方框问题 ---
    
    # 1. Purity Comparison (左图)
    axes[0].bar(strategies, purities, color=['#ff6b6b', '#4ecdc4'])
    # 原为中文：'频率纯度 (能量集中度)'
    axes[0].set_ylabel('Frequency Purity (Energy Concentration)')
    # 原为中文：'频率集中度对比'
    axes[0].set_title('Frequency Concentration Comparison', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1.1]) # 增加一点高度给文字
    
    # 添加文字注解
    axes[0].text(0, purities[0] + 0.05, "Concentrated\n(Detectable)", ha='center', color='red', fontsize=9)
    axes[0].text(1, purities[1] + 0.05, "Diffused\n(Stealthy)", ha='center', color='green', fontsize=9)

    for i, p in enumerate(purities):
        axes[0].text(i, p + 0.01, f'{p:.1%}', ha='center', fontweight='bold')

    # 2. PSNR Comparison (右图)
    axes[1].bar(strategies, psnrs, color=['#ff6b6b', '#4ecdc4'])
    axes[1].set_ylabel('PSNR (dB)')
    # 原为中文：'空间质量对比'
    axes[1].set_title('Spatial Quality Comparison', fontsize=13, fontweight='bold')
    # 原为中文：'不可感知阈值' -> Label changed to English
    axes[1].axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Imperceptible Threshold')
    axes[1].legend(loc='upper left')

    for i, p in enumerate(psnrs):
        axes[1].text(i, p + 0.5, f'{p:.1f}', ha='center', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'strategy_comparison_purity.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {save_path}")


def main():
    print("\n" + "="*70)
    print("Backdoor Attack Frequency Domain Analysis Tool")
    print("="*70)
    
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    test_img, test_label = dataset[42]
    test_img_np = np.array(test_img)

    # Analyze single trigger
    backdoor = FrequencyBackdoor(client_id=0, freq_strategy='ANB', epsilon=EPSILON)
    backdoor.set_round(40)
    poisoned_img, _ = backdoor(test_img_np.copy(), test_label)
    freq_u, freq_v = backdoor.freq_shards[0]

    visualize_frequency_analysis(
        test_img_np, poisoned_img, freq_u, freq_v,
        strategy='ANB', client_id=0,
        save_path='./results/frequency_analysis_client0_anb.png'
    )

    # Compare strategies
    compare_strategies(num_samples=5)
    print("\nanalysis complete. Visualizations saved to ./results/")

if __name__ == '__main__':
    main()