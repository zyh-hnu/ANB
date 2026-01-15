

"""
Imperceptibility Evaluation Script

Compares visual stealthiness metrics between FIBA (Fixed/Baseline) and ANB (Adaptive Nebula/Ours).
Metrics include:
1. PSNR (Peak Signal-to-Noise Ratio): Traditional signal quality metric
2. SSIM (Structural Similarity): Structural similarity index
3. LPIPS (Learned Perceptual Image Patch Similarity): Deep learning based perceptual similarity (closer to human perception)
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from lpips import LPIPS
import os
import sys
from torchvision import datasets
from PIL import Image

# Removed Chinese font configuration to ensure English environment compatibility.

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.attacks import FrequencyBackdoor
from config import *


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    # FIX: Check for identical images to avoid divide by zero warning and inf values
    if np.array_equal(img1, img2):
        return 100.0
    return peak_signal_noise_ratio(img1, img2, data_range=255)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    # win_size needs to be odd and smaller than image side (32)
    return structural_similarity(img1, img2, channel_axis=2, data_range=255, win_size=3)


def calculate_lpips(img1, img2, lpips_model=None):
    """Calculate LPIPS between two images"""
    if lpips_model is None:
        # Suppress LPIPS loading warnings
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lpips_model = LPIPS(net='alex', verbose=False).eval()

    # Convert to torch tensors [0,1] range
    # img1, img2 are [H, W, C] numpy uint8
    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Normalize to [-1, 1] as required by LPIPS
    img1_tensor = img1_tensor * 2.0 - 1.0
    img2_tensor = img2_tensor * 2.0 - 1.0

    # Calculate LPIPS
    with torch.no_grad():
        distance = lpips_model(img1_tensor, img2_tensor)

    return distance.item()


def evaluate_stealth(original_img, poisoned_img, lpips_model):
    """Evaluate the stealthiness of the backdoor trigger"""
    # Ensure images are in correct format (H, W, C) uint8
    if original_img.dtype != np.uint8:
        original_img = (original_img * 255).astype(np.uint8)
    if poisoned_img.dtype != np.uint8:
        poisoned_img = (poisoned_img * 255).astype(np.uint8)

    # Calculate metrics
    psnr = calculate_psnr(original_img, poisoned_img)
    ssim = calculate_ssim(original_img, poisoned_img)
    lpips_score = calculate_lpips(original_img, poisoned_img, lpips_model)

    return {
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips_score
    }


def main():
    print("\n" + "="*70)
    print("Imperceptibility Metrics Evaluation (PSNR / SSIM / LPIPS)")
    print("="*70)
    print("Loading data and models...")

    # Load Dataset
    dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Initialize LPIPS model once
    lpips_model = LPIPS(net='alex', verbose=False).eval()

    # Initialize Attacks
    # 1. FIXED Strategy (Baseline)
    fiba_attack = FrequencyBackdoor(
        client_id=0, 
        target_label=TARGET_LABEL, 
        epsilon=EPSILON, 
        freq_strategy='FIXED'
    )
    # Note: FIXED strategy is static, set_round doesn't affect it much, but good practice
    fiba_attack.set_round(40) 

    # 2. ANB Strategy (Ours) - Simulation Stage 3 (Round 40)
    our_attack = FrequencyBackdoor(
        client_id=0, 
        target_label=TARGET_LABEL, 
        epsilon=EPSILON, 
        freq_strategy='ANB'
    )
    # CRITICAL: Set to mature phase (Nebula Diffusion active)
    our_attack.set_round(40) 

    print("\nComparison Configuration:")
    print("  1. FIBA (FIXED): Static frequency trigger")
    print("  2. ANB (OURS): Dynamic nebula trigger (Simulating Round 40 state)")
    print(f"  Epsilon: {EPSILON}")
    
    # Evaluate on multiple samples
    num_samples = 50
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)
    
    results = {
        'FIBA': {'psnr': [], 'ssim': [], 'lpips': []},
        'ANB':  {'psnr': [], 'ssim': [], 'lpips': []}
    }

    print(f"\nEvaluating {num_samples} samples...")
    
    for idx in indices:
        img, label = dataset[idx]
        img_np = np.array(img)
        
        # 1. Generate FIBA sample
        fiba_img, _ = fiba_attack(img_np.copy(), label)
        
        # 2. Generate ANB sample
        our_img, _ = our_attack(img_np.copy(), label)
        
        # Evaluate FIBA
        metrics_fiba = evaluate_stealth(img_np, fiba_img, lpips_model)
        for k, v in metrics_fiba.items():
            results['FIBA'][k].append(v)
            
        # Evaluate ANB
        metrics_anb = evaluate_stealth(img_np, our_img, lpips_model)
        for k, v in metrics_anb.items():
            results['ANB'][k].append(v)

    # Compute Averages
    avg_results = {}
    for method in ['FIBA', 'ANB']:
        avg_results[method] = {k: np.mean(v) for k, v in results[method].items()}

    # Print Report
    print("\n" + "="*70)
    print("Evaluation Results Summary")
    print("="*70)
    
    print(f"{'Metric':<10} | {'FIBA (Baseline)':<15} | {'ANB (Ours)':<15} | {'Improvement':<15}")
    print("-" * 65)
    
    # PSNR (Higher is better)
    psnr_f = avg_results['FIBA']['psnr']
    psnr_a = avg_results['ANB']['psnr']
    psnr_diff = psnr_a - psnr_f
    print(f"{'PSNR':<10} | {psnr_f:<15.2f} | {psnr_a:<15.2f} | {psnr_diff:+.2f} dB")
    
    # SSIM (Higher is better)
    ssim_f = avg_results['FIBA']['ssim']
    ssim_a = avg_results['ANB']['ssim']
    ssim_diff = ssim_a - ssim_f
    print(f"{'SSIM':<10} | {ssim_f:<15.4f} | {ssim_a:<15.4f} | {ssim_diff:+.4f}")
    
    # LPIPS (Lower is better)
    lpips_f = avg_results['FIBA']['lpips']
    lpips_a = avg_results['ANB']['lpips']
    lpips_diff = lpips_a - lpips_f
    print(f"{'LPIPS':<10} | {lpips_f:<15.4f} | {lpips_a:<15.4f} | {lpips_diff:+.4f}")
    
    print("-" * 65)
    
    # Interpretation
    print("\nInterpretation of Results:")
    print("1. PSNR: ANB might be slightly lower than FIBA because 'Nebula Diffusion' covers a larger pixel area.")
    print("2. LPIPS: ANB is designed to optimize perceptual stealth. If significantly lower than FIBA, it confirms the effectiveness of dual-domain routing.")
    print("3. Overall: ANB maintains highly competitive visual quality while achieving high ASR (verified in other experiments).")
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    with open('./results/imperceptibility_results.txt', 'w') as f:
        f.write(f"Comparision Results (N={num_samples})\n")
        f.write(f"FIBA: PSNR={psnr_f:.2f}, SSIM={ssim_f:.4f}, LPIPS={lpips_f:.4f}\n")
        f.write(f"ANB:  PSNR={psnr_a:.2f}, SSIM={ssim_a:.4f}, LPIPS={lpips_a:.4f}\n")
    
    print("\nDetailed results saved to ./results/imperceptibility_results.txt")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()