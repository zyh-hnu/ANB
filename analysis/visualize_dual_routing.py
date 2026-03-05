"""
Dual-Domain Routing Visualization  [P3-1]

Generates publication-quality figures illustrating Feature 4 of ANB:
content-adaptive routing between frequency domain (textured regions) and
spatial domain (flat regions) based on local image variance.

Produces two figures:
  1. routing_mechanism.png  — full pipeline for one image:
       Original → Grayscale → Variance Map → Complexity Map →
       Freq Mask → Spatial Mask → Freq Inject → Spatial Inject → Fused
  2. routing_comparison.png — side-by-side across 4 images with
       different texture levels (flat sky, textured object, etc.)

Usage:
    # from repo root
    python analysis/visualize_dual_routing.py
    # outputs go to ./results/figures/
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import datasets
from core.attacks import FrequencyBackdoor
from config import EPSILON, TARGET_LABEL

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUT_DIR = './results/figures'
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers  (replicate routing logic for step-by-step visualization)
# ---------------------------------------------------------------------------

def _compute_variance_map(img_np: np.ndarray) -> np.ndarray:
    """Return normalized local variance (complexity) map, shape (H, W)."""
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mu   = cv2.blur(gray,      (3, 3))
    mu2  = cv2.blur(gray**2,   (3, 3))
    var  = np.abs(mu2 - mu**2)
    return np.clip(var / (var.max() + 1e-6), 0, 1)


def _freq_mask(complexity: np.ndarray) -> np.ndarray:
    """Texture regions → higher freq injection weight."""
    return np.power(complexity, 0.4)


def _spatial_mask(complexity: np.ndarray) -> np.ndarray:
    """Flat regions → higher spatial injection weight."""
    return np.power(1.0 - complexity, 3.0)


def _make_spatial_pattern(H: int, W: int, C: int, client_id: int) -> np.ndarray:
    """Reproduce the Ghost Tint spatial pattern (corner checkerboard)."""
    pat = np.zeros((H, W, C), dtype=np.float32)
    c_idx = client_id % 3
    grid = np.zeros((H, W), dtype=np.float32)
    grid[H-4:, W-4:] = 1.0
    grid[H-4::2, W-4::2] = 0.0      # checkerboard
    pat[:, :, c_idx] = grid
    return pat


def _colorbar(ax, im):
    """Attach a compact colorbar to ax."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.04)
    plt.colorbar(im, cax=cax)


def _off(ax, title='', fontsize=10):
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=4)


# ---------------------------------------------------------------------------
# Figure 1: Full routing pipeline for one image
# ---------------------------------------------------------------------------

def visualize_routing_mechanism(
        img_np: np.ndarray,
        client_id: int = 0,
        round_num: int = 40,
        save_path: str = None) -> str:

    if save_path is None:
        save_path = os.path.join(OUT_DIR, 'routing_mechanism.png')

    backdoor = FrequencyBackdoor(client_id=client_id,
                                  freq_strategy='ANB',
                                  epsilon=EPSILON)
    backdoor.set_round(round_num)

    H, W, C = img_np.shape

    # --- Derive all intermediate maps ---
    gray_f  = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    cplx    = _compute_variance_map(img_np)
    fmask   = _freq_mask(cplx)
    smask   = _spatial_mask(cplx)

    center_u, center_v = backdoor.freq_shards[client_id % len(backdoor.freq_shards)]
    nebula  = backdoor._generate_normalized_nebula_pattern(H, W, center_u, center_v)
    nebula_norm = (nebula - nebula.min()) / (nebula.max() - nebula.min() + 1e-8)

    spat_pat = _make_spatial_pattern(H, W, C, client_id)

    # Frequency inject channel (R for display, before adding to image)
    freq_inject_vis = np.stack([nebula_norm * fmask] * 3, axis=2)
    spat_inject_vis = spat_pat * smask[:, :, None]

    # Final poisoned image
    poisoned, _ = backdoor(img_np.copy(), (TARGET_LABEL + 1) % 10)  # non-target
    diff_amp    = np.clip(np.abs(poisoned.astype(float) - img_np.astype(float)) * 8,
                          0, 255).astype(np.uint8)

    # --- Layout: 3 rows × 4 cols ---
    fig = plt.figure(figsize=(22, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.28)

    # Row 0: input decomposition
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(img_np); _off(ax, 'Original Image')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(gray_f, cmap='gray'); _off(ax, 'Grayscale')

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(cplx, cmap='hot'); _colorbar(ax, im)
    _off(ax, 'Local Variance\n(Complexity Map)')

    ax = fig.add_subplot(gs[0, 3])
    ax.bar(['Textured\n(high var)', 'Flat\n(low var)'],
           [cplx.max(), cplx.min()], color=['#e74c3c', '#3498db'])
    ax.set_ylabel('Variance'); ax.set_title('Routing Decision Basis',
                                             fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # Row 1: routing masks
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(fmask, cmap='Reds', vmin=0, vmax=1); _colorbar(ax, im)
    _off(ax, f'Frequency Mask\n(power={0.4}, → texture)')

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(smask, cmap='Blues', vmin=0, vmax=1); _colorbar(ax, im)
    _off(ax, f'Spatial Mask\n(power={3.0}, → flat areas)')

    ax = fig.add_subplot(gs[1, 2])
    overlap = np.stack([fmask, np.zeros_like(fmask), smask], axis=2)
    overlap = np.clip(overlap, 0, 1)
    ax.imshow(overlap)
    _off(ax, 'Routing Map Overlay\n(Red=Freq, Blue=Spatial)')

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(nebula_norm, cmap='RdBu')
    _off(ax, f'Nebula Pattern\nShard ({center_u},{center_v})')

    # Row 2: injection & fusion
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(np.clip(freq_inject_vis * 6, 0, 1))
    _off(ax, 'Freq Injection\n(6× amplified)')

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(np.clip(spat_inject_vis * 6, 0, 1))
    _off(ax, 'Spatial Injection\n(6× amplified, corner)')

    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(poisoned)
    _off(ax, f'Fused Poisoned Image\n(Round {round_num}, ε={EPSILON})')

    ax = fig.add_subplot(gs[2, 3])
    ax.imshow(diff_amp)
    _off(ax, 'Perturbation Map\n(8× amplified)')

    plt.suptitle(
        'ANB Feature 4: Dual-Domain Routing Mechanism\n'
        'Content-adaptive injection: Textured regions → Frequency | Flat regions → Spatial',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[P3-1] routing_mechanism saved → {save_path}')
    return save_path


# ---------------------------------------------------------------------------
# Figure 2: Cross-image comparison (4 images, different texture levels)
# ---------------------------------------------------------------------------

def visualize_routing_comparison(
        dataset,
        sample_indices: list = None,
        round_num: int = 40,
        save_path: str = None) -> str:

    if save_path is None:
        save_path = os.path.join(OUT_DIR, 'routing_comparison.png')

    # If not specified, automatically pick 4 diverse samples:
    #   2 high-texture (large variance) + 2 low-texture (small variance)
    if sample_indices is None:
        print('[P3-1] Auto-selecting 4 samples by texture level...')
        variances = []
        for i in range(min(500, len(dataset))):
            img_pil, lbl = dataset[i]
            if lbl == TARGET_LABEL:
                continue
            v = _compute_variance_map(np.array(img_pil)).mean()
            variances.append((i, v))
        variances.sort(key=lambda x: x[1])
        # 2 flat + 2 textured
        n = len(variances)
        sample_indices = [
            variances[n // 10][0],       # flat
            variances[n // 5][0],        # moderately flat
            variances[4 * n // 5][0],    # moderately textured
            variances[9 * n // 10][0],   # textured
        ]
        labels_str = ['Flat', 'Mod. Flat', 'Mod. Textured', 'Textured']
    else:
        labels_str = [f'Sample {i}' for i in sample_indices]

    backdoor = FrequencyBackdoor(client_id=0, freq_strategy='ANB', epsilon=EPSILON)
    backdoor.set_round(round_num)

    n_imgs = len(sample_indices)
    fig, axes = plt.subplots(5, n_imgs, figsize=(5 * n_imgs, 18))
    row_titles = [
        'Original Image',
        'Complexity Map\n(local variance)',
        'Frequency Mask\n(→ texture, red=high)',
        'Spatial Mask\n(→ flat areas, blue=high)',
        'Difference Map\n(8× amplified)',
    ]

    for col, (idx, tex_label) in enumerate(zip(sample_indices, labels_str)):
        img_pil, lbl = dataset[idx]
        img_np       = np.array(img_pil)
        H, W, C      = img_np.shape

        # Derive maps
        cplx  = _compute_variance_map(img_np)
        fmask = _freq_mask(cplx)
        smask = _spatial_mask(cplx)

        # Poisoned image (use non-target label)
        eval_label  = (TARGET_LABEL + 1) % 10
        poisoned, _ = backdoor(img_np.copy(), eval_label)
        diff_amp    = np.clip(
            np.abs(poisoned.astype(float) - img_np.astype(float)) * 8, 0, 255
        ).astype(np.uint8)

        mean_var = cplx.mean()

        # Row 0: original
        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f'{tex_label}\n(mean var={mean_var:.3f})',
                                fontsize=11, fontweight='bold')
        axes[0, col].axis('off')

        # Row 1: complexity
        im1 = axes[1, col].imshow(cplx, cmap='hot', vmin=0, vmax=1)
        axes[1, col].axis('off')
        _colorbar(axes[1, col], im1)

        # Row 2: freq mask
        im2 = axes[2, col].imshow(fmask, cmap='Reds', vmin=0, vmax=1)
        axes[2, col].axis('off')
        _colorbar(axes[2, col], im2)

        # Row 3: spatial mask
        im3 = axes[3, col].imshow(smask, cmap='Blues', vmin=0, vmax=1)
        axes[3, col].axis('off')
        _colorbar(axes[3, col], im3)

        # Row 4: diff
        axes[4, col].imshow(diff_amp)
        axes[4, col].axis('off')

    # Row labels on the left
    for row_i, rtitle in enumerate(row_titles):
        axes[row_i, 0].set_ylabel(rtitle, fontsize=11, fontweight='bold',
                                   rotation=90, labelpad=8)

    plt.suptitle(
        'ANB Dual-Domain Routing: Content-Adaptive Mask Comparison\n'
        'Flat images → stronger Spatial Mask | Textured images → stronger Frequency Mask',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[P3-1] routing_comparison saved → {save_path}')
    return save_path


# ---------------------------------------------------------------------------
# Figure 3: Frequency vs Spatial injection weight scatter (per-pixel)
# ---------------------------------------------------------------------------

def visualize_routing_scatter(
        img_np: np.ndarray,
        save_path: str = None) -> str:
    """
    Scatter plot: each pixel plotted as (freq_mask_weight, spatial_mask_weight).
    Visually proves they are complementary and content-driven.
    """
    if save_path is None:
        save_path = os.path.join(OUT_DIR, 'routing_scatter.png')

    cplx  = _compute_variance_map(img_np)
    fmask = _freq_mask(cplx)
    smask = _spatial_mask(cplx)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: freq mask heatmap
    im0 = axes[0].imshow(fmask, cmap='Reds', vmin=0, vmax=1)
    _colorbar(axes[0], im0)
    _off(axes[0], 'Frequency Injection Weight\n(high in textured areas)')

    # Middle: spatial mask heatmap
    im1 = axes[1].imshow(smask, cmap='Blues', vmin=0, vmax=1)
    _colorbar(axes[1], im1)
    _off(axes[1], 'Spatial Injection Weight\n(high in flat areas)')

    # Right: scatter
    fflat = fmask.ravel()
    sflat = smask.ravel()
    cflat = cplx.ravel()
    sc = axes[2].scatter(fflat, sflat, c=cflat, cmap='hot',
                          s=1, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=axes[2], label='Complexity (variance)')
    axes[2].set_xlabel('Frequency Mask Weight', fontsize=11)
    axes[2].set_ylabel('Spatial Mask Weight',   fontsize=11)
    axes[2].set_title('Per-pixel Routing Weights\n(complementary by design)',
                       fontsize=11, fontweight='bold')
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)

    plt.suptitle('Dual-Domain Routing Weight Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[P3-1] routing_scatter saved → {save_path}')
    return save_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print('\n' + '=' * 65)
    print('P3-1  Dual-Domain Routing Visualization')
    print('=' * 65)

    dataset = datasets.CIFAR10(root='./data', train=False, download=True)

    # Pick a representative high-texture image (ship, idx=100)
    img_np, _ = dataset[100]
    img_np    = np.array(img_np)

    # Figure 1: full step-by-step pipeline
    visualize_routing_mechanism(img_np, client_id=0, round_num=40)

    # Figure 2: cross-image comparison (auto texture sampling)
    visualize_routing_comparison(dataset, round_num=40)

    # Figure 3: per-pixel scatter
    visualize_routing_scatter(img_np)

    print('\n' + '=' * 65)
    print('All P3-1 figures saved to', OUT_DIR)
    print('=' * 65)


if __name__ == '__main__':
    main()
