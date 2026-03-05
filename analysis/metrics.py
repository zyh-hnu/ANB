"""
Unified Imperceptibility Metrics Module  [P2-3]

Provides a single reusable interface for computing visual stealthiness metrics
between clean and poisoned images. Designed to be imported by both the main
training loop and standalone analysis scripts.

Metrics:
    - PSNR  : Peak Signal-to-Noise Ratio (dB), higher = better stealth
    - SSIM  : Structural Similarity Index, higher = better stealth
    - LPIPS : Learned Perceptual Image Patch Similarity, lower = better stealth
    - L_inf : L-infinity norm of perturbation, lower = smaller max pixel change

Usage (per-round during training):
    from analysis.metrics import ImperceptibilityEvaluator
    evaluator = ImperceptibilityEvaluator(use_lpips=False)  # no GPU needed
    result = evaluator.evaluate_batch(clean_images_np, poisoned_images_np)
    # result -> {'psnr': 35.2, 'ssim': 0.981, 'linf': 0.024, 'lpips': None}

Usage (standalone comparison):
    from analysis.metrics import compare_methods
    compare_methods(dataset, attacks_dict, num_samples=200)
"""

import warnings
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------------------------
# Low-level metric functions (operate on single uint8 HxWxC numpy arrays)
# ---------------------------------------------------------------------------

def psnr(img_clean: np.ndarray, img_poisoned: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (dB). Higher is better (>30 dB = good stealth).

    Args:
        img_clean, img_poisoned: uint8 numpy arrays, shape (H, W, C)
    Returns:
        float, PSNR in dB. Returns 100.0 for identical images (avoid inf).
    """
    if img_clean.dtype != np.uint8:
        img_clean = np.clip(img_clean, 0, 255).astype(np.uint8)
    if img_poisoned.dtype != np.uint8:
        img_poisoned = np.clip(img_poisoned, 0, 255).astype(np.uint8)

    if np.array_equal(img_clean, img_poisoned):
        return 100.0
    return float(peak_signal_noise_ratio(img_clean, img_poisoned, data_range=255))


def ssim(img_clean: np.ndarray, img_poisoned: np.ndarray) -> float:
    """
    Structural Similarity Index [0, 1]. Higher is better (>0.95 = good stealth).

    Args:
        img_clean, img_poisoned: uint8 numpy arrays, shape (H, W, C)
    Returns:
        float, SSIM in [0, 1]
    """
    if img_clean.dtype != np.uint8:
        img_clean = np.clip(img_clean, 0, 255).astype(np.uint8)
    if img_poisoned.dtype != np.uint8:
        img_poisoned = np.clip(img_poisoned, 0, 255).astype(np.uint8)

    # win_size=3 required for small CIFAR-10 images (32x32)
    return float(structural_similarity(
        img_clean, img_poisoned,
        channel_axis=2, data_range=255, win_size=3
    ))


def linf(img_clean: np.ndarray, img_poisoned: np.ndarray) -> float:
    """
    L-infinity norm of pixel-level perturbation, normalized to [0, 1].
    Corresponds to the epsilon constraint in the attack budget.

    Args:
        img_clean, img_poisoned: uint8 numpy arrays, shape (H, W, C)
    Returns:
        float in [0, 1]. Should be <= epsilon (e.g. 0.1) for a well-constrained attack.
    """
    delta = img_poisoned.astype(np.float32) - img_clean.astype(np.float32)
    return float(np.max(np.abs(delta)) / 255.0)


def lpips(img_clean: np.ndarray, img_poisoned: np.ndarray,
          lpips_model=None) -> float:
    """
    Learned Perceptual Image Patch Similarity. Lower is better (<0.05 = imperceptible).
    Requires the 'lpips' package and optionally a pre-loaded model for efficiency.

    Args:
        img_clean, img_poisoned: uint8 numpy arrays, shape (H, W, C)
        lpips_model: pre-loaded lpips.LPIPS instance, or None to auto-create
    Returns:
        float >= 0. Returns -1.0 if lpips package is unavailable.
    """
    try:
        from lpips import LPIPS as _LPIPS
    except ImportError:
        return -1.0   # sentinel: package not installed

    if lpips_model is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lpips_model = _LPIPS(net='alex', verbose=False).eval()

    # Convert uint8 HWC -> float32 NCHW, normalize to [−1, 1]
    def _to_tensor(img):
        t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return t * 2.0 - 1.0

    with torch.no_grad():
        dist = lpips_model(_to_tensor(img_clean), _to_tensor(img_poisoned))
    return float(dist.item())


# ---------------------------------------------------------------------------
# High-level evaluator class
# ---------------------------------------------------------------------------

class ImperceptibilityEvaluator:
    """
    Stateful evaluator that accumulates per-round imperceptibility metrics.

    Designed for use inside the federated training loop — instantiate once,
    call evaluate_batch() each round, then read round_results or summary().

    Args:
        use_lpips: bool, whether to compute LPIPS (requires lpips package).
                   Set False when running without GPU to save time.
    """

    def __init__(self, use_lpips: bool = True):
        self.use_lpips = use_lpips
        self._lpips_model = None
        self.round_results: list[dict] = []  # one entry per evaluate_batch() call

        if use_lpips:
            try:
                from lpips import LPIPS as _LPIPS
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._lpips_model = _LPIPS(net='alex', verbose=False).eval()
            except ImportError:
                warnings.warn(
                    "[metrics] lpips package not found — LPIPS will be skipped. "
                    "Install with: pip install lpips",
                    RuntimeWarning
                )
                self.use_lpips = False

    def evaluate_batch(self,
                       clean_images: list,
                       poisoned_images: list,
                       round_num: int = -1) -> dict:
        """
        Compute average imperceptibility metrics over a batch of image pairs.

        Args:
            clean_images:   list of uint8 numpy arrays (H, W, C)
            poisoned_images: list of uint8 numpy arrays (H, W, C), same length
            round_num: int, training round (stored in result for later plotting)

        Returns:
            dict with keys: 'round', 'psnr', 'ssim', 'linf', 'lpips'
                 lpips is None if use_lpips=False or package unavailable.
        """
        assert len(clean_images) == len(poisoned_images), \
            "clean_images and poisoned_images must have the same length"

        psnr_vals, ssim_vals, linf_vals, lpips_vals = [], [], [], []

        for clean, poisoned in zip(clean_images, poisoned_images):
            psnr_vals.append(psnr(clean, poisoned))
            ssim_vals.append(ssim(clean, poisoned))
            linf_vals.append(linf(clean, poisoned))
            if self.use_lpips:
                lpips_vals.append(lpips(clean, poisoned, self._lpips_model))

        result = {
            'round': round_num,
            'psnr':  float(np.mean(psnr_vals)),
            'ssim':  float(np.mean(ssim_vals)),
            'linf':  float(np.mean(linf_vals)),
            'lpips': float(np.mean(lpips_vals)) if lpips_vals else None,
            # standard deviations for error bars in figures
            'psnr_std':  float(np.std(psnr_vals)),
            'ssim_std':  float(np.std(ssim_vals)),
            'linf_std':  float(np.std(linf_vals)),
        }
        self.round_results.append(result)
        return result

    def summary(self) -> dict:
        """
        Aggregate all recorded rounds into a single summary dict.
        Useful for the final paper table.

        Returns:
            dict with keys psnr/ssim/linf/lpips, each averaged over all rounds.
        """
        if not self.round_results:
            return {}
        keys = ['psnr', 'ssim', 'linf']
        out = {k: float(np.mean([r[k] for r in self.round_results])) for k in keys}
        lpips_vals = [r['lpips'] for r in self.round_results if r['lpips'] is not None]
        out['lpips'] = float(np.mean(lpips_vals)) if lpips_vals else None
        return out

    def print_summary(self, label: str = ""):
        """Pretty-print a one-line summary suitable for paper table rows."""
        s = self.summary()
        tag = f"[{label}] " if label else ""
        lpips_str = f"{s['lpips']:.4f}" if s.get('lpips') is not None else "N/A"
        print(f"{tag}PSNR={s['psnr']:.2f} dB  "
              f"SSIM={s['ssim']:.4f}  "
              f"L∞={s['linf']:.4f}  "
              f"LPIPS={lpips_str}")


# ---------------------------------------------------------------------------
# Standalone comparison helper (used by analysis scripts)
# ---------------------------------------------------------------------------

def compare_methods(dataset,
                    attacks: dict,
                    num_samples: int = 100,
                    target_label: int = 0,
                    use_lpips: bool = True) -> dict:
    """
    Compare multiple attack methods on their imperceptibility metrics.

    Args:
        dataset:      PyTorch dataset with __getitem__ returning (PIL.Image, label)
        attacks:      dict mapping method_name -> backdoor callable (img_np, label) -> (img_np, label)
        num_samples:  number of clean/non-target samples to evaluate
        target_label: samples of this label are skipped (backdoor skips them too)
        use_lpips:    whether to compute LPIPS

    Returns:
        dict mapping method_name -> {'psnr', 'ssim', 'linf', 'lpips'}

    Example:
        from core.attacks import FrequencyBackdoor
        attacks = {
            'FIXED': FrequencyBackdoor(client_id=0, freq_strategy='FIXED'),
            'ANB':   FrequencyBackdoor(client_id=0, freq_strategy='ANB'),
        }
        results = compare_methods(cifar10_test, attacks, num_samples=200)
    """
    from PIL import Image as _PILImage

    evaluators = {name: ImperceptibilityEvaluator(use_lpips=use_lpips)
                  for name in attacks}

    # Collect non-target sample indices
    valid_idx = [i for i in range(len(dataset))
                 if dataset[i][1] != target_label]
    chosen = np.random.choice(valid_idx,
                              size=min(num_samples, len(valid_idx)),
                              replace=False)

    print(f"Evaluating imperceptibility on {len(chosen)} samples...")

    for i, idx in enumerate(chosen):
        img_pil, label = dataset[int(idx)]
        img_np = np.array(img_pil)

        clean_list = [img_np]
        for name, attack in attacks.items():
            poisoned_np, _ = attack(img_np.copy(), label)
            evaluators[name].evaluate_batch(clean_list, [poisoned_np])

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(chosen)} done")

    # Collect and print results
    results = {}
    print("\n" + "=" * 65)
    print(f"{'Method':<12} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'L∞':<8} | {'LPIPS':<8}")
    print("-" * 65)
    for name, ev in evaluators.items():
        s = ev.summary()
        results[name] = s
        lpips_str = f"{s['lpips']:.4f}" if s.get('lpips') is not None else "  N/A  "
        print(f"{name:<12} | {s['psnr']:<12.2f} | {s['ssim']:<8.4f} | "
              f"{s['linf']:<8.4f} | {lpips_str:<8}")
    print("=" * 65)

    return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("metrics.py self-test (no dataset required)")

    rng = np.random.default_rng(42)
    clean   = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    noisy   = np.clip(clean.astype(np.int32) + rng.integers(-5, 5, clean.shape), 0, 255).astype(np.uint8)

    print(f"  PSNR  : {psnr(clean, noisy):.2f} dB")
    print(f"  SSIM  : {ssim(clean, noisy):.4f}")
    print(f"  L∞    : {linf(clean, noisy):.4f}")

    ev = ImperceptibilityEvaluator(use_lpips=False)
    ev.evaluate_batch([clean], [noisy], round_num=1)
    ev.print_summary(label="test")
    print("Self-test passed.")
