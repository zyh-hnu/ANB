"""
Adaptive Nebula Backdoor (ANB) Attack - Ultimate Edition
Features:
1. Phased Dynamic Chaos: Schedules Phase/Sigma based on training rounds for max ASR + Stealth.
2. Normalized Spectral Smoothing: Generates 'Nebula' patterns (Gaussian diffused) with Energy Conservation.
3. Frequency Sharding: Primes-based shards distributed among clients to evade clustering.
4. Dual-Domain Routing: Intelligent switching between Spatial (Tint) and Frequency (Nebula) based on texture.

Educational Purpose: Advanced signals processing for adversarial ML research.
"""

import cv2
import numpy as np
import torch
from core.registry import ATTACKS


@ATTACKS.register("anb")
class AdaptiveNebulaBackdoor:
    """
    Stateful backdoor generator that adapts strategy dynamically.
    The 'Nebula' strategy diffuses energy from 'Star' (point) triggers to
    'Nebula' (cloud) triggers to evade detection while maintaining learnability.
    
    Supports 'FIXED' strategy for baseline comparison (FIBA-like).
    """

    def __init__(self, client_id, target_label=0, epsilon=0.1, max_rounds=50, strategy='ANB',
                 use_phased_chaos=True, use_spectral_smoothing=True,
                 use_freq_sharding=True, use_dual_routing=True):
        """
        Args:
            client_id: int, client identifier
            target_label: int, target class
            epsilon: float, injection strength
            max_rounds: int, total training rounds
            strategy: str, 'ANB' (Dynamic/Ours) or 'FIXED' (Baseline/Static)
            use_phased_chaos: bool, [P2-1 ablation] enable 3-stage phase scheduling
            use_spectral_smoothing: bool, [P2-1 ablation] enable Gaussian nebula diffusion
            use_freq_sharding: bool, [P2-1 ablation] enable per-client freq sharding
            use_dual_routing: bool, [P2-1 ablation] enable spatial/freq dual-domain routing
        """
        self.client_id = client_id
        self.target_label = target_label
        self.base_epsilon = epsilon
        self.max_rounds = max_rounds
        self.strategy = strategy
        self.current_round = 0  # Updated by training loop

        # [P2-4] Per-instance deterministic RNG for phase selection.
        # Seeded by (client_id, round) so the same (client, round) pair always
        # produces the same phase, regardless of global numpy RNG state.
        # This fixes non-reproducible ASR measurements across runs and epochs.
        self._phase_rng = np.random.RandomState(seed=client_id * 1000)

        # [P2-1] Ablation switches
        self.use_phased_chaos = use_phased_chaos
        self.use_spectral_smoothing = use_spectral_smoothing
        self.use_freq_sharding = use_freq_sharding
        self.use_dual_routing = use_dual_routing

        # ------------------------------------------------------------
        # Feature 3: Frequency Sharding Pool (Optimized Primes)
        # ------------------------------------------------------------
        # Defined as "safe zones" in frequency space [2, 7]
        self.freq_shards = [
            (2, 2), (2, 3), # Shard 0 (Client 0, 5...)
            (3, 2), (3, 3), # Shard 1 (Client 1, 6...)
            (2, 5), (5, 2), # Shard 2 (Client 2, 7...) - Asymmetric
            (3, 5), (5, 3), # Shard 3
            (4, 5), (5, 4), # Shard 4 - Mid Dense
            (4, 4)          # Fallback
        ]

        # ------------------------------------------------------------
        # Phase Pools for Phased Scheduling
        # ------------------------------------------------------------
        # Primary: 0, 90, 180, 270 deg (High Orthogonality)
        self.phase_pool_primary = [0, np.pi/2, np.pi, 3*np.pi/2]
        # Secondary: 45, 135, 225, 315 deg (Diversity)
        self.phase_pool_secondary = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]

    def set_round(self, round_num):
        """Called by Client.train() to update strategy state.

        [P2-4] Re-seeds the private phase RNG with (client_id, round_num) so
        that _get_current_phase() returns the same value for every call within
        the same round, making ASR measurements fully reproducible.
        """
        self.current_round = round_num
        # Deterministic seed: different clients AND different rounds get unique seeds
        self._phase_rng = np.random.RandomState(
            seed=(self.client_id * 1000 + round_num) % (2**31)
        )

    # ----------------------------------------------------------------
    # Feature 1: Phased Dynamic Chaos Controller
    # ----------------------------------------------------------------
    def _get_current_phase(self):
        """
        Dynamically schedule phase strategy based on learning stage.
        Idea: Stability early for learning -> Chaos late for stealth.
        Ablation: use_phased_chaos=False falls back to static phase 0.
        """
        # BASELINE: FIXED strategy or ablation disabled → static phase
        if self.strategy == 'FIXED' or not self.use_phased_chaos:
            return 0.0

        # ANB: Dynamic Scheduling
        # STAGE 1: Stabilization (Rounds 0-15) — deterministic per client
        if self.current_round < 15:
            idx = self.client_id % 4
            return self.phase_pool_primary[idx]

        # STAGE 2: Expansion (Rounds 15-35) — random from primary phases
        # [P2-4] Use private RNG seeded by (client_id, round) for reproducibility
        elif self.current_round < 35:
            idx = self._phase_rng.randint(0, 4)
            return self.phase_pool_primary[idx]

        # STAGE 3: Maximum Chaos (Rounds 35+) — all 8 phases
        # [P2-4] Same: private RNG guarantees same phase per (client, round) pair
        else:
            all_phases = self.phase_pool_primary + self.phase_pool_secondary
            idx = self._phase_rng.randint(0, 8)
            return all_phases[idx]

    def _get_adaptive_sigma(self):
        """
        Adaptive diffusion sigma.
        Early: 0.8 (Sharp, Star-like) -> High Signal Strength
        Late: 1.5 (Blurry, Nebula-like) -> High Stealth
        Ablation: use_spectral_smoothing=False keeps sigma fixed at 0.8 (Star mode).
        """
        # BASELINE or ablation disabled → always sharp (no smoothing)
        if self.strategy == 'FIXED' or not self.use_spectral_smoothing:
            return 0.8

        # ANB: Adaptive Smoothing
        if self.current_round < 20:
            return 0.8
        else:
            return 1.5

    def _get_scaling_factor(self, sigma):
        """
        Calculates Energy Compensation Factor.
        When a peak is smoothed, its amplitude drops. We boost it back
        to ensure the trigger remains "visible" to the neural network.
        """
        # Empirically tuned for CIFAR-10 range
        return 1.0 + (sigma * 1.5)

    # ----------------------------------------------------------------
    # Feature 2: Normalized Spectral Smoothing (Nebula Generator)
    # ----------------------------------------------------------------
    def _generate_normalized_nebula_pattern(self, H, W, center_u, center_v):
        """
        Generates the 'Nebula' Frequency Pattern.
        1. Creates Gaussian weights around center freq.
        2. Normalizes weights (Sum=1).
        3. Scales up peak amplitude (Compensation).
        4. Synthesizes composite wave.
        """
        sigma = self._get_adaptive_sigma()
        compensation = self._get_scaling_factor(sigma)

        # Prepare Pattern Canvas
        pattern = np.zeros((H, W), dtype=np.float32)

        # Grid pre-calculation
        x = np.arange(W)
        y = np.arange(H)
        grid_x, grid_y = np.meshgrid(x, y)

        # Get dynamic phase
        current_phase = self._get_current_phase()

        # Gaussian Kernel Window (3-sigma rule)
        window_r = int(np.ceil(2.5 * sigma))

        # Temporary storage
        components = []
        total_weight = 0.0

        for du in range(-window_r, window_r + 1):
            for dv in range(-window_r, window_r + 1):
                # Calculate absolute frequency
                curr_u, curr_v = center_u + du, center_v + dv

                # Filter valid positive frequencies (exclude DC and High-freq trash)
                if curr_u < 1 or curr_v < 1 or curr_u >= H//2 or curr_v >= W//2:
                    continue

                # Calculate Gaussian Weight
                dist_sq = du**2 + dv**2
                weight = np.exp(-dist_sq / (2 * sigma**2))

                # Pruning weak tails
                if weight < 0.01: continue

                components.append({'u': curr_u, 'v': curr_v, 'w': weight})
                total_weight += weight

        # Synthesize Pattern
        if total_weight < 1e-6: return pattern # Safe exit

        for comp in components:
            # 1. Normalized weight (fraction of total energy)
            # 2. Apply Compensation (boost visibility)
            real_amp = (comp['w'] / total_weight) * compensation

            # Note: Phase is consistent across the nebula for coherence
            wave = np.sin(2 * np.pi * comp['u'] * grid_x / W +
                          2 * np.pi * comp['v'] * grid_y / H +
                          current_phase)

            pattern += wave * real_amp

        return pattern

    # ----------------------------------------------------------------
    # Feature 4: Dual-Domain Routing Masks
    # ----------------------------------------------------------------
    def _compute_dual_routing_masks(self, image):
        """
        Calculates adaptive masks based on Local Complexity (Variance).
        High Complexity -> Frequency Injection
        Low Complexity  -> Spatial Injection
        Ablation: use_dual_routing=False returns uniform freq mask (no content-adaptive routing).
        """
        # BASELINE or ablation disabled → uniform frequency mask, no spatial routing
        if self.strategy == 'FIXED' or not self.use_dual_routing:
            return np.ones(image.shape[:2], dtype=np.float32), np.zeros(image.shape[:2], dtype=np.float32)

        # ANB: Adaptive Routing based on local variance
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        img_f = gray.astype(np.float32) / 255.0

        # Fast local variance (Complexity map)
        mu = cv2.blur(img_f, (3,3))
        mu2 = cv2.blur(img_f*img_f, (3,3))
        variance = np.abs(mu2 - mu*mu)

        # Normalize relative to image max (adaptive contrast)
        max_var = np.max(variance) + 1e-6
        complexity_map = np.clip(variance / max_var, 0, 1)

        # Freq Mask: enhance in textured areas
        freq_mask = np.power(complexity_map, 0.4)

        # Spatial Mask: only for truly flat areas
        spatial_base = 1.0 - complexity_map
        spatial_mask = np.power(spatial_base, 3.0)

        return freq_mask, spatial_mask

    # ----------------------------------------------------------------
    # Main Trigger Interface
    # ----------------------------------------------------------------
    def __call__(self, image, label):
        if label == self.target_label:
            return image, label

        H, W, C = image.shape
        img_float = image.astype(np.float32) / 255.0

        # 1. Routing
        freq_routing, spatial_routing = self._compute_dual_routing_masks(image)
        freq_routing_3d = np.stack([freq_routing]*C, axis=2)
        spatial_routing_3d = np.stack([spatial_routing]*C, axis=2)

        # 2. Frequency Branch (The Nebula)
        # Feature 3: Frequency Sharding — ablation: use_freq_sharding=False forces shard 0
        if self.strategy == 'FIXED' or not self.use_freq_sharding:
            # Baseline / ablation: all clients use the same frequency shard
            center_u, center_v = self.freq_shards[0]
        else:
            # ANB: dispersed shards based on client_id
            center_u, center_v = self.freq_shards[self.client_id % len(self.freq_shards)]
            
        nebula = self._generate_normalized_nebula_pattern(H, W, center_u, center_v)
        nebula_3d = np.stack([nebula]*C, axis=2)

        # Inject Freq
        # Feature 4 (dual routing): ANB boosts in textured areas; FIXED/ablation uses uniform mask
        if self.strategy == 'FIXED' or not self.use_dual_routing:
            freq_inject = nebula_3d * self.base_epsilon
        else:
            freq_inject = nebula_3d * freq_routing_3d * self.base_epsilon * 1.5

        # 3. Spatial Branch (The Ghost Tint)
        # Feature 4 (dual routing): only active when dual routing is enabled and strategy=ANB
        spatial_inject = 0.0
        if self.strategy == 'ANB' and self.use_dual_routing:
            # Strategy: "Grid Tint" - Checkerboard + Channel Bias
            spatial_pat = np.zeros_like(img_float)
            c_idx = self.client_id % 3

            # Create subtle corner grid (Spatial signature)
            corner_grid = np.zeros((H, W), dtype=np.float32)
            corner_grid[H-4:, W-4:] = 1.0
            corner_grid[H-4::2, W-4::2] = 0.0 # Make it a checkerboard

            spatial_pat[:, :, c_idx] = corner_grid

            # Inject Spatial: Lower epsilon (0.6x)
            spatial_inject = spatial_pat * spatial_routing_3d * self.base_epsilon * 0.6

        # 4. Fusion
        poisoned = img_float + freq_inject + spatial_inject
        poisoned = np.clip(poisoned, 0, 1)

        return (poisoned * 255).astype(np.uint8), self.target_label

    def poison_batch(self, images, labels):
        poisoned_images = []
        poisoned_labels = []
        for img, lbl in zip(images, labels):
            pi, pl = self(img, lbl)
            poisoned_images.append(pi)
            poisoned_labels.append(pl)
        return np.array(poisoned_images), np.array(poisoned_labels)


# --------------------------------------------------------
# Backward Compatibility Layer
# --------------------------------------------------------
@ATTACKS.register("frequency")
class FrequencyBackdoor(AdaptiveNebulaBackdoor):
    """
    Compatibility wrapper to maintain interface with existing codebase.
    Maps old FrequencyBackdoor calls to new AdaptiveNebulaBackdoor.
    """
    def __init__(self, client_id, target_label=0, epsilon=0.1, freq_strategy='ANB',
                 use_phased_chaos=True, use_spectral_smoothing=True,
                 use_freq_sharding=True, use_dual_routing=True):
        # Pass freq_strategy and ablation flags to parent
        super().__init__(
            client_id, target_label, epsilon, max_rounds=50, strategy=freq_strategy,
            use_phased_chaos=use_phased_chaos,
            use_spectral_smoothing=use_spectral_smoothing,
            use_freq_sharding=use_freq_sharding,
            use_dual_routing=use_dual_routing
        )


# --------------------------------------------------------
# Self-Test / Debug Block
# --------------------------------------------------------
if __name__ == '__main__':
    print("Initializing ANB Ultimate Self-Test...")
    
    # Test ANB (Dynamic)
    print("\n[TEST 1] ANB Strategy (Dynamic)")
    bd_anb = AdaptiveNebulaBackdoor(client_id=1, strategy='ANB')
    
    bd_anb.set_round(5)
    print(f"Round 5 (Start) Sigma: {bd_anb._get_adaptive_sigma()} (Expected 0.8)")
    
    bd_anb.set_round(40)
    print(f"Round 40 (End) Sigma: {bd_anb._get_adaptive_sigma()} (Expected 1.5)")
    
    # Test FIXED (Static)
    print("\n[TEST 2] FIXED Strategy (Static Baseline)")
    bd_fixed = AdaptiveNebulaBackdoor(client_id=1, strategy='FIXED')
    
    bd_fixed.set_round(40)
    print(f"Round 40 Sigma: {bd_fixed._get_adaptive_sigma()} (Expected 0.8)")
    print(f"Phase: {bd_fixed._get_current_phase()} (Expected 0)")
    
    print("\n✓ Core Attack Logic Patched Successfully!")
