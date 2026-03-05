from dataclasses import dataclass, asdict
import argparse


@dataclass
class Config:
    # Attack configuration
    attack_mode: str = "OURS"
    freq_strategy: str = "ANB"  # 'FIXED' or 'ANB'
    backdoor_name: str = "frequency"  # registry key

    # [P2-1] Ablation switches — set any to False to isolate component contribution.
    # Full ANB = all True. Each False removes one feature for ablation experiments.
    use_phased_chaos: bool = True       # Feature 1: 3-stage phase scheduling
    use_spectral_smoothing: bool = True # Feature 2: Gaussian-diffused nebula pattern
    use_freq_sharding: bool = True      # Feature 3: per-client frequency sharding
    use_dual_routing: bool = True       # Feature 4: spatial/frequency dual-domain routing

    # Dataset configuration
    dataset: str = "CIFAR10"
    image_size: int = 32
    num_classes: int = 10
    data_dir: str = "./data"

    # Federated learning parameters
    num_clients: int = 10
    poison_ratio: float = 0.2
    target_label: int = 0
    epsilon: float = 0.1
    # Poisoning rate inside each malicious client (fraction of non-target samples poisoned).
    poison_rate: float = 1.0
    alpha: float = 0.5
    # [P1-2] Model Replacement Scaling factor for malicious clients.
    # Compensates for FedAvg weight dilution when malicious clients are a minority.
    # Set to num_clients / num_malicious (e.g. 10/2=5.0) to match academic standard.
    # Set to 1.0 to disable scaling (ablation baseline).
    scaling_factor: float = 5.0
    # [Backdoor Boost] Weight for backdoor enhancement loss (0~1).
    # Helps malicious clients learn both main task and backdoor simultaneously.
    # Higher values strengthen backdoor learning. Set to 0 to disable.
    backdoor_boost_weight: float = 0.3

    # Training parameters
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    client_fraction: float = 1.0
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 42

    # Model
    model_name: str = "resnet18"

    # Defense configuration
    defense_enabled: bool = True
    # [P3-2] defense_method options:
    #   FreqFed variants : 'hdbscan' (default), 'kmeans', 'dbscan', 'freqfed'
    #   Additional       : 'fltrust'   (NDSS 2022)
    #                      'foolsgold' (CCS 2020)
    defense_method: str = "hdbscan"

    # Validation thresholds
    min_asr: float = 0.9
    max_psnr_drop: float = 5.0

    # Output
    results_dir: str = "./results"
    weights_dir: str = "./results/weights"


def _build_parser():
    parser = argparse.ArgumentParser(description="SAFB Experiment Configuration")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--poison-ratio", type=float, default=None)
    parser.add_argument("--poison-rate", type=float, default=None)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--scaling-factor", type=float, default=None)
    parser.add_argument("--backdoor-boost-weight", type=float, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--client-fraction", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pin-memory", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--attack-mode", type=str, default=None)
    parser.add_argument("--freq-strategy", type=str, default=None)
    parser.add_argument("--backdoor-name", type=str, default=None)
    # [P2-1] Ablation switches (pass 0 to disable, 1 to enable)
    parser.add_argument("--use-phased-chaos", type=int, default=None)
    parser.add_argument("--use-spectral-smoothing", type=int, default=None)
    parser.add_argument("--use-freq-sharding", type=int, default=None)
    parser.add_argument("--use-dual-routing", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--defense-enabled", type=int, default=None)
    parser.add_argument("--defense-method", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--weights-dir", type=str, default=None)
    return parser


def load_config(argv=None):
    cfg = Config()
    parser = _build_parser()
    args = parser.parse_args(argv)

    for field in asdict(cfg).keys():
        arg_name = field.replace("_", "-")
        if hasattr(args, field):
            value = getattr(args, field)
        else:
            value = getattr(args, field, None)
        if value is not None:
            setattr(cfg, field, value)

    if args.pin_memory is not None:
        cfg.pin_memory = bool(args.pin_memory)
    if args.defense_enabled is not None:
        cfg.defense_enabled = bool(args.defense_enabled)
    # [P2-1] Ablation bool switches
    if args.use_phased_chaos is not None:
        cfg.use_phased_chaos = bool(args.use_phased_chaos)
    if args.use_spectral_smoothing is not None:
        cfg.use_spectral_smoothing = bool(args.use_spectral_smoothing)
    if args.use_freq_sharding is not None:
        cfg.use_freq_sharding = bool(args.use_freq_sharding)
    if args.use_dual_routing is not None:
        cfg.use_dual_routing = bool(args.use_dual_routing)

    return cfg


_DEFAULT = Config()

# Backward-compatible module-level constants
ATTACK_MODE = _DEFAULT.attack_mode
FREQ_STRATEGY = _DEFAULT.freq_strategy
BACKDOOR_NAME = _DEFAULT.backdoor_name

DATASET = _DEFAULT.dataset
IMAGE_SIZE = _DEFAULT.image_size
NUM_CLASSES = _DEFAULT.num_classes

NUM_CLIENTS = _DEFAULT.num_clients
POISON_RATIO = _DEFAULT.poison_ratio
POISON_RATE = _DEFAULT.poison_rate
TARGET_LABEL = _DEFAULT.target_label
EPSILON = _DEFAULT.epsilon
ALPHA = _DEFAULT.alpha

NUM_ROUNDS = _DEFAULT.num_rounds
LOCAL_EPOCHS = _DEFAULT.local_epochs
BATCH_SIZE = _DEFAULT.batch_size
LEARNING_RATE = _DEFAULT.learning_rate

DEFENSE_ENABLED = _DEFAULT.defense_enabled
DEFENSE_METHOD = _DEFAULT.defense_method

MIN_ASR = _DEFAULT.min_asr
MAX_PSNR_DROP = _DEFAULT.max_psnr_drop
