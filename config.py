# Control Center - All hyperparameters and configuration

# Attack configuration
ATTACK_MODE = 'OURS'  # 'FIBA' (baseline) or 'OURS' (proposed method)
# CRITICAL FIX: Changed from 'DISPERSED' to 'ANB' to enable Spatial Tint in attacks.py
FREQ_STRATEGY = 'ANB'  # 'FIXED' or 'ANB'

# Dataset configuration
DATASET = 'CIFAR10'
IMAGE_SIZE = 32
NUM_CLASSES = 10

# Federated learning parameters
NUM_CLIENTS = 10
POISON_RATIO = 0.2  # 20% malicious clients
TARGET_LABEL = 0
EPSILON = 0.1  # Injection strength
ALPHA = 0.5  # Dirichlet concentration parameter for Non-IID

# Training parameters
NUM_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# Defense configuration
DEFENSE_ENABLED = True
DEFENSE_METHOD = 'hdbscan'  # 'kmeans', 'dbscan', or 'hdbscan'

# Validation thresholds
MIN_ASR = 0.9  # Minimum acceptable ASR for successful attack
MAX_PSNR_DROP = 5.0  # Maximum allowed PSNR drop for stealth