from pathlib import Path

# --------------------------------------------------------------------------
# Project root
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
DATASET_PATH = Path(r"D:\Datasets\BioHub\biohub-cell-tracking-during-development")
TRAIN_PATH = DATASET_PATH / "train"
TEST_PATH = DATASET_PATH / "test"

# --------------------------------------------------------------------------
# Project folders
# --------------------------------------------------------------------------
OUTPUT_PATH = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
LAST_MODEL_PATH = MODEL_DIR / "last_model.pth"
BEST_NORM_STATS_PATH = MODEL_DIR / "norm_stats.npz"

LOG_FILE = OUTPUT_PATH / "training.log"

# --------------------------------------------------------------------------
# Detection (used by src/detector.py)
# --------------------------------------------------------------------------
DETECTION_THRESHOLD = 800   # peak_local_max threshold_abs
GAUSSIAN_SIGMA = 2
CELL_RADIUS = 12              # peak_local_max min_distance

# --------------------------------------------------------------------------
# Patch extraction (used by notebooks/04_training.ipynb, src/predict.py)
# --------------------------------------------------------------------------
PATCH_SIZE = 32
NEGATIVE_EXCLUSION_RADIUS = 20  # min distance from a positive center for a valid negative sample

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Loss: "bce" (nn.BCEWithLogitsLoss) or "focal" (src/losses.py FocalLoss)
LOSS_FN = "bce"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

# LR scheduler (ReduceLROnPlateau, monitors val_loss)
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_MIN_LR = 1e-6

# Early stopping (monitors val_loss)
EARLY_STOPPING_PATIENCE = 6

# Augmentation (applied to training patches only, see src/augmentations.py)
USE_AUGMENTATION = True
BRIGHTNESS_MAX_DELTA = 0.2
GAUSSIAN_NOISE_SIGMA = 0.05

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
LOG_LEVEL = "INFO"
