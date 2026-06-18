"""
Centralized configuration for the Yoga Pose Detection & Correction Platform.
All hyperparameters, paths, and constants in one place.
"""
import os

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DATASET")
TRAIN_DIR = os.path.join(DATASET_DIR, "TRAIN")
TEST_DIR = os.path.join(DATASET_DIR, "TEST")

MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "yoga_pose_cnn.keras")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "yoga_pose_best.keras")

# ==============================================================================
# DATASET
# ==============================================================================
CLASS_NAMES = ["downdog", "goddess", "plank", "tree", "warrior2"]
NUM_CLASSES = len(CLASS_NAMES)

# ==============================================================================
# IMAGE SETTINGS
# ==============================================================================
IMG_SIZE = (224, 224)          # MobileNetV2 default input size
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

# ==============================================================================
# TRAINING - STAGE 1 (Feature Extraction)
# ==============================================================================
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3

# ==============================================================================
# TRAINING - STAGE 2 (Fine-Tuning)
# ==============================================================================
STAGE2_EPOCHS = 25
STAGE2_LR = 1e-5
FINE_TUNE_AT = 100  # Unfreeze layers from this index onward (~last 30% of MobileNetV2)

# ==============================================================================
# REGULARIZATION & SCHEDULING
# ==============================================================================
VALIDATION_SPLIT = 0.15
DROPOUT_1 = 0.5
DROPOUT_2 = 0.3
DENSE_UNITS = 256
L2_REG = 1e-4

LR_PATIENCE = 3         # ReduceLROnPlateau patience
LR_FACTOR = 0.5         # LR reduction factor
EARLY_STOP_PATIENCE = 7  # Early stopping patience

# ==============================================================================
# MEDIAPIPE POSE ESTIMATION
# ==============================================================================
MP_MIN_DETECTION_CONFIDENCE = 0.7
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ==============================================================================
# POSE CORRECTION
# ==============================================================================
ANGLE_TOLERANCE = 15  # Degrees of tolerance for angle comparison

# ==============================================================================
# APP SETTINGS
# ==============================================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CNN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to display pose classification

# Colors & UI (BGR for OpenCV)
COLOR_GREEN = (20, 255, 120)      # Softer Green
COLOR_RED = (50, 50, 255)         # Vibrant Red
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (245, 245, 245)
COLOR_BLACK = (20, 20, 20)
COLOR_BLUE = (255, 165, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)

# Premium UI Theme
COLOR_DARK_BG = (15, 15, 15)       # Deep matte black for the mentor side
COLOR_ACCENT = (255, 191, 0)       # Neon Blue for the AI Twin
COLOR_HUD_TEXT = (240, 240, 240)   # Off-white for readability
COLOR_IDEAL = (0, 255, 127)        # Spring Green for "perfect" alignment

# Layout
SPLIT_SCREEN_WIDTH = CAMERA_WIDTH * 2
UI_HEADER_HEIGHT = 80
UI_FOOTER_HEIGHT = 60
