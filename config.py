"""Project-wide paths and constants. Every module imports from here so sizes and paths never drift."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Data
DATASET_DIR = ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
SPLIT_DIRS = {
    "train": DATASET_DIR / "train",
    "validation": DATASET_DIR / "validation",
    "test": DATASET_DIR / "test",
    "external_test": DATASET_DIR / "external_test",  # Messidor-2, never used for training
}
MANIFEST_DIR = DATASET_DIR / "manifests"
# Preprocessed model-input arrays. RETINAGEN_ARRAY_DIR can point training at a copy on WSL's own disk: the 384 px
# training array (~12 GB) is memory-mapped, and random reads through /mnt/d are too slow.
ARRAY_DIR = Path(os.environ.get("RETINAGEN_ARRAY_DIR", DATASET_DIR / "arrays"))

# Models and run artifacts
MODELS_DIR = ROOT / "models"
CNN_MODEL_PATH = MODELS_DIR / "model.keras"
OUTPUTS_DIR = ROOT / "outputs"
EVAL_DIR = OUTPUTS_DIR / "eval"
REPORTS_DIR = ROOT / "reports"
STATIC_DIR = ROOT / "static"
DB_PATH = ROOT / "database" / "retinagen.db"

# DR stages (International Clinical DR scale)
STAGE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_STAGES = len(STAGE_NAMES)
MAX_STAGE = NUM_STAGES - 1
REFERABLE_STAGE = 2  # stage >= 2 is referable DR

# Image sizes. Every image is first cached at CACHE_SIZE; CNN and GAN inputs are downscaled from it.
CACHE_SIZE = 512
CNN_SIZE = 224  # default classifier input; each saved classifier records its own size in its input shape
CNN_SIZES = (224, 384)  # 384 px keeps microaneurysms (mild DR) visible
GAN_SIZE = 256

# Preprocessing parameters (shared by training and inference)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
DENOISE_KERNEL = 3  # Gaussian kernel size; light denoise only
BORDER_THRESHOLD = 10  # pixels darker than this (0-255) count as the black border
# Every model input is cut to one standard field of view: a circle the full width of the image, clipped to a
# horizontal band this fraction of its diameter tall. Cameras that clip the field top and bottom are commoner
# at higher stages (46.6% of stage-0 vs 65.9% of stage-4 training images), and the first 0->1 generator learned
# to clip fields instead of adding lesions. 0.75 leaves only 1.9% of images clipped beyond the template.
FIELD_HEIGHT_RATIO = 0.75

# Upload limits (FR-1)
MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Reproducibility
SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def generator_path(from_stage: int) -> Path:
    """Path of the k -> k+1 transition generator, e.g. models/generator_1to2.keras."""
    return MODELS_DIR / f"generator_{from_stage}to{from_stage + 1}.keras"


DISCLAIMER = (
    "Research and education prototype. Not a medical device and not a diagnosis. "
    "Simulated images show a general next-stage appearance learned from other patients, "
    "not a prediction of this patient's future. Consult an eye-care professional."
)
