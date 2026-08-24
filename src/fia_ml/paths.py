"""Project root and path helpers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DOCUMENTATION = PROJECT_ROOT / "documentation"
SCHEMA_CSV = DOCUMENTATION / "f1_dataset_example.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
DEFAULT_TRAINING_CONFIG = PROJECT_ROOT / "configs" / "xgboost.yaml"
DEFAULT_TRAINING_V2_CONFIG = PROJECT_ROOT / "configs" / "xgboost_v2.yaml"
DEFAULT_FEATURES_CONFIG = PROJECT_ROOT / "configs" / "features.yaml"
TARGET_MAPPING_CONFIG = PROJECT_ROOT / "configs" / "target_mapping.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
