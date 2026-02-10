from pathlib import Path

# Root projektu
PROJECT_ROOT = Path(__file__).parent.parent

# Foldery
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8m.pt"

