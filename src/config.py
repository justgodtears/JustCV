from pathlib import Path

# Root
PROJECT_ROOT = Path(__file__).parent.parent

# Folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEMO_DIR = PROJECT_ROOT / "demo"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
YOLO_MODEL_PATH = MODELS_DIR / "yolov8m.pt"
RAW_VIDEO = DEMO_DIR / "traffic.mp4"
PROCESSED_VIDEO = DEMO_DIR / "output_video.mp4"

