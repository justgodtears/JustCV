from src.core.image_detector import ObjectDetection
from src.config import RAW_DIR, PROCESSED_DIR, YOLO_MODEL_PATH
import datetime

date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

test = ObjectDetection("../models/yolov8x.pt")
result = test.detect(RAW_DIR / "3.jpg")
counts = test.count_objects(result)
print(f"Detected objects: {counts}")
test.save_annotations(result, PROCESSED_DIR / f"3-8N{date}.jpg")
print(f"✅ Saved in {PROCESSED_DIR / f'3{date}.jpg'}")

