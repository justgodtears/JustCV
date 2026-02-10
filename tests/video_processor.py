from src.core.video_processor import VideoProcessor
from src.core.image_detector import ObjectDetection
from src.config import RAW_DIR, PROCESSED_DIR

# Stwórz detector
detector = ObjectDetection()

# Stwórz video processor
processor = VideoProcessor(detector)

# Przetwórz wideo
processor.process_video(
    input_path=RAW_DIR / "traffic.mp4",
    output_path=PROCESSED_DIR / "output_video.mp4"
)