#video_processor.py
import cv2
from src.core.image_detector import ObjectDetection

class VideoProcessor:
    def __init__(self, detector: ObjectDetection):
        self.detector = detector


    def process_video(self, input_path, output_path):

        # Input path
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        # FPS, Resolution from video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


        # Looping detection for every from of video
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video ended")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

            results = self.detector.detect(frame)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        # Closing
        cap.release()
        out.release()
        print("Video processed!")






