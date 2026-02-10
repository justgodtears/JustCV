#image_detector.py
"""Detection of objects on images using YOLOv8"""
from ultralytics import YOLO, solutions
import cv2
from src.config import YOLO_MODEL_PATH
from collections import Counter


class ObjectDetection:
    def __init__(self, model_path=YOLO_MODEL_PATH):            #Default argument is yolov8m
        self.model = YOLO(str(model_path))


    def detect(self, image):
        results = self.model(image)
        return results


    def save_annotations(self,results, output_path):
        annotated_img = results[0].plot()
        cv2.imwrite(str(output_path), annotated_img)


    def count_objects(self, results):
        detected_objects = results[0].boxes.cls
        class_names = results[0].names
        detected_names = []
        for class_id in detected_objects:
            detected_names.append(class_names[int(class_id)])

        count = Counter(detected_names)
        return dict(count)

