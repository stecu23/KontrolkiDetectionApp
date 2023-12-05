# ObjectDetection.py
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        results = self.model.predict(frame, conf=0.3)
        boxes = results[0].boxes.xyxy.cpu()
        return boxes
