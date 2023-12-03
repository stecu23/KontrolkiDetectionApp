import cv2
from ObjectDetection import ObjectDetection
from Classification import Classification
from ImageProcessing import ImageProcessing
from DataVisualization import DataVisualization


class KontrolkiDetectionApp(ObjectDetection, ImageProcessing, DataVisualization):
    def __init__(self, yolo_model_path, classification_model_path, class_names):
        super().__init__(yolo_model_path)
        self.classification_model_path = classification_model_path
        self.class_names = class_names

    def process_image(self, image_path):
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error reading image file: {image_path}")
            return

        boxes = self.detect_objects(frame)

        cropped_images = []
        predicted_classes = []

        for box in boxes:
            crop_object_resized = self.crop_and_resize(frame, box)
            predicted_class = Classification.classify_object(
                crop_object_resized,
                self.classification_model_path,
                self.class_names
            )

            if predicted_class is not None:
                cropped_images.append(crop_object_resized)
                predicted_classes.append(predicted_class)

        self.display_results(cropped_images, predicted_classes)
