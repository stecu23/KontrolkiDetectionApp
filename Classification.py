import os
import numpy as np
from tensorflow.keras.models import load_model

class Classification:
    @staticmethod
    def _reshape_input(crop_object_resized):
        return crop_object_resized.reshape((1, 128, 128, 3))

    @staticmethod
    def classify_object(crop_object_resized, model_path, class_names):
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return None

        model = load_model(model_path)

        crop_object_resized = Classification._reshape_input(crop_object_resized)
        crop_object_preprocessed = crop_object_resized / 255.0
        classification_result = model.predict(crop_object_preprocessed)
        predicted_class_index = np.argmax(classification_result)
        predicted_class_name = class_names[predicted_class_index]
        return predicted_class_name


