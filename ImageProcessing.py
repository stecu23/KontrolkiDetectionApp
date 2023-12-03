# ImageProcessing.py
import cv2


class ImageProcessing:
    @staticmethod
    def crop_and_resize(frame, box, input_size=128):
        crop_object = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        crop_object_resized = cv2.resize(crop_object, (input_size, input_size))
        return crop_object_resized
