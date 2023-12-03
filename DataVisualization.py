# DataVisualization.py
import matplotlib.pyplot as plt
import cv2


class DataVisualization:
    @staticmethod
    def display_results(cropped_images, predicted_classes):
        num_images = len(cropped_images)
        num_cols = 4
        num_rows = (num_images // num_cols) + (num_images % num_cols > 0)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
        axs = axs.flatten()

        for i in range(num_images):
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(cropped_images[i], cv2.COLOR_BGR2RGB)

            axs[i].imshow(image_rgb)
            axs[i].set_title(predicted_classes[i])
            axs[i].axis('off')

        for j in range(num_images, len(axs)):
            fig.delaxes(axs[j])

        plt.show()
