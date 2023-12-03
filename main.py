from KontrolkiDetectionApp import KontrolkiDetectionApp

if __name__ == "__main__":
    yolo_model_path = "models/YOLO_kontrolka2.pt"
    classification_model_path = "models/DenseNet_model.h5"
    class_names = [
        'awaria poduszki powietrznej', 'awaria silnika',
        'awaria swiatel zewnetrznych', 'awaria swiec zarowych silnika',
        'awaria systemu ABS', 'awaria systemu hamulcowego',
        'awaria ukladu elektronicznego', 'awaria ukladu wspomagania kierownicy',
        'minimalny poziom oleju', 'niski poziom cisnienia w oponach',
        'oblodzona nawierzchnia drogi', 'rezerwa paliwa',
        'slaby akumulator', 'swiatla drogowe',
        'swiatla mijania', 'swiatla pozycyjne',
        'swiatla przeciwmgielne', 'system kontroli trakcji',
        'uszkodzenie skrzyni biegow', 'wylaczony system kontroli trakcji',
        'wysoka temperatura plynu chlodniczego', 'zaciagniety hamulec reczny',
        'zapinj pasy', 'zuzycie klockow hamulcowych'
    ]

    # Choose image path during runtime
    image_path = input("Enter the path to the input image: ")

    # Run the ImageProcessorApp
    image_processor_app = KontrolkiDetectionApp(yolo_model_path, classification_model_path, class_names)
    image_processor_app.process_image(image_path)
