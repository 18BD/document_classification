'''
Импортируем необходимые модули и библиотеки, такие как numpy (для работы с массивами), 
image (для обработки изображений) и load_model (для загрузки модели Keras).
'''
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.models import load_model


'''
Определяем размеры и форму входного изображения для модели.
'''
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)


'''
Определяем класс ImageClassifier, который инкапсулирует функциональность классификации изображений. 
В конструкторе класса инициализируются список классов (class_names) и загружается модель (load_model). 
Метод predict принимает путь к изображению, загружает его, преобразует в тензор, 
делает предсказание с помощью модели и возвращает предсказанный класс.
'''
class ImageClassifier:
    def __init__(self):
        self.class_names = ["Driver's license", "ID card", "Passport"]
        self.load_model()

    def load_model(self):
        self.model = load_model('model.h5')

    def predict(self, image_path):
        test_image = image.load_img(image_path, target_size=INPUT_SHAPE)
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        probability = self.model.predict(test_image)
        prediction = np.argmax(probability[0])
        return self.class_names[int(prediction)]


'''
Определяем функцию main, которая выполняет основную логику программы. 
В данном случае, задан путь к изображению test_id.jpg, создается экземпляр классификатора (ImageClassifier) 
и вызывается метод predict для выполнения предсказания класса изображения. Результат предсказания выводится на экран.
'''
def main():
    image_path = './dataset/single_prediction/test_id.jpg'
    classifier = ImageClassifier()
    prediction_result = classifier.predict(image_path)
    print(prediction_result)


'''
Проверяем, что скрипт запущен напрямую (а не импортирован как модуль), и если это так, вызываем функцию main.
'''
if __name__ == '__main__':
    main()
