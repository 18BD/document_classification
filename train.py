import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.optimizers import adam
from keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation


'''
Класс ImageClassifier представляет собой модель классификатора изображений. 
В целом, класс обеспечивает функциональность для создания, обучения и оценки модели на основе набора данных изображений.
Основные атрибуты класса включают параметры модели, такие как размеры изображений, количество слоев свертки, размеры слоев и количество полносвязных слоев. 
Кроме того, класс содержит коллбэки для обратного вызова во время обучения модели.
'''
class ImageClassifier:
    def __init__(self):
        self.IMAGE_WIDTH = 128
        self.IMAGE_HEIGHT = 128
        self.IMAGE_CHANNELS = 3
        self.INPUT_SHAPE = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_CHANNELS)
        self.EPOCHS = 100
        self.POOL_SIZE = 2
        self.LR = 0.0005
        self.CONV_LAYERS = [2]
        self.LAYER_SIZES = [64]
        self.DENSE_LAYERS = [2]
        self.callbacks = None
        self.model = None

    '''
    Функция create_callbacks создает коллбэки для обучения модели. 
    В данном случае используются три коллбэка: EarlyStopping 
    (останавливает обучение, если значение функции потерь на проверочных данных не улучшается в течение заданного количества эпох), 
    ModelCheckpoint (сохраняет веса модели с наилучшими показателями функции потерь на проверочных данных) 
    и TensorBoard (записывает логи для визуализации в TensorBoard).
    '''
    def create_callbacks(self):
        earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        checkpointer = ModelCheckpoint(filepath="best_weight.hdf5", monitor='val_loss', verbose=0, save_best_only=True)
        tensorboard = TensorBoard(log_dir="logs/{}".format(int(time.time())))
        self.callbacks = [earlystop, checkpointer, tensorboard]

    '''
    Функция build_model строит модель нейронной сети для классификации изображений. 
    Она принимает параметры conv_layers (количество слоев свертки), layer_size (размер слоев свертки и полносвязных слоев) 
    и dense_layers (количество полносвязных слоев). Модель состоит из слоев свертки (Conv2D), 
    функций активации (Activation), слоев пулинга (MaxPooling2D), слоев полносвязной нейронной сети (Dense) 
    и выходного слоя с сигмоидной функцией активации для бинарной классификации. 
    Модель компилируется с функцией потерь binary_crossentropy, оптимизатором adam и метрикой binary_accuracy.
    '''
    def build_model(self, conv_layers, layer_size, dense_layers):
        model = Sequential()
        model.add(Conv2D(layer_size, (3, 3), input_shape=self.INPUT_SHAPE, activation='relu'))
        model.add(MaxPooling2D(pool_size=(self.POOL_SIZE, self.POOL_SIZE)))
        for _ in range(conv_layers - 1):
            model.add(Conv2D(layer_size, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        for _ in range(dense_layers):
            model.add(Dense(layer_size))
            model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=adam.Adam(learning_rate=self.LR),
                      metrics=['binary_accuracy'])
        self.model = model

    '''
    Функция train выполняет процесс обучения модели на тренировочных данных. 
    Она принимает пути к тренировочному и тестовому наборам данных. Для аугментации тренировочных данных используется объект ImageDataGenerator, 
    а для тестовых данных используется рескейлинг. Затем создаются генераторы данных для тренировочного и тестового наборов.
    Далее, функция создает коллбэки с помощью метода create_callbacks. 
    Затем, происходит итерация по различным значениям количества слоев свертки, размера слоев и количества полносвязных слоев. 
    Для каждой комбинации вызывается метод build_model для построения модели и выполняется обучение модели с использованием метода fit_generator. 
    Затем вызываются методы plot_accuracy_loss для визуализации метрик обучения, evaluate_model для оценки модели на тестовом наборе данных 
    и generate_classification_report для создания отчета о классификации.
    '''
    def train(self, train_dir, test_dir):
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            width_shift_range=0.2,
            height_shift_range=0.2)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(train_dir,
                                                         target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
                                                         batch_size=32,
                                                         class_mode='binary',
                                                         shuffle=True,
                                                         seed=42)
        testing_set = test_datagen.flow_from_directory(test_dir,
                                                       target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
                                                       batch_size=1,
                                                       shuffle=False,
                                                       class_mode='binary')

        STEP_SIZE_TRAIN = training_set.n // training_set.batch_size
        STEP_SIZE_TEST = testing_set.n // testing_set.batch_size

        self.create_callbacks()

        for dense_layer in self.DENSE_LAYERS:
            for layer_size in self.LAYER_SIZES:
                for conv_layer in self.CONV_LAYERS:
                    self.build_model(conv_layer, layer_size, dense_layer)
                    history = self.model.fit_generator(training_set,
                                                       epochs=self.EPOCHS,
                                                       steps_per_epoch=STEP_SIZE_TRAIN,
                                                       validation_data=testing_set,
                                                       validation_steps=STEP_SIZE_TEST,
                                                       callbacks=self.callbacks)

                    self.plot_accuracy_loss(history)
                    self.evaluate_model(testing_set)
                    self.generate_classification_report(testing_set)

    '''
    Функция plot_accuracy_loss отображает графики точности и функции потерь во время обучения. 
    Она принимает историю обучения (history) и использует ее для извлечения значений точности, функции потерь и проверочной функции потерь. 
    Затем она строит два графика: точность обучения и функцию потерь обучения по эпохам, а также функцию потерь на проверочных данных по эпохам.
    '''
    @staticmethod
    def plot_accuracy_loss(history):
        acc = history.history['binary_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.title('Training accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.figure()

        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Testing loss')
        plt.title('Training and Testing loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

    '''
    Функция evaluate_model выполняет оценку модели на тестовом наборе данных. 
    Она создает простую модель с плоским слоем (Flatten), компилирует ее и выводит на экран метрики модели, 
    а также результат оценки модели на тестовом наборе данных.
    '''
    def evaluate_model(self, testing_set):
        model = Sequential()
        model.add(Flatten())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.metrics_names)
        print(model.evaluate(testing_set))

    '''
    Функция generate_classification_report создает отчет о классификации для модели. 
    Она получает метки классов из тестового набора данных, выполняет предсказания с помощью модели на тестовом наборе данных 
    и генерирует отчет о классификации, сравнивая предсказанные метки с фактическими метками тестового набора данных.
    '''
    def generate_classification_report(self, testing_set):
        labels = testing_set.class_indices
        Y_pred = self.model.predict_generator(testing_set, testing_set.n // testing_set.batch_size)
        y_pred = np.where(Y_pred > 0.5, 1, 0)

        print('Classification Report')
        print(classification_report(testing_set.classes, y_pred, target_names=labels))


'''
Функция main является точкой входа в программу. 
Она определяет пути к тренировочному и тестовому наборам данных, создает экземпляр класса ImageClassifier 
и вызывает метод train для обучения модели на указанных данных.
'''
def main():
    train_dataset_path = './dataset/train'
    test_dataset_path = './dataset/test'
    classifier = ImageClassifier()
    classifier.train(train_dataset_path, test_dataset_path)


if __name__ == '__main__':
    main()
