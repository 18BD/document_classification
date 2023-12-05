'''
Импортируем необходимые модули и библиотеки, такие как Flask (фреймворк для создания веб-приложений на Python), 
os (для работы с файловой системой), numpy (для работы с массивами), 
image (для обработки изображений) и load_model (для загрузки модели Keras).
'''
import os
import shutil
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


'''
Создаем экземпляр приложения Flask и настраиваем папку для загрузки файлов и разрешенные расширения файлов.
'''
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["ALLOWED_EXTENSIONS"] = {"jpg", "jpeg", "png"}


'''
Определяем размеры и форму входного изображения для модели.
'''
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)


'''
Определяем классы, которые модель может предсказывать. 
В данном случае, это "Водительское удостоверение", "Удостоверение личности" и "Паспорт".
'''
class_names = ["Driver's license", "ID card", "Passport"]


'''
Загружаем предварительно обученную модель из файла 'model.h5'.
'''
model = load_model('model.h5')


'''
Определяем функцию allowed_file, которая проверяет, 
соответствует ли расширение файла разрешенным расширениям, определенным в настройках приложения.
'''
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


'''
Определяем функцию predict_document_type, которая загружает изображение по заданному пути, 
обрабатывает его и предсказывает тип документа, используя загруженную модель.
'''
def predict_document_type(image_path):
    test_image = image.load_img(image_path, target_size=INPUT_SHAPE)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    probability = model.predict(test_image)
    prediction = np.argmax(probability[0])
    return class_names[int(prediction)]


'''
Определяем функцию save_document, 
которая сохраняет загруженный файл в соответствующей папке в зависимости от предсказанного типа документа.
'''
def save_document(file, filename, document_type, source):
    output_folder = os.path.join(app.config["UPLOAD_FOLDER"], document_type)
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, filename)
    shutil.copy(source, save_path)


'''
Определяем маршрут для главной страницы ("/"), который обрабатывает GET и POST запросы. 
При POST запросе получаем список загруженных файлов, классифицируем их и возвращаем результаты в шаблон HTML. 
При GET запросе просто отображаем главную страницу.
'''
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files = request.files.getlist("files")
        filenames, document_types = classify_documents(files)
        return render_template("index.html", filenames=filenames, document_types=document_types, uploaded=True)

    return render_template("index.html")


'''
Определяем функцию classify_documents, которая проходит по списку загруженных файлов, проверяет их расширения, 
классифицирует и сохраняет в соответствующие папки. Затем возвращает списки имен файлов и предсказанных типов документов.
'''
def classify_documents(files):
    filenames = []
    document_types = []

    for file in files:
        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            document_type = predict_document_type(save_path)
            save_document(file, filename, document_type, save_path)

            filenames.append(filename)
            document_types.append(document_type)

    return filenames, document_types


'''
Запускаем приложение Flask, если скрипт запущен напрямую (а не импортирован как модуль), 
с включенным режимом отладки (debug mode).
'''
if __name__ == "__main__":
    app.run(debug=True)
