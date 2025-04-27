import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                            QLineEdit, QPushButton, QFileDialog, QSizePolicy, QLabel)
from PyQt6.QtCore import Qt


from keras.models import load_model  # Правильный импорт  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def check_ai(path):
    

    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_Model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    result_text = f"Class: {class_name[2:]} Confidence Score: {confidence_score:.4f}"
    return result_text






selected_file_path = ""

def select_file():
    global selected_file_path  
    file_path, _ = QFileDialog.getOpenFileName(
        window,
        "Выберите файл",
        "",
        "Изображения (*.png *.jpg)"
    )
    if file_path:
        path_input.setText(file_path)
        selected_file_path = file_path  
      

def update_label():
    global selected_file_path
    selected_file_path = path_input.text() 
    result = check_ai(selected_file_path)
    path_label.setText(result)
    path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Выбор файла")
window.resize(500, 200)


path_input = QLineEdit()
path_input.setPlaceholderText("Введите путь или выберите файл...")



select_button = QPushButton("Выбрать")
select_button.clicked.connect(select_file)

check_button = QPushButton("Проверить")
check_button.clicked.connect(update_label)


path_label = QLabel("Здесь будет отображаться результат")
path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


path_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
select_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
check_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


input_layout = QHBoxLayout()
input_layout.addWidget(path_input)
input_layout.addWidget(select_button)


main_layout = QVBoxLayout()
main_layout.addStretch(1)
main_layout.addWidget(path_label, alignment=Qt.AlignmentFlag.AlignCenter)
main_layout.addStretch(1)
main_layout.addLayout(input_layout)
main_layout.addWidget(check_button)

check_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
window.setLayout(main_layout)
window.show()

sys.exit(app.exec())
