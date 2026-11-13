"""
Диагноз по одному фото листа.

Запуск (из корня проекта):
    python model/predict.py путь/к/фото.jpg

Пример ответа:
    Диагноз: Пирикуляриоз (blast)
    Уверенность: 91%
"""

import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
import tensorflow as tf
from PIL import Image
from tensorflow import keras

# Настройки берём из файла .env (если его нет - используем значения по умолчанию)
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/rice_doctor.keras")
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))  # 0.60 = 60%
CLASSES_PATH = "models/classes.json"

# Названия классов по-русски для красивого ответа
NAMES_RU = {
    "healthy": "Здоровый лист",
    "blast": "Пирикуляриоз",
    "brown_spot": "Бурая пятнистость",
    "bacterial_blight": "Бактериальный ожог",
}


def load_photo(path):
    """
    Открыть фото и подготовить его для сети ТОЧНО ТАК ЖЕ, как при обучении:
    1) уменьшить до 256 по длинной стороне (как в prepare_data.py);
    2) сжать до 224x224 тем же способом, что и при обучении (tf.image.resize).
    Если готовить фото по-другому, уверенность сети может немного "плыть".
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((256, 256))
    pixels = tf.image.resize(np.array(img), (224, 224)).numpy()  # 224 x 224 x 3, числа 0..255
    return np.expand_dims(pixels, axis=0)  # сеть ждёт пачку фото: 1 x 224 x 224 x 3


def predict(model, classes, path):
    """Вернуть (класс, уверенность от 0 до 1) для одного фото."""
    probabilities = model.predict(load_photo(path), verbose=0)[0]  # 4 вероятности
    best = int(np.argmax(probabilities))  # номер класса с наибольшей вероятностью
    return classes[best], float(probabilities[best])


def main():
    if len(sys.argv) != 2:
        print("Как запускать: python model/predict.py путь/к/фото.jpg")
        return

    model = keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    start = time.time()
    label, confidence = predict(model, classes, sys.argv[1])
    seconds = time.time() - start

    if confidence < THRESHOLD:
        # Сеть сомневается: честно говорим "не уверен", а не гадаем
        print(f"Не уверен (лучший вариант: {NAMES_RU[label]}, {confidence:.0%}).")
        print("Сфотографируйте лист ближе, при дневном свете, на однотонном фоне.")
    else:
        print(f"Диагноз: {NAMES_RU[label]} ({label})")
        print(f"Уверенность: {confidence:.0%}")
    print(f"Время ответа: {seconds:.2f} с")


if __name__ == "__main__":
    main()
