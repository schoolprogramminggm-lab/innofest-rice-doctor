"""
Диагноз по фото листа.

Запуск из консоли (из корня проекта):
    python model/predict.py путь/к/фото.jpg

Из кода (так делает бот):
    load_model()                      # один раз при старте
    label, confidence = predict(image_bytes)

Пример ответа:
    Диагноз: Пирикуляриоз (blast)
    Уверенность: 91%
"""

import io
import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
from PIL import Image

# Настройки берём из файла .env (если его нет - используем значения по умолчанию)
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/rice_doctor.keras")
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))  # 0.60 = 60%
CLASSES_PATH = "models/classes.json"

# Названия классов по-русски для красивого ответа в консоли
NAMES_RU = {
    "healthy": "Здоровый лист",
    "blast": "Пирикуляриоз",
    "brown_spot": "Бурая пятнистость",
    "bacterial_blight": "Бактериальный ожог",
}

# Модель и список классов грузим ОДИН раз и держим здесь.
# Бот вызывает load_model() при старте, а predict() только пользуется ими.
_model = None
_classes = None


def load_model():
    """Загрузить модель и список классов (один раз). Повторный вызов ничего не делает."""
    global _model, _classes
    if _model is None:
        # tensorflow тяжёлый, поэтому импортируем его только когда он правда нужен
        from tensorflow import keras

        _model = keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, encoding="utf-8") as f:
            _classes = json.load(f)
    return _model, _classes


def load_photo(image_bytes):
    """
    Подготовить фото (байты, как их скачал бот) для сети ТОЧНО ТАК ЖЕ, как при обучении:
    1) уменьшить до 256 по длинной стороне (как в prepare_data.py);
    2) сжать до 224x224 тем же способом, что и при обучении (tf.image.resize).
    Числа остаются 0..255: нормализация (x / 127.5 - 1) уже встроена в саму модель.
    Фото читается прямо из памяти, на диск ничего не пишется.
    """
    import tensorflow as tf

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((256, 256))
    pixels = tf.image.resize(np.array(img), (224, 224)).numpy()  # 224 x 224 x 3
    return np.expand_dims(pixels, axis=0)  # сеть ждёт пачку фото: 1 x 224 x 224 x 3


def predict(image_bytes):
    """Вернуть (класс, уверенность от 0 до 1) для одного фото (байты)."""
    model, classes = load_model()
    probabilities = model.predict(load_photo(image_bytes), verbose=0)[0]  # 4 вероятности
    best = int(np.argmax(probabilities))  # номер класса с наибольшей вероятностью
    return classes[best], float(probabilities[best])


def main():
    if len(sys.argv) != 2:
        print("Как запускать: python model/predict.py путь/к/фото.jpg")
        return

    load_model()  # грузим заранее, чтобы время ответа не включало загрузку модели
    with open(sys.argv[1], "rb") as f:
        image_bytes = f.read()

    start = time.time()
    label, confidence = predict(image_bytes)
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
