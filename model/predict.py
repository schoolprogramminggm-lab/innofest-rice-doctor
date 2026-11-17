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

Если на фото не лист риса, predict() вернёт класс "not_rice" (диагноза нет).
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
# Порог для "не лист риса": если вероятность not_rice не ниже него, диагноз не ставим.
# Это ЕДИНСТВЕННОЕ место, где он используется: боту и AgroQyzyl достаточно увидеть label == "not_rice".
NOT_RICE_THRESHOLD = float(os.getenv("NOT_RICE_THRESHOLD", "0.50"))
NOT_RICE = "not_rice"
MAX_PERCENT = 99  # больше 99% не показываем: нейросеть не может быть уверена на 100%
CLASSES_PATH = "models/classes.json"

# Названия классов по-русски для красивого ответа в консоли
NAMES_RU = {
    "healthy": "Здоровый лист",
    "blast": "Пирикуляриоз",
    "brown_spot": "Бурая пятнистость",
    "bacterial_blight": "Бактериальный ожог",
    "not_rice": "Не лист риса",
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
        # Модель и classes.json должны быть от одного обучения (одинаковое число классов)
        if _model.output_shape[-1] != len(_classes):
            raise ValueError(
                f"В модели {_model.output_shape[-1]} классов, а в {CLASSES_PATH} - {len(_classes)}. "
                "Скопируйте нужную модель в models/rice_doctor.keras (см. README).")
    return _model, _classes


def percent(confidence):
    """
    Уверенность 0..1 -> текст "91%". Не больше 99%: 0.9999 показываем как "99%", а не "100%".
    Везде, где показываем проценты, используем эту функцию.
    """
    return f"{min(round(confidence * 100), MAX_PERCENT)}%"


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


def choose_answer(probabilities, classes):
    """
    Из вероятностей сети выбрать ответ (класс, уверенность).
    1) Если вероятность "не лист риса" не ниже NOT_RICE_THRESHOLD -> ("not_rice", вероятность).
    2) Иначе берём лучшую из болезней (not_rice среди них не считаем).
    Отдельная функция, чтобы её можно было проверить без настоящей модели.
    """
    if NOT_RICE in classes:
        not_rice_index = classes.index(NOT_RICE)
        if probabilities[not_rice_index] >= NOT_RICE_THRESHOLD:
            return NOT_RICE, float(probabilities[not_rice_index])
        # вероятность not_rice обнуляем, чтобы argmax выбирал только среди болезней
        probabilities = np.array(probabilities, dtype=float)
        probabilities[not_rice_index] = 0
    best = int(np.argmax(probabilities))  # номер класса с наибольшей вероятностью
    return classes[best], float(probabilities[best])


def predict(image_bytes):
    """Вернуть (класс, уверенность от 0 до 1) для одного фото (байты)."""
    model, classes = load_model()
    probabilities = model.predict(load_photo(image_bytes), verbose=0)[0]  # 5 вероятностей
    return choose_answer(probabilities, classes)


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

    if label == NOT_RICE:
        print("Это не похоже на лист риса. Сфотографируйте лист ближе.")
    elif confidence < THRESHOLD:
        # Сеть сомневается: честно говорим "не уверен", а не гадаем
        print(f"Не уверен (лучший вариант: {NAMES_RU[label]}, {percent(confidence)}).")
        print("Сфотографируйте лист ближе, при дневном свете, на однотонном фоне.")
    else:
        print(f"Диагноз: {NAMES_RU[label]} ({label})")
        print(f"Уверенность: {percent(confidence)}")
    print(f"Время ответа: {seconds:.2f} с")


if __name__ == "__main__":
    main()
