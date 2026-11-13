"""
Улучшение готовой модели (версия 2).

Берём уже обученную модель models/rice_doctor.keras и ещё раз дообучаем
верхние 30 слоёв MobileNetV2, но по-другому:
- скорость обучения сначала 1e-4 (учимся быстрее), а когда точность перестаёт
  расти - автоматически уменьшается до 1e-5 (ReduceLROnPlateau);
- веса классов мягче: корень из обычного веса. Так сеть меньше "перестраховывается"
  и реже называет бактериальный ожог там, где его нет;
- до 25 эпох, остановка, если 4 эпохи подряд нет улучшения, лимит 45 минут.

Результат сохраняется в models/rice_doctor_v2.keras (основная модель не трогается).
Сравнить: python model/evaluate.py models/rice_doctor_v2.keras

Запуск (из корня проекта):
    python model/finetune.py
"""

import math
import time

from tensorflow import keras

from train import (CLASSES, TRAIN_DIR, VAL_DIR, TimeLimit, count_photos,
                   load_photos, minutes_since)
import train

START_MODEL = "models/rice_doctor.keras"
NEW_MODEL = "models/rice_doctor_v2.keras"

EPOCHS = 25
FINE_TUNE_LAYERS = 30
LR_START = 0.0001   # 1e-4
LR_MIN = 0.00001    # 1e-5
train.TIME_LIMIT_MIN = 45  # для этого прогона лимит 45 минут

keras.utils.set_random_seed(42)


def make_soft_class_weights(counts):
    """
    Мягкие веса: корень из обычного веса (всего / (классов * фото в классе)).
    Пример: обычный вес 2.6 -> мягкий 1.6.
    """
    total = sum(counts)
    return {i: math.sqrt(total / (len(counts) * n)) for i, n in enumerate(counts)}


def main():
    train_data = load_photos(TRAIN_DIR, shuffle=True)
    val_data = load_photos(VAL_DIR, shuffle=False)

    counts = count_photos()
    class_weight = make_soft_class_weights(counts)
    print("Мягкие веса классов:", {CLASSES[i]: round(w, 2) for i, w in class_weight.items()})

    # Загружаем лучшую модель и находим внутри неё MobileNetV2
    model = keras.models.load_model(START_MODEL)
    base = next(layer for layer in model.layers if layer.name.startswith("mobilenetv2"))

    _, accuracy = model.evaluate(val_data, verbose=0)
    print(f"Точность исходной модели: {accuracy:.1%}")

    # Размораживаем верхние 30 слоёв, остальные замораживаем
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR_START),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    start = time.time()
    callbacks = [
        # сохранять лучшую по точности на проверке
        keras.callbacks.ModelCheckpoint(
            NEW_MODEL, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1),
        # если 2 эпохи нет улучшения - скорость обучения в 10 раз меньше (1e-4 -> 1e-5)
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", mode="max", factor=0.1, patience=2,
            min_lr=LR_MIN, verbose=1),
        # если 4 эпохи нет улучшения - стоп, вернуть лучшие веса
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True),
        TimeLimit(start),
    ]
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS,
              class_weight=class_weight, callbacks=callbacks)

    _, accuracy = model.evaluate(val_data, verbose=0)
    print(f"\nТочность лучшей эпохи: {accuracy:.1%}")
    print(f"Время: {minutes_since(start):.1f} мин")
    print(f"Сохранено: {NEW_MODEL}")


if __name__ == "__main__":
    main()
