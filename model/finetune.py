"""
Улучшение готовой модели: версия 3 = модель v2 + пятый класс not_rice («не лист риса»).

Берём обученную модель v2 (4 болезни, models/rice_doctor_v2.keras) и добавляем ей
пятый выход not_rice. Выходы для первых четырёх классов копируем из v2 (сеть уже умеет
отличать болезни), выход для not_rice начинается «с нуля». Фото для not_rice:
python model/download_not_rice.py (400 для обучения, 100 для проверки).

Обучение в два шага. Главное правило: точность по болезням не должна стать ниже,
чем была (у v2 96,6%).
- Шаг 1: вся сеть заморожена, обучается ТОЛЬКО выход not_rice. Четыре выхода болезней
  после каждого шага возвращаются к значениям v2 (см. KeepOldOutputs), поэтому диагнозы
  остаются такими же, как у v2. Результат шага 1 сразу сохраняется как основа.
- Шаг 2: дообучаем верхние 30 слоёв MobileNetV2 маленькой скоростью (1e-5). Модель
  перезаписывается только если точность по болезням стала ВЫШЕ, чем после шага 1.
Веса классов мягкие (корень из обычного веса), как в v2, но у not_rice вес 1.0: иначе сеть
слишком часто называет настоящие листья «не листом». До 25 эпох, остановка, если 6 эпох
подряд нет улучшения, лимит 45 минут.

Лучшая эпоха выбирается по точности ТОЛЬКО на 4 болезнях (val_disease_accuracy). Как и у v2,
выбор идёт по проверочным фото, поэтому оценка немного оптимистична.

Результат сохраняется в models/rice_doctor_v3.keras (v2 и основная модель не трогаются)
и models/classes.json (теперь 5 классов).
Проверить: python model/evaluate.py models/rice_doctor_v3.keras
Сделать основной: cp models/rice_doctor_v3.keras models/rice_doctor.keras
(в Windows: copy models\\rice_doctor_v3.keras models\\rice_doctor.keras)

Запуск (из корня проекта):
    python model/finetune.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from tensorflow import keras

from train import (CLASSES, TRAIN_DIR, VAL_DIR, TimeLimit, count_photos,
                   load_photos, minutes_since)
import train

START_MODEL = "models/rice_doctor_v2.keras"
NEW_MODEL = "models/rice_doctor_v3.keras"

HEAD_EPOCHS = 5        # шаг 1: сколько эпох учим только выход not_rice
HEAD_LR = 0.003        # 3e-3: в этом слое всего 1281 число, скорость можно побольше
EPOCHS = 25            # шаг 2: максимум эпох
FINE_TUNE_LAYERS = 30
LR_START = 0.00001     # 1e-5: v2 уже хорошо обучена, трогаем её осторожно
LR_MIN = 0.000001      # 1e-6
NOT_RICE_WEIGHT = 1.0  # вес класса not_rice (мягче, чем по формуле: 1.48)
train.TIME_LIMIT_MIN = 45  # для этого прогона лимит 45 минут

keras.utils.set_random_seed(42)


def make_soft_class_weights(counts):
    """
    Мягкие веса: корень из обычного веса (всего / (классов * фото в классе)).
    Пример: обычный вес 2.6 -> мягкий 1.6.
    """
    total = sum(counts)
    return {i: math.sqrt(total / (len(counts) * n)) for i, n in enumerate(counts)}


class DiseaseAccuracy(keras.callbacks.Callback):
    """
    После каждой эпохи считает точность только по листьям (классы кроме not_rice)
    и кладёт её в logs["val_disease_accuracy"]. Следующие помощники (checkpoint,
    ReduceLROnPlateau, EarlyStopping) следят именно за этим числом.
    """

    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data
        # val_data не перемешивается (shuffle=False), поэтому порядок меток всегда один
        self.y_true = np.concatenate([labels.numpy() for _, labels in val_data])
        self.leaf = self.y_true != CLASSES.index("not_rice")

    def compute(self, model):
        """Доля верных ответов на фото листьев (ответ not_rice на листе считается ошибкой)."""
        y_pred = model.predict(self.val_data, verbose=0).argmax(axis=1)
        return float((y_pred[self.leaf] == self.y_true[self.leaf]).mean())

    def on_epoch_end(self, epoch, logs=None):
        logs["val_disease_accuracy"] = self.compute(self.model)
        print(f" - val_disease_accuracy: {logs['val_disease_accuracy']:.4f}")


class KeepOldOutputs(keras.callbacks.Callback):
    """
    Шаг 1: после каждого шага обучения возвращает выходы четырёх болезней к значениям
    из v2. Так обучается только выход not_rice, а диагнозы остаются как у v2.
    """

    def __init__(self, n_old):
        super().__init__()
        self.n_old = n_old

    def on_train_begin(self, logs=None):
        kernel, bias = self.model.layers[-1].get_weights()
        self.old_kernel = kernel[:, :self.n_old].copy()
        self.old_bias = bias[:self.n_old].copy()

    def on_train_batch_end(self, batch, logs=None):
        layer = self.model.layers[-1]
        kernel, bias = layer.get_weights()
        kernel[:, :self.n_old] = self.old_kernel
        bias[:self.n_old] = self.old_bias
        layer.set_weights([kernel, bias])


def add_not_rice_output(model):
    """
    Если у модели 4 выхода (болезни), делаем 5: добавляем выход для not_rice.
    Веса для первых четырёх выходов копируем из старого слоя, чтобы не потерять знания v2.
    Если выходов уже 5, возвращаем модель как есть.
    """
    n_old = model.output_shape[-1]
    if n_old == len(CLASSES):
        return model
    old_layer = model.layers[-1]                 # старый слой Dense на 4 класса
    new_layer = keras.layers.Dense(len(CLASSES), activation="softmax", name="classifier")
    outputs = new_layer(model.layers[-2].output)  # подключаем к тому же месту, что и старый
    new_model = keras.Model(model.input, outputs)

    kernel, bias = old_layer.get_weights()        # kernel: 1280 x 4, bias: 4
    new_kernel, new_bias = new_layer.get_weights()  # 1280 x 5, 5 (случайные начальные)
    new_kernel[:, :n_old] = kernel
    new_bias[:n_old] = bias
    new_layer.set_weights([new_kernel, new_bias])
    return new_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def main():
    train_data = load_photos(TRAIN_DIR, shuffle=True)
    val_data = load_photos(VAL_DIR, shuffle=False)

    counts = count_photos()
    class_weight = make_soft_class_weights(counts)
    class_weight[CLASSES.index("not_rice")] = NOT_RICE_WEIGHT
    print("Веса классов:", {CLASSES[i]: round(w, 2) for i, w in class_weight.items()})

    # Загружаем v2, добавляем пятый выход и находим внутри MobileNetV2
    old_model = keras.models.load_model(START_MODEL)
    n_old = old_model.output_shape[-1]
    model = add_not_rice_output(old_model)
    base = next(layer for layer in model.layers if layer.name.startswith("mobilenetv2"))
    disease_accuracy = DiseaseAccuracy(val_data)

    start = time.time()

    # --- Шаг 1: учим только выход not_rice ---
    base.trainable = False
    compile_model(model, HEAD_LR)
    print(f"\nШаг 1: учим только выход not_rice, {HEAD_EPOCHS} эпох")
    model.fit(train_data, validation_data=val_data, epochs=HEAD_EPOCHS,
              class_weight=class_weight, callbacks=[KeepOldOutputs(n_old)])
    step1_accuracy = disease_accuracy.compute(model)
    model.save(NEW_MODEL)  # основа: дальше перезаписываем, только если станет лучше
    print(f"Точность по болезням после шага 1: {step1_accuracy:.1%} (сохранено в {NEW_MODEL})")

    # --- Шаг 2: размораживаем верхние 30 слоёв, остальные оставляем замороженными ---
    print(f"\nШаг 2: дообучаем верхние {FINE_TUNE_LAYERS} слоёв MobileNetV2")
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    compile_model(model, LR_START)

    callbacks = [
        disease_accuracy,  # считает val_disease_accuracy (должен стоять первым)
        # сохранять, только если точность по болезням выше, чем после шага 1 и всех прошлых эпох
        keras.callbacks.ModelCheckpoint(
            NEW_MODEL, monitor="val_disease_accuracy", mode="max", save_best_only=True,
            initial_value_threshold=step1_accuracy, verbose=1),
        # если 3 эпохи нет улучшения - скорость обучения в 10 раз меньше (1e-5 -> 1e-6)
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_disease_accuracy", mode="max", factor=0.1, patience=3,
            min_lr=LR_MIN, verbose=1),
        # если 6 эпох нет улучшения - стоп
        keras.callbacks.EarlyStopping(monitor="val_disease_accuracy", mode="max", patience=6),
        TimeLimit(start),
    ]
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS,
              class_weight=class_weight, callbacks=callbacks)

    # В файле лежит лучшая модель (не обязательно последняя эпоха): проверяем её
    best = keras.models.load_model(NEW_MODEL)
    _, accuracy = best.evaluate(val_data, verbose=0)
    print(f"\nТочность по болезням: {disease_accuracy.compute(best):.1%}")
    print(f"Точность по всем 5 классам: {accuracy:.1%}")
    print(f"Время: {minutes_since(start):.1f} мин")
    print(f"Сохранено: {NEW_MODEL}")

    # Список классов для predict.py и evaluate.py (теперь 5 классов)
    with open(Path("models") / "classes.json", "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, ensure_ascii=False, indent=2)
    print("Сохранено: models/classes.json")
    print("Сделать v3 основной моделью: скопировать models/rice_doctor_v3.keras в models/rice_doctor.keras")


if __name__ == "__main__":
    main()
