"""
Обучение нейросети RiceDoctor.

Идея (transfer learning, "дообучение"):
- Берём готовую сеть MobileNetV2. Она уже обучена на миллионе фото (ImageNet)
  и умеет находить на картинках линии, пятна, текстуры.
- Шаг 1. "Замораживаем" её (эти знания не меняем), сверху добавляем свой
  маленький слой на 4 класса и обучаем только его.
- Шаг 2. Если точность на проверке ниже 80%, "размораживаем" верхние 30 слоёв
  сети и очень осторожно (маленькая скорость обучения) дообучаем их.
- Шаг 3. Если и после этого ниже 80%, размораживаем верхнюю половину сети.

Лучшая модель (с самой высокой точностью на проверке) сохраняется автоматически.
Всё обучение ограничено 20 минутами.

Перед запуском: python model/prepare_data.py
Запуск (из корня проекта):
    python model/train.py
"""

import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "rice_doctor.keras"

# Порядок классов важен: сеть выдаёт ответ номером (0, 1, 2, 3),
# а по этому списку мы понимаем, какой номер - какая болезнь.
CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight"]

IMAGE_SIZE = (224, 224)  # MobileNetV2 обучалась на картинках 224x224
BATCH_SIZE = 32          # сколько фото сеть смотрит за один шаг
TARGET_ACCURACY = 0.80   # требование конкурса: точность не ниже 80%
TIME_LIMIT_MIN = 20      # обучение не дольше 20 минут

# Шаг 1: только наш слой. Больше 5 эпох почти ничего не даёт (проверено).
EPOCHS_STEP1 = 5
# Шаг 2 и 3: дообучение. Эпох "с запасом" - остановимся раньше, если точность
# перестанет расти (EarlyStopping) или кончится время.
EPOCHS_STEP2 = 15
EPOCHS_STEP3 = 15
FINE_TUNE_LR = 0.00001   # маленькая скорость, чтобы не "сломать" готовые знания

# Фиксируем случайность, чтобы результат повторялся от запуска к запуску
keras.utils.set_random_seed(42)


def load_photos(folder, shuffle):
    """Загрузить фото из папки. Имя подпапки = класс фото."""
    data = keras.utils.image_dataset_from_directory(
        folder,
        class_names=CLASSES,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42,
    )
    # prefetch: пока сеть учится на одной пачке фото, следующая уже загружается
    return data.prefetch(tf.data.AUTOTUNE)


def count_photos():
    """Сколько обучающих фото в каждом классе."""
    return [len(list(Path(TRAIN_DIR, name).glob("*.jpg"))) for name in CLASSES]


def make_class_weights(counts):
    """
    Веса классов: редкий класс получает больший вес.
    Например, бактериального ожога в 3-4 раза меньше, чем здоровых листьев,
    поэтому ошибка на нём "штрафуется" в 3-4 раза сильнее.
    Формула: всего фото / (число классов * фото в этом классе).
    """
    total = sum(counts)
    return {i: total / (len(counts) * n) for i, n in enumerate(counts)}


def build_model():
    """Собрать сеть: аугментации -> MobileNetV2 -> наш слой на 4 класса."""
    # Аугментации: каждую эпоху фото слегка меняются (отражение, поворот,
    # приближение, яркость). Сеть видит "новые" фото и меньше зазубривает.
    # Работают только при обучении, при предсказании выключаются сами.
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomBrightness(0.2),
    ], name="augmentation")

    # Готовая сеть без последнего слоя (include_top=False):
    # последний слой у неё был на 1000 классов ImageNet, нам он не нужен.
    base = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # замораживаем

    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = augmentation(inputs)
    # MobileNetV2 ждёт пиксели от -1 до 1, а у нас от 0 до 255.
    # Это ровно то, что делает mobilenet_v2.preprocess_input: x / 127.5 - 1
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
    # training=False: внутренние настройки сети (BatchNorm) не трогаем,
    # даже когда позже разморозим слои
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)  # картинку признаков -> в список чисел
    x = keras.layers.Dropout(0.2)(x)  # случайно "выключаем" 20% чисел, против зазубривания
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax")(x)  # 4 вероятности

    return keras.Model(inputs, outputs), base


class TimeLimit(keras.callbacks.Callback):
    """Останавливает обучение, если с начала прошло больше TIME_LIMIT_MIN минут."""

    def __init__(self, start_time):
        super().__init__()
        self.start_time = start_time

    def on_epoch_end(self, epoch, logs=None):
        if minutes_since(self.start_time) > TIME_LIMIT_MIN:
            print(f"\nПрошло больше {TIME_LIMIT_MIN} минут - останавливаем обучение")
            self.model.stop_training = True


def minutes_since(start_time):
    return (time.time() - start_time) / 60


def unfreeze_top_layers(model, base, n_layers):
    """Разморозить верхние n_layers слоёв MobileNetV2 и заново "собрать" модель."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False  # нижние слои оставляем замороженными
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def main():
    train_data = load_photos(TRAIN_DIR, shuffle=True)
    val_data = load_photos(VAL_DIR, shuffle=False)

    counts = count_photos()
    class_weight = make_class_weights(counts)
    print("Фото для обучения:", dict(zip(CLASSES, counts)))
    print("Веса классов:", {CLASSES[i]: round(w, 2) for i, w in class_weight.items()})

    model, base = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    MODELS_DIR.mkdir(exist_ok=True)
    start = time.time()

    # Помощники при обучении (callbacks):
    # 1) сохранять модель в файл каждый раз, когда точность на проверке стала лучшей
    save_best = keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
    # 2) остановиться, если 3 эпохи подряд точность не растёт,
    #    и вернуть веса из лучшей эпохи (для каждого шага создаём заново)
    def early_stop():
        return keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True)
    # 3) не учиться дольше TIME_LIMIT_MIN минут
    time_limit = TimeLimit(start)

    results = []  # (шаг, точность) для итоговой таблицы

    # --- Шаг 1: обучаем только наш верхний слой ---
    print(f"\nШаг 1: обучение верхнего слоя, {EPOCHS_STEP1} эпох")
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS_STEP1,
              class_weight=class_weight, callbacks=[save_best, time_limit])
    _, accuracy = model.evaluate(val_data, verbose=0)
    results.append(("1: только верхний слой", accuracy))
    print(f"Точность после шага 1: {accuracy:.1%}  ({minutes_since(start):.1f} мин)")

    # --- Шаг 2: дообучаем верхние 30 слоёв ---
    if accuracy < TARGET_ACCURACY and minutes_since(start) < TIME_LIMIT_MIN:
        print("\nШаг 2: размораживаем верхние 30 слоёв MobileNetV2")
        unfreeze_top_layers(model, base, 30)
        model.fit(train_data, validation_data=val_data, epochs=EPOCHS_STEP2,
                  class_weight=class_weight, callbacks=[save_best, early_stop(), time_limit])
        _, accuracy = model.evaluate(val_data, verbose=0)
        results.append(("2: + верхние 30 слоёв", accuracy))
        print(f"Точность после шага 2: {accuracy:.1%}  ({minutes_since(start):.1f} мин)")

    # --- Шаг 3: дообучаем верхнюю половину сети ---
    if accuracy < TARGET_ACCURACY and minutes_since(start) < TIME_LIMIT_MIN:
        half = len(base.layers) // 2
        print(f"\nШаг 3: размораживаем верхнюю половину сети ({half} слоёв)")
        unfreeze_top_layers(model, base, half)
        model.fit(train_data, validation_data=val_data, epochs=EPOCHS_STEP3,
                  class_weight=class_weight, callbacks=[save_best, early_stop(), time_limit])
        _, accuracy = model.evaluate(val_data, verbose=0)
        results.append((f"3: + верхние {half} слоёв", accuracy))
        print(f"Точность после шага 3: {accuracy:.1%}  ({minutes_since(start):.1f} мин)")

    print(f"\nВремя обучения: {minutes_since(start):.1f} мин")
    print("\n| Шаг | Точность на проверке |")
    print("|---|---|")
    for step, acc in results:
        print(f"| {step} | {acc:.1%} |")
    print(f"\nЛучшая точность: {save_best.best:.1%} - эта модель сохранена в {MODEL_PATH}")

    # Список классов нужен для predict.py и evaluate.py
    with open(MODELS_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, ensure_ascii=False, indent=2)
    print("Сохранено: models/rice_doctor.keras и models/classes.json")


if __name__ == "__main__":
    main()
