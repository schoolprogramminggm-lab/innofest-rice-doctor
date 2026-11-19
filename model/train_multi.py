"""
Обучение на НЕСКОЛЬКИХ источниках (issue #23) и честная проверка «по источнику».

Зачем. Модель v3 училась только на Paddy Doctor и на чужих фото давала 21%. Теперь учим на фото
из разных мест и камер (список и лицензии: data/sources.csv, его делает model/download_sources.py).

Два режима (оба из корня проекта):
    python model/train_multi.py loso     # проверка «по источнику» (leave-one-source-out)
    python model/train_multi.py final    # финальная модель на всех источниках -> models/rice_doctor_v4.keras
    python model/train_multi.py all      # сначала loso, потом final

1. loso. Для КАЖДОГО источника учим сеть БЕЗ него и проверяем на нём. Так ведёт себя настоящий «чужой»
   фермер: сеть не видела ни его камеру, ни его поле. Результат: eval/loso_results.csv.
2. final. Учим на всех источниках. Проверки: фото Paddy Doctor, которых сеть не видела (часть test),
   посторонние фото (Open Images, часть test) и 33 чужих фото (python eval/run_eval.py models/rice_doctor_v4.keras).

Как учим (как в v1-v3: MobileNetV2, сначала голова, потом верхние слои):
- Старт: веса модели v3 (она уже знает Paddy Doctor на 96,5%), то есть это именно ДОобучение. Первая попытка
  «с нуля от ImageNet» за 5-9 эпох получилась недообученной (на Paddy Doctor всего 66%), см. docs/accuracy-external.md.
  Единственное исключение: когда ОТЛОЖЕН сам Paddy Doctor, стартуем от ImageNet (v3 его уже видела, это была бы утечка).
- Шаг 1: MobileNetV2 заморожена, учится только наш верхний слой.
- Шаг 2: размораживаем верхние 30 слоёв и учим их маленькой скоростью.
- Лучшая эпоха выбирается по ПРОВЕРОЧНОЙ части обучающих источников (часть val в sources.csv),
  а не по тесту: отложенный источник и 33 чужих фото при выборе не используются.
- Чтобы большой источник (Paddy Doctor) не заглушил остальные, берём не больше CAP_TRAIN фото
  из каждой пары (источник, класс).

Аугментации (каждую эпоху фото чуть меняются):
- поворот (в том числе на боку: EXIF при чтении фото тоже учитывается, см. download_sources.py и predict.py),
  отражение, масштаб, яркость, контраст;
- «кадр с однотонным фоном»: фото уменьшается и кладётся на белый / бумажный / серый / «ладонный»
  фон. Так сеть учится не зависеть от фона: лист на белом фоне - это тоже рис.
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # чтобы видеть папку model/ (choose_answer из predict.py)

from model.predict import choose_answer  # noqa: E402  (импорт после sys.path нужен специально)

DATA = ROOT / "data" / "multi"
SOURCES_CSV = ROOT / "data" / "sources.csv"
LOSO_CSV = ROOT / "eval" / "loso_results.csv"
FINAL_CSV = ROOT / "eval" / "v4_results.csv"
MODEL_V3 = ROOT / "models" / "rice_doctor_v3.keras"  # с неё начинаем дообучение
MODEL_V4 = ROOT / "models" / "rice_doctor_v4.keras"

# Порядок классов тот же, что у v1-v3 (и в models/classes.json)
CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight", "not_rice"]
NOT_RICE = "not_rice"
LEAF_SOURCES = ["paddy_doctor", "mendeley_bd", "riceleafbd", "uci", "dhan_shomadhan"]  # источники листьев

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CAP_TRAIN = 500       # не больше стольких обучающих фото из одной пары (источник, класс)
CAP_VAL = 80          # и стольких проверочных
FINE_TUNE_LAYERS = 30
HEAD_LR = 1e-3        # шаг 1: скорость для верхнего слоя, если стартуем от ImageNet
WARM_HEAD_LR = 3e-4   # шаг 1, если стартуем от v3: верхний слой уже обучен, трогаем осторожнее
TUNE_LR = 1e-4        # шаг 2: осторожнее, чтобы не «сломать» готовые знания
PLAIN_BACKGROUND_PART = 0.3  # у какой доли фото в обучении подменяем кадр на «однотонный фон»

# Сколько эпох: проверка «по источнику» короче (таких запусков 5), финальная модель дольше.
# Первое число - шаг 1 (голова), второе - шаг 2 (верхние слои).
LOSO_EPOCHS = (1, 5)
LOSO_PADDY_EPOCHS = (3, 5)  # Paddy Doctor отложен -> старт от ImageNet, голове нужно больше времени
FINAL_EPOCHS = (1, 8)

SEED = 42
keras.utils.set_random_seed(SEED)

# Цвета «однотонного фона»: белый, бумага, светло-серый, серый, три оттенка кожи (ладонь), тёмный стол
BACKGROUND_COLORS = [
    [255, 255, 255], [245, 243, 235], [215, 215, 215], [150, 150, 150],
    [241, 194, 160], [224, 172, 120], [198, 134, 90], [60, 50, 45],
]


# --- Данные --------------------------------------------------------------------------------
def read_rows():
    """Строки data/sources.csv (источник, класс, часть train / val / test)."""
    with open(SOURCES_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cap_per_group(rows, limit, rnd):
    """Оставить не больше limit фото из каждой пары (источник, класс), выбор случайный, но всегда одинаковый."""
    groups = {}
    for row in rows:
        groups.setdefault((row["source"], row["class"]), []).append(row)
    chosen = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["file"])
        rnd.shuffle(group)
        chosen.extend(group[:limit])
    return chosen


def to_arrays(rows):
    """Список строк -> (пути к файлам, номера классов)."""
    files = [str(DATA / r["file"]) for r in rows]
    labels = [CLASSES.index(r["class"]) for r in rows]
    return files, labels


def load_image(path, label):
    """Прочитать фото (оно уже уменьшено до 256) и сжать до 224x224 так же, как это делает бот (predict.py)."""
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)  # числа 0..255, нормализация внутри модели
    return image, label


def color_jitter(image, label):
    """Цвет: насыщенность и оттенок слегка меняются (у разных камер и погоды лист выглядит по-разному)."""
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.04)
    return tf.clip_by_value(image, 0.0, 1.0) * 255.0, label


def plain_background(image, label):
    """
    «Кадр с однотонным фоном»: с вероятностью PLAIN_BACKGROUND_PART фото уменьшается (45-90% кадра)
    и кладётся в случайное место на однотонный фон (белый, бумага, серый, ладонь, стол).
    Это относится ко ВСЕМ классам, включая not_rice, чтобы однотонный фон сам по себе ничего не значил.
    """
    if tf.random.uniform([]) > PLAIN_BACKGROUND_PART:
        return image, label
    size = tf.cast(tf.random.uniform([], 0.45, 0.9) * IMAGE_SIZE[0], tf.int32)
    small = tf.image.resize(image, (size, size))
    color = tf.gather(tf.constant(BACKGROUND_COLORS, tf.float32),
                      tf.random.uniform([], 0, len(BACKGROUND_COLORS), tf.int32))
    top = tf.random.uniform([], 0, IMAGE_SIZE[0] - size + 1, tf.int32)
    left = tf.random.uniform([], 0, IMAGE_SIZE[1] - size + 1, tf.int32)
    # (фото - цвет), дополненное нулями до 224x224, плюс цвет: снаружи получается ровно цвет фона
    padded = tf.pad(small - color, [[top, IMAGE_SIZE[0] - size - top],
                                    [left, IMAGE_SIZE[1] - size - left], [0, 0]])
    return padded + color, label


def make_dataset(files, labels, training):
    """Набор для сети: (фото, класс), пачками по BATCH_SIZE. При обучении фото перемешиваются и меняются."""
    data = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        data = data.shuffle(len(files), seed=SEED)
    data = data.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        data = data.map(color_jitter, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.map(plain_background, num_parallel_calls=tf.data.AUTOTUNE)
    return data.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --- Модель --------------------------------------------------------------------------------
def build_model():
    """Аугментации -> MobileNetV2 -> наш слой на 5 классов (как в train.py, аугментации богаче)."""
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.25, fill_mode="reflect"),  # до ±90°: фото могло лечь на бок
        keras.layers.RandomZoom((-0.3, 0.2)),                    # приблизить или отдалить лист
        keras.layers.RandomBrightness(0.3),                      # свет: день, тень, пасмурно
        keras.layers.RandomContrast(0.2),
    ], name="augmentation")  # работают только при обучении, при ответе выключаются сами

    base = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = augmentation(inputs)
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  # 0..255 -> -1..1, как ждёт MobileNetV2
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inputs, outputs), base


class KeepBest(keras.callbacks.Callback):
    """
    Запоминает веса лучшей эпохи по ПРОВЕРОЧНОЙ точности (val_accuracy) и в конце возвращает их.
    Один и тот же объект работает в обоих шагах (голова и верхние слои): лучшей считается лучшая эпоха из всех.
    """

    def __init__(self):
        super().__init__()
        self.best_accuracy = -1.0
        self.best_weights = None
        self.best_epoch = ""

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] > self.best_accuracy:
            self.best_accuracy = logs["val_accuracy"]
            self.best_weights = self.model.get_weights()
            self.best_epoch = f"эпоха {epoch + 1} этого шага"

    def restore(self):
        self.model.set_weights(self.best_weights)


def copy_weights_from_v3(model, base):
    """Перенести в новую модель веса MobileNetV2 и верхнего слоя из v3 (аугментации в v3 свои, веса у них нет)."""
    old = keras.models.load_model(MODEL_V3)
    old_base = next(layer for layer in old.layers if layer.name.startswith("mobilenetv2"))
    base.set_weights(old_base.get_weights())
    model.layers[-1].set_weights(old.layers[-1].get_weights())  # последний слой: Dense на 5 классов


def train_model(train_rows, val_rows, epochs, from_v3=True):
    """
    Два шага обучения (epochs = (шаг 1, шаг 2)). from_v3=True: старт с весов v3, иначе от ImageNet.
    Возвращает модель с весами лучшей эпохи (по проверочной выборке).
    """
    head_epochs, tune_epochs = epochs
    rnd = random.Random(SEED)
    train_rows = cap_per_group(train_rows, CAP_TRAIN, rnd)
    val_rows = cap_per_group(val_rows, CAP_VAL, rnd)
    train_files, train_labels = to_arrays(train_rows)
    val_files, val_labels = to_arrays(val_rows)
    train_data = make_dataset(train_files, train_labels, training=True)
    val_data = make_dataset(val_files, val_labels, training=False)

    # Мягкие веса классов: корень из обычного веса (редкий класс важнее, но без перегиба)
    counts = np.bincount(train_labels, minlength=len(CLASSES))
    class_weight = {i: float(np.sqrt(len(train_labels) / (len(CLASSES) * max(n, 1))))
                    for i, n in enumerate(counts)}
    print(f"Обучение: {len(train_files)} фото, проверка: {len(val_files)} фото")
    print("Фото по классам:", dict(zip(CLASSES, counts.tolist())))

    model, base = build_model()
    if from_v3:
        copy_weights_from_v3(model, base)
    keep_best = KeepBest()

    def compile_model(learning_rate):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Шаг 1: только верхний слой
    print(f"\nШаг 1: учим только верхний слой, {head_epochs} эп.")
    compile_model(HEAD_LR)
    model.fit(train_data, validation_data=val_data, epochs=head_epochs,
              class_weight=class_weight, callbacks=[keep_best], verbose=2)

    # Шаг 2: верхние 30 слоёв MobileNetV2, маленькая скорость; если проверка не растёт - скорость ещё меньше
    print(f"\nШаг 2: дообучаем верхние {FINE_TUNE_LAYERS} слоёв, {tune_epochs} эп.")
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    compile_model(TUNE_LR)
    model.fit(train_data, validation_data=val_data, epochs=tune_epochs,
              class_weight=class_weight, verbose=2,
              callbacks=[keep_best, keras.callbacks.ReduceLROnPlateau(
                  monitor="val_accuracy", factor=0.3, patience=1, verbose=1)])

    keep_best.restore()
    print(f"Лучшая эпоха: {keep_best.best_epoch}, проверочная точность {keep_best.best_accuracy:.1%}")
    return model


# --- Проверка ------------------------------------------------------------------------------
def answers(model, rows):
    """Ответы модели на фото (так, как их увидит пользователь: с порогом not_rice). -> список (ожидалось, ответ)."""
    files, labels = to_arrays(rows)
    probabilities = model.predict(make_dataset(files, labels, training=False), verbose=0)
    return [(CLASSES[labels[i]], choose_answer(probabilities[i], CLASSES)[0]) for i in range(len(rows))]


def accuracy_by_class(pairs):
    """Пары (ожидалось, ответ) -> {класс: (верно, всего)}."""
    result = {}
    for expected, got in pairs:
        right, total = result.get(expected, (0, 0))
        result[expected] = (right + (expected == got), total + 1)
    return result


def wrong_answers(pairs):
    """Пары (ожидалось, ответ) -> {класс: {неверный ответ: сколько раз}} (чтобы видеть, во что превращаются ошибки)."""
    result = {}
    for expected, got in pairs:
        if expected != got:
            counts = result.setdefault(expected, {})
            counts[got] = counts.get(got, 0) + 1
    return result


def run_loso(rows):
    """Для каждого источника: учим без него, проверяем на нём. Результат - eval/loso_results.csv."""
    leaf_classes = [c for c in CLASSES if c != NOT_RICE]
    results = []  # строки для CSV
    for held_out in LEAF_SOURCES:
        print(f"\n{'=' * 70}\nОтложенный источник: {held_out}\n{'=' * 70}")
        start = time.time()
        rest = [r for r in rows if r["source"] != held_out]
        paddy_out = held_out == "paddy_doctor"  # v3 училась на Paddy: тогда стартуем от ImageNet
        model = train_model([r for r in rest if r["part"] == "train"],
                            [r for r in rest if r["part"] == "val"],
                            LOSO_PADDY_EPOCHS if paddy_out else LOSO_EPOCHS, from_v3=not paddy_out)
        test_rows = [r for r in rows if r["source"] == held_out and r["class"] in leaf_classes]
        by_class = accuracy_by_class(answers(model, test_rows))
        right = sum(r for r, _ in by_class.values())
        total = sum(t for _, t in by_class.values())
        print(f"\n{held_out}: {right} из {total} = {right / total:.1%}  ({(time.time() - start) / 60:.1f} мин)")
        for cls, (r, t) in by_class.items():
            print(f"   {cls:<18}{r:>5} из {t:<5} {r / t:.1%}   ошибки -> {errors.get(cls, {})}")
            results.append({"held_out_source": held_out, "class": cls, "photos": t, "correct": r})
        results.append({"held_out_source": held_out, "class": "ALL", "photos": total, "correct": right})
        keras.backend.clear_session()  # освободить память перед следующим запуском

    with open(LOSO_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_source", "class", "photos", "correct"])
        writer.writeheader()
        writer.writerows(results)
    per_source = [r["correct"] / r["photos"] for r in results if r["class"] == "ALL"]
    print(f"\nСреднее по отложенным источникам: {np.mean(per_source):.1%}  (файл {LOSO_CSV})")
    no_paddy = [r["correct"] / r["photos"] for r in results
                if r["class"] == "ALL" and r["held_out_source"] != "paddy_doctor"]
    print(f"Среднее без Paddy Doctor: {np.mean(no_paddy):.1%}")


def run_final(rows):
    """Финальная модель на всех источниках: models/rice_doctor_v4.keras."""
    start = time.time()
    model = train_model([r for r in rows if r["part"] == "train"],
                        [r for r in rows if r["part"] == "val"], FINAL_EPOCHS)
    MODEL_V4.parent.mkdir(exist_ok=True)
    model.save(MODEL_V4)
    print(f"Сохранено: {MODEL_V4}  ({(time.time() - start) / 60:.1f} мин)")

    # Проверка на фото, которых сеть не видела: бывшая проверочная часть Paddy Doctor и Open Images
    results = []
    for source in ("paddy_doctor", "openimages"):
        test_rows = [r for r in rows if r["source"] == source and r["part"] == "test"]
        pairs = answers(model, test_rows)
        by_class = accuracy_by_class(pairs)
        errors = wrong_answers(pairs)
        right = sum(r for r, _ in by_class.values())
        total = sum(t for _, t in by_class.values())
        print(f"{source}: {right} из {total} = {right / total:.1%}")
        for cls, (r, t) in by_class.items():
            print(f"   {cls:<18}{r:>5} из {t:<5} {r / t:.1%}   ошибки -> {errors.get(cls, {})}")
            results.append({"test_set": source, "class": cls, "photos": t, "correct": r})
        results.append({"test_set": source, "class": "ALL", "photos": total, "correct": right})
    with open(FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test_set", "class", "photos", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print("Дальше: python eval/run_eval.py models/rice_doctor_v4.keras  (33 чужих фото)")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("loso", "final", "all"):
        print("Как запускать: python model/train_multi.py loso | final | all")
        return
    rows = read_rows()
    print(f"Фото в data/sources.csv: {len(rows)}")
    if mode in ("loso", "all"):
        run_loso(rows)
    if mode in ("final", "all"):
        run_final(rows)


if __name__ == "__main__":
    main()
