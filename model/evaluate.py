"""
Проверка точности модели на проверочных фото (data/val).
Эти фото сеть при обучении не видела, поэтому проверка честная.

Печатает:
- общую точность (по всем классам, включая not_rice);
- точность только по 4 болезням (фото листьев; ответ not_rice считается ошибкой);
- матрицу ошибок (что сеть путает с чем);
- precision / recall / F1 по каждому классу;
- как работает порог NOT_RICE_THRESHOLD из .env (сколько посторонних фото найдено,
  сколько листьев ошибочно названо «не лист»).

Запуск (из корня проекта):
    python model/evaluate.py
    python model/evaluate.py models/rice_doctor_v2.keras   # проверить другую модель
"""

import json
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras

from predict import NOT_RICE_THRESHOLD, choose_answer  # порог читается из .env в одном месте: predict.py

VAL_DIR = "data/val"
MODEL_PATH = "models/rice_doctor.keras"
CLASSES_PATH = "models/classes.json"


def main():
    # Путь к модели можно передать при запуске, иначе берём основную
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    print("Модель:", model_path)
    model = keras.models.load_model(model_path)
    with open(CLASSES_PATH, encoding="utf-8") as f:
        classes = json.load(f)

    # shuffle=False: порядок фото не перемешиваем, чтобы ответы совпали с правильными метками
    val_data = keras.utils.image_dataset_from_directory(
        VAL_DIR, class_names=classes, image_size=(224, 224),
        batch_size=32, shuffle=False,
    )

    # Правильные ответы (номера классов)
    y_true = np.concatenate([labels.numpy() for _, labels in val_data])
    # Ответы сети: для каждого фото вероятности всех классов
    probabilities = model.predict(val_data)
    # Ответ выбираем тем же правилом, что и бот (predict.choose_answer):
    # учитывается порог NOT_RICE_THRESHOLD, поэтому цифры ниже - это то, что увидит пользователь
    y_pred = np.array([classes.index(choose_answer(p, classes)[0]) for p in probabilities])

    # 1. Общая точность: доля фото, где сеть ответила правильно
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nТочность на проверочной части: {accuracy:.1%} ({len(y_true)} фото)\n")

    # 1б. Точность только по листьям с болезнями и здоровым (классы кроме not_rice)
    if "not_rice" in classes:
        leaf = y_true != classes.index("not_rice")
        leaf_accuracy = accuracy_score(y_true[leaf], y_pred[leaf])
        print(f"Точность по 4 болезням (без not_rice): {leaf_accuracy:.1%} ({leaf.sum()} фото)\n")

    # 2. Матрица ошибок: строка - правильный класс, столбец - ответ сети.
    #    Числа на диагонали - верные ответы, всё остальное - ошибки.
    matrix = confusion_matrix(y_true, y_pred)
    print("Матрица ошибок (строка - на самом деле, столбец - ответ сети):\n")
    print("| На самом деле \\ Ответ | " + " | ".join(classes) + " |")
    print("|---" * (len(classes) + 1) + "|")
    for name, row in zip(classes, matrix):
        print(f"| {name} | " + " | ".join(str(n) for n in row) + " |")

    # 3. По классам:
    #    precision - когда сеть сказала "blast", как часто это правда blast;
    #    recall    - из всех настоящих blast какую долю сеть нашла;
    #    F1        - среднее из этих двух.
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
    print("\n| Класс | Precision | Recall | F1 | Фото |")
    print("|---|---|---|---|---|")
    for i, name in enumerate(classes):
        print(f"| {name} | {precision[i]:.2f} | {recall[i]:.2f} | {f1[i]:.2f} | {support[i]} |")

    # 4. Порог "не лист риса": как в боте, смотрим на вероятность not_rice
    if "not_rice" in classes:
        p_not_rice = probabilities[:, classes.index("not_rice")]
        flagged = p_not_rice >= NOT_RICE_THRESHOLD  # эти фото бот назвал бы "не лист риса"
        n_other = (~leaf).sum()
        print(f"\nПорог NOT_RICE_THRESHOLD = {NOT_RICE_THRESHOLD}:")
        print(f"- посторонних фото распознано как «не лист»: {flagged[~leaf].sum()} из {n_other}")
        print(f"- листьев ошибочно названо «не лист»: {flagged[leaf].sum()} из {leaf.sum()}")


if __name__ == "__main__":
    main()
