"""
Проверка модели на «чужих» фото: их не было в Paddy Doctor, на котором сеть училась.

Что делает скрипт:
1. Читает eval/labels.csv (file, expected, source, license): какое фото, что на нём на самом деле.
2. Прогоняет каждое фото через НАСТОЯЩУЮ модель (models/rice_doctor.keras) так же, как это делает бот
   (model.predict.predict(): тот же размер, тот же порог not_rice).
3. Печатает: accuracy, precision / recall / F1 по классам, матрицу ошибок, время на фото
   и список ВСЕХ ошибок (файл, ожидалось, ответ модели, уверенность).
4. Сохраняет ответ по каждому фото в eval/results_<имя модели>.csv, например eval/results_rice_doctor_v4.csv
   (по нему пишется docs/accuracy-external.md; разные модели не затирают друг друга).

Модель здесь НЕ обучается и не меняется, только проверяется.

Запуск (из корня проекта, нужны models/rice_doctor.keras и models/classes.json):
    python eval/run_eval.py
    python eval/run_eval.py models/rice_doctor_v2.keras   # проверить другую модель (нужен свой classes.json)
"""

import csv
import sys
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # чтобы видеть папку model/

from model import predict  # noqa: E402  (импорт после sys.path нужен специально)

IMAGES = ROOT / "eval" / "images"
LABELS = ROOT / "eval" / "labels.csv"

# Порядок классов в таблицах: как в README
CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight", "not_rice"]


def main():
    # Пути в predict.py относительные, поэтому задаём их полностью (запускать можно из любой папки)
    predict.MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "rice_doctor.keras")
    predict.CLASSES_PATH = str(ROOT / "models" / "classes.json")
    predict.load_model()  # грузим заранее, чтобы время на фото не включало загрузку модели
    print("Модель:", predict.MODEL_PATH)
    results_file = ROOT / "eval" / f"results_{Path(predict.MODEL_PATH).stem}.csv"

    with open(LABELS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Фото в наборе: {len(rows)}\n")

    # Первый вызов сети всегда медленнее (TensorFlow «разогревается»), поэтому делаем один пробный,
    # а в время на фото его не включаем.
    predict.predict((IMAGES / rows[0]["file"]).read_bytes())

    results = []  # по одной записи на фото
    for row in rows:
        image_bytes = (IMAGES / row["file"]).read_bytes()
        start = time.time()
        label, confidence = predict.predict(image_bytes)
        seconds = time.time() - start
        results.append({
            "file": row["file"], "expected": row["expected"], "got": label,
            "confidence": round(confidence, 4), "seconds": round(seconds, 3),
        })

    expected = [r["expected"] for r in results]
    got = [r["got"] for r in results]

    # --- Точность ---
    right = sum(e == g for e, g in zip(expected, got))
    print(f"Accuracy: {accuracy_score(expected, got):.1%}  ({right} верно из {len(results)})")
    leaves = [r for r in results if r["expected"] != predict.NOT_RICE]
    leaves_right = sum(r["expected"] == r["got"] for r in leaves)
    print(f"Только листья риса (без not_rice): {leaves_right / len(leaves):.1%}  ({leaves_right} из {len(leaves)})\n")

    # --- Precision / recall / F1 по классам ---
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, got, labels=CLASSES, zero_division=0)
    print(f"{'класс':<18}{'precision':>10}{'recall':>8}{'F1':>7}{'фото':>6}")
    for i, name in enumerate(CLASSES):
        print(f"{name:<18}{precision[i]:>10.2f}{recall[i]:>8.2f}{f1[i]:>7.2f}{support[i]:>6}")
    print()

    # --- Матрица ошибок: строка = что на фото на самом деле, столбец = ответ модели ---
    matrix = confusion_matrix(expected, got, labels=CLASSES)
    print("Матрица ошибок (строка: на самом деле, столбец: ответ модели)")
    print(f"{'':<18}" + "".join(f"{name[:12]:>14}" for name in CLASSES))
    for i, name in enumerate(CLASSES):
        print(f"{name:<18}" + "".join(f"{n:>14}" for n in matrix[i]))
    print()

    # --- Время ---
    times = [r["seconds"] for r in results]
    print(f"Время на фото: в среднем {sum(times) / len(times):.2f} с, самое долгое {max(times):.2f} с\n")

    # --- Порог 60%: как увидит ответ фермер ---
    # Бот при уверенности ниже CONFIDENCE_THRESHOLD пишет «не уверен», а не диагноз.
    # Считаем отдельно, чтобы видеть, сколько ошибок порог прикрывает, а сколько верных ответов портит.
    # (not_rice порогом уверенности не проверяется: бот сразу отвечает «не лист риса»)
    diagnoses = [r for r in results if r["got"] != predict.NOT_RICE]
    unsure = [r for r in diagnoses if r["confidence"] < predict.THRESHOLD]
    print(f"Порог уверенности {predict.THRESHOLD:.0%}: из {len(diagnoses)} диагнозов бот ответил бы «не уверен» на {len(unsure)}:")
    print(f"  из них верных диагнозов {sum(r['expected'] == r['got'] for r in unsure)}, ошибочных {sum(r['expected'] != r['got'] for r in unsure)}\n")

    # --- ВСЕ ошибки ---
    errors = [r for r in results if r["expected"] != r["got"]]
    print(f"Ошибок: {len(errors)}")
    for r in errors:
        print(f"  {r['file']:<34} ожидалось {r['expected']:<17} ответ {r['got']:<17} уверенность {predict.percent(r['confidence'])}")

    # --- Сохраняем результаты по каждому фото ---
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "expected", "got", "confidence", "seconds"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nОтветы по каждому фото: {results_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
