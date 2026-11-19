"""
Точность модели по источникам и классам (issue #23): колонка «было» в отчёте.

Для каждого источника листьев считает, сколько фото модель назвала верно (так, как ответил бы бот:
с порогом not_rice). Для Paddy Doctor берётся только часть test (990 фото, сеть их при обучении не видела);
для остальных источников - все фото из data/sources.csv.

ВАЖНО: честная колонка «было» - у моделей, которые эти источники НЕ видели (v3 училась только на Paddy Doctor).
Для v4 такая таблица была бы нечестной (источники есть в обучении), поэтому для v4 смотрите
eval/loso_results.csv (источник отложен) и eval/v4_results.csv (Paddy Doctor test).

Запуск (из корня проекта):
    python model/evaluate_sources.py models/rice_doctor_v3.keras eval/sources_v3.csv
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import train_multi  # noqa: E402  (импорт после sys.path нужен специально)


def main():
    if len(sys.argv) != 3:
        print("Как запускать: python model/evaluate_sources.py models/rice_doctor_v3.keras eval/sources_v3.csv")
        return
    from tensorflow import keras
    model = keras.models.load_model(sys.argv[1])
    rows = train_multi.read_rows()

    results = []
    for source in train_multi.LEAF_SOURCES:
        test_rows = [r for r in rows if r["source"] == source
                     and (source != "paddy_doctor" or r["part"] == "test")]
        pairs = train_multi.answers(model, test_rows)
        by_class = train_multi.accuracy_by_class(pairs)
        errors = train_multi.wrong_answers(pairs)
        right = sum(r for r, _ in by_class.values())
        total = sum(t for _, t in by_class.values())
        print(f"{source}: {right} из {total} = {right / total:.1%}")
        for cls, (r, t) in by_class.items():
            print(f"   {cls:<18}{r:>5} из {t:<5} {r / t:.1%}   ошибки -> {errors.get(cls, {})}")
            results.append({"source": source, "class": cls, "photos": t, "correct": r})
        results.append({"source": source, "class": "ALL", "photos": total, "correct": right})

    with open(sys.argv[2], "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "class", "photos", "correct"])
        writer.writeheader()
        writer.writerows(results)
    print("Сохранено:", sys.argv[2])


if __name__ == "__main__":
    main()
