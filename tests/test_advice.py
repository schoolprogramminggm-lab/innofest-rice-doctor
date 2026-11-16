"""Проверка advice/advice.json: все классы и оба языка на месте."""

import json
from pathlib import Path

CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight"]
LANGS = ["kk", "ru"]

ADVICE_PATH = Path(__file__).resolve().parent.parent / "advice" / "advice.json"
ADVICE = json.loads(ADVICE_PATH.read_text(encoding="utf-8"))


def test_all_classes_present():
    assert sorted(ADVICE) == sorted(CLASSES)


def test_both_languages_for_every_class():
    for label in CLASSES:
        for lang in LANGS:
            entry = ADVICE[label][lang]
            assert entry["name"].strip(), f"{label}/{lang}: нет названия"
            assert entry["advice"], f"{label}/{lang}: нет совета"
            assert all(line.strip() for line in entry["advice"])


def test_advice_length_3_or_4_lines():
    # В задаче #7: совет 3-4 строки
    for label in CLASSES:
        for lang in LANGS:
            assert 3 <= len(ADVICE[label][lang]["advice"]) <= 4, f"{label}/{lang}"


def test_classes_match_model_classes():
    # Список классов в советах совпадает с тем, что знает модель (если файл есть локально)
    classes_file = Path(__file__).resolve().parent.parent / "models" / "classes.json"
    if classes_file.exists():
        assert sorted(json.loads(classes_file.read_text(encoding="utf-8"))) == sorted(ADVICE)
