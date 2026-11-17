"""Проверка advice/advice.json: все классы и оба языка на месте."""

import json
from pathlib import Path

CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight"]
LANGS = ["kk", "ru"]
# Разделы подробного ответа, которые хранятся списком строк
LIST_FIELDS = ["signs", "first_3_days", "prevention", "call_agronomist"]

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


def test_every_detail_field_in_both_languages():
    # Подробный ответ (кнопка «Подробнее»): пять разделов для каждой болезни на kk и ru
    for label in CLASSES:
        for lang in LANGS:
            entry = ADVICE[label][lang]
            assert isinstance(entry["danger"], str) and entry["danger"].strip(), f"{label}/{lang}: danger"
            for key in LIST_FIELDS:
                assert isinstance(entry[key], list) and entry[key], f"{label}/{lang}: нет {key}"
                assert all(line.strip() for line in entry[key]), f"{label}/{lang}: пустая строка в {key}"


def test_kazakh_text_differs_from_russian():
    # Защита от случайно скопированного русского текста в казахский раздел
    for label in CLASSES:
        for key in ["danger"] + LIST_FIELDS:
            assert ADVICE[label]["kk"][key] != ADVICE[label]["ru"][key], f"{label}/{key}"


def test_classes_match_model_classes():
    # Болезни в советах совпадают с классами модели без not_rice (если файл есть локально).
    # not_rice - не болезнь, советов для него нет: бот отвечает отдельным текстом.
    classes_file = Path(__file__).resolve().parent.parent / "models" / "classes.json"
    if classes_file.exists():
        model_classes = json.loads(classes_file.read_text(encoding="utf-8"))
        assert sorted(c for c in model_classes if c != "not_rice") == sorted(ADVICE)
