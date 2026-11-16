"""Проверка format_answer: языки kk/ru и порог уверенности 60%."""

import pytest

from bot import texts
from bot.texts import format_answer, get_text

CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight"]


def test_ru_diagnosis_with_percent_and_advice():
    answer = format_answer("blast", 0.91, "ru")
    assert "Диагноз: Пирикуляриоз" in answer
    assert "91%" in answer
    assert "Что делать" in answer
    assert "фунгицид" in answer  # совет из advice.json


def test_kk_diagnosis_with_percent_and_advice():
    answer = format_answer("healthy", 0.75, "kk")
    assert "Сау жапырақ" in answer
    assert "75%" in answer
    assert "Не істеу керек" in answer


def test_every_class_both_languages():
    for label in CLASSES:
        for lang in ("kk", "ru"):
            answer = format_answer(label, 0.9, lang)
            assert texts.NAMES[lang][label] in answer
            assert "90%" in answer


def test_below_threshold_says_not_sure():
    ru = format_answer("brown_spot", 0.40, "ru")
    kk = format_answer("brown_spot", 0.40, "kk")
    assert ru == get_text("not_sure", "ru")
    assert kk == get_text("not_sure", "kk")
    # диагноза и процентов в таком ответе нет
    assert "Диагноз" not in ru and "40%" not in ru
    assert "Диагноз" not in kk and "40%" not in kk


def test_threshold_is_60_percent():
    assert texts.CONFIDENCE_THRESHOLD == pytest.approx(0.60)
    assert format_answer("blast", 0.59, "ru") == get_text("not_sure", "ru")
    # ровно 60% уже считается уверенным ответом
    assert "Диагноз" in format_answer("blast", 0.60, "ru")


def test_unknown_language_falls_back_to_russian():
    assert format_answer("blast", 0.9, "en") == format_answer("blast", 0.9, "ru")
