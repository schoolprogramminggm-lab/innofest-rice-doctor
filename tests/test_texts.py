"""Проверка format_answer / format_details: языки kk/ru, порог 60%, максимум 99%, не лист риса."""

import re

import pytest

from bot import texts
from bot.texts import format_answer, format_details, get_text, needs_details
from model.predict import percent

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


def test_confidence_is_never_100_percent():
    # 0.9999 и даже 1.0 показываем как 99%
    for confidence in (0.9999, 0.996, 1.0):
        for lang in ("kk", "ru"):
            answer = format_answer("blast", confidence, lang)
            assert "99%" in answer
            assert "100%" not in answer
    assert percent(0.9999) == "99%"
    assert percent(0.91) == "91%"
    assert percent(0.29) == "29%"  # без ошибок округления (0.29 * 100 = 28.999...)


@pytest.mark.parametrize("lang", ["kk", "ru"])
def test_not_rice_has_no_diagnosis(lang):
    answer = format_answer("not_rice", 0.97, lang)
    assert answer == get_text("not_rice", lang)
    assert "Диагноз" not in answer and "%" not in answer
    assert needs_details("not_rice", 0.97) is False


def test_not_rice_message_text_ru():
    assert "Это не похоже на лист риса. Сфотографируйте лист ближе" in format_answer("not_rice", 0.9, "ru")


def test_details_button_only_for_a_sure_diagnosis():
    assert needs_details("blast", 0.91) is True
    assert needs_details("blast", 0.40) is False  # «не уверен»


@pytest.mark.parametrize("lang", ["kk", "ru"])
@pytest.mark.parametrize("label", CLASSES)
def test_details_have_all_sections(label, lang):
    details = format_details(label, lang)
    info = texts.ADVICE[label][lang]
    # все пять разделов с заголовками на нужном языке
    for key in ("signs", "danger", "first_3_days", "prevention", "call_agronomist"):
        assert get_text(key, lang) in details
    assert info["danger"] in details
    assert len(details) < 4096  # лимит Telegram на одно сообщение


def test_details_have_no_fungicide_names_or_doses():
    # Названий препаратов и доз быть не должно: их подбирает агроном
    doses = re.compile(r"\d+([.,]\d+)?\s*(мл|г|кг|л|литр|грамм|%)(?!\w)|г/л|кг/га|л/га|мл/га", re.IGNORECASE)
    names = ["трициклазол", "азоксистробин", "тебуконазол", "пропиконазол", "манкоцеб", "каратэ", "фундазол"]
    for label in CLASSES:
        for lang in ("kk", "ru"):
            info = texts.ADVICE[label][lang]
            text = format_details(label, lang) + " ".join(info["advice"])
            assert not doses.search(text), f"{label}/{lang}: похоже на дозу"
            assert not any(name in text.lower() for name in names), f"{label}/{lang}: название препарата"
