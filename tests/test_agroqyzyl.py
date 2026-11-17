"""Тесты bot/agroqyzyl.py: JSON для AgroQyzyl. Модель и интернет не нужны."""
import json
from pathlib import Path

import pytest

from bot import texts
from bot.agroqyzyl import to_agroqyzyl

ROOT = Path(__file__).resolve().parent.parent

# Обязательные ключи формата (docs/agroqyzyl-format.md)
KEYS = {"module", "version", "ok", "result", "message", "error"}
RESULT_KEYS = {"disease", "name", "confidence", "sure", "advice"}
CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight"]


def check_envelope(answer):
    """Общая проверка: все ключи, оба языка, текст не пустой."""
    assert set(answer) == KEYS
    assert answer["module"] == "rice_doctor"
    assert answer["version"] == 1
    assert set(answer["message"]) == {"kk", "ru"}
    assert answer["message"]["kk"].strip() and answer["message"]["ru"].strip()


def test_ok_answer_has_all_keys_and_both_languages():
    answer = to_agroqyzyl("blast", 0.91)
    check_envelope(answer)
    assert answer["ok"] is True
    assert answer["error"] is None

    result = answer["result"]
    assert set(result) == RESULT_KEYS
    assert result["disease"] == "blast"
    assert set(result["name"]) == {"kk", "ru"}
    assert result["confidence"] == pytest.approx(0.91)
    assert result["sure"] is True
    # советы на обоих языках, не пустые
    assert set(result["advice"]) == {"kk", "ru"}
    assert result["advice"]["kk"] and result["advice"]["ru"]


def test_every_class_both_languages():
    for label in CLASSES:
        answer = to_agroqyzyl(label, 0.9)
        check_envelope(answer)
        assert answer["ok"] is True
        for lang in ("kk", "ru"):
            assert answer["result"]["name"][lang] == texts.NAMES[lang][label]
            assert answer["result"]["advice"][lang] == texts.ADVICE[label][lang]["advice"]


def test_message_is_the_same_text_as_the_bot_shows():
    answer = to_agroqyzyl("brown_spot", 0.8)
    for lang in ("kk", "ru"):
        assert answer["message"][lang] == texts.format_answer("brown_spot", 0.8, lang)


def test_not_sure_is_ok_false_with_clear_error():
    answer = to_agroqyzyl("brown_spot", 0.40)
    check_envelope(answer)
    assert answer["ok"] is False
    assert "не уверен" in answer["error"]
    assert "40%" in answer["error"]
    # в message просьба переснять, на обоих языках
    assert answer["message"]["ru"] == texts.get_text("not_sure", "ru")
    assert answer["message"]["kk"] == texts.get_text("not_sure", "kk")
    # лучшая догадка в result есть, но sure=false и советов нет
    result = answer["result"]
    assert set(result) == RESULT_KEYS
    assert result["disease"] == "brown_spot"
    assert result["sure"] is False
    assert result["advice"] == {"kk": [], "ru": []}


def test_threshold_is_60_percent():
    assert to_agroqyzyl("blast", 0.59)["ok"] is False
    # ровно 60% уже считается уверенным ответом
    assert to_agroqyzyl("blast", 0.60)["ok"] is True


def test_unknown_class():
    answer = to_agroqyzyl("rust", 0.95)
    check_envelope(answer)
    assert answer["ok"] is False
    assert answer["result"] is None
    assert "rust" in answer["error"]


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_out_of_range(confidence):
    answer = to_agroqyzyl("blast", confidence)
    check_envelope(answer)
    assert answer["ok"] is False
    assert answer["result"] is None
    assert "от 0 до 1" in answer["error"]


def test_answer_is_json_serializable():
    # ответ уходит боту как JSON: проверяем, что он превращается в текст и обратно
    answer = to_agroqyzyl("healthy", 0.75)
    assert json.loads(json.dumps(answer, ensure_ascii=False)) == answer


def test_example_file_matches_format():
    # examples/agroqyzyl_rice_doctor.json должен соответствовать формату
    example = json.loads((ROOT / "examples" / "agroqyzyl_rice_doctor.json").read_text(encoding="utf-8"))
    check_envelope(example)
    assert example["ok"] is True
    assert set(example["result"]) == RESULT_KEYS
    assert set(example["result"]["advice"]) == {"kk", "ru"}
