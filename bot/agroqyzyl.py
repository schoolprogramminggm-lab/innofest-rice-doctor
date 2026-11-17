"""
agroqyzyl.py: ответ RiceDoctor для командного бота AgroQyzyl (формат: docs/agroqyzyl-format.md).

    {"module": "rice_doctor", "version": 1, "ok": true/false,
     "result": {...}, "message": {"kk": "...", "ru": "..."}, "error": null | "текст"}

Тексты берём из bot/texts.py (format_answer) и advice/advice.json, ничего не дублируем.

Проверка (из корня проекта): python -m bot.agroqyzyl
"""

import json
import sys

from bot import texts


def make_answer(ok, result, message, error):
    """Общая обёртка формата: все шесть ключей есть всегда."""
    return {
        "module": "rice_doctor",
        "version": 1,
        "ok": ok,
        "result": result,
        "message": message,
        "error": error,
    }


def to_agroqyzyl(label, confidence):
    """
    Ответ для AgroQyzyl по результату нейросети.
      label      - класс, например "blast"
      confidence - уверенность от 0 до 1, например 0.91

    Три случая:
      1. уверенность >= порога: ok=true, диагноз и советы;
      2. ниже порога («не уверен»): ok=false, но в result лежит лучшая догадка
         (sure=false, советов нет), а message просит переснять фото;
      3. неизвестный класс или уверенность не от 0 до 1: ok=false, result=null.
    """
    # Случай 3: входные данные неправильные
    if label not in texts.ADVICE:
        error = "неизвестный класс: %s" % label
        message = {lang: texts.get_text("error", lang) for lang in texts.LANGS}
        return make_answer(False, None, message, error)
    if not 0 <= confidence <= 1:
        error = "уверенность должна быть от 0 до 1, получено %s" % confidence
        message = {lang: texts.get_text("error", lang) for lang in texts.LANGS}
        return make_answer(False, None, message, error)

    sure = confidence >= texts.CONFIDENCE_THRESHOLD
    result = {
        "disease": label,
        "name": {lang: texts.NAMES[lang][label] for lang in texts.LANGS},
        "confidence": round(float(confidence), 4),
        "sure": sure,
        # Советы даём только когда уверены: на сомнительный диагноз советовать нельзя
        "advice": {
            lang: list(texts.ADVICE[label][lang]["advice"]) if sure else [] for lang in texts.LANGS
        },
    }
    # Готовый текст бота: при «не уверен» format_answer сам вернёт просьбу переснять
    message = {lang: texts.format_answer(label, confidence, lang) for lang in texts.LANGS}

    if sure:
        return make_answer(True, result, message, None)

    error = "не уверен: уверенность %.0f%% ниже порога %.0f%%" % (
        confidence * 100, texts.CONFIDENCE_THRESHOLD * 100)
    return make_answer(False, result, message, error)


# Быстрая проверка: python -m bot.agroqyzyl
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(to_agroqyzyl("blast", 0.91), ensure_ascii=False, indent=2))
