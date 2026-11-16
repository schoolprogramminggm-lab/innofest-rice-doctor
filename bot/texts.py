"""
Все тексты бота на казахском (kk) и русском (ru) + функция format_answer.

Названия болезней и советы лежат в advice/advice.json (их можно править, не трогая код).
Проверка: python bot/texts.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Порог уверенности: ниже него бот говорит «не уверен» (0.60 = 60%)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))

LANGS = ("kk", "ru")
DEFAULT_LANG = "ru"  # если человек ещё не выбрал язык

# advice.json лежит в корне проекта, на уровень выше папки bot/
ADVICE_PATH = Path(__file__).resolve().parent.parent / "advice" / "advice.json"
with open(ADVICE_PATH, encoding="utf-8") as f:
    ADVICE = json.load(f)  # {"blast": {"ru": {"name": ..., "advice": [...]}, "kk": {...}}, ...}

# Названия болезней: NAMES["ru"]["blast"] -> "Пирикуляриоз" (берём из advice.json)
NAMES = {lang: {label: ADVICE[label][lang]["name"] for label in ADVICE} for lang in LANGS}

# Обычные тексты бота: TEXTS[ключ][язык]
TEXTS = {
    # /start до выбора языка: пишем на двух языках сразу
    "choose_lang": {
        "kk": "🌾 Күріш дәрігері AI\nТілді таңдаңыз / Выберите язык:",
        "ru": "🌾 Күріш дәрігері AI\nТілді таңдаңыз / Выберите язык:",
    },
    "btn_kk": {"kk": "Қазақша", "ru": "Қазақша"},
    "btn_ru": {"kk": "Русский", "ru": "Русский"},
    # после выбора языка
    "lang_set": {
        "kk": "✅ Тіл таңдалды: қазақша.\n\n"
              "Күріш жапырағының суретін жіберіңіз, мен ауруын анықтап, кеңес беремін.\n"
              "Қалай дұрыс суретке түсіру керек: /help",
        "ru": "✅ Язык выбран: русский.\n\n"
              "Отправьте фото листа риса, я определю болезнь и дам совет.\n"
              "Как правильно фотографировать: /help",
    },
    "help": {
        "kk": "📸 Жапырақты қалай суретке түсіру керек:\n"
              "• Жапыраққа жақындаңыз: жапырақ кадрдың көп бөлігін алсын.\n"
              "• Күндіз, күн жарығында түсіріңіз, жарқылсыз.\n"
              "• Артқы фон біркелкі болсын (алақан, қағаз).\n"
              "• Сурет анық болсын, қозғалмаңыз.\n"
              "• Бір суретте бір жапырақ.\n\n"
              "Суретті осы чатқа жіберіңіз.",
        "ru": "📸 Как фотографировать лист:\n"
              "• Подойдите ближе: лист должен занимать большую часть кадра.\n"
              "• Снимайте днём, при дневном свете, без вспышки.\n"
              "• Фон однотонный (ладонь, лист бумаги).\n"
              "• Фото должно быть чётким, без смазывания.\n"
              "• Один лист на одном фото.\n\n"
              "Отправьте фото в этот чат.",
    },
    "wait": {
        "kk": "⏳ Суретті талдап жатырмын...",
        "ru": "⏳ Анализирую фото...",
    },
    "send_photo": {
        "kk": "Күріш жапырағының суретін жіберіңіз 📷. Қалай түсіру керек: /help",
        "ru": "Отправьте фото листа риса 📷. Как фотографировать: /help",
    },
    "error": {
        "kk": "😕 Суретті өңдеу мүмкін болмады. Басқа сурет жіберіп көріңіз.",
        "ru": "😕 Не удалось обработать фото. Попробуйте отправить другое.",
    },
    # ответ, когда уверенность ниже порога
    "not_sure": {
        "kk": "🤔 Сенімді емеспін. Жапырақты жақынырақ, күндіз, біркелкі фонда суретке түсіріңіз.",
        "ru": "🤔 Не уверен. Сфотографируйте лист ближе, при дневном свете, на однотонном фоне.",
    },
    # заголовки в ответе с диагнозом
    "diagnosis": {"kk": "🩺 Диагноз", "ru": "🩺 Диагноз"},
    "confidence": {"kk": "📊 Сенімділік", "ru": "📊 Уверенность"},
    "what_to_do": {"kk": "💡 Не істеу керек:", "ru": "💡 Что делать:"},
}


def get_text(key, lang):
    """Взять текст по ключу на нужном языке (если язык неизвестен - русский)."""
    if lang not in LANGS:
        lang = DEFAULT_LANG
    return TEXTS[key][lang]


def format_answer(label, confidence, lang):
    """
    Собрать готовый ответ бота.
      label      - класс, например "blast"
      confidence - уверенность от 0 до 1, например 0.91
      lang       - "kk" или "ru"
    Уверенность >= порога: диагноз + проценты + совет.
    Ниже порога: только «не уверен, переснимите».
    """
    if lang not in LANGS:
        lang = DEFAULT_LANG

    if confidence < CONFIDENCE_THRESHOLD:
        return get_text("not_sure", lang)

    lines = [
        f"{get_text('diagnosis', lang)}: {NAMES[lang][label]}",
        f"{get_text('confidence', lang)}: {confidence:.0%}",
        get_text("what_to_do", lang),
    ]
    # каждый пункт совета с новой строки, с точкой впереди
    lines += [f"• {line}" for line in ADVICE[label][lang]["advice"]]
    return "\n".join(lines)


# Быстрая проверка: python bot/texts.py
if __name__ == "__main__":
    # Консоль Windows (cp1251) не умеет печатать эмодзи, переключаем вывод на UTF-8
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print(format_answer("blast", 0.91, "ru"))
    print()
    print(format_answer("healthy", 0.75, "kk"))
    print()
    print(format_answer("brown_spot", 0.40, "ru"))
