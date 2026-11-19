"""
Все тексты бота на казахском (kk) и русском (ru) + функция format_answer.

Названия болезней и советы лежат в advice/advice.json (их можно править, не трогая код).
Ответ бота состоит из двух частей:
  1. короткий: диагноз, уверенность, 3-4 строки совета (format_answer);
  2. подробный, по кнопке «Подробнее»: как узнать, насколько опасно, первые 3 дня,
     профилактика, когда звать агронома (format_details).
Проверка (из корня проекта): python bot/texts.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Чтобы работало «python bot/texts.py», добавляем корень проекта в пути поиска модулей
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.predict import NOT_RICE, percent  # percent: уверенность не больше 99%

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
        "kk": "🤔 Сенімді емеспін. Жапырақты жақынырақ, күндіз суретке түсіріңіз.",
        "ru": "🤔 Не уверен. Сфотографируйте лист ближе, при дневном свете.",
    },
    # ответ, когда на фото не лист риса (диагноз не ставим)
    "not_rice": {
        "kk": "Бұл күріш жапырағына ұқсамайды. Жапырақты жақынырақ суретке түсіріңіз.",
        "ru": "Это не похоже на лист риса. Сфотографируйте лист ближе.",
    },
    # кнопка под коротким ответом
    "btn_more": {"kk": "Толығырақ", "ru": "Подробнее"},
    # заголовки подробного ответа
    "details_title": {"kk": "📖 Толығырақ", "ru": "📖 Подробнее"},
    "signs": {"kk": "🔍 Қалай анықтау керек:", "ru": "🔍 Как узнать:"},
    "danger": {"kk": "⚠️ Қаншалықты қауіпті:", "ru": "⚠️ Насколько опасно:"},
    "first_3_days": {"kk": "📅 Алғашқы 3 күнде не істеу керек:", "ru": "📅 Что сделать в первые 3 дня:"},
    "prevention": {
        "kk": "🌱 Келесі маусымға алдын алу:",
        "ru": "🌱 Профилактика на следующий сезон:",
    },
    "call_agronomist": {
        "kk": "📞 Агрономды қашан жедел шақыру керек:",
        "ru": "📞 Когда срочно звать агронома:",
    },
    "details_footer": {
        "kk": "ℹ️ Бұл суреттен шығарылған болжам, агрономның орнын баспайды. "
              "Өңдеу уақыты мен тәсілін агроном егістікте анықтайды.",
        "ru": "ℹ️ Это подсказка по фото, а не замена агронома. "
              "Сроки и способ обработки определяет агроном на месте.",
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
    Не лист риса (label == "not_rice"): только «это не лист риса», без диагноза.
    Уверенность >= порога: диагноз + проценты + совет.
    Ниже порога: только «не уверен, переснимите».
    """
    if lang not in LANGS:
        lang = DEFAULT_LANG

    if label == NOT_RICE:
        return get_text("not_rice", lang)

    if confidence < CONFIDENCE_THRESHOLD:
        return get_text("not_sure", lang)

    lines = [
        f"{get_text('diagnosis', lang)}: {NAMES[lang][label]}",
        f"{get_text('confidence', lang)}: {percent(confidence)}",
        get_text("what_to_do", lang),
    ]
    # каждый пункт совета с новой строки, с точкой впереди
    lines += [f"• {line}" for line in ADVICE[label][lang]["advice"]]
    return "\n".join(lines)


def needs_details(label, confidence):
    """Нужна ли кнопка «Подробнее»: только если поставлен диагноз (не «не уверен», не «не лист»)."""
    return label in ADVICE and confidence >= CONFIDENCE_THRESHOLD


def format_details(label, lang):
    """
    Подробный ответ по болезни (по кнопке «Подробнее»): как узнать, насколько опасно,
    первые 3 дня, профилактика, когда звать агронома. Всё берётся из advice.json.
    """
    if lang not in LANGS:
        lang = DEFAULT_LANG
    info = ADVICE[label][lang]

    def bullets(items):
        return [f"• {item}" for item in items]

    lines = [f"{get_text('details_title', lang)}: {info['name']}", ""]
    lines += [get_text("signs", lang)] + bullets(info["signs"]) + [""]
    lines += [get_text("danger", lang), info["danger"], ""]
    lines += [get_text("first_3_days", lang)] + bullets(info["first_3_days"]) + [""]
    lines += [get_text("prevention", lang)] + bullets(info["prevention"]) + [""]
    lines += [get_text("call_agronomist", lang)] + bullets(info["call_agronomist"]) + [""]
    lines += [get_text("details_footer", lang)]
    return "\n".join(lines)


# Быстрая проверка: python bot/texts.py
if __name__ == "__main__":
    # Консоль Windows (cp1251) не умеет печатать эмодзи, переключаем вывод на UTF-8
    sys.stdout.reconfigure(encoding="utf-8")

    print(format_answer("blast", 0.91, "ru"))
    print()
    print(format_answer("healthy", 0.75, "kk"))
    print()
    print(format_answer("brown_spot", 0.40, "ru"))
    print()
    print(format_answer("blast", 0.9999, "ru"))  # 99%, а не 100%
    print()
    print(format_answer("not_rice", 0.97, "ru"))
    print()
    print(format_details("blast", "ru"))
