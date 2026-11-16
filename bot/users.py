"""
Запоминаем язык каждого пользователя (kk или ru).

Хранится в маленьком файле data/user_langs.json: {"id пользователя": "kk"}.
Так язык не пропадает, если бота перезапустили. Фото и личные данные здесь не хранятся.
Папка data/ в .gitignore, в репозиторий файл не попадает.
"""

import json
from pathlib import Path

from bot.texts import DEFAULT_LANG, LANGS

LANGS_FILE = Path(__file__).resolve().parent.parent / "data" / "user_langs.json"

# Читаем файл один раз при старте, дальше работаем со словарём в памяти
_langs = {}
if LANGS_FILE.exists():
    try:
        _langs = json.loads(LANGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _langs = {}  # файл повреждён - начинаем с пустого, бот не падает


def get_lang(user_id):
    """Язык пользователя; если ещё не выбирал - русский."""
    return _langs.get(str(user_id), DEFAULT_LANG)


def set_lang(user_id, lang):
    """Запомнить язык пользователя (и записать в файл)."""
    if lang not in LANGS:
        return
    _langs[str(user_id)] = lang
    try:
        LANGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        LANGS_FILE.write_text(json.dumps(_langs, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # не смогли записать - язык всё равно запомнен до перезапуска
