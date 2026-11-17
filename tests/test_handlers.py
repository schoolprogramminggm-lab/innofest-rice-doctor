"""Кнопка «Подробнее»: клавиатура и обработчик нажатия. Telegram и модель не нужны."""

import asyncio

from bot import handlers, texts
from bot.keyboards import details_keyboard


class FakeMessage:
    """Притворяется сообщением Telegram: запоминает, что бот ответил."""

    def __init__(self):
        self.answers = []
        self.keyboard_removed = False

    async def answer(self, text):
        self.answers.append(text)

    async def edit_reply_markup(self, reply_markup=None):
        self.keyboard_removed = reply_markup is None


class FakeCallback:
    """Притворяется нажатием кнопки."""

    def __init__(self, data):
        self.data = data
        self.message = FakeMessage()
        self.from_user = type("User", (), {"id": 1})()
        self.answered = False

    async def answer(self):
        self.answered = True


def test_keyboard_has_one_button_per_language():
    for lang, title in (("ru", "Подробнее"), ("kk", "Толығырақ")):
        button = details_keyboard("blast", lang).inline_keyboard[0][0]
        assert button.text == title
        assert button.callback_data == "more:blast"


def test_press_details_sends_full_answer(monkeypatch):
    monkeypatch.setattr(handlers.users, "get_lang", lambda user_id: "kk")
    callback = FakeCallback("more:blast")
    asyncio.run(handlers.show_details(callback))
    assert callback.message.answers == [texts.format_details("blast", "kk")]
    assert callback.message.keyboard_removed  # повторно нажать нельзя
    assert callback.answered


def test_unknown_disease_in_button_is_ignored():
    callback = FakeCallback("more:rust")
    asyncio.run(handlers.show_details(callback))
    assert callback.message.answers == []
    assert callback.answered
