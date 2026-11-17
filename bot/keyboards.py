"""Кнопки бота."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.texts import get_text


def details_keyboard(label, lang):
    """Одна кнопка «Подробнее» под коротким ответом. В callback_data зашит диагноз: "more:blast"."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=get_text("btn_more", lang), callback_data="more:" + label),
    ]])


def language_keyboard():
    """Две кнопки под сообщением: «Қазақша» и «Русский»."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=get_text("btn_kk", "kk"), callback_data="lang:kk"),
        InlineKeyboardButton(text=get_text("btn_ru", "ru"), callback_data="lang:ru"),
    ]])
