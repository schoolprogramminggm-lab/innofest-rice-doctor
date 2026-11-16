"""Кнопки бота."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bot.texts import get_text


def language_keyboard():
    """Две кнопки под сообщением: «Қазақша» и «Русский»."""
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=get_text("btn_kk", "kk"), callback_data="lang:kk"),
        InlineKeyboardButton(text=get_text("btn_ru", "ru"), callback_data="lang:ru"),
    ]])
