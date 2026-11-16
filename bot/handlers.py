"""
Что бот делает на команды и сообщения: /start, выбор языка, /help, приём фото.
"""

import asyncio
import io
import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, Message

from bot import users
from bot.keyboards import language_keyboard
from bot.texts import LANGS, format_answer, get_text
from model import predict as model_predict

router = Router()


def answer_for_photo(image_bytes, lang):
    """Фото (байты) -> готовый текст ответа. Отдельная функция, чтобы её было легко тестировать."""
    label, confidence = model_predict.predict(image_bytes)
    return format_answer(label, confidence, lang)


@router.message(CommandStart())
async def cmd_start(message: Message):
    # Приветствие на двух языках и две кнопки выбора
    await message.answer(get_text("choose_lang", "ru"), reply_markup=language_keyboard())


@router.callback_query(F.data.startswith("lang:"))
async def choose_language(callback: CallbackQuery):
    lang = callback.data.split(":")[1]  # "lang:kk" -> "kk"
    if lang not in LANGS:
        await callback.answer()
        return
    users.set_lang(callback.from_user.id, lang)  # запоминаем выбор
    await callback.message.answer(get_text("lang_set", lang))
    await callback.answer()  # убираем «часики» на кнопке


@router.message(Command("help"))
async def cmd_help(message: Message):
    lang = users.get_lang(message.from_user.id)
    await message.answer(get_text("help", lang))


@router.message(F.photo)
async def handle_photo(message: Message):
    lang = users.get_lang(message.from_user.id)
    await message.answer(get_text("wait", lang))
    try:
        # Telegram присылает несколько размеров одного фото, последний - самый большой
        buffer = io.BytesIO()  # фото скачиваем в память, на диск не пишем
        await message.bot.download(message.photo[-1], destination=buffer)
        # Нейросеть считает долго, поэтому запускаем в отдельном потоке,
        # чтобы бот не «зависал» для других пользователей
        text = await asyncio.to_thread(answer_for_photo, buffer.getvalue(), lang)
    except Exception:
        logging.exception("Не удалось обработать фото")
        text = get_text("error", lang)
    await message.answer(text)


@router.message()
async def handle_other(message: Message):
    # Любое другое сообщение (текст, стикер...) - просим прислать фото
    lang = users.get_lang(message.from_user.id)
    await message.answer(get_text("send_photo", lang))
