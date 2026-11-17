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
from bot.keyboards import details_keyboard, language_keyboard
from bot.texts import ADVICE, LANGS, format_answer, format_details, get_text, needs_details
from model import predict as model_predict

router = Router()


def answer_for_photo(image_bytes, lang):
    """
    Фото (байты) -> (текст короткого ответа, диагноз для кнопки «Подробнее»).
    Диагноз равен None, если кнопка не нужна: бот не уверен или на фото не лист риса.
    Отдельная функция, чтобы её было легко тестировать.
    """
    label, confidence = model_predict.predict(image_bytes)
    text = format_answer(label, confidence, lang)
    return text, (label if needs_details(label, confidence) else None)


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
        text, label = await asyncio.to_thread(answer_for_photo, buffer.getvalue(), lang)
    except Exception:
        logging.exception("Не удалось обработать фото")
        text, label = get_text("error", lang), None
    # Кнопка «Подробнее» есть только у ответа с диагнозом
    keyboard = details_keyboard(label, lang) if label else None
    await message.answer(text, reply_markup=keyboard)


@router.callback_query(F.data.startswith("more:"))
async def show_details(callback: CallbackQuery):
    label = callback.data.split(":")[1]  # "more:blast" -> "blast"
    if label not in ADVICE:
        await callback.answer()
        return
    lang = users.get_lang(callback.from_user.id)
    await callback.message.answer(format_details(label, lang))
    # Убираем кнопку, чтобы подробный ответ не отправляли повторно
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.answer()


@router.message()
async def handle_other(message: Message):
    # Любое другое сообщение (текст, стикер...) - просим прислать фото
    lang = users.get_lang(message.from_user.id)
    await message.answer(get_text("send_photo", lang))
