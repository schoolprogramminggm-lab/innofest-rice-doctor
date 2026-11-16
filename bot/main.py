"""
Запуск бота (из корня проекта):
    python bot/main.py

Нужен файл .env с токеном: BOT_TOKEN=... (см. .env.example).
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Чтобы работало «python bot/main.py», добавляем корень проекта в пути поиска модулей
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from bot.handlers import router
from model import predict


async def main():
    load_dotenv()
    token = os.getenv("BOT_TOKEN")  # токен только из .env, в коде его нет
    if not token or token.startswith("вставьте"):
        sys.exit("Нет токена: впишите BOT_TOKEN в файл .env (см. .env.example)")

    # Модель грузим ОДИН раз при старте, а не на каждое фото
    logging.info("Загружаю модель...")
    try:
        predict.load_model()
    except Exception as error:
        sys.exit(f"Не удалось загрузить модель ({predict.MODEL_PATH}): {error}")
    logging.info("Модель загружена, бот запускается")

    bot = Bot(token=token)
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
