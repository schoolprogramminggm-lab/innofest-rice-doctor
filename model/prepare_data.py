"""
Подготовка фото для обучения.

Что делает скрипт:
1. Берёт из data/raw/train_images только 4 нужных класса.
2. Уменьшает каждое фото, чтобы длинная сторона была 256 пикселей
   (так обучение идёт в несколько раз быстрее).
3. Делит фото каждого класса: 80% -> data/train, 20% -> data/val.
4. Печатает таблицу: сколько фото в каждом классе.

Запуск (из корня проекта):
    python model/prepare_data.py
"""

import random
import shutil
from pathlib import Path

from PIL import Image

# Папка, куда распакован архив Paddy Doctor с Kaggle
RAW_DIR = Path("data/raw/train_images")

# Куда складываем готовые фото
TRAIN_DIR = Path("data/train")
VAL_DIR = Path("data/val")

# Какие папки из набора берём и как называем их у себя.
# Слева - имя папки в Paddy Doctor, справа - наше имя класса (как в README).
CLASSES = {
    "normal": "healthy",
    "blast": "blast",
    "brown_spot": "brown_spot",
    "bacterial_leaf_blight": "bacterial_blight",
}

MAX_SIDE = 256      # длинная сторона фото после уменьшения
VAL_PART = 0.2      # 20% фото идут на проверку
SEED = 42           # "зерно" случайности: при каждом запуске деление будет одинаковым


def save_small_copy(src, dst):
    """Открыть фото, уменьшить и сохранить копию."""
    img = Image.open(src).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))  # уменьшает, сохраняя пропорции
    img.save(dst, quality=95)


def main():
    if not RAW_DIR.exists():
        print("Не найдена папка", RAW_DIR)
        print("Скачайте Paddy Doctor с Kaggle и распакуйте в data/raw/")
        return

    # Удаляем старые train/val, чтобы при повторном запуске не было дублей
    for folder in (TRAIN_DIR, VAL_DIR):
        if folder.exists():
            shutil.rmtree(folder)

    rnd = random.Random(SEED)
    table = []  # строки для итоговой таблицы

    for raw_name, our_name in CLASSES.items():
        # Список всех фото класса, отсортированный (чтобы порядок был всегда одинаковым)
        photos = sorted((RAW_DIR / raw_name).glob("*.jpg"))
        rnd.shuffle(photos)  # перемешиваем

        n_val = round(len(photos) * VAL_PART)
        val_photos = photos[:n_val]
        train_photos = photos[n_val:]

        for part_dir, part_photos in ((TRAIN_DIR, train_photos), (VAL_DIR, val_photos)):
            out_dir = part_dir / our_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for photo in part_photos:
                save_small_copy(photo, out_dir / photo.name)

        table.append((our_name, len(photos), len(train_photos), len(val_photos)))
        print(f"{our_name}: готово")

    # Печатаем таблицу в формате Markdown, чтобы вставить её в README
    print()
    print("| Класс | Фото всего | Обучение (80%) | Проверка (20%) |")
    print("|---|---|---|---|")
    for name, total, n_train, n_val in table:
        print(f"| {name} | {total} | {n_train} | {n_val} |")
    total_all = sum(row[1] for row in table)
    train_all = sum(row[2] for row in table)
    val_all = sum(row[3] for row in table)
    print(f"| **Всего** | **{total_all}** | **{train_all}** | **{val_all}** |")


if __name__ == "__main__":
    main()
