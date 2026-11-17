"""
Скачивание посторонних фото для 5-го класса not_rice («это не лист риса»).

Источник: Open Images V7 (проверочная часть), https://storage.googleapis.com/openimages/web/
Лицензия: берём ТОЛЬКО фото с лицензией CC BY 2.0 (можно использовать, нужно указывать автора).
Метки Open Images: CC BY 4.0.

Что делает скрипт:
1. Скачивает три маленьких CSV-файла (описание классов, метки фото, список фото с лицензиями).
2. Для каждого понятия (кошка, человек, дерево, книга...) выбирает фото с этой меткой.
   Берём разное: другие растения, люди, животные, предметы, тексты и вывески.
3. Скачивает ~500 фото, уменьшает до 256 пикселей (как в prepare_data.py).
4. Делит 80% -> data/train/not_rice, 20% -> data/val/not_rice (seed = 42).
5. Пишет data/not_rice_sources.csv: у каждого фото автор, лицензия и ссылка (для README).

Папка data/ в git не попадает. Запуск (из корня проекта):
    python model/download_not_rice.py
"""

import csv
import io
import random
import shutil
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

BASE_URL = "https://storage.googleapis.com/openimages/"
CSV_FILES = {
    "classes": BASE_URL + "v7/oidv7-class-descriptions.csv",
    "labels": BASE_URL + "v5/validation-annotations-human-imagelabels.csv",
    "images": BASE_URL + "2018_04/validation/validation-images-with-rotation.csv",
}
CSV_DIR = Path("data/openimages")  # сюда складываем скачанные CSV

TRAIN_DIR = Path("data/train/not_rice")
VAL_DIR = Path("data/val/not_rice")
SOURCES_FILE = Path("data/not_rice_sources.csv")

LICENSE = "https://creativecommons.org/licenses/by/2.0/"  # CC BY 2.0
MAX_SIDE = 256   # как в prepare_data.py
VAL_PART = 0.2
SEED = 42

# Сколько фото берём по каждому понятию (всего 500).
# Больше всего «других растений»: именно их проще всего принять за лист риса.
QUOTAS = {
    # другие растения
    "Leaf": 50, "Houseplant": 35, "Tree": 25, "Grass": 30, "Flower": 30,
    # люди
    "Person": 60,
    # животные
    "Cat": 35, "Dog": 35, "Bird": 25,
    # предметы
    "Car": 25, "Building": 25, "Chair": 20, "Table": 20, "Laptop": 20,
    # текст
    "Book": 20, "Poster": 20, "Text": 25,
}


def download(url):
    """Скачать файл по ссылке и вернуть его байты."""
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def read_csv_file(name):
    """Скачать CSV (один раз, потом берём с диска) и прочитать строки как словари."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    path = CSV_DIR / (name + ".csv")
    if not path.exists():
        print("Скачиваю", CSV_FILES[name])
        path.write_bytes(download(CSV_FILES[name]))
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rnd = random.Random(SEED)

    # Название понятия -> его код в Open Images ("Cat" -> "/m/01yrx")
    # В файле классов нет заголовка, поэтому читаем его без DictReader.
    path = CSV_DIR / "classes.csv"
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(download(CSV_FILES["classes"]))
    with open(path, encoding="utf-8", newline="") as f:
        codes = {name: code for code, name in csv.reader(f)}

    # Информация о фото: только с нужной лицензией. ID фото -> строка CSV
    photos = {row["ImageID"]: row for row in read_csv_file("images") if row["License"] == LICENSE}

    # Какие фото имеют какую метку (Confidence = 1: метка точно верная)
    labelled = {}  # код понятия -> список ID фото
    for row in read_csv_file("labels"):
        if row["Confidence"] == "1" and row["ImageID"] in photos:
            labelled.setdefault(row["LabelName"], []).append(row["ImageID"])

    # Для каждого понятия случайно выбираем фото, скачиваем, пока не наберём норму.
    # Одно фото берём только один раз (даже если на нём и человек, и стул).
    used = set()
    chosen = []  # (ID фото, понятие, картинка)
    for concept, quota in QUOTAS.items():
        candidates = sorted(set(labelled.get(codes[concept], [])) - used)
        rnd.shuffle(candidates)
        got = 0
        for image_id in candidates:
            if got == quota:
                break
            try:
                data = download(photos[image_id]["Thumbnail300KURL"])
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            except Exception as error:  # фото могло исчезнуть с Flickr: берём следующее
                print("  пропуск", image_id, error)
                continue
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            used.add(image_id)
            chosen.append((image_id, concept, img))
            got += 1
        print(f"{concept}: {got} из {quota}")

    # Делим на обучение и проверку. Старые папки not_rice удаляем, чтобы не было дублей.
    for folder in (TRAIN_DIR, VAL_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    rnd.shuffle(chosen)
    n_val = round(len(chosen) * VAL_PART)

    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCES_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "part", "concept", "author", "license", "url"])
        for i, (image_id, concept, img) in enumerate(chosen):
            part, folder = ("val", VAL_DIR) if i < n_val else ("train", TRAIN_DIR)
            img.save(folder / (image_id + ".jpg"), quality=95)
            info = photos[image_id]
            writer.writerow([image_id + ".jpg", part, concept, info["Author"],
                             "CC BY 2.0", info["OriginalLandingURL"]])

    print(f"\nГотово: {len(chosen)} фото (обучение {len(chosen) - n_val}, проверка {n_val})")
    print("Авторы и ссылки:", SOURCES_FILE)


if __name__ == "__main__":
    main()
