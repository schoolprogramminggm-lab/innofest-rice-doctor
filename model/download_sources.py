"""
Сбор обучающего набора из НЕСКОЛЬКИХ открытых источников (issue #23).

Раньше сеть видела только Paddy Doctor и на чужих фото давала 21%. Теперь учим на разных источниках.

Что делает скрипт:
1. Скачивает архивы источников (если их ещё нет в data/raw_sources/). Все источники открытые, CC BY 4.0,
   логин не нужен. Paddy Doctor и Open Images уже скачаны прошлыми скриптами (prepare_data.py,
   download_not_rice.py), их берём из data/train и data/val.
2. Каждое фото читает с учётом EXIF-поворота, уменьшает до 256 пикселей (как prepare_data.py)
   и сохраняет в data/multi/<источник>/<класс>/<хеш>.jpg.
3. УТЕЧКА ЗАПРЕЩЕНА: 33 фото из eval/images/ в обучение не попадают. Проверка трёх видов:
   - хеш файла (sha256) совпал с хешем фото из eval;
   - «похожий хеш» (dHash): eval-фото были уменьшены до 512 px, поэтому sha256 оригинала не совпадёт,
     а dHash почти не меняется при уменьшении;
   - имя исходного файла из eval/labels.csv.
   Так же выбрасываются почти одинаковые фото из РАЗНЫХ источников (иначе «проверка по источнику»
   была бы нечестной: то же фото лежало бы и в обучении, и в проверке).
4. Пишет data/sources.csv: у каждого фото источник, лицензия, класс, исходное имя, хеш и часть
   (train / val / test). Этот CSV лежит в git, сами фото - нет.

Запуск (из корня проекта, нужен интернет, ~3 ГБ скачать):
    python model/download_sources.py
"""

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_sources"
OUT = ROOT / "data" / "multi"
SOURCES_CSV = ROOT / "data" / "sources.csv"

MAX_SIDE = 256        # как в prepare_data.py
VAL_PART = 0.15       # 15% фото каждого (источник, класс) идут на проверку при обучении
SEED = 42
EVAL_DISTANCE = 8     # dHash: расстояние до eval-фото не больше 8 из 64 бит -> считаем тем же фото
DUPLICATE_DISTANCE = 3  # то же для дублей между источниками

CC_BY = "CC BY 4.0"

# --- Описание источников -------------------------------------------------------------------
# Порядок важен: при дубле между источниками остаётся фото из того, что раньше в списке.
# mapping: часть пути внутри архива (папка) -> наш класс.
SOURCES = {
    "paddy_doctor": {
        "title": "Paddy Doctor (Индия, Тамилнад, поле)",
        "citation": "Petchiammal A. et al., CODS-COMAD 2023, doi:10.1145/3570991.3570994",
        "license": CC_BY,
    },
    "mendeley_bd": {
        "title": "Mendeley: Rice Leaf Bacterial and Fungal Disease Dataset (Бангладеш, поле)",
        "citation": "Hasan M., Khatun S., Raihan M.A., Uddin A.H., doi:10.17632/hx6f852hw4.2",
        "license": CC_BY,
        "mendeley_id": "hx6f852hw4", "zip": "Original Images.zip",
        "mapping": {"healthy rice leaf": "healthy", "leaf blast": "blast",
                    "brown spot": "brown_spot", "bacterial leaf blight": "bacterial_blight"},
    },
    "riceleafbd": {
        "title": "Mendeley: RiceLeafBD (Бангладеш, поле)",
        "citation": "RiceLeafBD, doi:10.17632/kx9rx8p2mz.1 (arXiv:2501.08912)",
        "license": CC_BY,
        "mendeley_id": "kx9rx8p2mz", "zip": "Original Images.zip",
        "mapping": {"healthy leaf": "healthy", "bacterial leaf blight": "bacterial_blight",
                    "brown spot": "brown_spot"},
    },
    "uci": {
        "title": "UCI Rice Leaf Diseases (Индия, лист на белом фоне)",
        "citation": "Shah J., Prajapati H., Dabhi V., doi:10.24432/C5R013",
        "license": CC_BY,
        "url": "https://archive.ics.uci.edu/static/public/486/rice+leaf+diseases.zip",
        "mapping": {"bacterial leaf blight": "bacterial_blight", "brown spot": "brown_spot"},
    },
    "dhan_shomadhan": {
        "title": "Dhan-Shomadhan (Бангладеш, Дакка: поле и белый фон)",
        "citation": "Hossain M.F. et al., doi:10.17632/znsxdctwtt.1 (arXiv:2309.07515)",
        "license": CC_BY,
        "mendeley_id": "znsxdctwtt",
        # в этом наборе все файлы лежат в одной папке, класс виден по началу имени файла
        "prefixes": {"rb_wb": "blast", "rbf": "blast", "bs_wb": "brown_spot", "bsf": "brown_spot"},
    },
    "openimages": {
        "title": "Open Images V7 (посторонние фото)",
        "citation": "Google, Open Images V7; автор каждого фото в data/not_rice_sources.csv",
        "license": "CC BY 2.0",
    },
}


# --- Помощники -----------------------------------------------------------------------------
def dhash(img):
    """
    «Отпечаток» картинки из 64 бит: сравниваем соседние пиксели уменьшенной серой копии.
    Одно и то же фото в разном размере даёт почти одинаковый отпечаток.
    """
    small = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def distance(a, b):
    """Сколько бит отличается у двух отпечатков."""
    return bin(a ^ b).count("1")


def load_small(data):
    """Открыть фото из байтов: учесть EXIF-поворот, перевести в RGB, уменьшить до MAX_SIDE."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def open_url(url):
    """Открыть ссылку. Mendeley не пускает стандартный User-Agent Python (ошибка 403), поэтому свой, общий."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (dataset download script)"})
    return urllib.request.urlopen(request, timeout=60)


def download(url, path):
    """Скачать файл, если его ещё нет (сначала во временный файл, чтобы не остался обрубок).
    url может быть функцией: тогда она вызывается, только если файла нет (чтобы зря не ходить в API)."""
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    url = url() if callable(url) else url
    print("Скачиваю", url)
    tmp = path.with_suffix(path.suffix + ".part")
    with open_url(url) as response, open(tmp, "wb") as f:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(path)


def mendeley_files(dataset_id):
    """Список файлов набора Mendeley Data: имя -> ссылка на скачивание (через открытый API)."""
    url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}"
    with open_url(url) as response:
        info = json.load(response)
    return {f["filename"]: f["content_details"]["download_url"] for f in info["files"]}


def is_image(name):
    return name.lower().endswith((".jpg", ".jpeg", ".png"))


# --- Чтение каждого источника: выдаёт (класс, исходное_имя, байты_файла) ---------------------
def read_zip(zip_path, mapping):
    """Фото из архива. Класс определяем по названию папки внутри архива (см. mapping)."""
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            if not is_image(name) or "__MACOSX" in name:
                continue
            folders = [p.lower().strip() for p in name.split("/")[:-1]]
            cls = next((mapping[f] for f in reversed(folders) if f in mapping), None)
            if cls:
                try:
                    yield cls, name, z.read(name)
                except zipfile.BadZipFile:  # повреждённый файл в архиве: пропускаем, а не падаем
                    print("  битый файл в архиве, пропуск:", name)


def read_dhan():
    """Dhan-Shomadhan: файлы лежат плоско, класс по префиксу имени (rb_wb_12.jpg -> blast)."""
    prefixes = SOURCES["dhan_shomadhan"]["prefixes"]
    folder = RAW / "dhan"
    # Скачиваем только blast и brown_spot (остальные болезни нам не нужны)
    def wanted(name):
        return is_image(name) and any(name.startswith(p + "_") or re.match(p + r"\d", name) for p in prefixes)

    for name, url in sorted(mendeley_files("znsxdctwtt").items()):  # download() пропускает уже скачанные
        if wanted(name):
            download(url, folder / name)
    for path in sorted(folder.glob("*.jp*g")):
        for prefix, cls in prefixes.items():
            if path.name.startswith(prefix + "_") or re.match(prefix + r"\d", path.name):
                yield cls, path.name, path.read_bytes()
                break


def read_paddy():
    """Paddy Doctor: берём уже уменьшенные копии из data/train и data/val, хеш - от оригинала."""
    raw_names = {"healthy": "normal", "blast": "blast", "brown_spot": "brown_spot",
                 "bacterial_blight": "bacterial_leaf_blight"}
    for part in ("train", "val"):
        for cls, raw_name in raw_names.items():
            for path in sorted((ROOT / "data" / part / cls).glob("*.jpg")):
                original = ROOT / "data" / "raw" / "train_images" / raw_name / path.name
                digest_from = original if original.exists() else path
                yield cls, f"{raw_name}/{path.name}", path.read_bytes(), part, digest_from.read_bytes()


def read_openimages():
    """Open Images (not_rice): уже скачано download_not_rice.py в data/train и data/val."""
    for part in ("train", "val"):
        for path in sorted((ROOT / "data" / part / "not_rice").glob("*.jpg")):
            data = path.read_bytes()
            yield "not_rice", path.name, data, part, data


def eval_fingerprints():
    """Хеши и отпечатки 33 фото из eval/images/ + исходные имена из eval/labels.csv."""
    sha, hashes, names = set(), [], set()
    for path in sorted((ROOT / "eval" / "images").glob("*.jp*g")):
        data = path.read_bytes()
        sha.add(hashlib.sha256(data).hexdigest())
        hashes.append(dhash(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")))
    with open(ROOT / "eval" / "labels.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            match = re.search(r"файл: (.+?\.\w+) \(", row["source"])  # «файл: папка/имя.jpg (уменьшено…»
            if match:
                names.add(match.group(1).split("/")[-1].lower())
    return sha, hashes, names


# --- Главная часть --------------------------------------------------------------------------
def collect():
    """Собрать список фото всех источников: (источник, класс, имя, байты, часть или None, байты для sha)."""
    # Скачиваем всё, чего нет
    for key in ("mendeley_bd", "riceleafbd"):
        info = SOURCES[key]
        download(lambda: mendeley_files(info["mendeley_id"])[info["zip"]], RAW / key / "original.zip")
    download(SOURCES["uci"]["url"], RAW / "uci" / "uci.zip")

    for source in SOURCES:
        if source == "paddy_doctor":
            items = read_paddy()
            for cls, name, data, part, sha_data in items:
                yield source, cls, name, data, part, sha_data
        elif source == "openimages":
            for cls, name, data, part, sha_data in read_openimages():
                yield source, cls, name, data, part, sha_data
        elif source == "dhan_shomadhan":
            for cls, name, data in read_dhan():
                yield source, cls, name, data, None, data
        else:
            zip_name = "uci.zip" if source == "uci" else "original.zip"
            for cls, name, data in read_zip(RAW / source / zip_name, SOURCES[source]["mapping"]):
                yield source, cls, name, data, None, data


def main():
    eval_sha, eval_hashes, eval_names = eval_fingerprints()
    print(f"Фото в eval/images: {len(eval_hashes)} (их в обучение не берём)")

    rnd = random.Random(SEED)
    seen = []  # (отпечаток, источник) уже принятых фото - для поиска дублей между источниками
    rows = []
    skipped = {"eval_sha": 0, "eval_hash": 0, "eval_name": 0, "duplicate": 0, "broken": 0}
    by_group = {}  # (источник, класс) -> список строк, чтобы потом разделить на train / val

    if OUT.exists():  # при повторном запуске начинаем с чистого листа
        import shutil
        shutil.rmtree(OUT)

    for source, cls, name, data, part, sha_data in collect():
        sha = hashlib.sha256(sha_data).hexdigest()
        if sha in eval_sha:
            skipped["eval_sha"] += 1
            continue
        if source != "paddy_doctor" and name.split("/")[-1].lower() in eval_names:
            skipped["eval_name"] += 1
            continue
        try:
            img = load_small(data)
        except Exception as error:  # битый файл
            print("  пропуск", name, error)
            skipped["broken"] += 1
            continue
        h = dhash(img)
        if any(distance(h, e) <= EVAL_DISTANCE for e in eval_hashes):
            skipped["eval_hash"] += 1
            print("  похоже на eval-фото, пропуск:", source, name)
            continue
        if any(distance(h, s) <= DUPLICATE_DISTANCE and src != source for s, src in seen):
            skipped["duplicate"] += 1
            continue
        seen.append((h, source))

        folder = OUT / source / cls
        folder.mkdir(parents=True, exist_ok=True)
        file = f"{sha[:16]}.jpg"
        img.save(folder / file, quality=95)
        row = {"file": f"{source}/{cls}/{file}", "source": source, "class": cls,
               "license": SOURCES[source]["license"], "citation": SOURCES[source]["citation"],
               "original_name": name, "sha256": sha,
               # бывшая проверочная часть Paddy / Open Images -> test (на ней считаем итоговую точность)
               "part": "test" if part == "val" else (part or "")}
        rows.append(row)
        by_group.setdefault((source, cls), []).append(row)

    # Остальные фото делим на обучение (85%) и проверку при обучении (15%): по проверке выбираем
    # лучшую эпоху. Часть test (бывшая проверка Paddy / Open Images) не трогаем.
    for group in by_group.values():
        pool = [r for r in group if r["part"] in ("", "train")]
        rnd.shuffle(pool)
        n_val = round(len(pool) * VAL_PART)
        for i, r in enumerate(pool):
            r["part"] = "val" if i < n_val else "train"

    with open(SOURCES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Итоговая таблица
    print("\nПропущено:", skipped)
    print(f"\n{'источник':<16}{'класс':<18}{'всего':>7}{'train':>7}{'val':>6}{'test':>6}")
    for (source, cls), group in sorted(by_group.items()):
        counts = {p: sum(r["part"] == p for r in group) for p in ("train", "val", "test")}
        print(f"{source:<16}{cls:<18}{len(group):>7}{counts['train']:>7}{counts['val']:>6}{counts['test']:>6}")
    print(f"\nВсего фото: {len(rows)}. Список: {SOURCES_CSV}")


if __name__ == "__main__":
    main()
