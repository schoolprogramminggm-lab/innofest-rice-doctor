"""
Защита от утечки (#23): 33 фото из eval/images/ не должны быть в обучающем наборе.
Проверяем по data/sources.csv (он лежит в git, сами фото в git не входят), поэтому тест работает без фото.
"""

import csv
import hashlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources.csv"


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.skipif(not SOURCES.exists(), reason="нет data/sources.csv (python model/download_sources.py)")
def test_eval_photos_are_not_in_training_set():
    rows = read_csv(SOURCES)
    train_hashes = {r["sha256"] for r in rows}
    train_names = {r["original_name"].split("/")[-1].lower() for r in rows}

    # 1) хеш файла: ни один eval-файл не совпадает по sha256 с обучающим фото
    for image in sorted((ROOT / "eval" / "images").glob("*.jp*g")):
        assert hashlib.sha256(image.read_bytes()).hexdigest() not in train_hashes, image.name

    # 2) исходное имя файла из eval/labels.csv (там записано, откуда взято фото)
    for row in read_csv(ROOT / "eval" / "labels.csv"):
        match = re.search(r"файл: (.+?\.\w+) \(", row["source"])
        if match:
            name = match.group(1).split("/")[-1].lower()
            assert name not in train_names, f"{row['file']} ({name}) есть в обучающем наборе"


@pytest.mark.skipif(not SOURCES.exists(), reason="нет data/sources.csv (python model/download_sources.py)")
def test_every_photo_has_source_license_and_class():
    rows = read_csv(SOURCES)
    assert len(rows) > 0
    sources = {r["source"] for r in rows}
    assert len(sources) >= 3  # задача #23: не меньше 3 источников
    for r in rows:
        assert r["license"] and r["citation"] and r["sha256"]
        assert r["class"] in {"healthy", "blast", "brown_spot", "bacterial_blight", "not_rice"}
        assert r["part"] in {"train", "val", "test"}
