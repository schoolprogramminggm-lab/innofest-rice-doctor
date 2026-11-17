"""
Тест с НАСТОЯЩЕЙ моделью на двух фото из tests/images/.
Если модели (models/rice_doctor.keras) или tensorflow нет, тесты пропускаются:
модель в репозиторий не загружается, на чужом компьютере её может не быть.

  rice_leaf.jpg - лист риса с пирикуляриозом: ожидаем диагноз;
  not_rice.jpg  - кошка: ожидаем «это не лист риса», без диагноза.

Источники и лицензии фото: tests/images/SOURCES.md
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "tests" / "images"
MODEL_FILE = ROOT / "models" / "rice_doctor.keras"
CLASSES_FILE = ROOT / "models" / "classes.json"


@pytest.fixture(scope="module")
def real_model():
    """Загрузить настоящую модель один раз для всех тестов файла (или пропустить тесты)."""
    if not MODEL_FILE.exists() or not CLASSES_FILE.exists():
        pytest.skip("нет models/rice_doctor.keras и models/classes.json")
    pytest.importorskip("tensorflow")
    if "not_rice" not in json.loads(CLASSES_FILE.read_text(encoding="utf-8")):
        pytest.skip("модель старая, без класса not_rice (см. README, раздел «Установка»)")

    from model import predict
    # пути в predict.py относительные (от корня проекта), поэтому задаём их полностью
    predict.MODEL_PATH = str(MODEL_FILE)
    predict.CLASSES_PATH = str(CLASSES_FILE)
    predict.load_model()
    return predict


def test_rice_leaf_gets_a_diagnosis(real_model):
    from bot import handlers

    image = (IMAGES / "rice_leaf.jpg").read_bytes()
    label, confidence = real_model.predict(image)
    assert label in ("healthy", "blast", "brown_spot", "bacterial_blight")
    assert 0 <= confidence <= 1

    text, button_label = handlers.answer_for_photo(image, "ru")
    assert "Диагноз" in text
    assert "не лист риса" not in text and "Это не похоже на лист риса" not in text
    assert "100%" not in text
    assert button_label == label  # у ответа с диагнозом есть кнопка «Подробнее»


def test_not_a_leaf_is_rejected(real_model):
    from bot import handlers

    image = (IMAGES / "not_rice.jpg").read_bytes()
    label, _ = real_model.predict(image)
    assert label == "not_rice"

    for lang, phrase in (("ru", "Это не похоже на лист риса"), ("kk", "Бұл күріш жапырағына ұқсамайды")):
        text, button_label = handlers.answer_for_photo(image, lang)
        assert phrase in text
        assert "Диагноз" not in text
        assert button_label is None  # кнопки «Подробнее» нет
