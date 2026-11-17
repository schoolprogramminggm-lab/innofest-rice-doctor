"""
Проверка predict(): модель подменена «пустышкой», настоящая модель и сеть не нужны.
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bot import handlers, texts
from model import predict as model_predict

CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight", "not_rice"]


class FakeModel:
    """Притворяется моделью: всегда возвращает заранее заданные вероятности."""

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0

    def predict(self, batch, verbose=0):
        self.calls += 1
        return np.array([self.probabilities])


def make_jpeg_bytes(size=(640, 480)):
    """Маленькое фото в памяти (файл на диск не пишем)."""
    buffer = io.BytesIO()
    Image.new("RGB", size, (30, 160, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def fake_model(monkeypatch):
    """Подставляем пустышку вместо настоящей модели и обходим tensorflow при подготовке фото."""
    model = FakeModel([0.02, 0.91, 0.03, 0.03, 0.01])  # 91% blast
    monkeypatch.setattr(model_predict, "_model", model)
    monkeypatch.setattr(model_predict, "_classes", CLASSES)
    monkeypatch.setattr(model_predict, "load_photo", lambda image_bytes: np.zeros((1, 224, 224, 3)))
    return model


def test_predict_returns_best_class_and_confidence(fake_model):
    label, confidence = model_predict.predict(make_jpeg_bytes())
    assert label == "blast"
    assert confidence == pytest.approx(0.91)


def test_model_is_loaded_once(fake_model):
    # load_model() не должна перезагружать уже загруженную модель
    assert model_predict.load_model() == (fake_model, CLASSES)
    model_predict.predict(make_jpeg_bytes())
    model_predict.predict(make_jpeg_bytes())
    assert model_predict.load_model()[0] is fake_model
    assert fake_model.calls == 2


def test_photo_is_prepared_224_and_0_255():
    pytest.importorskip("tensorflow")  # этот тест использует tensorflow для сжатия, но без модели
    batch = model_predict.load_photo(make_jpeg_bytes())
    assert batch.shape == (1, 224, 224, 3)
    assert 0 <= batch.min() and batch.max() <= 255  # нормализация внутри модели, не здесь


def test_low_confidence_gives_not_sure_answer(fake_model):
    fake_model.probabilities = [0.30, 0.30, 0.20, 0.10, 0.10]  # лучший вариант всего 30%
    ru, ru_label = handlers.answer_for_photo(make_jpeg_bytes(), "ru")
    kk, kk_label = handlers.answer_for_photo(make_jpeg_bytes(), "kk")
    assert "Не уверен" in ru and "Диагноз" not in ru
    assert "Сенімді емеспін" in kk and "Диагноз" not in kk
    assert ru_label is None and kk_label is None  # кнопки «Подробнее» нет


def test_high_confidence_gives_diagnosis(fake_model):
    ru, label = handlers.answer_for_photo(make_jpeg_bytes(), "ru")
    assert "Пирикуляриоз" in ru and "91%" in ru
    assert label == "blast"  # для кнопки «Подробнее»
    kk, _ = handlers.answer_for_photo(make_jpeg_bytes(), "kk")
    assert "Пирикуляриоз" in kk and "91%" in kk


def test_not_rice_gives_no_diagnosis(fake_model):
    # На фото кот: сеть уверена, что это not_rice. Диагноза нет, кнопки «Подробнее» нет.
    fake_model.probabilities = [0.01, 0.02, 0.01, 0.01, 0.95]
    assert model_predict.predict(make_jpeg_bytes()) == ("not_rice", pytest.approx(0.95))
    ru, ru_label = handlers.answer_for_photo(make_jpeg_bytes(), "ru")
    kk, kk_label = handlers.answer_for_photo(make_jpeg_bytes(), "kk")
    assert ru == "Это не похоже на лист риса. Сфотографируйте лист ближе."
    assert "Диагноз" not in ru and "%" not in ru
    assert kk == texts.get_text("not_rice", "kk")
    assert "Диагноз" not in kk and "%" not in kk
    assert ru_label is None and kk_label is None


def test_not_rice_wins_even_if_a_disease_has_a_high_probability(fake_model):
    # not_rice 55% >= порога 50%, хотя blast тоже набрал 40%: диагноз не ставим
    fake_model.probabilities = [0.02, 0.40, 0.02, 0.01, 0.55]
    assert model_predict.predict(make_jpeg_bytes())[0] == "not_rice"


def test_below_not_rice_threshold_best_disease_is_used(fake_model, monkeypatch):
    # not_rice всего 45% (порог 50%): берём лучшую из болезней, а не not_rice
    monkeypatch.setattr(model_predict, "NOT_RICE_THRESHOLD", 0.50)
    fake_model.probabilities = [0.02, 0.40, 0.08, 0.05, 0.45]
    label, confidence = model_predict.predict(make_jpeg_bytes())
    assert label == "blast" and confidence == pytest.approx(0.40)


def test_not_rice_threshold_comes_from_one_variable(fake_model, monkeypatch):
    # Меняем одну переменную NOT_RICE_THRESHOLD, и ответ меняется
    fake_model.probabilities = [0.02, 0.28, 0.10, 0.05, 0.55]
    monkeypatch.setattr(model_predict, "NOT_RICE_THRESHOLD", 0.60)
    assert model_predict.predict(make_jpeg_bytes())[0] == "blast"
    monkeypatch.setattr(model_predict, "NOT_RICE_THRESHOLD", 0.50)
    assert model_predict.predict(make_jpeg_bytes())[0] == "not_rice"


def test_old_model_without_not_rice_still_works(fake_model, monkeypatch):
    # Если у модели 4 класса (старая), ответ выбирается как раньше
    monkeypatch.setattr(model_predict, "_classes", CLASSES[:4])
    fake_model.probabilities = [0.02, 0.91, 0.04, 0.03]
    assert model_predict.predict(make_jpeg_bytes())[0] == "blast"


def test_confidence_9999_is_shown_as_99_percent(fake_model):
    fake_model.probabilities = [0.00005, 0.9999, 0.00002, 0.00002, 0.00001]
    ru, _ = handlers.answer_for_photo(make_jpeg_bytes(), "ru")
    kk, _ = handlers.answer_for_photo(make_jpeg_bytes(), "kk")
    assert "99%" in ru and "100%" not in ru
    assert "99%" in kk and "100%" not in kk


def test_not_rice_threshold_is_read_in_one_place():
    # Порог читается из .env только в model/predict.py: остальные файлы его импортируют
    root = Path(__file__).resolve().parent.parent
    places = [path.name for folder in ("bot", "model") for path in (root / folder).glob("*.py")
              if 'getenv("NOT_RICE_THRESHOLD"' in path.read_text(encoding="utf-8")]
    assert places == ["predict.py"]
