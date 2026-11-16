"""
Проверка predict(): модель подменена «пустышкой», настоящая модель и сеть не нужны.
"""

import io

import numpy as np
import pytest
from PIL import Image

from bot import handlers
from model import predict as model_predict

CLASSES = ["healthy", "blast", "brown_spot", "bacterial_blight"]


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
    model = FakeModel([0.02, 0.91, 0.04, 0.03])
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
    fake_model.probabilities = [0.30, 0.35, 0.20, 0.15]  # лучший вариант всего 35%
    ru = handlers.answer_for_photo(make_jpeg_bytes(), "ru")
    kk = handlers.answer_for_photo(make_jpeg_bytes(), "kk")
    assert "Не уверен" in ru and "Диагноз" not in ru
    assert "Сенімді емеспін" in kk and "Диагноз" not in kk


def test_high_confidence_gives_diagnosis(fake_model):
    ru = handlers.answer_for_photo(make_jpeg_bytes(), "ru")
    assert "Пирикуляриоз" in ru and "91%" in ru
    kk = handlers.answer_for_photo(make_jpeg_bytes(), "kk")
    assert "Пирикуляриоз" in kk and "91%" in kk
