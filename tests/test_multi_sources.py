"""
Проверка помощников для обучения на нескольких источниках (#23): без настоящих данных и без обучения.
- download_sources: «отпечаток» фото (dHash) находит уменьшенную копию, но не другое фото;
  фото из eval/images/ распознаются как запрещённые для обучения (защита от утечки);
- train_multi: «однотонный фон» не меняет размер, лимит фото по (источник, класс) работает;
- predict.load_photo: EXIF-поворот применяется.
"""

import io
import random

import numpy as np
from PIL import Image

from model import download_sources, predict, train_multi


def smooth_photo(seed, size=(400, 300)):
    """Гладкая картинка из нескольких цветных пятен: ближе к настоящему фото, чем шум."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.BICUBIC)


def test_dhash_same_photo_resized_is_close():
    big = smooth_photo(1)
    small = big.resize((200, 150), Image.LANCZOS)  # eval-фото тоже были уменьшены
    assert download_sources.distance(download_sources.dhash(big), download_sources.dhash(small)) <= 3


def test_dhash_different_photos_are_far():
    a = download_sources.dhash(smooth_photo(1))
    b = download_sources.dhash(smooth_photo(2))
    assert download_sources.distance(a, b) > download_sources.EVAL_DISTANCE


def test_eval_photos_are_recognised_as_forbidden():
    """Любое фото из eval/images/ (и его уменьшенная копия) имеет близкий отпечаток к одному из eval-хешей."""
    sha, hashes, names = download_sources.eval_fingerprints()
    assert len(hashes) == 33 and len(sha) == 33
    assert "healthy_rice_leaf  (124).jpg" in names  # исходное имя из eval/labels.csv
    for path in sorted((download_sources.ROOT / "eval" / "images").glob("*.jpg"))[:5]:
        img = Image.open(path).convert("RGB")
        small = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        h = download_sources.dhash(small)
        assert min(download_sources.distance(h, e) for e in hashes) <= download_sources.EVAL_DISTANCE


def test_load_small_applies_exif_rotation():
    """Фото, снятое боком (EXIF Orientation = 6), читается повёрнутым."""
    img = Image.new("RGB", (300, 100), "green")
    exif = Image.Exif()
    exif[0x0112] = 6  # повернуть на 90° по часовой
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", exif=exif)
    rotated = download_sources.load_small(buffer.getvalue())
    assert rotated.size == (100, 256) or rotated.height > rotated.width  # стало «высоким»


def test_predict_load_photo_applies_exif_rotation():
    img = Image.new("RGB", (300, 100), "green")
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", exif=exif)
    batch = predict.load_photo(buffer.getvalue())
    assert batch.shape == (1, 224, 224, 3)


def test_plain_background_keeps_shape_and_puts_photo_on_solid_color(monkeypatch):
    image = np.full((224, 224, 3), 100.0, dtype=np.float32)
    monkeypatch.setattr(train_multi, "PLAIN_BACKGROUND_PART", 1.1)  # 1.1 > любого случайного числа: меняем всегда
    # tf.random.uniform([]) <= 1.0, поэтому условие "больше доли" не срабатывает и фон подменяется
    out, label = train_multi.plain_background(image, 3)
    out = out.numpy()
    assert out.shape == (224, 224, 3) and int(label) == 3
    assert (out != 100.0).any()  # часть кадра теперь цвет фона


def test_cap_per_group_limits_each_source_and_class():
    rows = ([{"file": f"a/healthy/{i}.jpg", "source": "a", "class": "healthy"} for i in range(10)]
            + [{"file": f"b/healthy/{i}.jpg", "source": "b", "class": "healthy"} for i in range(3)])
    chosen = train_multi.cap_per_group(rows, 4, random.Random(0))
    assert sum(r["source"] == "a" for r in chosen) == 4
    assert sum(r["source"] == "b" for r in chosen) == 3  # меньше лимита - берём все
    # повторный вызов с тем же seed даёт тот же набор
    again = train_multi.cap_per_group(rows, 4, random.Random(0))
    assert [r["file"] for r in chosen] == [r["file"] for r in again]


def test_accuracy_by_class():
    pairs = [("blast", "blast"), ("blast", "healthy"), ("healthy", "healthy")]
    assert train_multi.accuracy_by_class(pairs) == {"blast": (1, 2), "healthy": (1, 1)}
