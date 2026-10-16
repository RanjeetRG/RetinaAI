import cv2
import numpy as np
import pytest

import config
from preprocessing import fundus as F
from preprocessing.quality import UploadError, check_quality, validate_upload
from tests.conftest import encode, make_fundus


def test_crop_border_removes_black_margin(fundus):
    cropped = F.crop_border(fundus)
    # The disc has radius 276 in a 600x800 frame: the crop must drop most of the 800 px width
    assert cropped.shape[1] < 600
    assert abs(cropped.shape[0] - cropped.shape[1]) < 10


def test_crop_border_keeps_blank_image():
    blank = np.zeros((100, 120, 3), np.uint8)
    assert F.crop_border(blank).shape == blank.shape


def test_pad_to_square_centres_image():
    img = np.full((100, 60, 3), 255, np.uint8)
    sq = F.pad_to_square(img)
    assert sq.shape == (100, 100, 3)
    assert sq[:, :20].max() == 0 and sq[:, 80:].max() == 0


def test_load_fundus_shape(fundus):
    out = F.load_fundus(fundus)
    assert out.shape == (config.CACHE_SIZE, config.CACHE_SIZE, 3) and out.dtype == np.uint8


def test_read_image_from_bytes_and_unicode_path(fundus, tmp_path):
    data = encode(fundus)
    path = tmp_path / "रेटिना_फोटो.png"  # non-ASCII Windows path
    path.write_bytes(data)
    assert np.array_equal(F.read_image(data), fundus)
    assert np.array_equal(F.read_image(path), fundus)


def test_read_image_rejects_garbage():
    with pytest.raises(ValueError):
        F.read_image(b"not an image")


def test_model_inputs(fundus):
    cached = F.load_fundus(fundus)
    cnn = F.prepare_cnn(cached)
    gan = F.prepare_gan(cached)
    assert cnn.shape == (config.CNN_SIZE, config.CNN_SIZE, 3) and cnn.dtype == np.uint8
    assert gan.shape == (config.GAN_SIZE, config.GAN_SIZE, 3) and gan.dtype == np.float32
    assert -1.0 <= gan.min() and gan.max() <= 1.0
    # The surround stays black after CLAHE
    assert cnn[0, 0].max() == 0 and gan[0, 0].max() == -1.0


def test_full_and_clipped_fields_get_the_same_shape(fundus):
    """Cameras that clip the field top and bottom are commoner at higher stages; after preprocessing the field
    shape must not reveal the camera (the first 0->1 generator learned to clip fields)."""
    diameter = int(fundus.shape[0] * 0.92)
    centre = fundus.shape[0] // 2
    keep = int(diameter * 0.85 / 2)  # clipped field: 85% of the diameter tall, more than the 0.75 template
    clipped = fundus.copy()
    clipped[: centre - keep] = 0
    clipped[centre + keep :] = 0

    def lit(img):
        return F.prepare_cnn(F.load_fundus(img)).max(axis=-1) > 0

    a, b = lit(fundus), lit(clipped)
    assert (a & b).sum() / (a | b).sum() > 0.97
    rows = np.where(a.any(axis=1))[0]
    assert (rows[-1] - rows[0] + 1) / config.CNN_SIZE == pytest.approx(config.FIELD_HEIGHT_RATIO, abs=0.03)


def test_preprocessing_is_deterministic(fundus):
    """Training and inference call the same functions, so the same file must give identical inputs."""
    a = F.prepare_cnn(F.load_fundus(encode(fundus)))
    b = F.prepare_cnn(F.load_fundus(encode(fundus)))
    assert np.array_equal(a, b)


def test_gan_range_round_trip(fundus):
    img = F.resize(F.pad_to_square(fundus), 64)
    assert np.array_equal(F.from_gan_range(F.to_gan_range(img)), img)


def test_quality_accepts_good_fundus(fundus):
    report = check_quality(F.load_fundus(fundus))
    assert report.ok, report.issues


def test_quality_flags_blank():
    report = check_quality(np.zeros((512, 512, 3), np.uint8))
    assert "blank" in report.issues and report.unusable


def test_quality_flags_blur(fundus):
    blurred = cv2.GaussianBlur(F.load_fundus(fundus), (0, 0), 6)
    assert "blurry" in check_quality(blurred).issues


def test_quality_flags_non_fundus_colours(fundus):
    blue = F.load_fundus(fundus)[..., ::-1].copy()  # swap red and blue
    assert "not_colour_fundus" in check_quality(blue).issues


def test_quality_flags_ordinary_photo():
    """A 4:3 photo gets black bars from padding, but its corners are lit, unlike a fundus disc."""
    rng = np.random.default_rng(0)
    photo = np.clip(rng.normal((180, 120, 90), 30, (300, 400, 3)), 0, 255).astype(np.uint8)
    assert "no_border" in check_quality(F.load_fundus(photo)).issues


def test_quality_accepts_clipped_fundus(fundus):
    """EyePACS often clips the top and bottom of the disc; its corners are still dark."""
    clipped = fundus[60:-60]
    assert "no_border" not in check_quality(F.load_fundus(clipped)).issues


@pytest.mark.parametrize("name", ["scan.gif", "scan.bmp", "notes.txt"])
def test_upload_rejects_other_types(name, fundus):
    with pytest.raises(UploadError):
        validate_upload(name, encode(fundus))


def test_upload_rejects_large_files():
    with pytest.raises(UploadError):
        validate_upload("big.jpg", b"\0" * (config.MAX_UPLOAD_MB * 1024 * 1024 + 1))


def test_upload_rejects_undecodable():
    with pytest.raises(UploadError):
        validate_upload("photo.jpg", b"definitely not a jpeg")


def test_upload_accepts_jpg(fundus):
    img = validate_upload("Photo.JPG", encode(fundus, ".jpg"))
    assert img.shape == fundus.shape


def test_different_synthetic_images_differ():
    assert not np.array_equal(make_fundus(1), make_fundus(2))
