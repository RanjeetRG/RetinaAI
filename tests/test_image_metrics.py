import numpy as np
import pytest

from preprocessing.fundus import load_fundus, resize
from tests.conftest import make_fundus
from utils.image_metrics import difference_map, psnr, ssim


@pytest.fixture
def img():
    return resize(load_fundus(make_fundus()), 256)


def test_identical_images(img):
    assert ssim(img, img) == pytest.approx(1.0)
    assert psnr(img, img) == float("inf")


def test_noise_lowers_similarity(img):
    rng = np.random.default_rng(0)
    noisy = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
    noisier = np.clip(img + rng.normal(0, 40, img.shape), 0, 255).astype(np.uint8)
    assert ssim(img, noisier) < ssim(img, noisy) < 1.0
    assert psnr(img, noisier) < psnr(img, noisy) < 40


def test_black_surround_does_not_inflate_ssim(img):
    """A changed fundus must score the same whatever the size of the identical black border."""
    altered = img.copy()
    altered[img.sum(axis=-1) > 0] //= 2
    assert ssim(img, altered) < 0.9


def test_difference_map_is_zero_outside_fundus(img):
    rng = np.random.default_rng(1)
    other = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    diff = difference_map(img, other)
    assert diff.shape == img.shape[:2]
    assert diff[0, 0] == 0 and diff[128, 128] > 0


def test_fid_and_kid():
    pytest.importorskip("keras")
    from gan.gan_metrics import fid, kid

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (400, 16))
    b = rng.normal(0, 1, (400, 16))
    shifted = rng.normal(1.5, 1, (400, 16))
    assert fid(a, b) < 1.0 < fid(a, shifted)
    assert abs(kid(a, b)) < 0.05 < kid(a, shifted)


def test_change_decomposition_separates_global_and_local_change(img):
    from utils.image_metrics import change_decomposition

    inside = img.max(axis=-1) > 25
    darker = img.astype(np.int16)
    darker[inside] -= 10  # exposure change only
    darker = np.clip(darker, 0, 255).astype(np.uint8)
    g, loc = change_decomposition(img[None], darker[None])
    assert g == pytest.approx(10, abs=0.5) and loc < 1.0

    spotted = img.copy()
    spotted[112:144, 112:144] = (150, 20, 20)  # a local, lesion-like change
    g, loc = change_decomposition(img[None], spotted[None])
    # A local change also nudges the image mean, so local is about 2x global here; a pure exposure change gives ~0
    assert loc > 1.5 * g


def test_change_map_correlation_flags_fixed_position_stamps():
    from tests.conftest import make_fundus
    from utils.image_metrics import change_map_correlation

    inputs = np.stack([resize(load_fundus(make_fundus(seed)), 256) for seed in range(6)])
    rng = np.random.default_rng(0)

    stamped = inputs.copy()
    stamped[:, 60:66, 90:96] = (150, 20, 20)  # the same dot at the same place in every eye
    varied = inputs.copy()
    for img in varied:  # one dot per eye, at a different place each time
        y, x = rng.integers(70, 180, 2)
        img[y : y + 6, x : x + 6] = (150, 20, 20)

    assert change_map_correlation(inputs, stamped) > 0.9
    assert change_map_correlation(inputs, varied) < 0.2
