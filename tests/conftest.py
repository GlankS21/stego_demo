import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.features import load_gray  # noqa: E402


@pytest.fixture(scope="session")
def cover():
    return load_gray(ROOT / "samples" / "cameraman.png", 256)


@pytest.fixture(scope="session")
def small_cover():
    return load_gray(ROOT / "samples" / "troi_va_co.png", 128)


@pytest.fixture(scope="session")
def saturated_cover():
    """Ảnh có nhiều pixel 0 và 255 để kiểm tra không tràn."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (128, 128)).astype(np.uint8)
    img[:40] = 255
    img[40:80, :64] = 0
    img[100:, 100:] = rng.choice([0, 255], size=(28, 28)).astype(np.uint8)
    return img
