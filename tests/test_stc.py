"""STC: H·y = m, tối ưu (so với vét cạn), khớp bản ghi vết trellis."""
from itertools import product

import numpy as np
import pytest

from core.stc import block_widths, generate_hhat, stc_embed, stc_embed_trace, syndrome


def _case(rng, h):
    n = int(rng.integers(64, 3000))
    alpha = float(rng.choice([0.02, 0.1, 0.25, 0.5, 0.9, 1.0]))
    m = max(1, int(n * alpha))
    x = rng.integers(0, 2, n).astype(np.uint8)
    rho = rng.random(n) * 10 + 1e-3
    msg = rng.integers(0, 2, m).astype(np.uint8)
    w = block_widths(n, m)
    cols = generate_hhat(h, int(w.max()), int(rng.integers(1 << 30)))
    return x, rho, msg, cols, w


@pytest.mark.parametrize("h", [1, 2, 3, 5, 7, 8, 9, 10])
def test_syndrome_equals_message_many_random(h):
    rng = np.random.default_rng(1000 + h)
    for _ in range(15):          # 8 giá trị h × 15 = 120 thông điệp ngẫu nhiên
        x, rho, msg, cols, w = _case(rng, h)
        r = stc_embed(x, rho, msg, h, cols, w)
        assert np.array_equal(syndrome(r.y, len(msg), h, cols, w), msg)
        assert r.cost == pytest.approx(float(np.sum(rho * (r.y != x))), rel=1e-9, abs=1e-9)


def test_stc_is_optimal_vs_bruteforce():
    rng = np.random.default_rng(7)
    for _ in range(30):
        n, m, h = 12, int(rng.integers(1, 6)), int(rng.integers(2, 5))
        x = rng.integers(0, 2, n).astype(np.uint8)
        rho = rng.random(n)
        msg = rng.integers(0, 2, m).astype(np.uint8)
        w = block_widths(n, m)
        cols = generate_hhat(h, int(w.max()), int(rng.integers(1 << 30)))
        r = stc_embed(x, rho, msg, h, cols, w)
        best = min(float(np.sum(rho * (np.array(y) != x))) for y in product((0, 1), repeat=n)
                   if np.array_equal(syndrome(np.array(y), m, h, cols, w), msg))
        assert r.cost == pytest.approx(best, abs=1e-12)


def test_trace_matches_fast_viterbi():
    rng = np.random.default_rng(3)
    for _ in range(20):
        x = rng.integers(0, 2, 16).astype(np.uint8)
        rho = rng.random(16)
        msg = rng.integers(0, 2, 4).astype(np.uint8)
        cols = generate_hhat(3, 4, int(rng.integers(1000)))
        a = stc_embed(x, rho, msg, 3, cols)
        b = stc_embed_trace(x, rho, msg, 3, cols)
        assert a.cost == pytest.approx(b.cost)
        assert np.array_equal(a.y, b.y)
        assert b.best_path[0] == 0 and b.best_path[-1] == 0


def test_empty_message_changes_nothing():
    x = np.array([0, 1, 1, 0], np.uint8)
    r = stc_embed(x, np.ones(4), np.zeros(0, np.uint8), 7, generate_hhat(7, 4, 0))
    assert np.array_equal(r.y, x) and r.cost == 0


def test_too_many_bits_rejected():
    with pytest.raises(ValueError):
        block_widths(10, 11)
