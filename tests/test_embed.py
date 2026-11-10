"""Nhúng / trích xuất qua đĩa, đối chiếu bảng thay đổi, chi phí, các trường hợp âm."""
import io

import numpy as np
import pytest
from PIL import Image

from core.costs import percentile_rank
from core.embed import (CapacityError, capacity_bytes, embed, extract, extract_from_file, lsb_sequential,
                        png_bytes, read_image_file, save_png, to_jpeg_bytes)

KEY = "khoa-test-🔑"
MESSAGES = {
    "tieng_viet": "Tiếng Việt có dấu: Đường đi khó không vì ngăn sông cách núi. Ứng dụng ổn định!",
    "emoji": "Emoji 😀🔐🌏👩‍💻 và ký hiệu ∑ρ·|x−y| 中文",
    "rong": "",
    "mot_ky_tu": "a",
}


def _roundtrip(tmp_path, cover, message, key=KEY, h=10, cost="HILL"):
    res = embed(cover, message, key, h, cost)
    p = tmp_path / "stego.png"
    save_png(res.stego, p)
    disk = read_image_file(p)          # đọc lại từ ĐĨA
    out = extract_from_file(p, key)    # trích xuất chỉ từ file + khoá
    return res, disk, out


@pytest.mark.parametrize("name", list(MESSAGES))
def test_roundtrip_disk_messages(tmp_path, cover, name):
    res, disk, out = _roundtrip(tmp_path, cover, MESSAGES[name])
    assert out.ok, out.reasons
    assert out.message == MESSAGES[name]
    assert out.checksum_ok


@pytest.mark.parametrize("h", [7, 8, 9, 10])
@pytest.mark.parametrize("cost", ["HILL", "WOW-like", "Đồng đều"])
def test_roundtrip_h_and_costs(tmp_path, small_cover, h, cost):
    msg = f"h={h}, cost={cost}, xin chào 👋"
    res, disk, out = _roundtrip(tmp_path, small_cover, msg, h=h, cost=cost)
    assert out.ok and out.message == msg and out.h == h


def test_roundtrip_near_capacity(tmp_path, cover):
    cap = capacity_bytes(cover.shape)
    msg = ("Tiếng Việt " * 1000).encode("utf-8")[: cap - 2].decode("utf-8", errors="ignore")
    msg += "x" * (cap - len(msg.encode("utf-8")))           # đúng bằng capacity
    assert len(msg.encode("utf-8")) == cap
    res, disk, out = _roundtrip(tmp_path, cover, msg, h=8)
    assert out.ok and out.message == msg


def test_change_table_matches_disk_file(tmp_path, cover, saturated_cover):
    for img in (cover, saturated_cover):
        res, disk, out = _roundtrip(tmp_path, img, "Kiểm tra bảng thay đổi " * 10)
        n_table = len(res.changes["hàng"])
        diff = disk.astype(int) - img.astype(int)
        assert n_table == np.count_nonzero(diff) == res.n_changes
        # cùng vị trí, cùng dấu, cùng giá trị mới
        rr, cc = res.changes["hàng"], res.changes["cột"]
        assert np.array_equal(diff[rr, cc], res.changes["±1"])
        assert np.array_equal(res.changes["giá trị mới"], disk[rr, cc])
        assert np.abs(diff).max() <= 1
        assert disk.min() >= 0 and disk.max() <= 255
        # pixel 0 chỉ được +1, pixel 255 chỉ được −1
        assert not np.any((img == 255) & (diff > 0))
        assert not np.any((img == 0) & (diff < 0))
        # bản đồ thay đổi từ STC = vị trí y ≠ x
        assert np.array_equal(res.change_mask, diff != 0)
        assert out.ok


def test_stc_cost_not_more_than_sequential_lsb(cover):
    for cost in ("HILL", "WOW-like", "Đồng đều"):
        for msg in ("ngắn", "Thông điệp dài hơn " * 30, "z" * capacity_bytes(cover.shape)):
            res = embed(cover, msg, KEY, 9, cost)
            bits = np.concatenate([res.payload.header_enc, res.payload.payload_enc])
            _, mask = lsb_sequential(cover, bits, KEY)
            lsb_cost = float(res.costs.rho[mask].sum())
            assert res.stc_cost == pytest.approx(float(res.costs.rho[res.change_mask].sum()))
            assert res.stc_cost <= lsb_cost + 1e-9


@pytest.mark.parametrize("cost", ["HILL", "WOW-like"])
def test_cheap_pixels_change_more(cover, cost):
    res = embed(cover, "Phân vị chi phí " * 40, KEY, 10, cost)
    pr = percentile_rank(res.costs.rho)
    low = res.change_mask[pr < 20].mean()
    high = res.change_mask[pr >= 80].mean()
    assert low > high
    assert low > 5 * max(high, 1e-9) or high == 0


def test_wrong_key_fails_safely(cover):
    res = embed(cover, "bí mật", KEY, 10)
    for k in ("", "khoa-sai", KEY + " ", "🔑"):
        out = extract(res.stego, k)
        assert not out.ok and out.message is None


def test_jpeg_fails_safely(cover):
    res = embed(cover, "bí mật qua JPEG", KEY, 10)
    for q in (100, 95, 75):
        out = extract_from_file(to_jpeg_bytes(res.stego, q), KEY)
        assert not out.ok
        assert not out.checksum_ok or not out.magic_ok


def test_one_pixel_modified_detected(cover):
    msg = "Sửa một pixel phải bị phát hiện"
    res = embed(cover, msg, KEY, 10)
    rng = np.random.default_rng(5)
    positions = [(128, 128), (0, 0), (255, 255)] + [tuple(rng.integers(0, 256, 2)) for _ in range(30)]
    for r, c in positions:
        mod = res.stego.copy()
        mod[r, c] = mod[r, c] + 1 if mod[r, c] < 255 else mod[r, c] - 1
        out = extract_from_file(png_bytes(mod), KEY)
        assert not out.ok, (r, c)


def test_garbage_inputs_do_not_crash():
    assert not extract_from_file(b"not an image", KEY).ok
    noise = np.random.default_rng(1).integers(0, 256, (64, 64)).astype(np.uint8)
    assert not extract(noise, KEY).ok
    assert not extract(np.zeros((10, 10), np.uint8), KEY).ok
    rgb = io.BytesIO()
    Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(rgb, format="PNG")
    assert not extract_from_file(rgb.getvalue(), KEY).ok


def test_capacity_exceeded_raises_before_embedding(cover):
    cap = capacity_bytes(cover.shape)
    with pytest.raises(CapacityError, match="vượt quá dung lượng"):
        embed(cover, "x" * (cap + 1), KEY, 10)
    with pytest.raises(CapacityError):
        embed(cover, "ệ" * cap, KEY, 10)      # 3 byte / ký tự
