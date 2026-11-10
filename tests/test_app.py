"""Smoke test giao diện Streamlit (chạy toàn bộ pipeline, không có exception)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_show_all_steps_no_exception():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    assert not at.exception
    next(b for b in at.button if b.label == "Xem toàn bộ").click().run()
    assert not at.exception
    labels = [m.value for m in at.markdown]
    assert any("KHỚP" in v for v in labels)                    # NT2: bảng STC ↔ file trên đĩa
    assert any("KHÔI PHỤC THÀNH CÔNG" in v for v in labels)     # bước 7


def test_app_capacity_error_shown():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.text_area[0].set_value("x" * 5000).run()
    assert not at.exception
    assert any("VƯỢT dung lượng" in e.value for e in at.error)
