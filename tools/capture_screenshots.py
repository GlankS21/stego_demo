import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8599"


def settle(page, ms=2500):
    page.wait_for_timeout(500)
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=60000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def shot(page, name):
    page.screenshot(path=str(OUT / name), full_page=True)
    print("saved", name)


def nxt(page, n=1):
    for _ in range(n):
        page.get_by_role("button", name="Bước tiếp theo ▶").click()
        settle(page)


with sync_playwright() as p:
    try:
        b = p.chromium.launch(channel="msedge")
    except Exception:
        b = p.chromium.launch(channel="chrome")
    page = b.new_page(viewport={"width": 1500, "height": 2600}, device_scale_factor=1)
    page.goto(URL)
    page.get_by_role("button", name="Bước tiếp theo ▶").wait_for(timeout=120000)
    settle(page, 4000)
    shot(page, "01_buoc1_anh_goc.png")
    nxt(page)
    shot(page, "02_buoc2_dac_trung.png")
    nxt(page)
    shot(page, "03_buoc3_ban_do_Q.png")
    nxt(page)
    # NT1: click vào vùng cỏ (phía dưới ảnh cameraman) trên ảnh truy vết
    frame_el = page.locator("iframe[title*='streamlit_image_coordinates']").first
    frame_el.scroll_into_view_if_needed()
    box = frame_el.bounding_box()
    if box:
        page.mouse.click(box["x"] + box["width"] * 0.30, box["y"] + box["height"] * 0.88)
        settle(page, 3000)
    shot(page, "04_NT1_truy_vet_pixel_co.png")
    # pixel bầu trời
    frame_el = page.locator("iframe[title*='streamlit_image_coordinates']").first
    box = frame_el.bounding_box()
    if box:
        page.mouse.click(box["x"] + box["width"] * 0.80, box["y"] + box["height"] * 0.10)
        settle(page, 3000)
    shot(page, "05_NT1_truy_vet_pixel_bau_troi.png")
    nxt(page)
    shot(page, "06_buoc5_STC_trellis.png")
    nxt(page)
    shot(page, "07_NT2_pixel_thay_doi.png")
    page.get_by_role("tab", name="NT3 · Khác cover bao nhiêu").click()
    settle(page)
    shot(page, "08_NT3_khac_biet.png")
    page.get_by_role("tab", name="NT1 · Vì sao vùng này").click()
    settle(page)
    page.get_by_role("button", name="⚖️ So sánh: thích nghi + STC vs LSB tuần tự vs chi phí đồng đều").click()
    settle(page, 5000)
    page.get_by_role("tab", name="NT1 · Vì sao vùng này").click()
    settle(page)
    shot(page, "09_NT1_vung_va_so_sanh.png")
    nxt(page)
    shot(page, "10_buoc7_trich_xuat.png")
    # Người nhận
    page.get_by_role("tab", name="📥 Người nhận").click()
    settle(page)
    page.get_by_text("Đọc file trên đĩa", exact=False).first.click()
    settle(page)
    page.get_by_placeholder("nhập khoá bí mật").fill("khoa-bi-mat-2026")
    page.keyboard.press("Enter")
    settle(page)
    import hashlib
    sha = hashlib.sha256(("Xin chào! Đây là thông điệp bí mật 🔐 được giấu bằng HILL + STC. "
                          "Tiếng Việt có dấu vẫn khôi phục chính xác.").encode()).hexdigest()
    page.locator("input[aria-label^='SHA-256']:visible").first.fill(sha)
    page.keyboard.press("Enter")
    settle(page)
    page.get_by_role("button", name="🔓 Trích xuất").click()
    settle(page, 4000)
    page.get_by_role("button", name="Chạy 3 trường hợp âm").click()
    settle(page, 5000)
    shot(page, "11_NT4_nguoi_nhan_va_truong_hop_am.png")
    b.close()
