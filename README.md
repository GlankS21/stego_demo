# Demo trực quan: giấu tin thích nghi trên ảnh xám 8-bit

Pipeline: **Ảnh gốc → Đặc trưng → Q → Chi phí ρ → STC → Ảnh stego → Trích xuất**

Chương trình chạy **thuật toán nhúng thật**: chi phí HILL / WOW-like, Syndrome-Trellis Codes
cài bằng Viterbi (tự viết, không dùng thư viện steganography), LSB matching ±1, header + CRC32,
mã hoá dòng SHA-256 và hoán vị pixel theo khoá.

## Cài đặt

```bash
cd stego_demo
py -3.11 -m venv .venv            # Linux/macOS: python3.11 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Chạy kiểm thử: `python -m pytest -v`. Báo cáo lần chạy gần nhất: `screenshots/pytest_report.txt` (40/40 đạt).

Ảnh mẫu (`samples/`) lấy từ bộ dữ liệu công khai của scikit-image; tạo lại bằng `python samples/make_samples.py`.
`cameraman.png` có cả bầu trời trơn và bãi cỏ texture, nên dùng nó cho demo.

## Cấu trúc

| File | Nội dung |
|---|---|
| `app.py` | Giao diện Streamlit: tab **Người gửi** (7 bước, nút "Bước tiếp theo" / "Xem toàn bộ") và tab **Người nhận** |
| `core/features.py` | Residual KB, σ cục bộ, gradient Sobel, wavelet db8 (LH/HL/HH), bản đồ Q |
| `core/costs.py` | HILL, WOW-like, đồng đều; ρ⁺ = ∞ tại pixel 255, ρ⁻ = ∞ tại pixel 0 |
| `core/stc.py` | STC: sinh Ĥ, Viterbi vector hoá, truy vết ngược, syndrome (= trích xuất), bản ghi vết trellis |
| `core/embed.py` | Header 88 bit (magic, h, độ dài, CRC32), mã hoá, hoán vị, ±1, lưu/đọc PNG, trích xuất |
| `core/metrics.py` | PSNR, SSIM, MSE, thống kê theo phân vị ρ, câu kết luận tự động |
| `viz/plots.py` | Heatmap, overlay, trellis, bản đồ thay đổi, histogram, zoom 16×16 |
| `tests/` | pytest (STC, round-trip qua đĩa, các trường hợp âm, smoke test giao diện) |
| `screenshots/` | Ảnh chụp chứng minh NT1–NT4 (tạo bằng `tools/capture_screenshots.py`) |
| `docs/bao_cao_thuat_toan.md` | Báo cáo mô tả thuật toán cho luận văn (công thức, giả mã, kết quả thực nghiệm) |

## Thiết kế chính

- **Hai vùng STC.** Sau khi hoán vị, 1024 pixel đầu là vùng header (STC h = 7 cố định, 88 bit). Phần còn lại
  là vùng payload (STC với h người dùng chọn). Vì h và độ dài nằm trong header, **người nhận chỉ cần file + khoá**.
- **Độ rộng khối thay đổi** `w_i = ⌊(i+1)n/M⌋ − ⌊i·n/M⌋`: mọi pixel trong vùng đều được dùng. Hàng đầu và hàng cuối
  của Ĥ toàn 1, nên luôn có nghiệm và sửa bất kỳ LSB nào cũng làm sai syndrome (CRC báo lỗi).
- **Bản đồ thay đổi** lấy trực tiếp từ `y ≠ x` của STC rồi ánh xạ ngược qua hoán vị. Giao diện đối chiếu nó
  với file PNG đọc lại từ đĩa.
- **Dung lượng**: tối đa 0,5 bpp ở vùng payload (256×256 → 4032 byte). Nếu vượt, hệ thống báo lỗi trước khi nhúng.

## Kịch bản demo 5 phút

Để mặc định: `cameraman.png`, resize 256×256, HILL, h = 10, khoá `khoa-bi-mat-2026`.

**0:00–1:00 · Pipeline.** Bấm **Bước tiếp theo** qua bước 1–3. Bước 2 có 4 heatmap đặc trưng. Bước 3 có Q chồng
lên ảnh: bãi cỏ màu vàng (Q cao), bầu trời màu tím (Q thấp).

**1:00–2:00 · NT1, vì sao vùng này.** Ở bước 4, click một điểm trên **bãi cỏ**. Bảng truy vết đi từ giá trị pixel
qua 4 đặc trưng, Q, ρ, thứ hạng ("rẻ hơn ~78% số pixel") đến kết luận **ĐƯỢC ƯU TIÊN**. Sau đó click lên
**bầu trời**: Q ≈ 0,03, "rẻ hơn ~9%", **ÍT ĐƯỢC ƯU TIÊN**. Chọn "Block 8×8" để xem theo khối.
Bước 5 có trellis 12 bit, chi phí Viterbi trùng với kết quả vét cạn 4096 khả năng.

**2:00–3:00 · NT2, pixel nào bị đổi.** Ở bước 6, tab NT2: bản đồ đỏ (+1) và xanh (−1) nằm ở bãi cỏ và chân máy.
Ô xanh **"KHỚP: 197 = 197"** đối chiếu bảng STC với file `output/stego.png` đọc lại từ đĩa. Bảng chi tiết có lọc,
sắp xếp và nút **Xuất CSV**. Σρ của STC khoảng 565, của LSB tuần tự khoảng 59 138 (đắt hơn ~100 lần).
Tab NT1: bảng vùng A (trời) và B (cỏ), biểu đồ tỉ lệ thay đổi giảm dần theo phân vị ρ, nút **So sánh**
đặt 3 bản đồ cạnh nhau.

**3:00–4:00 · NT3, khác bao nhiêu.** Tab NT3 có câu kết luận tự động (ví dụ "Thay đổi 197/65536 pixel (0.30%),
tối đa ±1, PSNR 73.4 dB…"), bảng số liệu, thanh trượt cover|stego, ảnh hiệu khuếch đại ×255, histogram chồng nhau
và zoom 16×16 có giá trị số.

**4:00–5:00 · NT4, trích xuất.** Tab **Người nhận**: chọn "Đọc file trên đĩa" (hoặc upload `stego.png` đã tải về),
nhập khoá, dán SHA-256 gốc, bấm **Trích xuất**. Kết quả: CRC KHỚP, SHA-256 KHỚP. Bấm **Chạy 3 trường hợp âm**:
sai khoá, JPEG q95 và sửa 1 pixel đều cho **KHÔNG KHÔI PHỤC ĐƯỢC** (magic hoặc checksum sai) và không crash.

## Tài liệu tham khảo

- T. Filler, J. Judas, J. Fridrich, *Minimizing Additive Distortion in Steganography Using Syndrome-Trellis Codes*, IEEE TIFS 6(3), 2011.
- B. Li, M. Wang, J. Huang, X. Li, *A New Cost Function for Spatial Image Steganography* (HILL), ICIP 2014.
- V. Holub, J. Fridrich, *Designing Steganographic Distortion Using Directional Filters* (WOW), WIFS 2012.
- V. Holub, J. Fridrich, T. Denemark, *Universal Distortion Function for Steganography in an Arbitrary Domain* (S-UNIWARD), EURASIP JIS 2014.
