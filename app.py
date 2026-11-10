"""Demo trực quan: giấu tin thích nghi trên ảnh xám 8-bit (HILL / WOW-like + STC).

Chạy:  streamlit run app.py

Pipeline: Ảnh gốc → Đặc trưng → Q → Chi phí ρ → STC → Ảnh stego → Trích xuất
"""
from __future__ import annotations

import hashlib
import io
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.costs import COST_NAMES, compute_costs, percentile_rank  # noqa: E402
from core.embed import (HEADER_BITS, HEADER_H, HEADER_PIXELS, CapacityError, bits_to_bytes,  # noqa: E402
                        capacity_bytes, embed, extract_from_file, lsb_sequential, png_bytes,
                        read_image_file, save_png, sha256_text, to_jpeg_bytes)
from core.features import compute_features, load_gray  # noqa: E402
from core.metrics import (change_rate_by_cost_percentile, conclusion_sentence, image_metrics,  # noqa: E402
                          region_stats)
from core.stc import generate_hhat, hhat_matrix, stc_embed_trace, syndrome  # noqa: E402
from viz import plots  # noqa: E402

SAMPLES = ROOT / "samples"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
STEGO_PATH = OUTPUT / "stego.png"

STEPS = ["1. Ảnh gốc", "2. Đặc trưng", "3. Bản đồ Q", "4. Chi phí ρ", "5. STC", "6. Ảnh stego", "7. Trích xuất"]

st.set_page_config(page_title="Demo giấu tin thích nghi", page_icon="🔐", layout="wide")
st.markdown("""
<style>
.pipe {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:4px 0 10px 0}
.pipe .chip {padding:6px 12px; border-radius:16px; border:1px solid #cbd5e1; font-size:0.9rem;
             background:#f8fafc; color:#334155}
.pipe .chip.done {background:#e0f2fe; border-color:#7dd3fc}
.pipe .chip.on {background:#0f4c75; color:white; border-color:#0f4c75; font-weight:600}
.pipe .arrow {color:#94a3b8}
.verdict-ok {padding:10px 14px; border-radius:8px; background:#dcfce7; color:#14532d; font-weight:600}
.verdict-bad {padding:10px 14px; border-radius:8px; background:#fee2e2; color:#7f1d1d; font-weight:600}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cache tính toán
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_cover(data: bytes, resize: bool) -> np.ndarray:
    return load_gray(data, 256 if resize else None)


@st.cache_data(show_spinner=False)
def cached_features(cover: np.ndarray):
    f = compute_features(cover)
    return f.as_dict()


@st.cache_data(show_spinner="Đang chạy Viterbi trên trellis STC…")
def cached_embed(cover: np.ndarray, message: str, key: str, h: int, cost_name: str):
    return embed(cover, message, key, h, cost_name)


@st.cache_data(show_spinner=False)
def cached_costs(cover: np.ndarray, cost_name: str):
    return compute_costs(cover, cost_name)


@st.cache_data(show_spinner=False)
def cached_rank(rho: np.ndarray) -> np.ndarray:
    return percentile_rank(rho)


def fmt(v, d=3):
    if isinstance(v, float):
        if not np.isfinite(v):
            return "∞"
        return f"{v:.{d}g}" if abs(v) < 1e-2 or abs(v) >= 1e4 else f"{v:.{d}f}"
    return str(v)


def explain(text: str):
    st.markdown(f"<div style='color:#475569;font-size:0.95rem'>{text}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Tham số")
sample_files = sorted(p.name for p in SAMPLES.glob("*.png"))
default_sample = sample_files.index("cameraman.png") if "cameraman.png" in sample_files else 0
src = st.sidebar.selectbox("Ảnh gốc", sample_files + ["⬆ Tải ảnh lên…"], index=default_sample)
if src == "⬆ Tải ảnh lên…":
    up = st.sidebar.file_uploader("Chọn ảnh (PNG/JPG/BMP)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
    cover_bytes = up.getvalue() if up else (SAMPLES / sample_files[default_sample]).read_bytes()
else:
    cover_bytes = (SAMPLES / src).read_bytes()
resize = st.sidebar.checkbox("Resize về 256×256", value=True)
message = st.sidebar.text_area("Thông điệp bí mật", height=110,
                               value="Xin chào! Đây là thông điệp bí mật 🔐 được giấu bằng HILL + STC. "
                                     "Tiếng Việt có dấu vẫn khôi phục chính xác.")
key = st.sidebar.text_input("Khoá bí mật", value="khoa-bi-mat-2026", type="default")
h = st.sidebar.slider("Tham số h của STC (chiều cao Ĥ)", 7, 10, 10,
                      help="Trellis có 2^h trạng thái. h lớn ⇒ nhúng hiệu quả hơn (ít thay đổi hơn) nhưng chậm hơn.")
cost_name = st.sidebar.radio("Hàm chi phí", COST_NAMES, index=0,
                             help="HILL (Li et al. 2014), WOW-like (Holub & Fridrich 2012), Đồng đều = đối chứng.")

cover = cached_cover(cover_bytes, resize)
H_img, W_img = cover.shape
cap = capacity_bytes(cover.shape)
msg_len = len(message.encode("utf-8"))
st.sidebar.markdown("**Dung lượng (capacity)**")
st.sidebar.progress(min(msg_len / max(cap, 1), 1.0))
st.sidebar.caption(f"{msg_len} / {cap} byte  ·  payload {8 * msg_len + HEADER_BITS} bit "
                   f"= {(8 * msg_len + HEADER_BITS) / cover.size:.4f} bpp  ·  ảnh {W_img}×{H_img}")
capacity_error = None
if msg_len > cap:
    capacity_error = (f"Thông điệp {msg_len} byte VƯỢT dung lượng {cap} byte — không nhúng. "
                      "Hãy rút ngắn thông điệp hoặc tắt resize/dùng ảnh lớn hơn.")
    st.sidebar.error(capacity_error)
if not key:
    st.sidebar.warning("Khoá đang để trống.")

# ---------------------------------------------------------------------------
# Tính toán chính
# ---------------------------------------------------------------------------

feats = cached_features(cover)
costs = cached_costs(cover, cost_name)
rank = cached_rank(costs.rho)
res = None
if capacity_error is None:
    try:
        res = cached_embed(cover, message, key, h, cost_name)
    except CapacityError as e:
        capacity_error = str(e)
if res is not None:
    stego_png = png_bytes(res.stego)
    save_png(res.stego, STEGO_PATH)  # lưu ra đĩa — NT2/NT4 đọc lại file này

st.title("🔐 Giấu tin thích nghi trên ảnh xám 8-bit")
st.caption("Thuật toán nhúng thật: HILL / WOW-like + Syndrome-Trellis Codes (Viterbi) + LSB matching ±1. "
           "Không dùng thư viện steganography có sẵn.")

tab_send, tab_recv = st.tabs(["📤 Người gửi", "📥 Người nhận"])


# ---------------------------------------------------------------------------
# Truy vết pixel / block (NT1)
# ---------------------------------------------------------------------------

def trace_table(r0, c0, size):
    sl = (slice(r0, r0 + size), slice(c0, c0 + size))
    agg = (lambda a: float(a[sl].mean())) if size > 1 else (lambda a: float(a[r0, c0]))
    rho_v = agg(costs.rho)
    cheaper = 100.0 * float((costs.rho > rho_v).mean())
    pv = cover[sl]
    rows = [("Giá trị pixel", f"{int(pv[0, 0])}" if size == 1 else f"TB {pv.mean():.1f} (min {pv.min()}, max {pv.max()})")]
    for name, arr in feats.items():
        rows.append((name, fmt(agg(arr))))
    rows.append(("Q (độ texture, 0–1)", fmt(agg(costs.Q))))
    rows.append(("ρ (chi phí nhị phân)", fmt(rho_v)))
    if size == 1:
        rows.append(("ρ⁺ / ρ⁻", f"{fmt(float(costs.rho_plus[r0, c0]))} / {fmt(float(costs.rho_minus[r0, c0]))}"))
    rows.append(("Thứ hạng chi phí", f"rẻ hơn {cheaper:.1f}% số pixel"))
    forbidden = ""
    if size == 1 and cover[r0, c0] == 255:
        forbidden = " Hướng +1 BỊ CẤM (255+1 tràn), chỉ được −1."
    if size == 1 and cover[r0, c0] == 0:
        forbidden = " Hướng −1 BỊ CẤM (0−1 tràn), chỉ được +1."
    if cost_name == "Đồng đều":
        verdict = "Chi phí đồng đều — mọi pixel như nhau, STC chỉ tối thiểu số thay đổi."
    elif cheaper >= 70:
        verdict = "✅ ĐƯỢC ƯU TIÊN — vùng texture, thay đổi ±1 khó bị phát hiện."
    elif cheaper >= 30:
        verdict = "➖ Ưu tiên trung bình."
    else:
        verdict = "⛔ ÍT ĐƯỢC ƯU TIÊN — vùng trơn/dễ dự đoán, thay đổi dễ lộ."
    rows.append(("Kết luận", verdict + forbidden))
    if res is not None:
        d = res.delta[sl]
        if size == 1:
            actual = {1: "ĐÃ BỊ ĐỔI +1", -1: "ĐÃ BỊ ĐỔI −1", 0: "không đổi"}[int(d[0, 0])]
        else:
            actual = f"{int(np.count_nonzero(d))}/{size * size} pixel bị đổi"
        rows.append(("Thực tế (đầu ra STC)", actual))
    return pd.DataFrame(rows, columns=["Đại lượng", "Giá trị"])


def nt1_pixel_trace(key_prefix: str):
    st.markdown("#### 🔍 NT1 · Truy vết: vì sao pixel/block này được (hoặc không được) ưu tiên")
    mode = st.radio("Đơn vị", ["Pixel", "Block 8×8"], horizontal=True, key=f"{key_prefix}_mode")
    size = 1 if mode == "Pixel" else 8
    scale = max(1, 512 // max(H_img, W_img))
    ss = st.session_state
    kr, kc = f"{key_prefix}_r", f"{key_prefix}_c"
    ss.setdefault(kr, H_img // 4)
    ss.setdefault(kc, W_img // 2)
    ss[kr] = min(int(ss[kr]), H_img - 1)
    ss[kc] = min(int(ss[kc]), W_img - 1)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.caption("Click vào ảnh (nền: ảnh gốc, màu: ρ thang log — sáng = đắt, tối = rẻ) để chọn vị trí.")
        r, c = ss[kr], ss[kc]
        r0, c0 = ((r // 8) * 8, (c // 8) * 8) if size == 8 else (r, c)
        base = plots.overlay_rgb(cover, costs.rho, cmap="magma", alpha=0.5, log=True)
        im = plots.draw_boxes(plots.upscale(base, scale), [(r0, r0 + size, c0, c0 + size, "#00e5ff", "")], scale)
        click = streamlit_image_coordinates(im, key=f"{key_prefix}_click")
        if click and click.get("width") and ss.get(f"{key_prefix}_last") != click:
            ss[f"{key_prefix}_last"] = click
            rr = int(click["y"] * im.height / click["height"]) // scale
            cc = int(click["x"] * im.width / click["width"]) // scale
            ss[kr], ss[kc] = min(max(rr, 0), H_img - 1), min(max(cc, 0), W_img - 1)
            st.rerun()
    with c2:
        a, b = st.columns(2)
        a.number_input("Hàng", 0, H_img - 1, key=kr)
        b.number_input("Cột", 0, W_img - 1, key=kc)
        r, c = int(ss[kr]), int(ss[kc])
        r0, c0 = ((r // 8) * 8, (c // 8) * 8) if size == 8 else (r, c)
        r0, c0 = min(r0, H_img - size), min(c0, W_img - size)
        st.markdown(f"**{'Pixel' if size == 1 else 'Block 8×8 tại'} ({r0}, {c0})**")
        st.table(trace_table(r0, c0, size).set_index("Đại lượng"))


# ---------------------------------------------------------------------------
# Các khung bước
# ---------------------------------------------------------------------------

def step_cover():
    st.subheader("Bước 1 · Ảnh gốc (cover)")
    explain("Ảnh được chuyển sang ảnh xám 8-bit (mỗi pixel 0–255). Ảnh tốt cho demo có cả vùng trơn "
            "(bầu trời) và vùng texture (cỏ, tóc): thuật toán thích nghi sẽ đối xử hai vùng này rất khác nhau.")
    c1, c2 = st.columns([1, 1])
    c1.plotly_chart(plots.gray_figure(cover, f"Cover {W_img}×{H_img}"), key="cov")
    c2.dataframe(pd.DataFrame({
        "Thuộc tính": ["Kích thước", "Số pixel", "Min / Max", "Trung bình", "Pixel = 0", "Pixel = 255",
                       "SHA-256 (8 byte đầu)"],
        "Giá trị": [f"{W_img}×{H_img}", f"{cover.size}", f"{cover.min()} / {cover.max()}", f"{cover.mean():.2f}",
                    f"{int((cover == 0).sum())}", f"{int((cover == 255).sum())}",
                    hashlib.sha256(cover.tobytes()).hexdigest()[:16]],
    }), hide_index=True, width="stretch")
    c2.caption("Pixel = 0 hoặc 255 chỉ được thay đổi theo một hướng (không được tràn khỏi [0,255]).")


def step_features():
    st.subheader("Bước 2 · Đặc trưng cục bộ")
    explain("Bốn đặc trưng đo mức độ 'khó dự đoán' quanh mỗi pixel. Residual high-pass (KB, wavelet) ≈ 0 ở vùng "
            "trơn vì pixel dễ đoán từ lân cận; ở vùng texture residual lớn, nên một thay đổi ±1 bị nhiễu tự nhiên che đi.")
    cols = st.columns(2)
    for i, (name, arr) in enumerate(feats.items()):
        cols[i % 2].plotly_chart(plots.heatmap(arr, name, "Blues", cbar_title="", height=320,
                                               hover_name=name.split(" ")[0]), key=f"feat{i}")
    st.caption("KB = [[-1,2,-1],[2,-4,2],[-1,2,-1]] (Ker–Böhme, dùng trong HILL). Wavelet: bộ lọc Daubechies-8 "
               "định hướng LH/HL/HH như WOW / S-UNIWARD (Holub & Fridrich).")


def step_q():
    st.subheader("Bước 3 · Bản đồ Q ∈ [0,1]")
    if cost_name == "HILL":
        formula = "Q = clip( (|I ⋆ KB| ⋆ L₁) / p₉₉ , 0, 1 ), L₁ = trung bình 3×3  (high-pass → |·| → low-pass)"
    elif cost_name == "WOW-like":
        formula = "ξ_k = |I ⋆ K_k| ⋆ |K_k|,  ξ = (Σ_k 1/ξ_k)⁻¹,  Q = clip(ξ / p₉₉, 0, 1)  (k = LH, HL, HH)"
    else:
        formula = "Q = 0.5 mọi nơi (đối chứng: bỏ qua nội dung ảnh)"
    explain("Q tổng hợp các đặc trưng thành một số duy nhất cho mỗi pixel: Q cao = texture mạnh = thay đổi khó "
            "bị phát hiện; Q thấp = vùng trơn = thay đổi dễ lộ. "
            f"Công thức ({cost_name}): <code>{formula}</code>")
    c1, c2 = st.columns(2)
    c1.image(plots.upscale(plots.overlay_rgb(cover, costs.Q, cmap="viridis", alpha=0.6, vmin=0, vmax=1),
                           max(1, 512 // W_img)),
             caption="Q chồng lên ảnh gốc (vàng = Q cao/texture, tím = Q thấp/trơn)", width="stretch")
    c2.plotly_chart(plots.heatmap(costs.Q, "Heatmap Q (rê chuột để xem giá trị)", "Viridis", height=420,
                                  hover_name="Q"), key="qmap")


def step_rho():
    st.subheader("Bước 4 · Chi phí biến đổi ρ")
    if cost_name == "HILL":
        formula = "ρ = L₂ ⋆ 1/(Q + ε),  L₂ = trung bình 15×15, ε = 10⁻³"
    elif cost_name == "WOW-like":
        formula = "ρ = Σ_k 1/ξ_k  ∝ 1/(Q + ε)"
    else:
        formula = "ρ = 1"
    explain(f"Chi phí là 'giá phải trả' khi đổi pixel ±1: <code>{formula}</code>. Hướng làm pixel tràn khỏi 0–255 "
            "có ρ = ∞ (ρ⁺ tại 255, ρ⁻ tại 0). STC sẽ tìm cách nhúng với tổng chi phí nhỏ nhất, nên các pixel "
            "rẻ (texture) gần như gánh hết thay đổi.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(plots.heatmap(costs.rho, "ρ (thang log₁₀) — sáng = đắt", "OrRd", log=True, height=420,
                                  hover_name="ρ"), key="rhomap")
    c2.plotly_chart(plots.rho_histogram(costs.rho, res.change_mask if res is not None else None, height=420),
                    key="rhohist")
    n_inf = int((~np.isfinite(costs.rho_plus)).sum() + (~np.isfinite(costs.rho_minus)).sum())
    st.caption(f"ρ min = {costs.rho.min():.3g}, trung vị = {np.median(costs.rho):.3g}, max = {costs.rho.max():.3g}. "
               f"Số hướng bị cấm (ρ = ∞): {n_inf}.")
    nt1_pixel_trace("s4")


def step_stc():
    st.subheader("Bước 5 · Syndrome-Trellis Codes (Viterbi)")
    explain("Tìm vector LSB y gần x nhất theo chi phí ρ sao cho <b>H·y = m (mod 2)</b>. H ghép từ ma trận con Ĥ "
            "(h×w) chạy chéo; Viterbi duyệt trellis 2^h trạng thái và chọn đường đi có Σρ·|x−y| nhỏ nhất "
            "(Filler, Judas, Fridrich 2011).")
    if res is None:
        st.error(capacity_error or "Chưa nhúng được.")
        return
    pl = res.payload
    st.markdown("##### 5a · Xử lý thông điệp")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(pd.DataFrame({
            "Trường header": ["magic", "h", "độ dài (byte)", "CRC32"],
            "Giá trị": ["0x5354 ('ST')", str(pl.h), str(len(pl.msg_bytes)), f"{pl.crc:08X}"],
            "Số bit": [16, 8, 32, 32]}), hide_index=True, width="stretch")
        st.caption(f"UTF-8: {len(pl.message)} ký tự → {len(pl.msg_bytes)} byte. "
                   f"Hex đầu: {pl.msg_bytes[:24].hex(' ')}{' …' if len(pl.msg_bytes) > 24 else ''}")
    with c2:
        nb = min(48, len(pl.payload_plain))
        st.code("bit rõ (payload)  : " + "".join(map(str, pl.payload_plain[:nb])) + "\n"
                "dòng khoá SHA-256 : " + "".join(map(str, (pl.payload_plain ^ pl.payload_enc)[:nb])) + "\n"
                "bit mã hoá (XOR)  : " + "".join(map(str, pl.payload_enc[:nb])) + "\n\n"
                "hoán vị pixel (PRNG seed = SHA-256(khoá)):\n  " +
                ", ".join(f"#{i}→({p // W_img},{p % W_img})" for i, p in enumerate(res.perm[:6])) + ", …",
                language=None)
    st.markdown("##### 5b · Tham số STC")
    n_pay = cover.size - HEADER_PIXELS
    L = len(pl.payload_enc)
    pw = res.payload_widths
    st.dataframe(pd.DataFrame({
        "Vùng": ["header", "payload"],
        "Số pixel n": [HEADER_PIXELS, n_pay],
        "Số bit M": [HEADER_BITS, L],
        "Tỉ lệ α = M/n": [f"{HEADER_BITS / HEADER_PIXELS:.4f}", f"{L / n_pay:.4f}"],
        "h": [HEADER_H, h],
        "Độ rộng khối w": [f"{int(res.header_widths.min())}–{int(res.header_widths.max())}",
                           f"{int(pw.min())}–{int(pw.max())}" if L else "—"],
        "Số trạng thái 2^h": [2 ** HEADER_H, 2 ** h],
    }), hide_index=True, width="stretch")
    if res.payload_cols is not None:
        ncol = min(32, len(res.payload_cols))
        st.plotly_chart(plots.heatmap(hhat_matrix(res.payload_cols[:ncol], h), f"Ma trận con Ĥ (h={h}, {ncol} cột đầu); "
                                      "hàng đầu và hàng cuối toàn 1 ⇒ luôn có nghiệm", "Greys", height=230,
                                      hover_name="bit"), key="hhat")

    st.markdown("##### 5c · Minh hoạ trellis thu nhỏ (12 bit, h = 3, 3 bit thông điệp)")
    explain("Lấy 12 pixel THẬT đầu tiên của vùng payload (sau hoán vị) với LSB x và chi phí ρ thật, giấu 3 bit "
            "thông điệp. Mỗi cột là một bit y_j; cạnh ngang = y_j = 0 (giữ trạng thái), cạnh xiên = y_j = 1 (XOR với "
            "cột Ĥ); cột 'm=…' chỉ giữ trạng thái có bit thấp đúng bằng bit thông điệp. Đường đậm = đường Viterbi rẻ nhất.")
    xs = res.x_bits[HEADER_PIXELS:HEADER_PIXELS + 12]
    rr = costs.rho.ravel()[res.perm[HEADER_PIXELS:HEADER_PIXELS + 12]]
    mm = pl.payload_enc[:3] if L >= 3 else np.array([1, 0, 1], np.uint8)
    toy_cols = generate_hhat(3, 4, 7)
    tr = stc_embed_trace(xs, rr, mm, 3, toy_cols)
    st.plotly_chart(plots.trellis_figure(tr), key="trellis")
    # kiểm chứng tối ưu bằng vét cạn 2^12
    best = min((float(np.sum(rr * (np.array(y) != xs))), y) for y in product((0, 1), repeat=12)
               if np.array_equal(syndrome(np.array(y), 3, 3, toy_cols, tr.widths), mm))
    toy = pd.DataFrame({"j": range(1, 13), "x (LSB cover)": xs, "ρ": np.round(rr, 3), "y (Viterbi)": tr.y,
                        "đổi?": np.where(tr.y != xs, "✱", "")})
    c1, c2 = st.columns([1.3, 1])
    c1.dataframe(toy.T, width="stretch")
    c2.markdown(f"Ĥ (3×4) = cột {[f'{c:03b}'[::-1] for c in toy_cols]}  \n"
                f"m = {''.join(map(str, mm))} · H·y = {''.join(map(str, syndrome(tr.y, 3, 3, toy_cols, tr.widths)))} ✓  \n"
                f"Chi phí Viterbi = **{tr.cost:.4f}** · vét cạn 4096 khả năng: **{best[0]:.4f}** "
                f"{'✓ tối ưu' if abs(best[0] - tr.cost) < 1e-9 else '✗'}")

    st.markdown("##### 5d · Kết quả STC trên toàn ảnh")
    lsb_stego, lsb_mask = lsb_sequential(cover, np.concatenate([pl.header_enc, pl.payload_enc]), key)
    lsb_cost = float(costs.rho[lsb_mask].sum())
    a, b, c, d = st.columns(4)
    a.metric("Bit đã nhúng", f"{res.total_bits}")
    b.metric("Pixel bị STC đổi", f"{res.n_changes}")
    c.metric("Σρ của STC", f"{res.stc_cost:.1f}")
    d.metric("Σρ của LSB tuần tự", f"{lsb_cost:.1f}", delta=f"{lsb_cost / max(res.stc_cost, 1e-9):.1f}× đắt hơn",
             delta_color="inverse")


def step_stego():
    st.subheader("Bước 6 · Ảnh stego (LSB matching ±1)")
    explain("Tại mỗi vị trí STC yêu cầu đổi LSB (y ≠ x), pixel được cộng hoặc trừ 1 (dấu ngẫu nhiên theo khoá; "
            "bắt buộc +1 tại 0 và −1 tại 255 để không tràn). Ảnh được lưu thành PNG (nén không mất mát).")
    if res is None:
        st.error(capacity_error or "Chưa nhúng được.")
        return
    c1, c2 = st.columns([1, 2])
    c1.download_button("⬇ Tải ảnh stego (PNG)", stego_png, file_name="stego.png", mime="image/png", type="primary")
    c2.caption(f"Đã lưu ra đĩa: `{STEGO_PATH.relative_to(ROOT)}` — SHA-256 file: "
               f"`{hashlib.sha256(stego_png).hexdigest()[:24]}…`")
    t2, t3, t1 = st.tabs(["NT2 · Pixel nào bị đổi", "NT3 · Khác cover bao nhiêu", "NT1 · Vì sao vùng này"])
    with t2:
        nt2()
    with t3:
        nt3()
    with t1:
        nt1_regions()


def nt2():
    st.markdown("#### 🎯 NT2 · STC đã thay đổi những pixel nào")
    explain("Bản đồ lấy TRỰC TIẾP từ đầu ra STC (vị trí y ≠ x, ánh xạ ngược qua hoán vị), không suy ra lại từ ảnh. "
            "Đỏ = +1, xanh = −1.")
    c1, c2 = st.columns([1.2, 1])
    fig = plots.go.Figure(plots.go.Image(z=plots.change_map_rgb(cover, res.delta),
                                         hovertemplate="hàng %{y}, cột %{x}<extra></extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    c1.plotly_chart(plots._layout(fig, f"{res.n_changes} pixel bị đổi (đỏ +1: {int((res.delta > 0).sum())}, "
                                        f"xanh −1: {int((res.delta < 0).sum())})", 520), key="chmap")
    # Đối chiếu với file đọc lại từ đĩa
    disk = read_image_file(STEGO_PATH)
    n_disk = int(np.count_nonzero(disk.astype(np.int16) - cover.astype(np.int16)))
    n_table = len(res.changes["hàng"])
    same_pos = disk.shape == cover.shape and np.array_equal(disk != cover, res.change_mask)
    with c2:
        st.markdown("**Đối chiếu bảng STC ↔ file stego trên đĩa**")
        st.dataframe(pd.DataFrame({
            "Nguồn": ["Bảng thay đổi (đầu ra STC)", f"File {STEGO_PATH.name} đọc lại từ đĩa"],
            "Số pixel": [n_table, n_disk]}), hide_index=True, width="stretch")
        if n_table == n_disk and same_pos:
            st.markdown(f"<div class='verdict-ok'>✓ KHỚP: {n_table} = {n_disk}, trùng cả vị trí từng pixel.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='verdict-bad'>✗ LỆCH: bảng STC có {n_table} pixel nhưng file trên đĩa khác "
                        f"cover ở {n_disk} pixel (trùng vị trí: {same_pos}). File có thể đã bị ghi đè/sửa!</div>",
                        unsafe_allow_html=True)
        lsb_stego, lsb_mask = lsb_sequential(cover, np.concatenate([res.payload.header_enc, res.payload.payload_enc]), key)
        st.markdown("**Tổng chi phí với cùng thông điệp**")
        st.dataframe(pd.DataFrame({
            "Phương pháp": ["STC + ρ thích nghi", "LSB tuần tự (raster)"],
            "Số pixel đổi": [res.n_changes, int(lsb_mask.sum())],
            "Σρ": [round(res.stc_cost, 2), round(float(costs.rho[lsb_mask].sum()), 2)],
        }), hide_index=True, width="stretch")
        st.caption(f"Kiểm tra: Σρ tại các pixel trong bảng = {costs.rho[res.change_mask].sum():.2f} "
                   f"= chi phí Viterbi {res.stc_cost:.2f}.")

    st.markdown("**Bảng đầy đủ các pixel bị thay đổi** (click tiêu đề cột để sắp xếp)")
    df = pd.DataFrame(res.changes)
    f1, f2, f3 = st.columns([1, 1, 2])
    sign_f = f1.multiselect("Dấu", ["+1", "−1"], default=["+1", "−1"])
    reg_f = f2.multiselect("Vùng", ["header", "payload"], default=["header", "payload"])
    lo, hi = float(df["ρ"].min()) if len(df) else 0.0, float(df["ρ"].max()) if len(df) else 1.0
    rho_f = f3.slider("Khoảng ρ", lo, max(hi, lo + 1e-6), (lo, max(hi, lo + 1e-6))) if len(df) else (0, 1)
    sel = df[df["±1"].isin([1 if s == "+1" else -1 for s in sign_f]) & df["vùng"].isin(reg_f)
             & df["ρ"].between(rho_f[0] - 1e-9, rho_f[1] + 1e-9)]
    st.dataframe(sel, hide_index=True, width="stretch", height=300,
                 column_config={"ρ": st.column_config.NumberColumn(format="%.4f"),
                                "Q": st.column_config.NumberColumn(format="%.4f")})
    st.download_button("⬇ Xuất CSV (đang lọc)", sel.to_csv(index=False).encode("utf-8-sig"),
                       file_name="pixel_thay_doi.csv", mime="text/csv")
    st.caption(f"Đang hiển thị {len(sel)}/{len(df)} dòng.")


def nt3():
    st.markdown("#### 📏 NT3 · Ảnh stego khác cover bao nhiêu")
    disk = read_image_file(STEGO_PATH)
    m = image_metrics(cover, disk, res.total_bits)
    cheap_half = rank < 50
    share = 100.0 * float((res.change_mask & cheap_half).sum()) / max(res.n_changes, 1)
    st.markdown(f"<div class='verdict-ok'>{conclusion_sentence(m, share)}</div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([1, 1.4])
    c1.dataframe(pd.DataFrame({
        "Chỉ số": ["Số pixel thay đổi", "% pixel thay đổi", "Biên độ lớn nhất |stego − cover|", "MSE", "PSNR",
                   "SSIM", "Payload", "Payload (bpp)", "Hiệu suất nhúng", "Miền giá trị stego"],
        "Giá trị": [f"{m['n_changed']} / {m['n_pixels']}", f"{m['pct_changed']:.3f}%",
                    f"{m['max_abs_diff']} {'✓ (≤ 1)' if m['max_abs_diff'] <= 1 else '✗'}", f"{m['mse']:.5f}",
                    f"{m['psnr']:.2f} dB", f"{m['ssim']:.6f}", f"{m['payload_bits']} bit",
                    f"{m['bpp']:.4f}", f"{m['efficiency']:.2f} bit / pixel đổi",
                    f"[{m['min_val']}, {m['max_val']}] ⊂ [0,255]"]}), hide_index=True, width="stretch")
    c1.caption("Số liệu tính trên file stego ĐỌC LẠI TỪ ĐĨA. Payload gồm 88 bit header + bit thông điệp.")
    with c2:
        pos = st.slider("Thanh trượt so sánh: trái = cover | phải = stego", 0, 100, 50, key="cmp_slider")
        cut = int(W_img * pos / 100)
        comp = np.concatenate([cover[:, :cut], disk[:, cut:]], axis=1)
        rgb = np.repeat(comp[..., None], 3, axis=2)
        if 0 < cut < W_img:
            rgb[:, cut] = (255, 200, 0)
        st.image(plots.upscale(rgb, max(1, 512 // W_img)), width="stretch",
                 caption=f"Cột 0–{cut - 1}: cover · cột {cut}–{W_img - 1}: stego. Đường vàng = ranh giới.")
    c3, c4 = st.columns(2)
    amp = st.slider("Hệ số khuếch đại ảnh hiệu", 1, 255, 255, key="amp")
    diff = np.abs(disk.astype(np.int16) - cover.astype(np.int16))
    c3.image(plots.upscale(np.clip(diff * amp, 0, 255).astype(np.uint8), max(1, 512 // W_img)), width="stretch",
             caption=f"|stego − cover| × {amp} (trắng = pixel bị đổi; không khuếch đại thì ảnh đen hoàn toàn)")
    c4.plotly_chart(plots.histogram_overlay(cover, disk), key="histov")

    st.markdown("**Zoom 16×16 — giá trị số trước / sau**")
    # mặc định: block 16×16 có nhiều thay đổi nhất
    cnt = res.change_mask.reshape(H_img // 16, 16, W_img // 16, 16).sum(axis=(1, 3)) if H_img % 16 == 0 and W_img % 16 == 0 else None
    if cnt is not None:
        br, bc = np.unravel_index(np.argmax(cnt), cnt.shape)
        dr, dc = int(br * 16), int(bc * 16)
    else:
        dr, dc = 0, 0
    z1, z2 = st.columns(2)
    zr = z1.number_input("Hàng bắt đầu", 0, H_img - 16, dr, key="zr")
    zc = z2.number_input("Cột bắt đầu", 0, W_img - 16, dc, key="zc")
    blk_c = cover[zr:zr + 16, zc:zc + 16]
    blk_s = disk[zr:zr + 16, zc:zc + 16]
    dd = blk_s.astype(int) - blk_c.astype(int)
    a, b = st.columns(2)
    a.plotly_chart(plots.zoom_values(blk_c, dd, f"Cover ({zr}:{zr + 16}, {zc}:{zc + 16})"), key="zc_fig")
    b.plotly_chart(plots.zoom_values(blk_s, dd, f"Stego — {int(np.count_nonzero(dd))} ô đổi (viền đỏ +1, xanh −1)"),
                   key="zs_fig")


def auto_regions(size=32):
    """Gợi ý: block size×size có Q trung bình thấp nhất (trơn) và cao nhất (texture)."""
    best_lo, best_hi = None, None
    for r in range(0, H_img - size + 1, size // 2):
        for c in range(0, W_img - size + 1, size // 2):
            q = float(costs.Q[r:r + size, c:c + size].mean()) if cost_name != "Đồng đều" else \
                float(feats["|KB residual|"][r:r + size, c:c + size].mean())
            if best_lo is None or q < best_lo[0]:
                best_lo = (q, r, c)
            if best_hi is None or q > best_hi[0]:
                best_hi = (q, r, c)
    return best_lo[1:], best_hi[1:]


def nt1_regions():
    st.markdown("#### 🧭 NT1 · Vì sao một vùng được ưu tiên")
    (ar, ac), (br_, bc_) = auto_regions()
    st.caption("Mặc định: A = khối 32×32 trơn nhất (Q thấp nhất), B = khối texture nhất. Có thể chỉnh tay.")
    cols = st.columns(6)
    size = cols[0].number_input("Cạnh vùng", 8, min(H_img, W_img), 32, step=8, key="rg_size")
    a_r = cols[1].number_input("A: hàng", 0, H_img - size, min(ar, H_img - size), key="a_r")
    a_c = cols[2].number_input("A: cột", 0, W_img - size, min(ac, W_img - size), key="a_c")
    b_r = cols[3].number_input("B: hàng", 0, H_img - size, min(br_, H_img - size), key="b_r")
    b_c = cols[4].number_input("B: cột", 0, W_img - size, min(bc_, W_img - size), key="b_c")
    c1, c2 = st.columns([1, 1.2])
    scale = max(1, 512 // W_img)
    im = plots.draw_boxes(plots.upscale(plots.change_map_rgb(cover, res.delta), scale),
                          [(a_r, a_r + size, a_c, a_c + size, "#f59e0b", "A"),
                           (b_r, b_r + size, b_c, b_c + size, "#10b981", "B")], scale)
    c1.image(im, caption="Vùng A (cam) và B (xanh lá) trên bản đồ pixel thay đổi", width="stretch")
    sa = region_stats(costs.Q, costs.rho, res.change_mask, a_r, a_r + size, a_c, a_c + size)
    sb = region_stats(costs.Q, costs.rho, res.change_mask, b_r, b_r + size, b_c, b_c + size)
    tbl = pd.DataFrame({"Chỉ số": list(sa.keys()), "Vùng A": [fmt(v) for v in sa.values()],
                        "Vùng B": [fmt(v) for v in sb.values()]})
    c2.dataframe(tbl, hide_index=True, width="stretch")
    c2.caption("Vùng có Q cao hơn → ρ thấp hơn → STC đặt nhiều thay đổi hơn ở đó.")
    edges, rate, changed, counts = change_rate_by_cost_percentile(costs.rho, res.change_mask, 10)
    c2.plotly_chart(plots.percentile_chart(edges, rate, changed, counts, height=300), key="pct")

    st.markdown("**So sánh với cách nhúng không thích nghi** (cùng thông điệp, cùng khoá)")
    if st.button("⚖️ So sánh: thích nghi + STC vs LSB tuần tự vs chi phí đồng đều", type="primary"):
        st.session_state["cmp_on"] = True
    if st.session_state.get("cmp_on"):
        bits = np.concatenate([res.payload.header_enc, res.payload.payload_enc])
        lsb_st, lsb_mask = lsb_sequential(cover, bits, key)
        uni = cached_embed(cover, message, key, h, "Đồng đều") if cost_name != "Đồng đều" else res
        maps = [(f"(a) {cost_name} + STC", cover, res.delta),
                ("(b) LSB tuần tự", cover, (lsb_st.astype(np.int16) - cover).astype(np.int8)),
                ("(c) Đồng đều + STC", cover, uni.delta)]
        st.plotly_chart(plots.side_by_side_change_maps(maps, height=400), key="cmpmaps")
        rows = []
        for name, mask, st_img in ((maps[0][0], res.change_mask, res.stego), (maps[1][0], lsb_mask, lsb_st),
                                   (maps[2][0], uni.change_mask, uni.stego)):
            mm = image_metrics(cover, st_img, res.total_bits)
            rows.append({"Phương pháp": name, "Pixel đổi": int(mask.sum()),
                         f"Σρ ({cost_name})": round(float(costs.rho[mask].sum()), 1),
                         "% thay đổi nằm ở 20% ρ rẻ nhất": round(100 * float((mask & (rank < 20)).sum()) / max(mask.sum(), 1), 1),
                         "% thay đổi nằm ở 20% ρ đắt nhất": round(100 * float((mask & (rank >= 80)).sum()) / max(mask.sum(), 1), 1),
                         "PSNR (dB)": round(mm["psnr"], 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("LSB tuần tự dồn mọi thay đổi vào các hàng đầu ảnh bất kể nội dung (thường là bầu trời trơn); "
                   "chi phí đồng đều rải đều khắp ảnh; chỉ cách (a) gom thay đổi vào vùng texture.")


def step_extract():
    st.subheader("Bước 7 · Trích xuất")
    explain("Người nhận chỉ có <b>file stego + khoá</b>: tạo lại hoán vị từ khoá, đọc LSB, tính m' = H·LSB(y) "
            "cho vùng header (h = 7 cố định) → giải mã header → biết h và độ dài → tính tiếp vùng payload → "
            "giải mã → kiểm tra CRC32. Dùng tab <b>📥 Người nhận</b> để thử độc lập (upload file).")
    if res is None:
        st.error(capacity_error or "Chưa nhúng được.")
        return
    r = extract_from_file(STEGO_PATH, key)
    show_extract_result(r, sha256_text(message))


def show_extract_result(r, sha_orig: str | None):
    if r.ok:
        st.markdown(f"<div class='verdict-ok'>✓ {r.status}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='verdict-bad'>✗ {r.status} — checksum sai / dữ liệu hỏng</div>",
                    unsafe_allow_html=True)
    st.write("")
    sha_rec = hashlib.sha256(r.msg_bytes).hexdigest() if (r.msg_bytes is not None and r.ok) else None
    rows = [
        ("Magic header", "0x5354 ✓" if r.magic_ok else "SAI ✗"),
        ("h (đọc từ header)", str(r.h) if r.h is not None else "—"),
        ("Độ dài (byte, từ header)", str(r.length) if r.length is not None else "—"),
        ("CRC32 trong header", f"{r.crc_header:08X}" if r.crc_header is not None else "—"),
        ("CRC32 tính lại", f"{r.crc_computed:08X}" if r.crc_computed is not None else "—"),
        ("Checksum", "KHỚP ✓" if r.checksum_ok else "SAI ✗"),
    ]
    if sha_orig:
        rows.append(("SHA-256 thông điệp gốc", sha_orig))
        rows.append(("SHA-256 thông điệp khôi phục", sha_rec or "—"))
        rows.append(("So sánh SHA-256", "KHỚP ✓" if sha_rec == sha_orig else "KHÔNG KHỚP ✗"))
    st.dataframe(pd.DataFrame(rows, columns=["Mục", "Giá trị"]), hide_index=True, width="stretch")
    if r.ok:
        st.text_area("Thông điệp khôi phục", r.message, height=100, disabled=True,
                     key=f"rec_{hashlib.md5(r.message.encode()).hexdigest()}_{id(r)}")
    for reason in r.reasons:
        st.warning(reason)


# ---------------------------------------------------------------------------
# Tab Người gửi
# ---------------------------------------------------------------------------

with tab_send:
    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("show_all", False)
    chips = []
    for i, name in enumerate(STEPS):
        cls = "on" if (ss.show_all or i == ss.step) else ("done" if i < ss.step else "")
        chips.append(f"<span class='chip {cls}'>{name}</span>")
    st.markdown("<div class='pipe'>" + "<span class='arrow'>→</span>".join(chips) + "</div>",
                unsafe_allow_html=True)
    b1, b2, b3, b4, _ = st.columns([1, 1.2, 1, 1, 3])
    if b1.button("◀ Bước trước", disabled=ss.step == 0 and not ss.show_all):
        ss.show_all = False
        ss.step = max(0, ss.step - 1)
        st.rerun()
    if b2.button("Bước tiếp theo ▶", type="primary", disabled=ss.step == len(STEPS) - 1 and not ss.show_all):
        ss.show_all = False
        ss.step = min(len(STEPS) - 1, ss.step + 1)
        st.rerun()
    if b3.button("Xem toàn bộ"):
        ss.show_all = True
        st.rerun()
    if b4.button("Về bước 1"):
        ss.show_all = False
        ss.step = 0
        st.rerun()
    if capacity_error:
        st.error("⚠️ " + capacity_error)

    renderers = [step_cover, step_features, step_q, step_rho, step_stc, step_stego, step_extract]
    for i, fn in enumerate(renderers):
        if ss.show_all or i == ss.step:
            with st.container(border=True):
                fn()


# ---------------------------------------------------------------------------
# Tab Người nhận — độc lập: chỉ dùng FILE + khoá
# ---------------------------------------------------------------------------

with tab_recv:
    st.subheader("📥 Người nhận: trích xuất từ file stego")
    explain("Tab này KHÔNG dùng bất kỳ dữ liệu nào của phiên nhúng (cover, ρ, biến tạm). Đầu vào duy nhất là "
            "<b>bytes của file ảnh</b> và <b>khoá</b>; h và độ dài thông điệp được đọc từ header nhúng trong ảnh.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        src_mode = st.radio("Nguồn file", ["Upload file", f"Đọc file trên đĩa ({STEGO_PATH.relative_to(ROOT)})"],
                            horizontal=True)
        file_bytes, fname = None, None
        if src_mode == "Upload file":
            upf = st.file_uploader("File stego (PNG; có thể thử JPG để thấy trường hợp hỏng)",
                                   type=["png", "jpg", "jpeg", "bmp"], key="recv_up")
            if upf:
                file_bytes, fname = upf.getvalue(), upf.name
        elif STEGO_PATH.exists():
            file_bytes, fname = STEGO_PATH.read_bytes(), STEGO_PATH.name
        rkey = st.text_input("Khoá", value="", key="recv_key", placeholder="nhập khoá bí mật")
        sha_in = st.text_input("SHA-256 của thông điệp gốc (tuỳ chọn, chỉ để đối chiếu)", key="recv_sha",
                               help="Người gửi có thể gửi kèm hash để người nhận kiểm chứng. "
                                    "Không dùng trong quá trình trích xuất.")
        if res is not None:
            st.caption(f"SHA-256 thông điệp bên tab Người gửi (để copy): `{sha256_text(message)}`")
    with c2:
        if file_bytes:
            st.image(file_bytes, caption=f"{fname} · {len(file_bytes)} byte · SHA-256 "
                                         f"{hashlib.sha256(file_bytes).hexdigest()[:16]}…", width=280)
    if file_bytes and st.button("🔓 Trích xuất", type="primary"):
        st.session_state["recv_result"] = (extract_from_file(file_bytes, rkey), sha_in.strip().lower() or None)
    if st.session_state.get("recv_result"):
        show_extract_result(*st.session_state["recv_result"])

    st.divider()
    st.subheader("🧪 Các trường hợp âm (phải thất bại an toàn, không treo/crash)")
    explain("Tạo 3 biến thể từ file đang chọn ở trên rồi trích xuất từng cái chỉ từ bytes của file.")
    if file_bytes and st.button("Chạy 3 trường hợp âm"):
        base = read_image_file(file_bytes)
        mod = base.copy()
        r_, c_ = base.shape[0] // 2, base.shape[1] // 2
        mod[r_, c_] = mod[r_, c_] + 1 if mod[r_, c_] < 255 else mod[r_, c_] - 1
        cases = [
            ("Sai khoá", file_bytes, (rkey or "") + "-sai"),
            ("Chuyển sang JPEG (quality 95)", to_jpeg_bytes(base, 95), rkey),
            (f"Sửa 1 pixel tại ({r_},{c_}): {base[r_, c_]}→{mod[r_, c_]}", png_bytes(mod), rkey),
            ("Đối chứng: file gốc, đúng khoá", file_bytes, rkey),
        ]
        out = []
        for name, data, k in cases:
            rr = extract_from_file(data, k)
            out.append({"Trường hợp": name, "Kết quả": rr.status, "Magic": "✓" if rr.magic_ok else "✗",
                        "Checksum": "KHỚP" if rr.checksum_ok else "SAI",
                        "Lý do": " | ".join(rr.reasons) if rr.reasons else "—"})
        st.session_state["neg_cases"] = out
    if st.session_state.get("neg_cases"):
        st.dataframe(pd.DataFrame(st.session_state["neg_cases"]), hide_index=True, width="stretch")
