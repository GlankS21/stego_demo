"""Hình minh hoạ (plotly + PIL) cho từng bước của pipeline."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from matplotlib import colormaps
from PIL import Image, ImageDraw

INK = "#1f2937"
MUTED = "#6b7280"
PLUS = "#d62728"     # +1 = đỏ
MINUS = "#1f5fd6"    # −1 = xanh
BAR = "#2a6f97"


def _layout(fig, title=None, height=360, **kw):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=INK)) if title else None,
        height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **kw)
    return fig


def heatmap(arr, title, colorscale="Blues", log=False, cbar_title="", height=340, hover_name="giá trị"):
    z = np.log10(np.maximum(arr, 1e-12)) if log else arr
    fig = go.Figure(go.Heatmap(
        z=z, colorscale=colorscale, customdata=arr,
        hovertemplate=f"hàng %{{y}}, cột %{{x}}<br>{hover_name} = %{{customdata:.4g}}<extra></extra>",
        colorbar=dict(title=("log₁₀ " if log else "") + cbar_title, thickness=12)))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", showgrid=False, visible=False)
    fig.update_xaxes(showgrid=False, visible=False)
    return _layout(fig, title, height)


def gray_figure(img, title=None, height=340):
    fig = go.Figure(go.Heatmap(z=img, colorscale="gray", zmin=0, zmax=255, showscale=False,
                               hovertemplate="hàng %{y}, cột %{x}<br>giá trị = %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", visible=False)
    fig.update_xaxes(visible=False)
    return _layout(fig, title, height)


def overlay_rgb(img: np.ndarray, field: np.ndarray, cmap="inferno", alpha=0.55,
                vmin=None, vmax=None, log=False) -> np.ndarray:
    """Chồng một bản đồ (Q hoặc ρ) lên ảnh xám → ảnh RGB uint8."""
    f = np.log10(np.maximum(field, 1e-12)) if log else field.astype(np.float64)
    vmin = np.nanmin(f) if vmin is None else vmin
    vmax = np.nanmax(f) if vmax is None else vmax
    t = np.clip((f - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    col = colormaps[cmap](t)[..., :3]
    base = np.repeat((img / 255.0)[..., None], 3, axis=2)
    return (255 * ((1 - alpha) * base + alpha * col)).astype(np.uint8)


def change_map_rgb(cover: np.ndarray, delta: np.ndarray, dim=0.45) -> np.ndarray:
    """Bản đồ pixel thay đổi: nền = ảnh gốc làm mờ, đỏ = +1, xanh = −1."""
    base = (cover.astype(np.float64) * dim + 255 * (1 - dim) * 0.6).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)
    rgb[delta > 0] = (214, 39, 40)
    rgb[delta < 0] = (31, 95, 214)
    return rgb


def upscale(rgb_or_gray: np.ndarray, factor: int) -> Image.Image:
    im = Image.fromarray(rgb_or_gray)
    return im.resize((im.width * factor, im.height * factor), Image.NEAREST)


def draw_boxes(im: Image.Image, boxes, scale: int) -> Image.Image:
    """Vẽ khung (r0,r1,c0,c1,màu,nhãn) lên ảnh PIL đã phóng to."""
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for r0, r1, c0, c1, color, label in boxes:
        d.rectangle([c0 * scale, r0 * scale, c1 * scale - 1, r1 * scale - 1], outline=color, width=3)
        if label:
            d.rectangle([c0 * scale, r0 * scale, c0 * scale + 14 * len(label), r0 * scale + 18], fill=color)
            d.text((c0 * scale + 3, r0 * scale + 3), label, fill="white")
    return im


def rho_histogram(rho, change_mask=None, height=300):
    lr = np.log10(rho.ravel())
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=lr, nbinsx=60, name="mọi pixel", marker_color="#9ca3af",
                               hovertemplate="log₁₀ρ ≈ %{x:.2f}<br>%{y} pixel<extra></extra>"))
    if change_mask is not None and change_mask.any():
        fig.add_trace(go.Histogram(x=np.log10(rho[change_mask]), nbinsx=60, name="pixel bị STC đổi",
                                   marker_color=PLUS,
                                   hovertemplate="log₁₀ρ ≈ %{x:.2f}<br>%{y} pixel đổi<extra></extra>"))
    fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=0.99, x=0.99, xanchor="right",
                                                     bgcolor="rgba(255,255,255,0.8)"))
    fig.update_traces(opacity=0.8)
    fig.update_xaxes(title="log₁₀ ρ (chi phí)")
    fig.update_yaxes(title="số pixel", type="log")
    return _layout(fig, "Histogram chi phí ρ", height)


def percentile_chart(edges, rate, changed, counts, height=330):
    centers = [f"{int(edges[i])}–{int(edges[i + 1])}%" for i in range(len(rate))]
    fig = go.Figure(go.Bar(
        x=centers, y=100 * rate, marker_color=BAR, customdata=np.stack([changed, counts], 1),
        hovertemplate="phân vị ρ %{x}<br>tỉ lệ đổi = %{y:.2f}%<br>%{customdata[0]}/%{customdata[1]} pixel<extra></extra>"))
    fig.update_xaxes(title="Phân vị của ρ (trái = rẻ nhất / texture, phải = đắt nhất / trơn)")
    fig.update_yaxes(title="% pixel bị thay đổi", gridcolor="#e5e7eb")
    return _layout(fig, "Xác suất thay đổi theo chi phí", height, bargap=0.15)


def histogram_overlay(cover, stego, height=320):
    v = np.arange(256)
    hc = np.bincount(cover.ravel(), minlength=256)
    hs = np.bincount(stego.ravel(), minlength=256)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v, y=hc, name="cover", line=dict(color="#374151", width=2),
                             hovertemplate="giá trị %{x}: %{y} pixel<extra>cover</extra>"))
    fig.add_trace(go.Scatter(x=v, y=hs, name="stego", line=dict(color=PLUS, width=2, dash="dot"),
                             hovertemplate="giá trị %{x}: %{y} pixel<extra>stego</extra>"))
    fig.update_xaxes(title="giá trị pixel")
    fig.update_yaxes(title="số pixel", gridcolor="#e5e7eb")
    fig.update_layout(legend=dict(orientation="h", y=1.12), hovermode="x unified")
    return _layout(fig, "Histogram cover vs stego (chồng nhau)", height)


def zoom_values(block, changed, title, height=430):
    """Phóng to 16×16 kèm giá trị số; ô bị đổi có viền đỏ/xanh."""
    text = [[str(int(v)) for v in row] for row in block]
    fig = go.Figure(go.Heatmap(z=block, colorscale="gray", zmin=0, zmax=255, showscale=False,
                               text=text, texttemplate="%{text}", textfont=dict(size=9),
                               hovertemplate="(%{y},%{x}) = %{z}<extra></extra>"))
    rr, cc = np.nonzero(changed)
    for r, c in zip(rr, cc):
        color = PLUS if changed[r, c] > 0 else MINUS
        fig.add_shape(type="rect", x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
                      line=dict(color=color, width=3))
    fig.update_yaxes(autorange="reversed", scaleanchor="x", visible=False)
    fig.update_xaxes(visible=False)
    return _layout(fig, title, height)


def trellis_figure(tr, height=460):
    """Vẽ trellis của một khối nhỏ: nút = (bước, trạng thái), cạnh y=0 / y=1 / dịch,
    đường Viterbi chi phí thấp nhất tô đậm."""
    S = 1 << tr.h
    K = len(tr.weights)
    fig = go.Figure()
    best = tr.best_path
    best_edges = set()
    for k in range(K - 1):
        best_edges.add((k, best[k], best[k + 1]))
    # cạnh thường
    for kind_y, color, name in ((0, "#c7cdd6", "cạnh y=0 (giữ trạng thái)"), (1, "#f2b8b5", "cạnh y=1"),
                                (None, "#d9c99a", "dịch (kiểm tra bit m)")):
        xs, ys = [], []
        for k, e in enumerate(tr.edges):
            for s, t, yb, _c in e:
                if yb == kind_y and (k, s, t) not in best_edges:
                    xs += [k, k + 1, None]
                    ys += [s, t, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.2),
                                 name=name, hoverinfo="skip"))
    # đường tối ưu
    fig.add_trace(go.Scatter(x=list(range(K)), y=best, mode="lines", line=dict(color=INK, width=4),
                             name="đường Viterbi tối ưu", hoverinfo="skip"))
    # nút
    xs, ys, txt, col = [], [], [], []
    for k, w in enumerate(tr.weights):
        for s in range(S):
            if np.isfinite(w[s]):
                xs.append(k)
                ys.append(s)
                txt.append(f"bước {k}, trạng thái {s:0{tr.h}b}<br>chi phí tích luỹ = {w[s]:.3f}")
                col.append(INK if best[k] == s else "#6b7280")
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(size=9, color=col,
                             line=dict(color="white", width=2)), text=txt,
                             hovertemplate="%{text}<extra></extra>", name="trạng thái"))
    # nhãn bit y trên đường tối ưu
    ann_y = []
    bit_i = 0
    for k, kind in enumerate(tr.kinds):
        if kind == "bit":
            yv = int(tr.y[bit_i])
            ch = yv != int(tr.x[bit_i])
            ann_y.append(dict(x=k + 0.5, y=(best[k] + best[k + 1]) / 2, text=f"y={yv}" + ("✱" if ch else ""),
                              showarrow=False, font=dict(size=10, color=PLUS if ch else INK),
                              bgcolor="rgba(255,255,255,0.85)"))
            bit_i += 1
    fig.update_layout(annotations=ann_y)
    ticks = ["start"] + tr.labels
    fig.update_xaxes(tickvals=list(range(K)), ticktext=ticks, tickangle=0, showgrid=False)
    fig.update_yaxes(tickvals=list(range(S)), ticktext=[f"{s:0{tr.h}b}" for s in range(S)],
                     title="trạng thái (syndrome đang mở)", autorange="reversed", showgrid=False)
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    return _layout(fig, None, height)


def side_by_side_change_maps(maps: list[tuple[str, np.ndarray, np.ndarray]], height=360):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(maps), subplot_titles=[m[0] for m in maps],
                        horizontal_spacing=0.02)
    for i, (_t, cover, delta) in enumerate(maps, start=1):
        rgb = change_map_rgb(cover, delta)
        fig.add_trace(go.Image(z=rgb, hovertemplate="hàng %{y}, cột %{x}<extra></extra>"), 1, i)
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, row=1, col=i)
    _layout(fig, None, height)
    fig.update_layout(margin=dict(t=40))
    return fig
