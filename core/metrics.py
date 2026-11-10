"""Số liệu so sánh cover/stego (NT3) và thống kê theo chi phí (NT1)."""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .costs import percentile_rank


def image_metrics(cover: np.ndarray, stego: np.ndarray, payload_bits: int) -> dict:
    c = cover.astype(np.int16)
    s = stego.astype(np.int16)
    diff = s - c
    n = cover.size
    n_changed = int(np.count_nonzero(diff))
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0 else float(peak_signal_noise_ratio(cover, stego, data_range=255))
    ssim = float(structural_similarity(cover, stego, data_range=255))
    return {
        "n_pixels": n,
        "n_changed": n_changed,
        "pct_changed": 100.0 * n_changed / n,
        "max_abs_diff": int(np.abs(diff).max()) if n else 0,
        "n_plus": int((diff > 0).sum()),
        "n_minus": int((diff < 0).sum()),
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "payload_bits": int(payload_bits),
        "bpp": payload_bits / n,
        "efficiency": (payload_bits / n_changed) if n_changed else float("inf"),
        "min_val": int(s.min()),
        "max_val": int(s.max()),
    }


def change_rate_by_cost_percentile(rho: np.ndarray, change_mask: np.ndarray, n_bins: int = 10):
    """Tỉ lệ pixel bị đổi trong từng nhóm phân vị của ρ (nhóm 1 = rẻ nhất)."""
    pr = percentile_rank(rho).ravel()
    m = change_mask.ravel()
    bins = np.minimum((pr / 100.0 * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    changed = np.bincount(bins, weights=m.astype(float), minlength=n_bins)
    rate = np.divide(changed, counts, out=np.zeros(n_bins), where=counts > 0)
    edges = np.linspace(0, 100, n_bins + 1)
    return edges, rate, changed.astype(int), counts


def region_stats(Q, rho, change_mask, r0, r1, c0, c1) -> dict:
    sl = (slice(r0, r1), slice(c0, c1))
    n = max((r1 - r0) * (c1 - c0), 1)
    k = int(change_mask[sl].sum())
    return {
        "Kích thước": f"{r1 - r0}×{c1 - c0}",
        "Q trung bình": float(Q[sl].mean()),
        "ρ trung bình": float(rho[sl].mean()),
        "ρ trung vị": float(np.median(rho[sl])),
        "Số pixel bị đổi": k,
        "% pixel bị đổi": 100.0 * k / n,
    }


def conclusion_sentence(m: dict, texture_share: float | None) -> str:
    """Câu kết luận tự động sinh từ số liệu (NT3)."""
    parts = [
        f"Thay đổi {m['n_changed']}/{m['n_pixels']} pixel ({m['pct_changed']:.2f}%)",
        f"mỗi pixel tối đa ±{m['max_abs_diff']}",
        f"PSNR {m['psnr']:.1f} dB" if np.isfinite(m["psnr"]) else "PSNR = ∞ (không đổi pixel nào)",
        f"SSIM {m['ssim']:.5f}",
    ]
    s = ", ".join(parts)
    if texture_share is not None and m["n_changed"]:
        s += (f"; {texture_share:.0f}% số thay đổi nằm ở nửa ảnh có ρ thấp (vùng texture)")
    if m["max_abs_diff"] <= 1 and (not np.isfinite(m["psnr"]) or m["psnr"] >= 48):
        s += ". Mắt người không phân biệt được ảnh stego với ảnh gốc."
    elif m["max_abs_diff"] <= 1:
        s += ". Thay đổi chỉ ±1 nhưng khá nhiều pixel — vẫn khó thấy bằng mắt, song dễ bị steganalysis hơn."
    else:
        s += ". CẢNH BÁO: có pixel đổi quá ±1!"
    return s
