"""Bước 1–3 của pipeline: ảnh gốc → đặc trưng → bản đồ Q.

Các đặc trưng đo "độ phức tạp cục bộ" của ảnh. Ý tưởng chung của giấu tin thích nghi:
một thay đổi ±1 ở vùng texture (cỏ, tóc) bị "chìm" trong nhiễu tự nhiên, còn ở vùng trơn
(bầu trời) thì nó tạo ra một dị thường thống kê dễ bị steganalysis phát hiện.

Tài liệu tham khảo
------------------
* B. Li, M. Wang, J. Huang, X. Li, "A new cost function for spatial image steganography",
  ICIP 2014 (HILL): R = I ⋆ KB,  ξ = |R| ⋆ L1,  ρ = (1/ξ) ⋆ L2.
* V. Holub, J. Fridrich, "Designing steganographic distortion using directional filters",
  WIFS 2012 (WOW): residual theo 3 hướng wavelet (LH, HL, HH) dùng bộ lọc Daubechies-8.
* V. Holub, J. Fridrich, T. Denemark, "Universal distortion function for steganography in an
  arbitrary domain", EURASIP JIS 2014 (S-UNIWARD) — cùng ngân hàng bộ lọc với WOW.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from math import comb

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import convolve2d

# ---------------------------------------------------------------------------
# Bước 1: ảnh gốc
# ---------------------------------------------------------------------------

def load_gray(source, resize: int | None = 256) -> np.ndarray:
    """Đọc ảnh (đường dẫn, bytes hoặc file-like) → ảnh xám 8-bit (uint8, H×W).

    Chuyển xám theo ITU-R 601 (mode "L" của Pillow). Nếu `resize` khác None thì
    resize về resize×resize bằng LANCZOS (chỉ áp dụng cho ẢNH GỐC, không bao giờ cho stego).
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source)
    img = img.convert("L")
    if resize:
        img = img.resize((resize, resize), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8).copy()


# ---------------------------------------------------------------------------
# Các bộ lọc
# ---------------------------------------------------------------------------

# Bộ lọc KB (Ker–Böhme) 3×3 — bộ lọc high-pass được HILL dùng làm bước đầu.
KB = np.array([[-1, 2, -1],
               [2, -4, 2],
               [-1, 2, -1]], dtype=np.float64)


def daubechies_lowpass(p: int = 8) -> np.ndarray:
    """Hệ số bộ lọc thông thấp Daubechies-p (2p tap) tính bằng phân tích phổ.

    P(y) = Σ_{k<p} C(p-1+k, k) y^k; mỗi nghiệm y_j cho một nghiệm z_j trong đường tròn
    đơn vị; h(z) ∝ (1+z)^p · Π(z - z_j). Chuẩn hoá Σh = √2. (Thuật toán giống
    `scipy.signal.daub` cũ — đã bị xoá khỏi SciPy ≥1.15 nên tự cài lại.)
    """
    coeffs = [comb(p - 1 + k, k) for k in range(p)][::-1]
    roots_y = np.roots(coeffs)
    q = np.poly1d([1.0])
    for y in roots_y[: p - 1]:
        part = 2 * np.sqrt(y * (y - 1))
        const = 1 - 2 * y
        z = const + part
        if abs(z) < 1:
            z = const - part
        q = q * np.poly1d([1, -z])
    q = np.poly1d([1, 1]) ** p * np.real(q)
    h = q.c[::-1].real
    h = h / h.sum() * np.sqrt(2)
    return h


def wavelet_kernels(p: int = 8) -> dict[str, np.ndarray]:
    """Ba bộ lọc định hướng của WOW / S-UNIWARD: K_LH = h·gᵀ, K_HL = g·hᵀ, K_HH = g·gᵀ.

    h: thông thấp Daubechies, g: thông cao tương ứng g[n] = (-1)^n h[L-1-n].
    """
    h = daubechies_lowpass(p)
    n = np.arange(len(h))
    g = ((-1) ** n) * h[::-1]
    return {
        "LH": np.outer(h, g),  # nhạy cạnh ngang
        "HL": np.outer(g, h),  # nhạy cạnh dọc
        "HH": np.outer(g, g),  # nhạy cạnh chéo
    }


def _conv_same(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Tích chập 2D 'same' với biên phản xạ (symmetric) — tránh hiệu ứng biên giả."""
    ph, pw = k.shape[0] // 2, k.shape[1] // 2
    padded = np.pad(img, ((ph, k.shape[0] - 1 - ph), (pw, k.shape[1] - 1 - pw)), mode="symmetric")
    return convolve2d(padded, k, mode="valid")


def box(img: np.ndarray, size: int) -> np.ndarray:
    """Bộ lọc thông thấp trung bình size×size (L1, L2 của HILL)."""
    return ndimage.uniform_filter(img.astype(np.float64), size=size, mode="reflect")


# ---------------------------------------------------------------------------
# Bước 2: đặc trưng
# ---------------------------------------------------------------------------

def kb_residual(img: np.ndarray) -> np.ndarray:
    """R = I ⋆ KB — phần dư high-pass (≈0 ở vùng trơn, lớn ở vùng texture/cạnh)."""
    return _conv_same(img.astype(np.float64), KB)


def local_std(img: np.ndarray, size: int = 5) -> np.ndarray:
    """Độ lệch chuẩn cục bộ σ = sqrt(E[I²] − E[I]²) trên cửa sổ size×size."""
    f = img.astype(np.float64)
    m = box(f, size)
    m2 = box(f * f, size)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """|∇I| = sqrt(Sobel_x² + Sobel_y²)."""
    f = img.astype(np.float64)
    gx = ndimage.sobel(f, axis=1, mode="reflect")
    gy = ndimage.sobel(f, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def wavelet_residuals(img: np.ndarray) -> dict[str, np.ndarray]:
    """W_k = I ⋆ K_k cho k ∈ {LH, HL, HH} (residual định hướng của WOW/S-UNIWARD)."""
    f = img.astype(np.float64)
    return {k: _conv_same(f, K) for k, K in wavelet_kernels().items()}


@dataclass
class FeatureSet:
    kb_residual: np.ndarray          # |I ⋆ KB|
    local_std: np.ndarray            # σ cục bộ 5×5
    gradient: np.ndarray             # |∇I|
    wavelet: np.ndarray              # Σ_k |W_k|
    wavelet_dirs: dict = field(default_factory=dict)  # |W_LH|, |W_HL|, |W_HH|

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "|KB residual|": self.kb_residual,
            "Độ lệch chuẩn cục bộ σ (5×5)": self.local_std,
            "Gradient |∇I| (Sobel)": self.gradient,
            "Wavelet residual Σ|W_k| (db8)": self.wavelet,
        }


def compute_features(img: np.ndarray) -> FeatureSet:
    wr = wavelet_residuals(img)
    wabs = {k: np.abs(v) for k, v in wr.items()}
    return FeatureSet(
        kb_residual=np.abs(kb_residual(img)),
        local_std=local_std(img),
        gradient=gradient_magnitude(img),
        wavelet=sum(wabs.values()),
        wavelet_dirs=wabs,
    )


# ---------------------------------------------------------------------------
# Bước 3: bản đồ Q ∈ [0, 1]
# ---------------------------------------------------------------------------

def normalize01(xi: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """Chuẩn hoá robust về [0,1]: Q = clip(ξ / percentile_99(ξ), 0, 1)."""
    scale = np.percentile(xi, pct)
    if scale <= 0:
        scale = xi.max() if xi.max() > 0 else 1.0
    return np.clip(xi / scale, 0.0, 1.0)


def hill_texture(img: np.ndarray) -> np.ndarray:
    """ξ_HILL = |I ⋆ KB| ⋆ L1 (L1 = trung bình 3×3): high-pass → |·| → low-pass."""
    return box(np.abs(kb_residual(img)), 3)


def wow_texture(img: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """ξ_WOW = (Σ_k ξ_k^{-1})^{-1}, ξ_k = |W_k| ⋆ |K_k| (tổng hợp kiểu Hölder p = −1).

    Một pixel chỉ "có texture" theo WOW khi nó phức tạp theo CẢ BA hướng; một cạnh thẳng
    (chỉ phức tạp theo 1 hướng) vẫn bị coi là dễ dự đoán → chi phí cao.
    """
    kernels = wavelet_kernels()
    f = img.astype(np.float64)
    xi = {}
    for k, K in kernels.items():
        w = np.abs(_conv_same(f, K))
        xi[k] = _conv_same(w, np.abs(K)[::-1, ::-1])
    harmonic = 1.0 / sum(1.0 / (v + 1e-10) for v in xi.values())
    return harmonic, xi


def compute_Q(img: np.ndarray, cost_name: str) -> np.ndarray:
    """Q ∈ [0,1]: 1 = texture rất cao (thay đổi khó lộ), 0 = vùng trơn (thay đổi dễ lộ)."""
    if cost_name == "HILL":
        return normalize01(hill_texture(img))
    if cost_name == "WOW-like":
        return normalize01(wow_texture(img)[0])
    if cost_name == "Đồng đều":
        return np.full(img.shape, 0.5)
    raise ValueError(f"Hàm chi phí không hỗ trợ: {cost_name}")
