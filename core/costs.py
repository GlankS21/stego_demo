"""Bước 4: chi phí biến đổi ρ.

ρ_ij là "giá" (độ méo thống kê) khi thay đổi pixel (i,j) đi ±1. STC sẽ tìm cách nhúng
sao cho tổng Σ ρ_ij·|x_ij − y_ij| nhỏ nhất ⇒ pixel rẻ (texture) được ưu tiên.

* HILL (Li et al. 2014):   ρ = L2 ⋆ 1/(Q + ε), L2 = trung bình 15×15
  (Q = (|I⋆KB| ⋆ L1)/scale; bước low-pass L2 "lan" chi phí cao của vùng trơn sang lân cận,
  để các thay đổi gom vào vùng texture lớn).
* WOW-like (Holub & Fridrich 2012):  ρ = Σ_k ξ_k^{-1} = 1/ξ_WOW  ⇒ ρ = 1/(Q + ε).
* Đồng đều (đối chứng): ρ = 1 ⇒ STC chỉ tối thiểu hoá SỐ pixel thay đổi, không quan tâm vị trí.

Hướng thay đổi làm tràn [0,255] bị cấm: ρ⁺ = ∞ tại pixel 255, ρ⁻ = ∞ tại pixel 0.
STC nhị phân (thay đổi LSB) dùng ρ = min(ρ⁺, ρ⁻) — với ảnh 8-bit luôn có ít nhất
một hướng hợp lệ nên ρ luôn hữu hạn.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import box, compute_Q

COST_NAMES = ("HILL", "WOW-like", "Đồng đều")
EPS = 1e-3          # ε trong 1/(Q + ε) — giới hạn trên của chi phí ≈ 1/ε
HILL_L2 = 15        # kích thước bộ lọc L2 của HILL (theo bài báo)


@dataclass
class CostMaps:
    Q: np.ndarray           # bản đồ Q ∈ [0,1]
    rho: np.ndarray         # chi phí nhị phân (dùng cho STC), hữu hạn
    rho_plus: np.ndarray    # chi phí của +1 (∞ nếu pixel = 255)
    rho_minus: np.ndarray   # chi phí của −1 (∞ nếu pixel = 0)
    name: str


def rho_from_Q(Q: np.ndarray, cost_name: str) -> np.ndarray:
    if cost_name == "HILL":
        return box(1.0 / (Q + EPS), HILL_L2)
    if cost_name == "WOW-like":
        return 1.0 / (Q + EPS)
    if cost_name == "Đồng đều":
        return np.ones_like(Q, dtype=np.float64)
    raise ValueError(f"Hàm chi phí không hỗ trợ: {cost_name}")


def compute_costs(img: np.ndarray, cost_name: str = "HILL") -> CostMaps:
    Q = compute_Q(img, cost_name)
    rho = rho_from_Q(Q, cost_name)
    rho_plus = rho.copy()
    rho_minus = rho.copy()
    rho_plus[img == 255] = np.inf    # +1 sẽ tràn lên 256
    rho_minus[img == 0] = np.inf     # −1 sẽ tràn xuống −1
    rho_bin = np.minimum(rho_plus, rho_minus)
    return CostMaps(Q=Q, rho=rho_bin, rho_plus=rho_plus, rho_minus=rho_minus, name=cost_name)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Phân vị (0..100) của từng phần tử: tỉ lệ % số pixel có ρ NHỎ HƠN pixel này."""
    flat = values.ravel()
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(flat.size)
    return (ranks / max(flat.size - 1, 1) * 100.0).reshape(values.shape)
