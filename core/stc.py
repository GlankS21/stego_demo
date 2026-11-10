"""Bước 5: Syndrome-Trellis Codes (STC) — cài đặt thật bằng thuật toán Viterbi.

Tài liệu: T. Filler, J. Judas, J. Fridrich, "Minimizing Additive Distortion in Steganography
Using Syndrome-Trellis Codes", IEEE TIFS 6(3), 2011.

Bài toán
--------
Cho vector LSB của cover x ∈ {0,1}^n, chi phí ρ ∈ R+^n và thông điệp m ∈ {0,1}^M.
Tìm y ∈ {0,1}^n sao cho  H·y = m (mod 2)  và  D(x,y) = Σ ρ_i |x_i − y_i|  nhỏ nhất.

Ma trận kiểm tra H (M×n) được ghép từ ma trận con Ĥ (h×w) đặt chéo xuống từng hàng:
khối i (gồm w_i cột) chiếm các hàng i, i+1, …, i+h−1 (cắt bớt ở cuối). Vì H có cấu trúc
dải băng (band), mọi nghiệm y có thể biểu diễn là đường đi trong một trellis có 2^h trạng
thái; Viterbi tìm đường đi chi phí nhỏ nhất trong O(n·2^h).

Quy ước bit
-----------
Mỗi cột Ĥ lưu thành một số nguyên h-bit: bit r = Ĥ[r, j]. Trạng thái trellis là h bit
syndrome "đang mở" (bit 0 = hàng hiện tại i). Sau khi xử lý hết khối i, bit 0 phải bằng m_i;
ta "đẩy" bit này ra (dịch phải 1 bit) và chuyển sang hàng i+1.

Ở đây Ĥ có hàng đầu và hàng cuối toàn 1 (như khuyến nghị của bài báo) ⇒ bài toán luôn có
nghiệm, và mỗi bit y_j ảnh hưởng tới ít nhất bit m_i của khối mình.

Độ rộng khối thay đổi: w_i = ⌊(i+1)n/M⌋ − ⌊i·n/M⌋ (tỉ lệ nhúng α = M/n tuỳ ý, MỌI pixel
trong vùng đều được dùng ⇒ sửa bất kỳ LSB nào cũng làm sai syndrome).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def generate_hhat(h: int, w: int, seed: int) -> np.ndarray:
    """Sinh ma trận con Ĥ (h×w) giả ngẫu nhiên, trả về w số nguyên h-bit (mỗi số = 1 cột).

    Bit 0 (hàng đầu) và bit h−1 (hàng cuối) luôn = 1.
    """
    if not 1 <= h <= 20:
        raise ValueError("h phải trong khoảng 1..20")
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, 1 << h, size=max(w, 1), dtype=np.int64)
    cols |= 1 | (1 << (h - 1))
    return cols


def hhat_matrix(cols: np.ndarray, h: int) -> np.ndarray:
    """Chuyển các cột số nguyên → ma trận nhị phân h×w (để hiển thị)."""
    return ((cols[None, :] >> np.arange(h)[:, None]) & 1).astype(np.uint8)


def block_widths(n: int, m: int) -> np.ndarray:
    """Độ rộng từng khối sao cho Σ w_i = n và các w_i chênh nhau tối đa 1."""
    if m <= 0:
        return np.zeros(0, dtype=np.int64)
    if n < m:
        raise ValueError(f"Không đủ pixel: n={n} < M={m} bit")
    b = (np.arange(m + 1, dtype=np.int64) * n) // m
    return np.diff(b)


def syndrome(y: np.ndarray, m: int, h: int, cols: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Tính m' = H·y (mod 2) — đây cũng chính là thuật toán TRÍCH XUẤT.

    Với mỗi vị trí j thuộc khối i, nếu y_j = 1 thì syndrome hàng i+r bị đảo khi Ĥ[r, j_local]=1.
    """
    if m == 0:
        return np.zeros(0, dtype=np.uint8)
    y = np.asarray(y, dtype=np.int64)
    n = int(widths.sum())
    blk = np.repeat(np.arange(m, dtype=np.int64), widths)           # khối của từng vị trí
    local = np.arange(n, dtype=np.int64) - np.repeat(np.cumsum(widths) - widths, widths)
    colv = cols[local]
    s = np.zeros(m + h, dtype=np.int64)
    for r in range(h):
        bit = ((colv >> r) & 1) & y[:n]
        s += np.bincount(blk + r, weights=bit, minlength=m + h).astype(np.int64)
    return (s[:m] % 2).astype(np.uint8)


@dataclass
class STCResult:
    y: np.ndarray            # LSB stego (0/1), độ dài n
    cost: float              # Σ ρ_i |x_i − y_i| tối ưu
    widths: np.ndarray
    cols: np.ndarray


def stc_embed(x: np.ndarray, rho: np.ndarray, msg: np.ndarray, h: int, cols: np.ndarray,
              widths: np.ndarray | None = None) -> STCResult:
    """Viterbi trên trellis: trả về y với H·y = msg và Σρ|x−y| nhỏ nhất.

    Chi phí cạnh: chọn y_j = x_j tốn 0, chọn y_j ≠ x_j tốn ρ_j.
    Cạnh "y_j = 0": trạng thái giữ nguyên; cạnh "y_j = 1": trạng thái XOR với cột Ĥ.
    """
    x = np.asarray(x, dtype=np.uint8).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    msg = np.asarray(msg, dtype=np.uint8).ravel()
    n, m = x.size, msg.size
    if m == 0:
        return STCResult(y=x.copy(), cost=0.0, widths=np.zeros(0, np.int64), cols=cols)
    if widths is None:
        widths = block_widths(n, m)
    if int(widths.sum()) != n:
        raise ValueError("Tổng độ rộng khối phải bằng n")
    if np.any(widths > len(cols)):
        raise ValueError("Ĥ có ít cột hơn độ rộng khối")

    S = 1 << h
    states = np.arange(S, dtype=np.int64)
    INF = np.inf
    wght = np.full(S, INF)
    wght[0] = 0.0
    nbytes = (S + 7) // 8
    path = np.empty((n, nbytes), dtype=np.uint8)   # bit = 1 ⇔ cạnh tốt nhất là y_j = 1
    rho = np.where(np.isfinite(rho), rho, 1e300)   # an toàn số học (∞·0 = nan)

    pos = 0
    full_mask = S - 1
    for i in range(m):
        rows_left = min(h, m - i)                   # cắt Ĥ ở các khối cuối
        mask = (1 << rows_left) - 1
        for j in range(int(widths[i])):
            col = int(cols[j]) & mask if mask != full_mask else int(cols[j])
            r = rho[pos]
            if x[pos]:
                w0 = wght + r                        # y=0 ≠ x=1 → tốn ρ
                w1 = wght[states ^ col]              # y=1 = x → miễn phí
            else:
                w0 = wght                            # y=0 = x → miễn phí
                w1 = wght[states ^ col] + r          # y=1 ≠ x → tốn ρ
            choose1 = w1 < w0
            wght = np.where(choose1, w1, w0)
            path[pos] = np.packbits(choose1)
            pos += 1
        # Kết thúc khối i: bit syndrome hàng i (bit 0 của trạng thái) phải = m_i → dịch ra.
        bit = int(msg[i])
        wght = np.concatenate([wght[bit::2], np.full(S // 2, INF)])

    cost = float(wght[0])
    if not np.isfinite(cost):
        raise RuntimeError("STC không tìm được nghiệm (không thể xảy ra nếu Ĥ hợp lệ)")

    # Truy vết ngược (backtracking)
    y = np.zeros(n, dtype=np.uint8)
    state = 0
    pos = n
    for i in range(m - 1, -1, -1):
        state = 2 * state + int(msg[i])
        rows_left = min(h, m - i)
        mask = (1 << rows_left) - 1
        for j in range(int(widths[i]) - 1, -1, -1):
            pos -= 1
            b = (path[pos, state >> 3] >> (7 - (state & 7))) & 1
            y[pos] = b
            if b:
                state ^= int(cols[j]) & mask
    assert state == 0
    return STCResult(y=y, cost=cost, widths=widths, cols=cols)


# ---------------------------------------------------------------------------
# Phiên bản "có ghi vết" cho trellis nhỏ (minh hoạ trên giao diện)
# ---------------------------------------------------------------------------

@dataclass
class TrellisTrace:
    h: int
    n: int
    m: int
    x: np.ndarray
    rho: np.ndarray
    msg: np.ndarray
    cols: np.ndarray
    widths: np.ndarray
    # weights[k] = vector chi phí 2^h trạng thái tại "cột" k của trellis
    # (gồm cả các cột sau phép dịch cuối khối)
    weights: list
    # edges[k] = list (s_from, s_to, y, cost_edge) giữa cột k và k+1, kind = 'bit' | 'shift'
    edges: list
    kinds: list
    labels: list
    best_path: list          # chuỗi trạng thái trên đường tối ưu
    y: np.ndarray
    cost: float


def stc_embed_trace(x, rho, msg, h, cols, widths=None) -> TrellisTrace:
    """Viterbi viết bằng vòng lặp thuần (dễ đọc), lưu toàn bộ trellis để vẽ."""
    x = np.asarray(x, dtype=np.uint8)
    rho = np.asarray(rho, dtype=np.float64)
    msg = np.asarray(msg, dtype=np.uint8)
    n, m = len(x), len(msg)
    if widths is None:
        widths = block_widths(n, m)
    S = 1 << h
    INF = float("inf")
    wght = [INF] * S
    wght[0] = 0.0
    weights = [list(wght)]
    edges, kinds, labels, back = [], [], [], []
    pos = 0
    for i in range(m):
        mask = (1 << min(h, m - i)) - 1
        for j in range(int(widths[i])):
            col = int(cols[j]) & mask
            new = [INF] * S
            bp = [None] * S
            e = []
            for s in range(S):
                if wght[s] == INF:
                    continue
                for yb in (0, 1):
                    t = s ^ col if yb else s
                    c = rho[pos] if yb != x[pos] else 0.0
                    e.append((s, t, yb, c))
                    # hoà chi phí → ưu tiên y=0 (giống hệt bản vector hoá)
                    if wght[s] + c < new[t] or (wght[s] + c == new[t] and yb == 0):
                        new[t] = wght[s] + c
                        bp[t] = (s, yb)
            edges.append(e)
            kinds.append("bit")
            labels.append(f"x{pos + 1}")
            back.append(bp)
            wght = new
            weights.append(list(wght))
            pos += 1
        bit = int(msg[i])
        new = [INF] * S
        bp = [None] * S
        e = []
        for s in range(S):
            if wght[s] < INF and (s & 1) == bit:
                new[s >> 1] = wght[s]
                bp[s >> 1] = (s, None)
                e.append((s, s >> 1, None, 0.0))
        edges.append(e)
        kinds.append("shift")
        labels.append(f"m{i + 1}={bit}")
        back.append(bp)
        wght = new
        weights.append(list(wght))

    # truy vết
    state = 0
    best = [state]
    y = np.zeros(n, dtype=np.uint8)
    pos = n
    for k in range(len(back) - 1, -1, -1):
        s, yb = back[k][state]
        if kinds[k] == "bit":
            pos -= 1
            y[pos] = yb
        state = s
        best.append(state)
    best.reverse()
    return TrellisTrace(h=h, n=n, m=m, x=x, rho=rho, msg=msg, cols=np.asarray(cols), widths=widths,
                        weights=weights, edges=edges, kinds=kinds, labels=labels,
                        best_path=best, y=y, cost=float(wght[0]))
