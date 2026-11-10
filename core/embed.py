"""Bước 5–7: chuẩn bị payload, nhúng bằng STC + LSB matching ±1, trích xuất.

Định dạng dữ liệu nhúng
-----------------------
Thông điệp UTF-8 → bytes. Header 88 bit (big-endian):

    magic  16 bit  = 0x5354 ("ST")
    h       8 bit  = tham số h của STC dùng cho phần payload
    length 32 bit  = số byte của thông điệp
    crc32  32 bit  = CRC32 của thông điệp gốc (bản rõ)

Header và payload được MÃ HOÁ bằng XOR với dòng khoá SHA-256(key ‖ nhãn ‖ bộ đếm).
Thứ tự pixel bị HOÁN VỊ bằng PRNG có seed = SHA-256(key ‖ "perm").
Sau hoán vị: 1024 pixel đầu là vùng header (STC h=7 cố định — người nhận biết trước),
phần còn lại là vùng payload (STC với h đọc từ header, độ rộng khối suy ra từ length).

Nhờ vậy người nhận CHỈ cần file stego + khoá; mọi tham số khác nằm trong header.

LSB matching ±1: STC quyết định pixel nào phải đổi LSB; dấu ±1 được chọn ngẫu nhiên (PRNG
theo khoá) nhưng bắt buộc +1 tại pixel 0 và −1 tại pixel 255 (không tràn [0,255]).
Cả +1 và −1 đều đảo LSB nên người nhận chỉ cần đọc LSB.
"""
from __future__ import annotations

import hashlib
import io
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .costs import CostMaps, compute_costs
from .stc import block_widths, generate_hhat, stc_embed, syndrome

MAGIC = 0x5354
HEADER_BITS = 16 + 8 + 32 + 32
HEADER_PIXELS = 1024
HEADER_H = 7
MAX_PAYLOAD_RATE = 0.5          # tối đa 0.5 bit / pixel trong vùng payload (w ≥ 2)
H_RANGE = (1, 14)


class CapacityError(ValueError):
    """Thông điệp vượt quá dung lượng — báo TRƯỚC khi nhúng."""


# ---------------------------------------------------------------------------
# Tiện ích khoá, bit, mã hoá
# ---------------------------------------------------------------------------

def derive_seed(key: str, label: str) -> int:
    d = hashlib.sha256(key.encode("utf-8") + b"|" + label.encode("utf-8")).digest()
    return int.from_bytes(d[:8], "big")


def keystream_bits(key: str, label: str, nbits: int) -> np.ndarray:
    """Dòng khoá SHA-256 chế độ bộ đếm (counter mode)."""
    nbytes = (nbits + 7) // 8
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        out += hashlib.sha256(f"{key}|{label}|{ctr}".encode("utf-8")).digest()
        ctr += 1
    return np.unpackbits(np.frombuffer(bytes(out[:nbytes]), dtype=np.uint8))[:nbits]


def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)) if b else np.zeros(0, np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes() if len(bits) else b""


def pixel_permutation(key: str, n: int) -> np.ndarray:
    return np.random.default_rng(derive_seed(key, "perm")).permutation(n)


def capacity_bytes(shape) -> int:
    """Dung lượng tối đa (byte) cho ảnh kích thước `shape`."""
    n = int(np.prod(shape))
    return max(0, int((n - HEADER_PIXELS) * MAX_PAYLOAD_RATE) // 8)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class Payload:
    message: str
    msg_bytes: bytes
    crc: int
    h: int
    header_plain: np.ndarray
    header_enc: np.ndarray
    payload_plain: np.ndarray
    payload_enc: np.ndarray


def build_payload(message: str, key: str, h: int) -> Payload:
    data = message.encode("utf-8")
    crc = zlib.crc32(data) & 0xFFFFFFFF
    header = struct.pack(">HBII", MAGIC, h, len(data), crc)
    hp = bytes_to_bits(header)
    pp = bytes_to_bits(data)
    return Payload(
        message=message, msg_bytes=data, crc=crc, h=h,
        header_plain=hp, header_enc=hp ^ keystream_bits(key, "header", len(hp)),
        payload_plain=pp, payload_enc=pp ^ keystream_bits(key, "payload", len(pp)),
    )


# ---------------------------------------------------------------------------
# Nhúng
# ---------------------------------------------------------------------------

@dataclass
class EmbedResult:
    cover: np.ndarray
    stego: np.ndarray
    costs: CostMaps
    payload: Payload
    perm: np.ndarray
    x_bits: np.ndarray                 # LSB cover theo thứ tự hoán vị
    y_bits: np.ndarray                 # LSB stego do STC trả về (cùng thứ tự)
    change_mask: np.ndarray            # (H×W) bool — lấy TRỰC TIẾP từ y ≠ x
    delta: np.ndarray                  # (H×W) int8 ∈ {−1,0,+1}
    changes: dict = field(default_factory=dict)   # bảng pixel thay đổi (cột → mảng)
    stc_cost: float = 0.0              # Σ ρ tại pixel bị đổi (= cost Viterbi)
    header_widths: np.ndarray | None = None
    payload_widths: np.ndarray | None = None
    payload_cols: np.ndarray | None = None
    header_cols: np.ndarray | None = None

    @property
    def n_changes(self) -> int:
        return int(self.change_mask.sum())

    @property
    def total_bits(self) -> int:
        return len(self.payload.header_enc) + len(self.payload.payload_enc)


def _signs(key: str, values: np.ndarray, rho_plus: np.ndarray, rho_minus: np.ndarray) -> np.ndarray:
    """Chọn dấu ±1 cho các pixel phải đổi: không bao giờ tràn khỏi [0,255]."""
    rng = np.random.default_rng(derive_seed(key, "sign"))
    s = np.where(rng.random(values.size) < 0.5, 1, -1).astype(np.int16)
    s[~np.isfinite(rho_plus)] = -1      # pixel 255 → chỉ được −1
    s[~np.isfinite(rho_minus)] = 1      # pixel 0   → chỉ được +1
    return s


def embed(cover: np.ndarray, message: str, key: str, h: int = 10,
          cost_name: str = "HILL", costs: CostMaps | None = None) -> EmbedResult:
    if cover.dtype != np.uint8 or cover.ndim != 2:
        raise ValueError("Cover phải là ảnh xám uint8 2 chiều")
    if not H_RANGE[0] <= h <= H_RANGE[1]:
        raise ValueError(f"h phải trong {H_RANGE}")
    N = cover.size
    data_len = len(message.encode("utf-8"))
    cap = capacity_bytes(cover.shape)
    if data_len > cap:
        raise CapacityError(
            f"Thông điệp dài {data_len} byte vượt quá dung lượng {cap} byte "
            f"(ảnh {cover.shape[1]}×{cover.shape[0]}, tối đa {MAX_PAYLOAD_RATE} bpp vùng payload). "
            "Hãy rút ngắn thông điệp hoặc dùng ảnh lớn hơn.")
    if costs is None:
        costs = compute_costs(cover, cost_name)
    pl = build_payload(message, key, h)

    perm = pixel_permutation(key, N)
    flat = cover.ravel()
    x = (flat[perm] & 1).astype(np.uint8)
    rho_p = costs.rho.ravel()[perm]

    # --- vùng header (h cố định = 7) ---
    hw = block_widths(HEADER_PIXELS, HEADER_BITS)
    hcols = generate_hhat(HEADER_H, int(hw.max()), derive_seed(key, "H-header"))
    rh = stc_embed(x[:HEADER_PIXELS], rho_p[:HEADER_PIXELS], pl.header_enc, HEADER_H, hcols, hw)

    # --- vùng payload (h do người dùng chọn) ---
    n_reg = N - HEADER_PIXELS
    L = len(pl.payload_enc)
    if L:
        pw = block_widths(n_reg, L)
        pcols = generate_hhat(h, int(pw.max()), derive_seed(key, f"H-payload-{h}"))
        rp = stc_embed(x[HEADER_PIXELS:], rho_p[HEADER_PIXELS:], pl.payload_enc, h, pcols, pw)
        y_pay, cost_pay = rp.y, rp.cost
    else:
        pw, pcols, y_pay, cost_pay = np.zeros(0, np.int64), None, x[HEADER_PIXELS:].copy(), 0.0
    y = np.concatenate([rh.y, y_pay])

    # --- bản đồ thay đổi lấy TRỰC TIẾP từ đầu ra STC: vị trí y ≠ x ---
    changed_perm_pos = np.flatnonzero(y != x)
    flat_idx = perm[changed_perm_pos]
    rho_plus = costs.rho_plus.ravel()[flat_idx]
    rho_minus = costs.rho_minus.ravel()[flat_idx]
    signs = _signs(key, flat[flat_idx], rho_plus, rho_minus)

    delta = np.zeros(N, dtype=np.int16)
    delta[flat_idx] = signs
    stego_flat = flat.astype(np.int16) + delta
    assert stego_flat.min() >= 0 and stego_flat.max() <= 255
    stego = stego_flat.astype(np.uint8).reshape(cover.shape)
    mask = np.zeros(N, dtype=bool)
    mask[flat_idx] = True

    W = cover.shape[1]
    order = np.argsort(flat_idx)
    fi = flat_idx[order]
    changes = {
        "hàng": fi // W,
        "cột": fi % W,
        "giá trị cũ": flat[fi].astype(int),
        "giá trị mới": stego_flat[fi].astype(int),
        "±1": delta[fi].astype(int),
        "ρ": costs.rho.ravel()[fi],
        "Q": costs.Q.ravel()[fi],
        "vùng": np.where(changed_perm_pos[order] < HEADER_PIXELS, "header", "payload"),
    }
    return EmbedResult(
        cover=cover, stego=stego, costs=costs, payload=pl, perm=perm, x_bits=x, y_bits=y,
        change_mask=mask.reshape(cover.shape), delta=delta.reshape(cover.shape).astype(np.int8),
        changes=changes, stc_cost=float(rh.cost + cost_pay),
        header_widths=hw, payload_widths=pw, payload_cols=pcols, header_cols=hcols,
    )


def lsb_sequential(cover: np.ndarray, bits: np.ndarray, key: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Đối chứng: ghi lần lượt từng bit vào LSB pixel theo thứ tự quét (raster), ±1 matching.

    Trả về (stego, change_mask). Không thích nghi, không mã sửa sai.
    """
    flat = cover.ravel().astype(np.int16)
    k = len(bits)
    if k > flat.size:
        raise CapacityError("Quá nhiều bit cho LSB tuần tự")
    need = (flat[:k] & 1) != bits
    idx = np.flatnonzero(need)
    rng = np.random.default_rng(derive_seed(key, "lsb-seq"))
    s = np.where(rng.random(idx.size) < 0.5, 1, -1)
    s[flat[idx] == 255] = -1
    s[flat[idx] == 0] = 1
    out = flat.copy()
    out[idx] += s
    mask = np.zeros(flat.size, dtype=bool)
    mask[idx] = True
    return out.astype(np.uint8).reshape(cover.shape), mask.reshape(cover.shape)


# ---------------------------------------------------------------------------
# Lưu / đọc file
# ---------------------------------------------------------------------------

def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def save_png(img: np.ndarray, path) -> None:
    Image.fromarray(img, mode="L").save(path, format="PNG")


def read_image_file(source) -> np.ndarray:
    """Đọc ảnh stego từ file/bytes — KHÔNG resize, chỉ chuyển sang xám nếu cần."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    img = Image.open(source)
    if img.mode != "L":
        img = img.convert("L")
    return np.asarray(img, dtype=np.uint8).copy()


def to_jpeg_bytes(img: np.ndarray, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Trích xuất — chỉ dùng ảnh stego + khoá
# ---------------------------------------------------------------------------

@dataclass
class ExtractResult:
    ok: bool
    status: str
    message: str | None = None
    msg_bytes: bytes | None = None
    magic_ok: bool = False
    h: int | None = None
    length: int | None = None
    crc_header: int | None = None
    crc_computed: int | None = None
    reasons: list = field(default_factory=list)

    @property
    def checksum_ok(self) -> bool:
        return self.crc_header is not None and self.crc_header == self.crc_computed


def extract(stego: np.ndarray, key: str) -> ExtractResult:
    """m' = H·LSB(y) → giải mã → đọc header → kiểm tra CRC32. Không bao giờ ném ngoại lệ."""
    try:
        return _extract(stego, key)
    except Exception as exc:  # phòng thủ: dữ liệu rác không được làm treo/crash ứng dụng
        return ExtractResult(ok=False, status="KHÔNG KHÔI PHỤC ĐƯỢC",
                             reasons=[f"Lỗi khi trích xuất: {type(exc).__name__}: {exc}"])


def _extract(stego: np.ndarray, key: str) -> ExtractResult:
    if stego.ndim != 2 or stego.dtype != np.uint8:
        return ExtractResult(False, "KHÔNG KHÔI PHỤC ĐƯỢC", reasons=["Ảnh không phải xám 8-bit"])
    N = stego.size
    if N < HEADER_PIXELS + 8:
        return ExtractResult(False, "KHÔNG KHÔI PHỤC ĐƯỢC", reasons=["Ảnh quá nhỏ"])
    perm = pixel_permutation(key, N)
    y = (stego.ravel()[perm] & 1).astype(np.uint8)

    hw = block_widths(HEADER_PIXELS, HEADER_BITS)
    hcols = generate_hhat(HEADER_H, int(hw.max()), derive_seed(key, "H-header"))
    hdr_enc = syndrome(y[:HEADER_PIXELS], HEADER_BITS, HEADER_H, hcols, hw)
    hdr = bits_to_bytes(hdr_enc ^ keystream_bits(key, "header", HEADER_BITS))
    magic, h, length, crc_h = struct.unpack(">HBII", hdr)
    res = ExtractResult(False, "KHÔNG KHÔI PHỤC ĐƯỢC", magic_ok=(magic == MAGIC), h=h,
                        length=length, crc_header=crc_h)
    if not res.magic_ok:
        res.reasons.append(f"Magic header sai (0x{magic:04X} ≠ 0x{MAGIC:04X}) → sai khoá hoặc ảnh đã bị thay đổi")
    cap = capacity_bytes(stego.shape)
    if not H_RANGE[0] <= h <= H_RANGE[1]:
        res.reasons.append(f"Tham số h={h} trong header không hợp lệ")
        return res
    if length > cap:
        res.reasons.append(f"Độ dài trong header ({length} byte) vượt dung lượng ảnh ({cap} byte)")
        return res

    L = 8 * length
    if L:
        n_reg = N - HEADER_PIXELS
        pw = block_widths(n_reg, L)
        pcols = generate_hhat(h, int(pw.max()), derive_seed(key, f"H-payload-{h}"))
        p_enc = syndrome(y[HEADER_PIXELS:], L, h, pcols, pw)
        data = bits_to_bytes(p_enc ^ keystream_bits(key, "payload", L))
    else:
        data = b""
    res.msg_bytes = data
    res.crc_computed = zlib.crc32(data) & 0xFFFFFFFF
    if not res.checksum_ok:
        res.reasons.append(f"Checksum SAI: CRC32 header = {crc_h:08X}, CRC32 tính lại = {res.crc_computed:08X}")
        return res
    if not res.magic_ok:
        return res
    try:
        res.message = data.decode("utf-8")
    except UnicodeDecodeError:
        res.reasons.append("Dữ liệu không phải UTF-8 hợp lệ")
        return res
    res.ok = True
    res.status = "KHÔI PHỤC THÀNH CÔNG"
    return res


def extract_from_file(source, key: str) -> ExtractResult:
    """Người nhận: đọc FILE (đường dẫn hoặc bytes) + khoá. Không dùng gì từ phiên nhúng."""
    try:
        img = read_image_file(source)
    except Exception as exc:
        return ExtractResult(False, "KHÔNG KHÔI PHỤC ĐƯỢC", reasons=[f"Không đọc được file ảnh: {exc}"])
    return extract(img, key)
