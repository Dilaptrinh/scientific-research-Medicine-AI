"""Chấm điểm các tổ hợp thuốc: loại tổ hợp có xung đột, cộng điểm thưởng cho cặp hỗ trợ, chọn top K."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from embedding import tinh_score_ho_tro, tinh_score_xung_dot

# (thuốc a, thuốc b, điểm hỗ trợ, điểm xung đột, điểm cặp, ghi chú)
ChiTietCap = Tuple[str, str, float, float, float, str]


@dataclass
class KetQua:
    top: List[Tuple[float, Tuple[str, ...], List[ChiTietCap]]] = field(default_factory=list)
    so_kich_ban: int = 0
    so_xung_dot: int = 0
    so_trung_ten: int = 0
    so_hop_le: int = 0


def tinh_diem_cap_cache(a: str, b: str, cache: dict, model, tf) -> Tuple[float, float, float]:
    """Điểm PyKEEN của 1 cặp thuốc (có cache). Trả về (hỗ trợ, xung đột, hỗ trợ - xung đột)."""
    key = frozenset({a, b})
    if key not in cache:
        ht = tinh_score_ho_tro(model, tf, a, b)
        xd = tinh_score_xung_dot(model, tf, a, b)
        cache[key] = (ht, xd, ht - xd)
    return cache[key]


def tinh_diem_kich_ban(kich_ban: Sequence[str], cache: dict,
                       cap_xung_dot: Dict[FrozenSet[str], str],
                       cap_ho_tro: Dict[FrozenSet[str], str],
                       model, tf, diem_thuong_ho_tro: float):
    """Điểm tổng của 1 kịch bản (mỗi viên đã chốt 1 tên thuốc).

    Trả về (None, lý do) nếu có cặp xung đột đã khai báo -> loại luôn,
    ngược lại trả về (điểm tổng, chi tiết từng cặp).
    """
    tong, chi_tiet = 0.0, []
    for a, b in combinations(kich_ban, 2):
        key = frozenset({a, b})

        if key in cap_xung_dot:                       # xung đột -> loại cả tổ hợp
            return None, f"{a} <-> {b}: {cap_xung_dot[key]}"

        ht, xd, diem = tinh_diem_cap_cache(a, b, cache, model, tf)

        ghi_chu = ""
        if key in cap_ho_tro:                         # hỗ trợ -> cộng điểm thưởng
            diem += diem_thuong_ho_tro
            ghi_chu = f"[HỖ TRỢ +{diem_thuong_ho_tro}: {cap_ho_tro[key]}]"

        tong += diem
        chi_tiet.append((a, b, ht, xd, diem, ghi_chu))
    return tong, chi_tiet


def tinh_top_to_hop(danh_sach_vien: Sequence[Sequence[str]],
                    cap_xung_dot, cap_ho_tro, model, tf,
                    top_k: int = 3, toi_da_kich_ban: int = 5000,
                    diem_thuong_ho_tro: float = 0.5) -> KetQua:
    """
    danh_sach_vien: mỗi viên là list 1–3 tên thuốc.
      ["A"]          -> chắc chắn là A
      ["A","B","C"]  -> nghi ngờ là A hoặc B hoặc C
    Trả về KetQua (top_k tổ hợp điểm cao nhất nằm trong `.top`).
    Ném ValueError nếu đầu vào không hợp lệ.
    """
    if len(danh_sach_vien) < 2:
        raise ValueError("Cần ít nhất 2 viên thuốc.")

    so_kb = 1
    for v in danh_sach_vien:
        so_kb *= len(v)
    if so_kb > toi_da_kich_ban:
        raise ValueError(f"Quá nhiều kịch bản ({so_kb} > {toi_da_kich_ban}). "
                         "Hãy giảm bớt số viên nghi ngờ.")

    kq = KetQua(so_kich_ban=so_kb)
    cache: dict = {}
    ket_qua = []
    for kb in product(*danh_sach_vien):
        if len(set(kb)) < len(kb):                    # 2 viên cùng 1 tên thuốc -> bỏ qua
            kq.so_trung_ten += 1
            continue
        diem, chi_tiet = tinh_diem_kich_ban(kb, cache, cap_xung_dot, cap_ho_tro, model, tf,
                                            diem_thuong_ho_tro)
        if diem is None:                              # có xung đột -> loại
            kq.so_xung_dot += 1
            continue
        ket_qua.append((diem, kb, chi_tiet))

    ket_qua.sort(key=lambda x: x[0], reverse=True)
    kq.so_hop_le = len(ket_qua)
    kq.top = ket_qua[:top_k]
    return kq


def in_ket_qua(kq: KetQua, top_k: int) -> None:
    print("\n" + "=" * 60)
    print(f"TOP {top_k} TỔ HỢP CÓ ĐIỂM CAO NHẤT")
    print("=" * 60)
    print(f"  Tổng {kq.so_kich_ban} kịch bản | loại do xung đột: {kq.so_xung_dot} | "
          f"loại do trùng tên: {kq.so_trung_ten} | còn lại: {kq.so_hop_le}\n")

    if not kq.top:
        print("  Không còn tổ hợp nào hợp lệ (tất cả đều xung đột hoặc trùng tên).")
        return

    for hang, (diem, kb, ct) in enumerate(kq.top, 1):
        print(f"  HẠNG {hang}: {' + '.join(kb)}")
        print(f"    Điểm tổng: {diem:+.4f}")
        for a, b, ht, xd, d, gc in ct:
            print(f"      {a} <-> {b}: hỗ trợ={ht:+.4f}, xung đột={xd:+.4f}, "
                  f"điểm cặp={d:+.4f} {gc}")
        print()
    print("=" * 60)
