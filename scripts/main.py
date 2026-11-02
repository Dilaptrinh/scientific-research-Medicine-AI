#!/usr/bin/env python3
"""Điểm vào của chương trình: xây đồ thị Neo4j -> train PyKEEN -> chọn top tổ hợp thuốc.

Chế độ tương tác (nhập thuốc từ bàn phím):
    python main.py

Chế độ không tương tác:
    python main.py --drugs "Paracetamol" "Aspirin / Ibuprofen / Naproxen"

Trong Google Colab, dùng `%run main.py ...` (không dùng `!python`) nếu cần nhập bằng bàn phím.
"""
from __future__ import annotations

import argparse
import re
from typing import List, Optional

from config import load_config
from embedding import train_pykeen
from neo4j_graph import xay_do_thi_va_lay_du_lieu
from scoring import in_ket_qua, tinh_top_to_hop

TOI_DA_TEN_MOI_VIEN = 3


def parse_vien(s: str) -> List[str]:
    """'A' -> ['A'];  'A / B, C' -> ['A','B','C'] (bỏ tên trùng, tối đa 3 tên)."""
    ung_vien = [x.strip() for x in re.split(r"[/,]", s) if x.strip()]
    ung_vien = list(dict.fromkeys(ung_vien))          # bỏ tên trùng, giữ thứ tự
    if len(ung_vien) > TOI_DA_TEN_MOI_VIEN:
        print(f"  [!] Tối đa {TOI_DA_TEN_MOI_VIEN} tên cho 1 viên, chỉ lấy {TOI_DA_TEN_MOI_VIEN} tên đầu.")
        ung_vien = ung_vien[:TOI_DA_TEN_MOI_VIEN]
    return ung_vien


def canh_bao_thuoc_la(ung_vien: List[str], tf) -> None:
    if tf is None:
        return
    la = [t for t in ung_vien if t not in tf.entity_to_id]
    if la:
        print(f"  [!] Chưa có trong đồ thị (điểm sẽ = 0): {', '.join(la)}")


def nhap_danh_sach_vien(tf) -> List[List[str]]:
    print("Nhập từng VIÊN thuốc:")
    print("  - Chắc chắn      : gõ 1 tên          (vd: Paracetamol)")
    print("  - Nghi ngờ (≤ 3) : nhiều tên cách nhau bằng '/'  (vd: Aspirin / Ibuprofen / Naproxen)")
    print("  - Enter trống để dừng.\n")

    danh_sach: List[List[str]] = []
    while True:
        s = input(f"Viên #{len(danh_sach) + 1}: ").strip()
        if not s:
            break
        ung_vien = parse_vien(s)
        if not ung_vien:
            continue
        canh_bao_thuoc_la(ung_vien, tf)
        danh_sach.append(ung_vien)
        loai = "chắc chắn" if len(ung_vien) == 1 else f"nghi ngờ ({len(ung_vien)} khả năng)"
        print(f"  -> Viên #{len(danh_sach)}: {' | '.join(ung_vien)}  [{loai}]")
    return danh_sach


def chay_mot_lan(danh_sach_vien, cap_xung_dot, cap_ho_tro, model, tf, cfg) -> None:
    try:
        kq = tinh_top_to_hop(danh_sach_vien, cap_xung_dot, cap_ho_tro, model, tf,
                             top_k=cfg.top_k, toi_da_kich_ban=cfg.toi_da_kich_ban,
                             diem_thuong_ho_tro=cfg.diem_thuong_ho_tro)
    except ValueError as exc:
        print(f"\n[!] {exc}")
        return
    in_ket_qua(kq, cfg.top_k)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chọn top tổ hợp thuốc bằng đồ thị Neo4j + PyKEEN.")
    p.add_argument("--file-ho-tro", help="CSV quan hệ hỗ trợ (mặc định: data/support.csv hoặc $FILE_HO_TRO)")
    p.add_argument("--file-xung-dot", help="CSV quan hệ xung đột (mặc định: data/conflict.csv hoặc $FILE_XUNG_DOT)")
    p.add_argument("--model", dest="pykeen_model", help="Mô hình PyKEEN (mặc định TransE)")
    p.add_argument("--epochs", dest="pykeen_epochs", type=int, help="Số epoch PyKEEN (mặc định 100)")
    p.add_argument("--bonus", dest="diem_thuong_ho_tro", type=float, help="Điểm thưởng cho cặp hỗ trợ (mặc định 0.5)")
    p.add_argument("--top-k", type=int, help="Số tổ hợp tốt nhất cần in (mặc định 3)")
    p.add_argument("--max-scenarios", dest="toi_da_kich_ban", type=int, help="Giới hạn số kịch bản (mặc định 5000)")
    p.add_argument("--drugs", nargs="+", metavar="VIEN",
                   help="Danh sách viên thuốc, mỗi viên là 1 chuỗi; nghi ngờ thì cách nhau bằng '/'. "
                        "Nếu bỏ trống sẽ chuyển sang nhập tay.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.file_ho_tro, args.file_xung_dot, args.pykeen_model, args.pykeen_epochs,
                      args.diem_thuong_ho_tro, args.top_k, args.toi_da_kich_ban)

    cap_xung_dot, cap_ho_tro, all_triples = xay_do_thi_va_lay_du_lieu(cfg)
    model, tf = train_pykeen(all_triples, cfg.pykeen_model, cfg.pykeen_epochs, cfg.seed)

    if args.drugs:                                    # không tương tác
        danh_sach = [parse_vien(s) for s in args.drugs]
        danh_sach = [v for v in danh_sach if v]
        for v in danh_sach:
            canh_bao_thuoc_la(v, tf)
        chay_mot_lan(danh_sach, cap_xung_dot, cap_ho_tro, model, tf, cfg)
        return

    while True:                                       # tương tác: thử nhiều danh sách mà không train lại
        danh_sach = nhap_danh_sach_vien(tf)
        if danh_sach:
            chay_mot_lan(danh_sach, cap_xung_dot, cap_ho_tro, model, tf, cfg)
        else:
            print("\nCần nhập ít nhất 2 viên thuốc.")
        if input("\nThử danh sách khác? (y/N): ").strip().lower() not in {"y", "yes", "c", "co", "có"}:
            break


if __name__ == "__main__":
    main()
