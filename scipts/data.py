"""Đọc 2 file CSV và cắt thông tin mỗi thuốc thành các đoạn theo mục."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd

# Mỗi mục: (nhãn, [các cột gộp vào đoạn này])
MUC = [
    ("Thông tin chung", ["hoat_chat", "dang_bao_che"]),
    ("Chỉ định",        ["chi_dinh"]),
    ("Liều dùng",       ["lieu_dung"]),
    ("Chống chỉ định",  ["chong_chi_dinh"]),
    ("Tác dụng phụ",    ["tac_dung_phu"]),
    ("Lưu ý",           ["luu_y"]),
]
NHAN_COT = {"hoat_chat": "Hoạt chất", "dang_bao_che": "Dạng bào chế"}

COT_QA = ("cau_hoi", "cau_tra_loi", "ten_thuoc")


@dataclass
class KhoThuoc:
    """Các đoạn thông tin thuốc; phần tử thứ i của 4 list ứng với cùng một đoạn."""
    van: List[str] = field(default_factory=list)     # nội dung đoạn (có tên thuốc ở đầu)
    ten: List[str] = field(default_factory=list)     # tên thuốc của đoạn
    muc: List[str] = field(default_factory=list)     # mục: Chỉ định, Liều dùng, ...
    nguon: List[str] = field(default_factory=list)   # nguồn tham khảo

    @property
    def ten_goc(self) -> List[str]:
        """Danh sách tên thuốc, không trùng, giữ thứ tự."""
        return list(dict.fromkeys(self.ten))


@dataclass
class BoQA:
    """Các cặp hỏi-đáp mẫu."""
    hoi: List[str]
    dap: List[str]
    ten: List[str]

    def __len__(self) -> int:
        return len(self.hoi)


def doc_csv(path) -> pd.DataFrame:
    """Đọc CSV, thử lần lượt các encoding phổ biến của file tiếng Việt."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Không thấy file: {path}")
    for enc in ("utf-8-sig", "cp1258", "cp1252", "latin-1"):
        try:
            d = pd.read_csv(path, encoding=enc).fillna("")
            print(f"Đọc OK: {path} | encoding = {enc}")
            return d
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Không đọc được file: {path}")


def sach(v) -> str:
    """Chuẩn hoá ô dữ liệu; ô trống hoặc bắt đầu bằng '<' (placeholder) coi như rỗng."""
    v = str(v).strip()
    return "" if (not v or v.startswith("<")) else v


def tao_kho_thuoc(df: pd.DataFrame) -> KhoThuoc:
    """Cắt mỗi thuốc thành các đoạn theo mục. Mỗi đoạn ghi rõ tên thuốc ở đầu."""
    if "ten_thuoc" not in df.columns:
        raise ValueError(f"File thuốc thiếu cột 'ten_thuoc'. Các cột hiện có: {list(df.columns)}")

    kho = KhoThuoc()
    for _, r in df.iterrows():
        ten = sach(r.get("ten_thuoc", ""))
        if not ten:
            continue
        nguon = sach(r.get("nguon", ""))
        for nhan, cols in MUC:
            mau = []
            for c in cols:
                v = sach(r.get(c, ""))
                if v:
                    mau.append(f"{NHAN_COT.get(c, nhan)}: {v}" if len(cols) > 1 else v)
            if not mau:
                continue
            if len(cols) == 1:
                van = f"Thuốc: {ten}\n{nhan}: " + "; ".join(mau)
            else:
                van = f"Thuốc: {ten}\n" + "\n".join(mau)
            kho.van.append(van)
            kho.ten.append(ten)
            kho.muc.append(nhan)
            kho.nguon.append(nguon)
    return kho


def tao_bo_qa(dqa: pd.DataFrame) -> BoQA:
    thieu = [c for c in COT_QA if c not in dqa.columns]
    if thieu:
        raise ValueError(f"File hỏi-đáp thiếu cột {thieu}. Các cột hiện có: {list(dqa.columns)}")
    return BoQA(
        hoi=[str(x).strip() for x in dqa["cau_hoi"]],
        dap=[str(x).strip() for x in dqa["cau_tra_loi"]],
        ten=[str(x).strip() for x in dqa["ten_thuoc"]],
    )


def tai_du_lieu(file_thuoc, file_qa):
    """Đọc cả 2 file. Trả về (KhoThuoc, BoQA)."""
    df = doc_csv(file_thuoc)
    dqa = doc_csv(file_qa)
    print(f"File thuốc: {len(df)} dòng | cột: {list(df.columns)}")
    print(f"File hỏi-đáp: {len(dqa)} dòng | cột: {list(dqa.columns)}")
    kho, qa = tao_kho_thuoc(df), tao_bo_qa(dqa)
    print(f"Số thuốc dùng được: {len(kho.ten_goc)} | Số đoạn: {len(kho.van)}")
    return kho, qa


def kiem_tra_ten_khop(kho: KhoThuoc, qa: BoQA) -> None:
    """Báo các tên thuốc không khớp giữa 2 file (tên lệch sẽ làm lọc theo thuốc bị sai)."""
    a, b = set(qa.ten), set(kho.ten_goc)
    print("Có trong file hỏi-đáp nhưng KHÔNG có trong file thuốc:", sorted(a - b) or "không có (tốt)")
    print("Có trong file thuốc nhưng KHÔNG có trong file hỏi-đáp:", sorted(b - a) or "không có (tốt)")
