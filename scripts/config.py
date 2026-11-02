"""Cấu hình cho toàn bộ chương trình.

Thông tin nhạy cảm (URI, user, password Neo4j) KHÔNG được ghi trong code mà đọc từ
biến môi trường hoặc file `.env` (xem `.env.example`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # tuỳ chọn: tự đọc file .env nếu đã cài python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Config:
    # Dữ liệu
    file_ho_tro: Path
    file_xung_dot: Path
    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    # PyKEEN
    pykeen_model: str = "TransE"
    pykeen_epochs: int = 100
    seed: int = 42
    # Chấm điểm
    diem_thuong_ho_tro: float = 0.5   # điểm cộng cho mỗi cặp đã khai báo HỖ TRỢ
    top_k: int = 3                    # số tổ hợp tốt nhất cần in
    toi_da_kich_ban: int = 5000       # giới hạn số kịch bản (tránh bùng nổ tổ hợp)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Thiếu biến môi trường {name}. Hãy đặt nó (xem .env.example) rồi chạy lại."
        )
    return value


def load_config(
    file_ho_tro: Optional[str] = None,
    file_xung_dot: Optional[str] = None,
    pykeen_model: Optional[str] = None,
    pykeen_epochs: Optional[int] = None,
    diem_thuong_ho_tro: Optional[float] = None,
    top_k: Optional[int] = None,
    toi_da_kich_ban: Optional[int] = None,
) -> Config:
    """Tạo Config: tham số truyền vào > biến môi trường > giá trị mặc định."""
    defaults = Config.__dataclass_fields__
    return Config(
        file_ho_tro=Path(file_ho_tro or os.environ.get("FILE_HO_TRO", "data/support.csv")),
        file_xung_dot=Path(file_xung_dot or os.environ.get("FILE_XUNG_DOT", "data/conflict.csv")),
        neo4j_uri=_require_env("NEO4J_URI"),
        neo4j_user=_require_env("NEO4J_USER"),
        neo4j_password=_require_env("NEO4J_PASSWORD"),
        pykeen_model=pykeen_model or os.environ.get("PYKEEN_MODEL", defaults["pykeen_model"].default),
        pykeen_epochs=pykeen_epochs or int(os.environ.get("PYKEEN_EPOCHS", defaults["pykeen_epochs"].default)),
        diem_thuong_ho_tro=(diem_thuong_ho_tro if diem_thuong_ho_tro is not None
                            else float(os.environ.get("DIEM_THUONG_HO_TRO", defaults["diem_thuong_ho_tro"].default))),
        top_k=top_k or int(os.environ.get("TOP_K", defaults["top_k"].default)),
        toi_da_kich_ban=toi_da_kich_ban or int(os.environ.get("TOI_DA_KICH_BAN", defaults["toi_da_kich_ban"].default)),
    )
