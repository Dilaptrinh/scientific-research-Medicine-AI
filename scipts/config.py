"""Cấu hình chung của hệ thống RAG hỏi đáp thông tin thuốc.

Giá trị mặc định lấy từ notebook gốc. Có thể ghi đè bằng biến môi trường (xem `Config.tu_moi_truong`)
hoặc bằng tham số dòng lệnh trong `main.py`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # Dữ liệu
    file_thuoc: str = "data/thuoc_48.csv"
    file_qa: str = "data/bo_tra_loi_cau_hoi.csv"

    # Mô hình
    embed_model: str = "BAAI/bge-m3"              # embedding hỗ trợ tiếng Việt
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"   # chạy được trên T4 miễn phí
    load_in_4bit: bool = False                    # bật khi dùng mô hình lớn (vd Qwen2.5-7B-Instruct)

    # Truy xuất
    top_k_doan: int = 6        # số đoạn thông tin thuốc đưa cho mô hình
    top_k_qa: int = 3          # số cặp hỏi-đáp mẫu đưa cho mô hình
    dung_qa: bool = True       # dùng thêm cặp hỏi-đáp mẫu khi trả lời

    # Sinh câu trả lời
    nguong_co_ten: float = 0.78      # ngưỡng khớp câu mẫu khi câu hỏi đã nhắc tên thuốc
    nguong_khong_ten: float = 0.88   # ngưỡng khi câu hỏi không nhắc tên thuốc
    bo_sung: bool = True             # thêm thông tin bổ sung sau câu trả lời mẫu
    k_bo_sung: int = 4               # số đoạn thông tin dùng để bổ sung
    max_tokens: int = 450

    @classmethod
    def tu_moi_truong(cls) -> "Config":
        """Tạo Config, cho phép ghi đè các trường chuỗi bằng biến môi trường (viết hoa)."""
        cfg = cls()
        for ten in ("file_thuoc", "file_qa", "embed_model", "llm_model"):
            gia_tri = os.environ.get(ten.upper())
            if gia_tri:
                setattr(cfg, ten, gia_tri)
        return cfg
