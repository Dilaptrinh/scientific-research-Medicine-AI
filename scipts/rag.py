"""Luồng trả lời của hệ thống RAG.

Ưu tiên 1: câu hỏi khớp một câu hỏi mẫu (độ giống >= ngưỡng) -> dùng câu trả lời mẫu và
           (tuỳ chọn) nhờ mô hình viết thêm phần bổ sung từ tài liệu.
Ưu tiên 2: câu hỏi ngoài bộ mẫu -> mô hình viết câu trả lời từ tài liệu, bắt chước giọng văn các câu mẫu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import Config
from generator import Generator
from retriever import KetQuaTim, Retriever

PROMPT_BO_SUNG = (
    "Bạn là trợ lý cung cấp thông tin thuốc. Bạn được cho một CÂU TRẢ LỜI GỐC đã hoàn chỉnh. "
    "Nhiệm vụ: viết PHẦN BỔ SUNG nối tiếp ngay sau câu trả lời gốc, dài 2 đến 3 câu. "
    "Quy tắc: (1) chỉ viết thông tin MỚI, có trong TÀI LIỆU và liên quan trực tiếp đến câu hỏi; "
    "(2) không lặp lại những gì câu trả lời gốc đã nói; (3) không bịa số liệu, liều dùng; "
    "(4) giọng văn tự nhiên, nhẹ nhàng, giống câu trả lời gốc, viết liền mạch thành đoạn văn, "
    "không gạch đầu dòng, không chào hỏi, không mở đầu lại; "
    "(5) nếu TÀI LIỆU không có gì mới đáng bổ sung, chỉ trả lời đúng: [KHONG_CO]."
)

PROMPT_NGOAI_KHUON = (
    "Bạn là trợ lý cung cấp thông tin thuốc. "
    "Nội dung chuyên môn CHỈ lấy từ phần TÀI LIỆU, không được bịa. "
    "Nếu tài liệu không có, nói rõ 'Tài liệu không có thông tin này'. "
    "Về CÁCH VIẾT: bắt chước sát giọng văn, cách mở đầu câu (ví dụ 'Thuốc X có...'), "
    "độ dài của các câu trả lời mẫu trong hội thoại trước. Viết thành đoạn văn liền mạch, tiếng Việt."
)


@dataclass
class TraLoi:
    tra_loi: str
    doan: KetQuaTim                       # các đoạn thông tin thuốc đã dùng
    qa: KetQuaTim                         # các cặp hỏi-đáp mẫu gần nhất
    tu_cau_mau: bool = False              # True: lấy từ câu trả lời mẫu (ưu tiên 1)
    co_bo_sung: bool = False
    cau_mau: Optional[str] = None         # câu hỏi mẫu đã khớp
    do_giong: Optional[float] = None


class RagAssistant:
    def __init__(self, retriever: Retriever, generator: Generator, cfg: Config) -> None:
        self.retriever, self.generator, self.cfg = retriever, generator, cfg

    def _ghep_tai_lieu(self, doan: KetQuaTim) -> str:
        van = self.retriever.kho.van
        return "\n\n".join(f"[Đoạn {n + 1}]\n{van[i]}" for n, (i, _) in enumerate(doan))

    def hoi(self, cau_hoi: str, dung_qa: Optional[bool] = None, max_tokens: Optional[int] = None) -> TraLoi:
        cfg, rt = self.cfg, self.retriever
        dung_qa = cfg.dung_qa if dung_qa is None else dung_qa
        max_tokens = max_tokens or cfg.max_tokens
        doan, qa, thuoc_nhac = rt.truy_xuat(cau_hoi, cfg.top_k_doan, cfg.top_k_qa, dung_qa=dung_qa)

        # ===== ƯU TIÊN 1: có câu hỏi tương ứng trong file -> câu trả lời mẫu + bổ sung =====
        nguong = cfg.nguong_co_ten if thuoc_nhac else cfg.nguong_khong_ten
        if qa and qa[0][1] >= nguong:
            j, s = qa[0]
            goc = rt.qa.dap[j]
            them = ""
            doan_bs = rt.doan_cua_thuoc(cau_hoi, rt.qa.ten[j], cfg.k_bo_sung) if cfg.bo_sung else []

            if doan_bs:
                messages = [
                    {"role": "system", "content": PROMPT_BO_SUNG},
                    {"role": "user", "content":
                     f"CÂU HỎI: {cau_hoi}\n\nCÂU TRẢ LỜI GỐC:\n{goc}\n\n"
                     f"TÀI LIỆU:\n{self._ghep_tai_lieu(doan_bs)}\n\nPHẦN BỔ SUNG:"},
                ]
                them = self.generator.sinh(messages, max_tokens=220)
                if "[KHONG_CO]" in them or len(them) < 20:
                    them = ""

            return TraLoi(goc + (" " + them if them else ""), doan_bs or doan, qa,
                          tu_cau_mau=True, co_bo_sung=bool(them), cau_mau=rt.qa.hoi[j], do_giong=s)

        # ===== ƯU TIÊN 2: câu hỏi ngoài bộ mẫu -> mô hình viết theo giọng văn các câu mẫu =====
        messages = [{"role": "system", "content": PROMPT_NGOAI_KHUON}]
        for i, _ in reversed(qa):
            messages.append({"role": "user", "content": rt.qa.hoi[i]})
            messages.append({"role": "assistant", "content": rt.qa.dap[i]})
        messages.append({"role": "user",
                         "content": f"TÀI LIỆU:\n{self._ghep_tai_lieu(doan)}\n\nCÂU HỎI: {cau_hoi}"})
        return TraLoi(self.generator.sinh(messages, max_tokens), doan, qa)

    def in_tra_loi(self, kq: TraLoi) -> None:
        kho = self.retriever.kho
        print(f"\nTRẢ LỜI:\n{kq.tra_loi}")
        if kq.tu_cau_mau:
            print(f"\n(Khớp câu mẫu: \"{kq.cau_mau}\" | độ giống {kq.do_giong:.3f} | "
                  f"{'có bổ sung' if kq.co_bo_sung else 'không bổ sung'})")
        else:
            print("\nNGUỒN:")
            for i, s in kq.doan:
                print(f"  - {kho.ten[i]} | mục: {kho.muc[i]} | độ giống: {s:.3f} | {kho.nguon[i]}")
