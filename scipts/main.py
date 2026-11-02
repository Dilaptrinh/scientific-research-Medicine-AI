#!/usr/bin/env python3
"""Điểm vào: hỏi đáp thông tin thuốc bằng RAG, và các lệnh đo độ chính xác.

    python main.py chat                       # hỏi đáp tương tác
    python main.py chat -q "Liều dùng của Becacold-E là bao nhiêu?"
    python main.py eval-retrieval             # đo độ chính xác truy xuất (không cần tải LLM)
    python main.py eval-answer --so-cau 10    # đo chất lượng câu trả lời so với đáp án gốc

Trong Google Colab dùng `%run main.py chat` (không dùng `!python`) nếu cần gõ câu hỏi bằng bàn phím.
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from config import Config
from data import kiem_tra_ten_khop, tai_du_lieu
from retriever import Retriever


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG hỏi đáp thông tin thuốc.")
    chung = argparse.ArgumentParser(add_help=False)
    chung.add_argument("--file-thuoc", help="CSV thông tin thuốc (mặc định data/thuoc_48.csv)")
    chung.add_argument("--file-qa", help="CSV hỏi-đáp mẫu (mặc định data/bo_tra_loi_cau_hoi.csv)")
    chung.add_argument("--embed-model", help="Mô hình embedding (mặc định BAAI/bge-m3)")
    chung.add_argument("--llm-model", help="Mô hình sinh câu trả lời (mặc định Qwen/Qwen2.5-3B-Instruct)")
    chung.add_argument("--load-in-4bit", action="store_true", default=None,
                       help="Nạp LLM 4-bit (cần bitsandbytes), dùng cho mô hình 7B trở lên")
    chung.add_argument("--top-k-doan", type=int, help="Số đoạn thông tin thuốc đưa cho mô hình (mặc định 6)")
    chung.add_argument("--top-k-qa", type=int, help="Số cặp hỏi-đáp mẫu (mặc định 3)")

    sub = p.add_subparsers(dest="lenh", required=True)
    c = sub.add_parser("chat", parents=[chung], help="Hỏi đáp")
    c.add_argument("-q", "--question", help="Một câu hỏi duy nhất; bỏ trống để vào chế độ nhập tay")
    c.add_argument("--no-qa", action="store_true", help="Không dùng cặp hỏi-đáp mẫu")
    c.add_argument("--no-bo-sung", action="store_true", help="Không viết phần bổ sung sau câu trả lời mẫu")
    sub.add_parser("eval-retrieval", parents=[chung], help="Đo top-1 / top-k accuracy của truy xuất")
    e = sub.add_parser("eval-answer", parents=[chung], help="Đo độ giống ngữ nghĩa với đáp án gốc")
    e.add_argument("--so-cau", type=int, default=50, help="Số câu hỏi ngẫu nhiên (mặc định 50)")
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--output", default="results/ket_qua_danh_gia.csv", help="Nơi lưu kết quả chi tiết")
    return p.parse_args()


def tao_config(args: argparse.Namespace) -> Config:
    cfg = Config.tu_moi_truong()
    ghi_de = {k: getattr(args, k) for k in
              ("file_thuoc", "file_qa", "embed_model", "llm_model", "load_in_4bit", "top_k_doan", "top_k_qa")
              if getattr(args, k, None) is not None}
    if getattr(args, "no_qa", False):
        ghi_de["dung_qa"] = False
    if getattr(args, "no_bo_sung", False):
        ghi_de["bo_sung"] = False
    return replace(cfg, **ghi_de)


def main() -> None:
    args = parse_args()
    cfg = tao_config(args)

    kho, qa = tai_du_lieu(cfg.file_thuoc, cfg.file_qa)
    kiem_tra_ten_khop(kho, qa)
    retriever = Retriever(kho, qa, cfg.embed_model)

    if args.lenh == "eval-retrieval":            # không cần LLM
        from evaluate import danh_gia_truy_xuat
        danh_gia_truy_xuat(retriever, cfg.top_k_doan)
        return

    from generator import Generator
    from rag import RagAssistant
    assistant = RagAssistant(retriever, Generator(cfg.llm_model, cfg.load_in_4bit), cfg)

    if args.lenh == "eval-answer":
        from evaluate import danh_gia_cau_tra_loi
        danh_gia_cau_tra_loi(assistant, args.so_cau, args.seed, args.output)
        return

    if args.question:                            # chat, một câu
        assistant.in_tra_loi(assistant.hoi(args.question))
        return
    while True:                                  # chat, tương tác
        cau = input("\nNhập câu hỏi (gõ 'thoat' để dừng): ").strip()
        if not cau or cau.lower() == "thoat":
            print("Đã dừng.")
            break
        assistant.in_tra_loi(assistant.hoi(cau))


if __name__ == "__main__":
    main()
