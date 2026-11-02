"""Đo chất lượng hệ thống (tuỳ chọn, để lấy số liệu báo cáo)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from retriever import Retriever
from rag import RagAssistant


def danh_gia_truy_xuat(retriever: Retriever, top_k: int, so_sai_in: int = 10) -> dict:
    """Với mỗi câu hỏi trong file hỏi-đáp, kiểm tra thuốc đúng có nằm trong các đoạn tìm được không.

    Tắt lọc theo tên thuốc và tắt cặp hỏi-đáp mẫu để đo thuần khả năng tìm kiếm của embedding.
    """
    qa, kho = retriever.qa, retriever.kho
    top1 = topk = 0
    sai = []
    for h, ten in zip(qa.hoi, qa.ten):
        doan, _, _ = retriever.truy_xuat(h, top_k, 0, dung_qa=False, loc_ten=False)
        kq = [kho.ten[i] for i, _ in doan]
        top1 += bool(kq) and kq[0] == ten
        topk += ten in kq
        if ten not in kq:
            sai.append((ten, h, list(dict.fromkeys(kq))))

    n = len(qa)
    ket_qua = {"so_cau_hoi": n, "top1": top1 / n, f"top{top_k}": topk / n, "so_cau_sai": len(sai)}
    print(f"Số câu hỏi: {n}")
    print(f"Top-1 accuracy: {top1 / n:.2%}")
    print(f"Top-{top_k} accuracy: {topk / n:.2%}")
    print(f"Số câu truy xuất sai: {len(sai)}")
    for ten, h, kq in sai[:so_sai_in]:
        print(f"  - Cần: {ten} | Hỏi: {h} | Nhận được: {kq}")
    return ket_qua


def danh_gia_cau_tra_loi(assistant: RagAssistant, so_cau: int = 50, seed: int = 42,
                         duong_luu: Optional[str] = "results/ket_qua_danh_gia.csv") -> pd.DataFrame:
    """Cho hệ thống trả lời `so_cau` câu ngẫu nhiên (KHÔNG dùng cặp hỏi-đáp mẫu để khỏi chép đáp án),
    rồi so với đáp án gốc bằng độ giống ngữ nghĩa (0 đến 1)."""
    qa, rt = assistant.retriever.qa, assistant.retriever
    rng = np.random.RandomState(seed)
    chon = rng.choice(len(qa), size=min(so_cau, len(qa)), replace=False)

    rows = []
    for n_, i in enumerate(chon, 1):
        kq = assistant.hoi(qa.hoi[i], dung_qa=False)
        v = rt.encode([kq.tra_loi, qa.dap[i]])
        rows.append({
            "ten_thuoc": qa.ten[i], "cau_hoi": qa.hoi[i],
            "dap_an_goc": qa.dap[i], "he_thong_tra_loi": kq.tra_loi,
            "do_giong": round(float(np.dot(v[0], v[1])), 4),
        })
        if n_ % 10 == 0:
            print(f"  đã xong {n_}/{len(chon)} câu")

    df = pd.DataFrame(rows)
    print(f"\nĐộ giống trung bình với đáp án gốc: {df['do_giong'].mean():.3f}")
    print(f"Tỉ lệ câu có độ giống >= 0.8: {(df['do_giong'] >= 0.8).mean():.2%}")
    if duong_luu:
        Path(duong_luu).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(duong_luu, index=False, encoding="utf-8-sig")
        print("Đã lưu kết quả chi tiết:", duong_luu)
    return df
