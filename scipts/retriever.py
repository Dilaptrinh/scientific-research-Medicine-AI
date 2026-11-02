"""Embedding + FAISS: tìm đoạn thông tin thuốc và cặp hỏi-đáp mẫu gần nhất với câu hỏi."""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from data import BoQA, KhoThuoc

KetQuaTim = List[Tuple[int, float]]   # [(chỉ số, độ giống), ...]


def ten_co_ban(t: str) -> str:
    """'Coxileb 200mg' -> 'coxileb'; 'Becacold-E' -> 'becacold-e'."""
    return re.sub(r"\s*\d.*$", "", t).strip().lower()


class Retriever:
    def __init__(self, kho: KhoThuoc, qa: BoQA, embed_model: str, batch_size: int = 16) -> None:
        self.kho, self.qa, self.batch_size = kho, qa, batch_size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Tải embedding {embed_model} trên {device} ...")
        self.embedder = SentenceTransformer(embed_model, device=device)

        self.base = {t: ten_co_ban(t) for t in kho.ten_goc}

        self.idx_doan = self._tao_index(self.ma_hoa(kho.van))
        # Cặp hỏi-đáp đánh chỉ mục theo CÂU HỎI để khớp với câu hỏi người dùng
        self.idx_qa = self._tao_index(self.ma_hoa(qa.hoi))
        print("Xong. FAISS đoạn thuốc:", self.idx_doan.ntotal, "| FAISS hỏi-đáp:", self.idx_qa.ntotal)

    @staticmethod
    def _tao_index(vectors: np.ndarray):
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def ma_hoa(self, ds: List[str]) -> np.ndarray:
        """Mã hoá nhiều văn bản (có thanh tiến trình)."""
        return self.embedder.encode(ds, batch_size=self.batch_size, show_progress_bar=True,
                                    normalize_embeddings=True).astype("float32")

    def encode(self, ds: List[str]) -> np.ndarray:
        """Mã hoá nhanh (không thanh tiến trình), dùng cho câu hỏi và khi đánh giá."""
        return self.embedder.encode(ds, normalize_embeddings=True).astype("float32")

    def tim_ten_thuoc(self, cau_hoi: str) -> List[str]:
        """Các thuốc được nhắc tên trong câu hỏi."""
        q = cau_hoi.lower()
        return [t for t, b in self.base.items() if b and b in q]

    def truy_xuat(self, cau_hoi: str, k: int, k_qa: int, dung_qa: bool = True,
                  loc_ten: bool = True) -> Tuple[KetQuaTim, KetQuaTim, List[str]]:
        """Trả về (đoạn thuốc, cặp hỏi-đáp, thuốc được nhắc tên).

        Nếu loc_ten và câu hỏi có nhắc tên thuốc, chỉ giữ kết quả của đúng thuốc đó.
        """
        qv = self.encode([cau_hoi])
        thuoc_nhac = self.tim_ten_thuoc(cau_hoi) if loc_ten else []

        D, I = self.idx_doan.search(qv, self.idx_doan.ntotal)
        doan: KetQuaTim = []
        for i, s in zip(I[0], D[0]):
            if thuoc_nhac and self.kho.ten[i] not in thuoc_nhac:
                continue
            doan.append((int(i), float(s)))
            if len(doan) == k:
                break

        qa: KetQuaTim = []
        if dung_qa:
            D, I = self.idx_qa.search(qv, self.idx_qa.ntotal)
            for i, s in zip(I[0], D[0]):
                if thuoc_nhac and self.qa.ten[i] not in thuoc_nhac:
                    continue
                qa.append((int(i), float(s)))
                if len(qa) == k_qa:
                    break
        return doan, qa, thuoc_nhac

    def doan_cua_thuoc(self, cau_hoi: str, ten: str, k: int) -> KetQuaTim:
        """Các đoạn liên quan nhất, nhưng CHỈ của đúng thuốc `ten`."""
        qv = self.encode([cau_hoi])
        D, I = self.idx_doan.search(qv, self.idx_doan.ntotal)
        kq: KetQuaTim = []
        for i, s in zip(I[0], D[0]):
            if self.kho.ten[i] == ten:
                kq.append((int(i), float(s)))
                if len(kq) == k:
                    break
        return kq
