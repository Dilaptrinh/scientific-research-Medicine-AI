"""Train PyKEEN (Knowledge Graph Embedding) và chấm điểm một cặp thuốc theo từng quan hệ."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

MIN_TRIPLE = 3


def train_pykeen(triples: List[Tuple[str, str, str]], model_name: str = "TransE",
                 epochs: int = 100, seed: int = 42):
    """Train mô hình embedding. Trả về (model, triples_factory) hoặc (None, None) nếu quá ít dữ liệu."""
    print("=" * 60)
    print("GIAI ĐOẠN 2: TRAIN PYKEEN (Knowledge Graph Embedding)")
    print("=" * 60)

    if len(triples) < MIN_TRIPLE:
        print(f"  [CẢNH BÁO] Chỉ có {len(triples)} triple, cần ≥ {MIN_TRIPLE}. Bỏ qua PyKEEN.")
        return None, None

    tf_full = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))

    if len(triples) >= 10:
        train_tf, test_tf, valid_tf = tf_full.split([0.8, 0.1, 0.1], random_state=seed)
        result = pipeline(training=train_tf, testing=test_tf, validation=valid_tf,
                          model=model_name, training_kwargs=dict(num_epochs=epochs),
                          random_seed=seed)
    else:
        train_tf, test_tf = tf_full.split([0.8, 0.2], random_state=seed)
        result = pipeline(training=train_tf, testing=test_tf,
                          model=model_name, training_kwargs=dict(num_epochs=epochs),
                          random_seed=seed)

    print(f"  Train xong! Loss cuối: {result.losses[-1]:.4f}\n")
    return result.model, tf_full


def _tinh_score(model, tf, quan_he: str, thuoc_a: str, thuoc_b: str) -> float:
    """Điểm (head=a, relation=quan_he, tail=b). Trả 0.0 nếu thiếu mô hình hoặc thuốc chưa có trong đồ thị."""
    if model is None or tf is None:
        return 0.0
    e2id, r2id = tf.entity_to_id, tf.relation_to_id
    if thuoc_a not in e2id or thuoc_b not in e2id or quan_he not in r2id:
        return 0.0
    # Đặt tensor cùng thiết bị với mô hình (CPU/GPU), tránh lỗi device mismatch khi train bằng GPU.
    device = getattr(model, "device", torch.device("cpu"))
    hrt = torch.tensor([[e2id[thuoc_a], r2id[quan_he], e2id[thuoc_b]]], device=device)
    model.eval()
    with torch.no_grad():
        score = model.score_hrt(hrt)
    return float(score.item())


def tinh_score_ho_tro(model, tf, a: str, b: str) -> float:
    return _tinh_score(model, tf, "HO_TRO", a, b)


def tinh_score_xung_dot(model, tf, a: str, b: str) -> float:
    return _tinh_score(model, tf, "XUNG_DOT", a, b)
