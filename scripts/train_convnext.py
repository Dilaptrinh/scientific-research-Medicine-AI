#!/usr/bin/env python3
"""
train_convnext.py
=================
Fine-tune ConvNeXt-Tiny (ImageNet pretrained) cho bài toán phân loại ảnh nhiều lớp.

Quy trình (rút ra từ notebook Train_cnn_lan4):
  1. Quét thư mục dữ liệu, kiểm tra ảnh hỏng, loại ảnh trùng lặp (SHA-256).
  2. Chia train / val / test một lần và lưu lại (splits.csv) để tái lập.
  3. Fine-tune: đóng băng backbone vài epoch đầu, sau đó mở khóa; LR riêng cho head/backbone.
  4. Chọn checkpoint chỉ dựa trên validation (macro F1). Giữ top-K checkpoint.
  5. Model soup (uniform + greedy) từ top-K checkpoint.
  6. Đánh giá trên test đúng một lần cho: best / uniform soup / greedy soup.
  7. Xuất `final_model.pt` (chọn theo validation) để dùng với predict_convnext.py.

Cấu trúc dữ liệu mong đợi:
    data_dir/
        CLASS_A/ *.jpg
        CLASS_B/ *.png
        CLASS_C/ ...

Cách chạy:
  * Google Colab (khuyến nghị dùng GPU):  chỉ cần  `%run train_convnext.py`
      - tự mount Google Drive, copy dữ liệu từ  /content/drive/MyDrive/data_cnn
        về /content/project/Data (giống notebook gốc),
      - lưu kết quả vào /content/drive/MyDrive/project_outputs/convnext
  * Máy tính cá nhân:  `python train_convnext.py`
      - tự tìm thư mục ./Data, ./data_cnn hoặc ./data; kết quả lưu vào ./outputs
  * Chỉ định đường dẫn thủ công ở bất kỳ đâu:
      python train_convnext.py --data-dir D:/data_cnn --output-dir D:/ket_qua
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # chạy được cả khi không có màn hình
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MODEL_ID = "convnext_tiny"


# --------------------------------------------------------------------------- #
# Tham số dòng lệnh
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ConvNeXt-Tiny (transfer learning) + model soup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dữ liệu / đầu ra
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Thư mục chứa các thư mục con, mỗi thư mục là một lớp. "
                        "Bỏ trống để tự tìm (xem hướng dẫn ở đầu file).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Nơi lưu kết quả. Bỏ trống: Colab -> Drive, máy tính -> ./outputs")
    p.add_argument("--drive-data", type=Path, default=Path("/content/drive/MyDrive/data_cnn"),
                   help="(Colab) Thư mục dữ liệu trên Google Drive.")
    p.add_argument("--drive-output", type=Path, default=Path("/content/drive/MyDrive/project_outputs"),
                   help="(Colab) Thư mục gốc lưu kết quả trên Google Drive.")
    p.add_argument("--class-names", nargs="+", default=None,
                   help="Danh sách lớp theo thứ tự nhãn. Mặc định: tên thư mục con (sắp xếp A-Z).")
    # Huấn luyện
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--lr-head", type=float, default=1e-4, help="LR của classification head.")
    p.add_argument("--lr-backbone", type=float, default=1e-5, help="LR của backbone.")
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=1, help="Số bước tích lũy gradient.")
    p.add_argument("--patience", type=int, default=6, help="Early stopping patience (theo val macro F1).")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
                   help="Đóng băng backbone lúc đầu.")
    p.add_argument("--unfreeze-epoch", type=int, default=3,
                   help="Epoch bắt đầu mở khóa backbone (chỉ dùng khi --freeze-backbone).")
    p.add_argument("--imbalance", choices=["weighted_loss", "weighted_sampler", "none"],
                   default="weighted_loss", help="Chiến lược xử lý mất cân bằng lớp.")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="Dùng trọng số ImageNet (cần internet lần đầu).")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="Mixed precision (chỉ có tác dụng trên GPU CUDA).")
    # Chia dữ liệu
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True,
                   help="Chia theo tỉ lệ lớp (stratified).")
    # Checkpoint / soup
    p.add_argument("--top-k", type=int, default=5, help="Số checkpoint tốt nhất giữ lại để làm soup.")
    p.add_argument("--soup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--greedy-min-delta", type=float, default=0.0)
    # Hệ thống
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                   help="Bật chế độ xác định (chậm hơn một chút nhưng tái lập tốt hơn).")
    p.add_argument("--device", default=None, help="cuda / cpu. Mặc định tự chọn.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Tự nhận diện Colab / máy tính và tìm đường dẫn dữ liệu
# --------------------------------------------------------------------------- #
def is_colab() -> bool:
    return ("google.colab" in sys.modules or "COLAB_GPU" in os.environ
            or "COLAB_RELEASE_TAG" in os.environ or Path("/content/sample_data").exists())


def ensure_drive_mounted() -> bool:
    """Mount Google Drive nếu chưa mount. Chỉ mount được khi chạy bằng %run (không phải !python)."""
    if Path("/content/drive/MyDrive").is_dir():
        return True
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
    except Exception as exc:  # noqa: BLE001
        print(f"Không tự mount được Google Drive ({type(exc).__name__}). "
              "Hãy chạy cell:  from google.colab import drive; drive.mount('/content/drive')")
    return Path("/content/drive/MyDrive").is_dir()


def has_class_dirs(path: Path, class_names: Optional[Sequence[str]]) -> bool:
    if not path.is_dir():
        return False
    if class_names:
        return all((path / c).is_dir() for c in class_names)
    return sum(1 for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")) >= 2


def resolve_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return args.data_dir.expanduser()
    candidates: List[Path] = []
    if os.environ.get("PROJECT_ROOT"):
        candidates.append(Path(os.environ["PROJECT_ROOT"]) / "Data")
    if is_colab():
        local = Path("/content/project/Data")
        candidates.append(local)
        for c in candidates:
            if has_class_dirs(c, args.class_names):
                return c
        if ensure_drive_mounted() and args.drive_data.is_dir():
            print(f"Copy dữ liệu từ Drive: {args.drive_data} -> {local} (chỉ làm một lần)...")
            shutil.copytree(args.drive_data, local, dirs_exist_ok=True)
            return local
        sys.exit(f"Không tìm thấy dữ liệu. Hãy đặt ảnh tại {args.drive_data}/<tên_lớp>/*.jpg "
                 "hoặc truyền --data-dir / --drive-data.")
    candidates += [Path.cwd() / "Data", Path.cwd() / "data_cnn", Path.cwd() / "data"]
    for c in candidates:
        if has_class_dirs(c, args.class_names):
            return c
    sys.exit("Không tìm thấy thư mục dữ liệu (đã thử ./Data, ./data_cnn, ./data). "
             "Hãy truyền --data-dir <đường dẫn>.")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir.expanduser()
    if is_colab():
        if ensure_drive_mounted():
            return args.drive_output / "convnext"
        print("Cảnh báo: chưa mount Drive, kết quả lưu ở /content/outputs sẽ MẤT khi phiên Colab kết thúc.")
        return Path("/content/outputs")
    return Path("outputs")


# --------------------------------------------------------------------------- #
# Tiện ích chung
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_: int) -> None:
    s = torch.initial_seed() % (2**32)
    np.random.seed(s)
    random.seed(s)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def json_safe(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Quét dữ liệu + chia tập
# --------------------------------------------------------------------------- #
def scan_dataset(data_dir: Path, class_names: Sequence[str], out_dir: Path) -> pd.DataFrame:
    """Đọc toàn bộ ảnh, bỏ ảnh hỏng, loại ảnh trùng nội dung (giữ bản đầu tiên)."""
    rows: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    for label, cls in enumerate(class_names):
        files = sorted(p for p in (data_dir / cls).rglob("*") if p.is_file())
        for path in tqdm(files, desc=f"Quét {cls}"):
            if path.suffix.lower() not in VALID_EXTENSIONS:
                invalid.append({"path": str(path), "class_name": cls, "error": "định dạng không hỗ trợ"})
                continue
            try:
                if path.stat().st_size == 0:
                    raise ValueError("file rỗng")
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.convert("RGB").load()
                rows.append({"path": str(path.resolve()), "class_name": cls, "label": label,
                             "sha256": sha256_file(path)})
            except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
                invalid.append({"path": str(path), "class_name": cls, "error": f"{type(exc).__name__}: {exc}"})

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"Không tìm thấy ảnh hợp lệ trong {data_dir}")

    # Cùng nội dung nhưng khác nhãn => dữ liệu mâu thuẫn, dừng lại.
    conflict = df.groupby("sha256")["class_name"].nunique()
    bad = conflict[conflict > 1].index
    if len(bad):
        df[df["sha256"].isin(bad)].to_csv(out_dir / "label_conflicts.csv", index=False)
        raise ValueError(f"Có ảnh giống hệt nhau nhưng khác nhãn. Xem {out_dir / 'label_conflicts.csv'}")

    dup_mask = df.duplicated("sha256", keep="first")
    if dup_mask.any():
        df[dup_mask].to_csv(out_dir / "excluded_duplicates.csv", index=False)
    if invalid:
        pd.DataFrame(invalid).to_csv(out_dir / "invalid_images.csv", index=False)
    print(f"Ảnh hợp lệ: {len(df)} | trùng bị loại: {int(dup_mask.sum())} | lỗi/không hỗ trợ: {len(invalid)}")
    return df[~dup_mask].reset_index(drop=True)


def make_splits(df: pd.DataFrame, val_size: float, test_size: float, seed: int, stratify: bool) -> pd.DataFrame:
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("val-size và test-size phải > 0 và tổng < 1.")
    if (df.groupby("class_name").size() < 3).any():
        raise ValueError("Mỗi lớp cần ít nhất 3 ảnh hợp lệ.")
    strat = df["label"] if stratify else None
    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True, stratify=strat)
    strat2 = train_val["label"] if stratify else None
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  random_state=seed, shuffle=True, stratify=strat2)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")],
                    ignore_index=True)
    assert out.groupby("sha256")["split"].nunique().max() == 1, "Rò rỉ dữ liệu giữa các tập!"
    return out


# --------------------------------------------------------------------------- #
# Dataset / transform
# --------------------------------------------------------------------------- #
def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    resize = int(round(image_size * 256 / 224))
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.75, 1.33),
                                     interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.2, 0.03)], p=0.7),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        row = self.frame.iloc[i]
        with Image.open(row["path"]) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(row["label"]), row["path"]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def create_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    try:
        model = models.convnext_tiny(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError("Không tải được trọng số ImageNet. Kiểm tra internet hoặc dùng --no-pretrained.") from exc
        raise
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad = trainable
    for p in model.classifier.parameters():  # head luôn học
        p.requires_grad = True


def build_optimizer(model: nn.Module, lr_head: float, lr_backbone: float, wd: float):
    head_ids = {id(p) for p in model.classifier[2].parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    return torch.optim.AdamW([
        {"params": backbone, "lr": lr_backbone, "group_name": "backbone"},
        {"params": head, "lr": lr_head, "group_name": "head"},
    ], weight_decay=wd)


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def autocast_ctx(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> float:
    return float(precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)[2])


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, frozen, accum, max_norm, n_classes):
    if frozen:  # backbone đóng băng -> giữ ở eval, chỉ head ở train
        model.eval()
        model.classifier.train()
    else:
        model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, n = 0.0, 0
    ys, ps = [], []
    for step, (x, y, _) in enumerate(tqdm(loader, desc="  train", leave=False)):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss / accum).backward()
        if (step + 1) % accum == 0 or step + 1 == len(loader):
            if max_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        ys += y.cpu().tolist()
        ps += logits.detach().argmax(1).cpu().tolist()
    return {"loss": total_loss / n, "accuracy": accuracy_score(ys, ps), "macro_f1": macro_f1(ys, ps, n_classes)}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp, n_classes, collect=False):
    model.eval()
    total_loss, n = 0.0, 0
    ys, ps, probs, paths = [], [], [], []
    for x, y, p in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(amp):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        ys += y.cpu().tolist()
        ps += logits.argmax(1).cpu().tolist()
        if collect:
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            paths += list(p)
    out = {"loss": total_loss / n, "accuracy": accuracy_score(ys, ps), "macro_f1": macro_f1(ys, ps, n_classes)}
    if collect:
        out.update({"y_true": ys, "y_pred": ps, "probs": np.vstack(probs), "paths": paths})
    return out


# --------------------------------------------------------------------------- #
# Checkpoint + soup
# --------------------------------------------------------------------------- #
def save_bundle(path: Path, state_dict: Dict[str, torch.Tensor], meta: Dict[str, Any]) -> None:
    """Mọi file .pt đều có cùng định dạng để predict_convnext.py đọc trực tiếp."""
    tmp = path.with_suffix(".tmp")
    torch.save({**meta, "model_state_dict": {k: v.detach().cpu() for k, v in state_dict.items()}}, tmp)
    tmp.replace(path)


def load_state(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")["model_state_dict"]


class TopKCheckpoints:
    """Giữ K checkpoint có val macro F1 cao nhất (đồng hạng thì ưu tiên val loss thấp hơn)."""

    def __init__(self, directory: Path, k: int, meta: Dict[str, Any]) -> None:
        self.dir, self.k, self.meta = directory, k, meta
        self.items: List[Dict[str, Any]] = []
        directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _rank(it): return (it["val_f1"], -it["val_loss"])

    def offer(self, model: nn.Module, epoch: int, m: Dict[str, float]) -> None:
        cand = {"epoch": epoch, "val_f1": m["macro_f1"], "val_loss": m["loss"],
                "val_acc": m["accuracy"], "path": self.dir / f"epoch_{epoch:03d}.pt"}
        if len(self.items) >= self.k and self._rank(cand) <= min(self._rank(i) for i in self.items):
            return
        save_bundle(cand["path"], model.state_dict(),
                    {**self.meta, "variant": "checkpoint", "epoch": epoch, "val_macro_f1": cand["val_f1"]})
        self.items.append(cand)
        self.items.sort(key=self._rank, reverse=True)
        for old in self.items[self.k:]:
            old["path"].unlink(missing_ok=True)
        self.items = self.items[: self.k]

    @property
    def best(self): return self.items[0]


def float_state(state):
    return {k: (v.float().clone() if torch.is_floating_point(v) else v.clone()) for k, v in state.items()}


def uniform_soup(paths: Sequence[Path]) -> Dict[str, torch.Tensor]:
    avg = float_state(load_state(paths[0]))
    for n, p in enumerate(paths[1:], start=2):
        s = load_state(p)
        for k in avg:
            if torch.is_floating_point(avg[k]):
                avg[k].mul_((n - 1) / n).add_(s[k].float(), alpha=1 / n)
    return avg


def greedy_soup(paths: Sequence[Path], model, val_loader, criterion, device, amp, n_classes, min_delta):
    """Bắt đầu từ checkpoint tốt nhất; chỉ thêm checkpoint khác nếu val macro F1 không giảm."""
    sources = [paths[0]]
    state = float_state(load_state(paths[0]))
    model.load_state_dict(state)
    best = evaluate(model, val_loader, criterion, device, amp, n_classes)
    for p in paths[1:]:
        cand = load_state(p)
        n = len(sources)
        prop = {k: (v * (n / (n + 1)) + cand[k].float() / (n + 1)) if torch.is_floating_point(v) else v.clone()
                for k, v in state.items()}
        model.load_state_dict(prop)
        m = evaluate(model, val_loader, criterion, device, amp, n_classes)
        if m["macro_f1"] >= best["macro_f1"] + min_delta:
            state, best = prop, m
            sources.append(p)
    return state, best, sources


# --------------------------------------------------------------------------- #
# Vẽ biểu đồ
# --------------------------------------------------------------------------- #
def plot_history(h: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (key, title) in zip(axes, [("loss", "Loss"), ("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]):
        ax.plot(h["epoch"], h[f"train_{key}"], label="Train")
        ax.plot(h["epoch"], h[f"val_{key}"], label="Validation")
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: Sequence[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted", ylabel="True", title=title)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args)
    args.output_dir = resolve_output_dir(args)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Dữ liệu : {args.data_dir}\nKết quả : {out}")
    set_seed(args.seed, args.deterministic)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(args.amp and device.type == "cuda")
    print(f"Device: {device} | AMP: {amp}")
    if device.type == "cpu":
        print("Cảnh báo: đang chạy CPU, quá trình train sẽ rất chậm.")

    # ----- Lớp -----
    if args.class_names:
        class_names = list(args.class_names)
    else:
        class_names = sorted(d.name for d in args.data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    missing = [c for c in class_names if not (args.data_dir / c).is_dir()]
    if missing or len(class_names) < 2:
        raise SystemExit(f"Cần >= 2 thư mục lớp trong {args.data_dir}. Thiếu: {missing}")
    n_classes = len(class_names)
    print(f"Các lớp ({n_classes}): {class_names}")
    (out / "class_mapping.json").write_text(json.dumps({c: i for i, c in enumerate(class_names)}, indent=2))

    # ----- Dữ liệu -----
    manifest = scan_dataset(args.data_dir, class_names, out)
    splits = make_splits(manifest, args.val_size, args.test_size, args.seed, args.stratify)
    splits.to_csv(out / "splits.csv", index=False)
    print(pd.crosstab(splits["class_name"], splits["split"]).reindex(class_names), "\n")

    train_df = splits[splits["split"] == "train"]
    val_df = splits[splits["split"] == "val"]
    test_df = splits[splits["split"] == "test"]
    train_tf, eval_tf = build_transforms(args.image_size)

    common = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.num_workers > 0, worker_init_fn=seed_worker)
    counts = train_df["label"].value_counts().reindex(range(n_classes), fill_value=0).to_numpy(dtype=np.float64)
    class_weights = torch.tensor(len(train_df) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    print("Class weights:", {c: round(float(w), 3) for c, w in zip(class_names, class_weights)})

    g = make_generator(args.seed)
    train_ds = ImageDataset(train_df, train_tf)
    if args.imbalance == "weighted_sampler":
        sw = class_weights[torch.tensor(train_df["label"].to_numpy())].double()
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True, generator=g)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, generator=g, **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, **common)
    val_loader = DataLoader(ImageDataset(val_df, eval_tf), batch_size=args.batch_size, shuffle=False,
                            generator=make_generator(args.seed + 1), **common)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if args.imbalance == "weighted_loss" else None,
        label_smoothing=args.label_smoothing)

    # ----- Model -----
    model = create_model(n_classes, args.pretrained)
    set_backbone_trainable(model, trainable=not args.freeze_backbone)
    model.to(device)
    optimizer = build_optimizer(model, args.lr_head, args.lr_backbone, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    meta = {"model_id": MODEL_ID, "class_names": class_names, "image_size": args.image_size,
            "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD), "seed": args.seed}
    ckpt_dir = out / "checkpoints"
    topk = TopKCheckpoints(ckpt_dir, args.top_k, meta)

    # ----- Vòng lặp huấn luyện -----
    history: List[Dict[str, Any]] = []
    best_f1, bad_epochs = -math.inf, 0
    frozen = args.freeze_backbone
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if frozen and epoch >= args.unfreeze_epoch:
            set_backbone_trainable(model, True)
            frozen = False
            print(f"Epoch {epoch}: mở khóa backbone.")
        t_ep = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp, frozen,
                             args.grad_accum, args.max_grad_norm, n_classes)
        va = evaluate(model, val_loader, criterion, device, amp, n_classes)
        scheduler.step(va["macro_f1"])
        topk.offer(model, epoch, va)

        improved = va["macro_f1"] > best_f1 + args.min_delta
        if improved:
            best_f1, bad_epochs = va["macro_f1"], 0
        else:
            bad_epochs += 1
        history.append({"epoch": epoch,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()},
                        "lr_backbone": optimizer.param_groups[0]["lr"],
                        "lr_head": optimizer.param_groups[1]["lr"],
                        "seconds": time.perf_counter() - t_ep})
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} "
              f"F1 {tr['macro_f1']:.4f} | val loss {va['loss']:.4f} acc {va['accuracy']:.4f} "
              f"F1 {va['macro_f1']:.4f} | {history[-1]['seconds']:.0f}s")
        if bad_epochs >= args.patience:
            print(f"Early stopping tại epoch {epoch}.")
            break

    train_seconds = time.perf_counter() - t0
    hist = pd.DataFrame(history)
    hist.to_csv(out / "history.csv", index=False)
    plot_history(hist, out / "training_curves.png")

    # ----- Chọn ứng viên (chỉ dùng validation) -----
    best = topk.best
    best_path = out / "best_model.pt"
    save_bundle(best_path, load_state(best["path"]),
                {**meta, "variant": "best", "epoch": best["epoch"], "val_macro_f1": best["val_f1"]})
    candidates: Dict[str, Dict[str, Any]] = {
        "best": {"path": best_path, "val_f1": best["val_f1"], "n_ckpt": 1}}

    if args.soup and len(topk.items) > 1:
        paths = [i["path"] for i in topk.items]
        eval_model = create_model(n_classes, pretrained=False).to(device)

        u_state = uniform_soup(paths)
        eval_model.load_state_dict(u_state)
        u_val = evaluate(eval_model, val_loader, criterion, device, amp, n_classes)
        u_path = out / "uniform_soup.pt"
        save_bundle(u_path, u_state, {**meta, "variant": "uniform_soup", "val_macro_f1": u_val["macro_f1"],
                                      "sources": [str(p) for p in paths]})
        candidates["uniform_soup"] = {"path": u_path, "val_f1": u_val["macro_f1"], "n_ckpt": len(paths)}

        g_state, g_val, g_src = greedy_soup(paths, eval_model, val_loader, criterion, device, amp,
                                            n_classes, args.greedy_min_delta)
        g_path = out / "greedy_soup.pt"
        save_bundle(g_path, g_state, {**meta, "variant": "greedy_soup", "val_macro_f1": g_val["macro_f1"],
                                      "sources": [str(p) for p in g_src]})
        candidates["greedy_soup"] = {"path": g_path, "val_f1": g_val["macro_f1"], "n_ckpt": len(g_src)}
        del eval_model
        print("Validation macro F1 -> " + " | ".join(f"{k}: {v['val_f1']:.4f}" for k, v in candidates.items()))

    # Chọn mô hình cuối theo validation; hòa thì ưu tiên 'best' (đơn giản nhất).
    order = ["best", "uniform_soup", "greedy_soup"]
    chosen = max((k for k in order if k in candidates), key=lambda k: (candidates[k]["val_f1"], -order.index(k)))
    print(f"Mô hình được chọn theo validation: {chosen}")

    # ----- Đánh giá test (lần duy nhất) -----
    test_loader = DataLoader(ImageDataset(test_df, eval_tf), batch_size=args.batch_size, shuffle=False,
                             generator=make_generator(args.seed + 2), **common)
    eval_model = create_model(n_classes, pretrained=False).to(device)
    results: List[Dict[str, Any]] = []
    (out / "test").mkdir(exist_ok=True)
    for name in [k for k in order if k in candidates]:
        eval_model.load_state_dict(load_state(candidates[name]["path"]))
        r = evaluate(eval_model, test_loader, criterion, device, amp, n_classes, collect=True)
        labels = list(range(n_classes))
        report = classification_report(r["y_true"], r["y_pred"], labels=labels, target_names=class_names,
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels)
        plot_confusion(cm, class_names, f"ConvNeXt-T — {name}", out / "test" / f"confusion_{name}.png")
        (out / "test" / f"report_{name}.json").write_text(json.dumps(json_safe(report), indent=2))
        pred_df = pd.DataFrame({"path": r["paths"],
                                "true_class": [class_names[i] for i in r["y_true"]],
                                "pred_class": [class_names[i] for i in r["y_pred"]]})
        for i, c in enumerate(class_names):
            pred_df[f"prob_{c}"] = r["probs"][:, i]
        pred_df.to_csv(out / "test" / f"predictions_{name}.csv", index=False)
        results.append({"variant": name, "chosen_by_val": name == chosen, "val_macro_f1": candidates[name]["val_f1"],
                        "n_checkpoints": candidates[name]["n_ckpt"], "test_accuracy": r["accuracy"],
                        "test_macro_f1": r["macro_f1"], "test_loss": r["loss"],
                        "test_weighted_f1": report["weighted avg"]["f1-score"], "test_samples": len(r["y_true"])})
        print(f"\n[{name}] test acc {r['accuracy']:.4f} | macro F1 {r['macro_f1']:.4f}")
        print(classification_report(r["y_true"], r["y_pred"], labels=labels, target_names=class_names,
                                    zero_division=0, digits=4))
    res_df = pd.DataFrame(results)
    res_df.to_csv(out / "results.csv", index=False)

    # ----- Xuất mô hình cuối để suy luận -----
    final_path = out / "final_model.pt"
    src = torch.load(candidates[chosen]["path"], map_location="cpu")
    src["variant"] = f"final ({chosen})"
    torch.save(src, final_path)

    (out / "run_config.json").write_text(json.dumps(json_safe({
        "args": vars(args), "class_names": class_names, "chosen_variant": chosen,
        "train_seconds": train_seconds, "epochs_completed": len(hist),
        "torch": torch.__version__, "device": str(device)}), indent=2))

    print("\n" + res_df.to_string(index=False))
    print(f"\nHoàn tất. Mô hình dùng để suy luận: {final_path}")
    del eval_model, model
    gc.collect()


if __name__ == "__main__":
    main()
