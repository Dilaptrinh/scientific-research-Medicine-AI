#!/usr/bin/env python3
"""
predict_convnext.py
===================
Dùng mô hình ConvNeXt-Tiny đã train bằng train_convnext.py để phân loại ảnh.

Ví dụ:
    # Một ảnh
    python predict_convnext.py --model outputs/final_model.pt --input anh.jpg

    # Cả thư mục (có thể thêm --recursive), lưu kết quả ra CSV
    python predict_convnext.py --model outputs/final_model.pt --input ./anh_moi --output ket_qua.csv

    # Dùng TTA (lật ngang) và gắn nhãn UNCERTAIN nếu độ tin cậy < 0.6
    python predict_convnext.py --model outputs/final_model.pt --input ./anh_moi --tta --min-confidence 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dự đoán lớp ảnh bằng ConvNeXt-Tiny đã train.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=Path, default=None,
                   help="File .pt do train_convnext.py tạo. Bỏ trống: tự tìm outputs/final_model.pt "
                        "hoặc bản trên Google Drive.")
    p.add_argument("--input", type=Path, required=True, help="Đường dẫn một ảnh hoặc một thư mục ảnh.")
    p.add_argument("--output", type=Path, default=None, help="Lưu kết quả ra file CSV (tùy chọn).")
    p.add_argument("--recursive", action="store_true", help="Duyệt cả thư mục con.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=3, help="Số lớp xác suất cao nhất hiển thị.")
    p.add_argument("--tta", action="store_true", help="Test-time augmentation: trung bình ảnh gốc + lật ngang.")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="Nếu xác suất cao nhất thấp hơn ngưỡng này, gắn nhãn UNCERTAIN.")
    p.add_argument("--device", default=None, help="cuda / cpu. Mặc định tự chọn.")
    p.add_argument("--quiet", action="store_true", help="Không in từng ảnh, chỉ lưu CSV / tóm tắt.")
    return p.parse_args()


def load_model(path: Path, device: torch.device):
    if not path.is_file():
        sys.exit(f"Không tìm thấy file mô hình: {path}")
    ckpt = torch.load(path, map_location="cpu")
    class_names: List[str] = list(ckpt["class_names"])
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    size = int(ckpt.get("image_size", 224))
    tf = transforms.Compose([
        transforms.Resize(int(round(size * 256 / 224)), interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(ckpt.get("mean", (0.485, 0.456, 0.406)), ckpt.get("std", (0.229, 0.224, 0.225))),
    ])
    return model, class_names, tf, ckpt


def collect_images(root: Path, recursive: bool) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        sys.exit(f"Đường dẫn không tồn tại: {root}")
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)


@torch.inference_mode()
def predict(model, tf, paths: Sequence[Path], device, batch_size: int, tta: bool):
    """Trả về (danh sách (path, prob_tensor)) và danh sách file lỗi."""
    results, failed = [], []
    use_amp = device.type == "cuda"
    for i in tqdm(range(0, len(paths), batch_size), desc="Dự đoán", disable=len(paths) <= batch_size):
        batch, ok = [], []
        for p in paths[i:i + batch_size]:
            try:
                with Image.open(p) as im:
                    batch.append(tf(im.convert("RGB")))
                ok.append(p)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                failed.append((p, str(exc)))
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            probs = torch.softmax(model(x).float(), dim=1)
            if tta:
                probs = (probs + torch.softmax(model(torch.flip(x, dims=[3])).float(), dim=1)) / 2
        results += list(zip(ok, probs.cpu()))
    return results, failed


def main() -> None:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.model is None:
        for cand in (Path("outputs/final_model.pt"),
                     Path("/content/drive/MyDrive/project_outputs/convnext/final_model.pt")):
            if cand.is_file():
                args.model = cand
                break
        else:
            sys.exit("Không tự tìm thấy final_model.pt, hãy truyền --model <đường dẫn>.")
    model, class_names, tf, ckpt = load_model(args.model, device)
    print(f"Mô hình: {args.model} | biến thể: {ckpt.get('variant', '?')} | lớp: {class_names} | device: {device}")

    paths = collect_images(args.input, args.recursive)
    if not paths:
        sys.exit(f"Không có ảnh hợp lệ trong {args.input}")

    results, failed = predict(model, tf, paths, device, args.batch_size, args.tta)

    rows = []
    k = min(args.top_k, len(class_names))
    for path, probs in results:
        conf, idx = probs.max(0)
        label = class_names[int(idx)]
        if args.min_confidence is not None and float(conf) < args.min_confidence:
            label = "UNCERTAIN"
        top = probs.topk(k)
        rows.append({"path": str(path), "prediction": label, "confidence": float(conf),
                     **{f"prob_{c}": float(probs[i]) for i, c in enumerate(class_names)}})
        if not args.quiet:
            top_str = ", ".join(f"{class_names[int(i)]} {float(v):.1%}" for v, i in zip(top.values, top.indices))
            print(f"{path.name}: {label} ({float(conf):.1%})  [{top_str}]")

    df = pd.DataFrame(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Đã lưu kết quả: {args.output}")

    print(f"\nĐã dự đoán {len(df)} ảnh" + (f", bỏ qua {len(failed)} ảnh lỗi" if failed else "") + ".")
    if not df.empty and len(df) > 1:
        print("Phân bố dự đoán:\n" + df["prediction"].value_counts().to_string())
    for p, err in failed:
        print(f"  [LỖI] {p}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
