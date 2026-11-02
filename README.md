# Phân loại ảnh bằng ConvNeXt-Tiny (Transfer Learning + Model Soup)

Bộ script này được tách ra từ notebook `Train_cnn_lan4` và chỉ giữ lại phần **ConvNeXt-Tiny**.
Mô hình ImageNet pretrained của `torchvision` được fine-tune cho bài toán phân loại ảnh nhiều lớp
(mặc định notebook dùng 3 lớp: `HOP`, `VI`, `VIEN`, nhưng script tự nhận số lớp từ thư mục dữ liệu).

| File | Vai trò |
|---|---|
| `train_convnext.py` | Quét dữ liệu, chia tập, huấn luyện, tạo model soup, đánh giá test, xuất `final_model.pt` |
| `predict_convnext.py` | Dùng `final_model.pt` để dự đoán ảnh mới (một ảnh hoặc cả thư mục) |
| `README.md` | Tài liệu này |

---

## 1. Cài đặt

Yêu cầu Python ≥ 3.9.

```bash
pip install torch torchvision numpy pandas pillow scikit-learn matplotlib tqdm
```

> Nên cài PyTorch theo đúng phiên bản CUDA của máy tại https://pytorch.org/get-started/.
> Lần chạy đầu tiên cần internet để tải trọng số ImageNet của ConvNeXt-Tiny (~110 MB).
> Train trên CPU vẫn chạy được nhưng rất chậm, nên dùng GPU (hoặc Google Colab).

## 2. Chuẩn bị dữ liệu

Mỗi lớp là một thư mục con chứa ảnh:

```text
Data/
├── HOP/    *.jpg, *.png, ...
├── VI/
└── VIEN/
```

Định dạng hỗ trợ: `.jpg .jpeg .png .bmp .tif .tiff .webp`.
Thứ tự nhãn mặc định là thứ tự A-Z của tên thư mục; có thể chỉ định bằng `--class-names`.

## 3. Huấn luyện

### 3a. Trên Google Colab (giống cấu trúc notebook gốc)

Đặt dữ liệu trên Drive tại `MyDrive/data_cnn/` (gồm các thư mục `HOP`, `VI`, `VIEN`). Chọn **Runtime → Change runtime type → GPU**, upload `train_convnext.py` và `predict_convnext.py` lên Colab, rồi chạy **một cell duy nhất**:

```python
%run /content/train_convnext.py
```

Script sẽ tự làm:
- mount Google Drive,
- copy `MyDrive/data_cnn` về `/content/project/Data` (chỉ copy lần đầu, vì đọc trực tiếp từ Drive rất chậm),
- train và lưu toàn bộ kết quả vào `MyDrive/project_outputs/convnext/` (không mất khi phiên Colab ngắt).

> Dùng `%run` (không phải `!python`) để script mount được Drive. Nếu muốn dùng `!python`, hãy chạy trước cell
> `from google.colab import drive; drive.mount('/content/drive')`.

Muốn đổi tham số, thêm vào sau lệnh, ví dụ: `%run /content/train_convnext.py --epochs 15 --batch-size 16`.

### 3b. Trên máy tính

```bash
# Tự tìm thư mục ./Data, ./data_cnn hoặc ./data; kết quả lưu vào ./outputs
python train_convnext.py

# Hoặc chỉ định đường dẫn thủ công
python train_convnext.py --data-dir D:/data_cnn --output-dir D:/ket_qua
```

### Tùy chọn hay dùng

```bash
# Giữ đúng thứ tự lớp như notebook
python train_convnext.py --data-dir ./Data --class-names HOP VI VIEN

# GPU nhỏ bị hết bộ nhớ: giảm batch, hoặc dùng tích lũy gradient
python train_convnext.py --data-dir ./Data --batch-size 8 --grad-accum 3

# Chạy thử nhanh để kiểm tra pipeline
python train_convnext.py --data-dir ./Data --epochs 2 --top-k 2 --num-workers 0

# Xử lý mất cân bằng lớp bằng sampler thay vì weighted loss
python train_convnext.py --data-dir ./Data --imbalance weighted_sampler

# Không đóng băng backbone / không làm soup / không dùng pretrained
python train_convnext.py --data-dir ./Data --no-freeze-backbone --no-soup --no-pretrained
```

Xem đầy đủ: `python train_convnext.py --help`.

### Giá trị mặc định (lấy từ notebook)

| Tham số | Mặc định | Ghi chú |
|---|---|---|
| `--epochs` | 25 | có early stopping |
| `--batch-size` | 24 | batch size ConvNeXt trong notebook |
| `--image-size` | 224 | |
| `--lr-head` / `--lr-backbone` | 1e-4 / 1e-5 | AdamW, LR riêng cho head và backbone |
| `--weight-decay` | 0.05 | |
| `--label-smoothing` | 0.1 | |
| `--freeze-backbone`, `--unfreeze-epoch` | bật, 3 | head học trước, backbone mở khóa từ epoch 3 |
| `--imbalance` | `weighted_loss` | cross-entropy có trọng số theo lớp |
| `--patience` | 6 | theo validation macro F1 |
| `--top-k` | 5 | số checkpoint dùng làm soup |
| `--val-size` / `--test-size` | 0.15 / 0.15 | tức 70 / 15 / 15 |
| `--seed` | 43 | |

Scheduler là `ReduceLROnPlateau` theo validation macro F1 (factor 0.5, patience 1); có clip gradient (1.0) và AMP khi có GPU.

### Quy trình bên trong

1. **Quét dữ liệu**: kiểm tra từng ảnh có giải mã được không, tính SHA-256. Ảnh trùng nội dung bị loại (giữ bản đầu tiên, không xoá file). Nếu cùng một ảnh nằm ở hai lớp khác nhau thì script dừng lại và ghi `label_conflicts.csv`.
2. **Chia tập** train / val / test một lần và lưu vào `splits.csv`.
3. **Fine-tune** ConvNeXt-Tiny: thay lớp Linear cuối, đóng băng backbone vài epoch rồi mở khóa.
4. **Chọn checkpoint chỉ bằng validation** (macro F1, hòa thì xét val loss). Giữ top-K.
5. **Model soup** từ top-K checkpoint:
   - *Uniform soup*: trung bình đều tất cả.
   - *Greedy soup*: bắt đầu từ checkpoint tốt nhất, chỉ thêm checkpoint khác nếu val macro F1 không giảm.
6. **Đánh giá test một lần** cho 3 biến thể: `best`, `uniform_soup`, `greedy_soup`.
7. **`final_model.pt`** là biến thể có val macro F1 cao nhất (hòa thì ưu tiên `best`). Việc chọn này không dùng tập test, nên kết quả test không bị "chọn lọc" theo test.

### Kết quả đầu ra

```text
outputs/
├── final_model.pt          # <-- dùng cho predict_convnext.py
├── best_model.pt           # checkpoint tốt nhất theo validation
├── uniform_soup.pt
├── greedy_soup.pt
├── checkpoints/            # top-K checkpoint (epoch_XXX.pt)
├── class_mapping.json
├── splits.csv              # đường dẫn, nhãn, hash, split của từng ảnh
├── history.csv             # loss/acc/F1 theo epoch
├── training_curves.png
├── results.csv             # so sánh test của 3 biến thể
├── run_config.json         # toàn bộ tham số, phiên bản, thời gian train
├── test/
│   ├── report_<variant>.json
│   ├── confusion_<variant>.png
│   └── predictions_<variant>.csv   # xác suất từng lớp cho từng ảnh test
├── excluded_duplicates.csv # (nếu có)
└── invalid_images.csv      # (nếu có)
```

Mọi file `.pt` đều tự chứa tên lớp, kích thước ảnh, mean/std nên không cần file cấu hình đi kèm khi dự đoán.

## 4. Dự đoán ảnh mới

```bash
# Một ảnh
python predict_convnext.py --model outputs/final_model.pt --input anh.jpg

# Cả thư mục, lưu ra CSV
python predict_convnext.py --model outputs/final_model.pt --input ./anh_moi --output ket_qua.csv

# Duyệt cả thư mục con, dùng TTA (lật ngang), gắn UNCERTAIN nếu độ tin cậy < 0.6
python predict_convnext.py --model outputs/final_model.pt --input ./anh_moi \
    --recursive --tta --min-confidence 0.6 --output ket_qua.csv
```

Ví dụ kết quả in ra:

```text
anh1.jpg: VI (97.4%)  [VI 97.4%, HOP 2.1%, VIEN 0.5%]
```

File CSV có các cột: `path`, `prediction`, `confidence`, và `prob_<tên lớp>` cho từng lớp.

Dùng trong code Python:

```python
import torch
from PIL import Image
from predict_convnext import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names, transform, _ = load_model("outputs/final_model.pt", device)

img = Image.open("anh.jpg").convert("RGB")
with torch.inference_mode():
    probs = torch.softmax(model(transform(img).unsqueeze(0).to(device)), dim=1)[0]
print(class_names[int(probs.argmax())], float(probs.max()))
```

## 5. Lưu ý

- **Tái lập**: seed cố định và chế độ deterministic được bật mặc định (chậm hơn một chút). Tắt bằng `--no-deterministic` nếu cần tốc độ. Kết quả vẫn có thể lệch nhẹ giữa các phiên bản PyTorch/CUDA/phần cứng khác nhau.
- **Chia tập**: trong notebook gốc, đoạn code thực tế chia ngẫu nhiên *không* stratify (dù phần mô tả ghi stratified). Script này mặc định dùng **stratified** để mỗi tập đều có đủ các lớp; thêm `--no-stratify` nếu muốn giống hệt notebook. Lưu ý: dù cùng seed, kết quả chia sẽ khác notebook gốc.
- **Ảnh cùng nguồn**: script chỉ loại ảnh trùng *byte-by-byte*. Nếu dữ liệu có nhiều ảnh gần giống nhau (cùng một vật thể/bệnh nhân/phiên chụp), chúng có thể rơi vào cả train lẫn test và làm điểm test cao hơn thực tế. Khi đó nên chia theo nhóm.
- **Lớp ít ảnh**: mỗi lớp cần ít nhất 3 ảnh hợp lệ; với `--stratify`, mỗi lớp cần đủ ảnh để chia vào cả 3 tập.
- **Hết bộ nhớ GPU**: giảm `--batch-size` (kết hợp `--grad-accum` để giữ batch hiệu dụng), hoặc giảm `--image-size`.
- **Ảnh HEIC** không được hỗ trợ sẵn; cần chuyển sang JPG/PNG hoặc cài `pillow-heif`.
- **Bảo mật khi tải checkpoint**: chỉ dùng `torch.load` với file `.pt` do chính bạn tạo hoặc nguồn tin cậy.
