# Medicine AI Research

Backend skeleton cho nghiên cứu nhận diện nhiều loại thuốc trong một ảnh:

```text
Ảnh thuốc -> YOLO detection/crop
           -> few-shot embedding cho ảnh viên -> FAISS retrieval -> VLM trả lời
```

> Đây là dự án nghiên cứu. Kết quả nhận diện không thay thế tư vấn của bác sĩ
> hoặc hướng dẫn sử dụng thuốc chính thức.

## 1. Yêu cầu môi trường

- Windows, Python 3.10 trở lên
- Git
- Có thể dùng PowerShell hoặc terminal tích hợp trong VS Code

Kiểm tra Python:

```powershell
py -3 --version
```

Nếu lệnh trên không chạy, cài Python từ python.org và chọn **Add Python to PATH**
trong quá trình cài đặt.

## 2. Cài đặt lần đầu

Mở PowerShell tại thư mục repository:

```powershell
cd D:\My_File\code\NCKH
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Nếu PowerShell chặn activate script, chỉ cần chạy Python trực tiếp trong
virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 3. Chạy và kiểm tra skeleton

Chạy test:

```powershell
pytest
```

Khởi động API:

```powershell
uvicorn app.main:app --reload
```

Mở tài liệu API tại <http://127.0.0.1:8000/docs>.

Kiểm tra server:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

Kết quả:

```json
{"status": "ok"}
```

Endpoint dự đoán hiện chỉ tích hợp YOLO detection/crop và cần có YOLO weights.

## 3.1. Đưa model YOLO hiện có vào project

Code adapter YOLO nằm tại
[`app/services/detection.py`](./app/services/detection.py). Model weights không
được commit vào Git. Hãy tạo thư mục `models/yolo/` và copy file:

```text
D:\My_File\code\python\New folder\Model\Mo_Hinh_YOLO\best.pt
```

vào:

```text
models/yolo/best.pt
```

Cài thêm dependency cho YOLO:

```powershell
pip install -e ".[ml,dev]"
```

Adapter nhận ảnh OpenCV dạng `numpy.ndarray`, chạy YOLO, lưu các crop vào
`data/processed/` và trả mỗi detection theo format:

```json
{
  "label": "hop",
  "confidence": 0.92,
  "bounding_box": [100, 80, 350, 500],
  "crop_path": "data/processed/image_001_crop_0.jpg"
}
```

## 4. Cấu trúc thư mục

```text
app/
  api/
    routes.py          # Nhận request và trả response HTTP
  core/                # Cấu hình, logging và hạ tầng dùng chung
  schemas/
    prediction.py      # Format input/output chuẩn
  services/
    detection.py       # Adapter YOLO và lưu crop
    retrieval.py       # Sẽ chứa adapter FAISS
    vlm.py             # Sẽ chứa adapter VLM
  main.py              # Khởi tạo FastAPI

data/
  raw/                 # Ảnh gốc; không commit ảnh thật
  processed/           # Ảnh crop và kết quả tiền xử lý
  annotations/         # Nhãn YOLO và metadata
  pill_dataset/        # Ảnh train embedding, chia theo tên thuốc
  pill_queries/        # Ảnh viên dùng để truy vấn
  pill_index/          # FAISS index và metadata

models/                # Weights/index; không commit file lớn
tests/                 # Unit/API tests
pyproject.toml         # Dependency và cấu hình test
```

`data/` và `models/` chỉ có `.gitkeep` trong Git. Ảnh thật, weights (`.pt`,
`.pth`, `.onnx`) và FAISS index cần được quản lý bằng storage riêng hoặc Git
LFS theo quy định của nhóm.

## 5. Quy ước output giữa các module

YOLO nhận ảnh gốc và trả các detection/crop:

```python
{
    "crop_path": "data/processed/image_001_crop_01.jpg",
    "bounding_box": [100, 80, 300, 400],
    "confidence": 0.95,
}
```

Embedding model nhận ảnh viên thuốc và trả vector. FAISS nhận vector đó và trả
các kết quả gần nhất kèm metadata. VLM
chỉ tổng hợp từ ảnh và các kết quả có nguồn, không tự thay thế dữ liệu model.

## 6. Cách chia việc

### Nhánh data/YOLO

```text
data/raw/
data/annotations/
data/processed/
app/services/detection.py
```

Phụ trách thu thập ảnh, gán bounding box, train YOLO, inference và crop.

### Nhánh few-shot ảnh viên

```text
notebooks/train_pill_embedding_colab.py
data/pill_dataset/
data/pill_queries/
data/pill_index/
```

Phụ trách train embedding model cho ảnh viên, xây FAISS index và ánh xạ kết quả
tới metadata thuốc.

### Các bước tiếp theo

1. Hoàn thiện dataset và YOLO.
2. Train embedding model cho ảnh viên.
3. Xây FAISS index và metadata.
4. Thêm VLM sau khi retrieval đã trả metadata đáng tin cậy.

## 7. Quy trình Git và Pull Request

`main` chỉ chứa code đã review. Mỗi task làm trên một branch riêng:

```text
main
├── feature/data-preparation
├── feature/yolo-detection
├── feature/mobilenet-classification
├── feature/api-pipeline
├── feature/faiss-retrieval
└── feature/vlm-answering
```

Tạo branch mới:

```powershell
git switch main
git pull origin main
git switch -c feature/<ten-task>
```

Sau khi hoàn thành một lát cắt có thể chạy/test được:

```powershell
git add .
git commit -m "feat: <mo-ta-ngan>"
git push -u origin feature/<ten-task>
```

Tạo Pull Request từ branch đó vào `main`. PR nên có:

- Mục đích và các file đã thay đổi
- Input/output của module
- Lệnh test đã chạy
- Phần chưa triển khai hoặc giới hạn hiện tại

Không commit secrets, ảnh người dùng, weights lớn hoặc dữ liệu thật vào
repository. Không code trực tiếp trên `main`.
