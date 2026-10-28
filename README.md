# Medicine AI Research

Backend skeleton cho nghiên cứu nhận diện nhiều loại thuốc trong một ảnh:

```text
Ảnh thuốc -> YOLO detection/crop -> MobileNet classification
           -> few-shot embedding -> FAISS retrieval -> VLM trả lời
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

Ở skeleton ban đầu, endpoint dự đoán trả `501 Not Implemented` vì chưa tích hợp
model thật. Branch tích hợp bên dưới chuyển endpoint sang pipeline thật, nhưng
chỉ chạy được sau khi có đủ hai weights tương thích.

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

### 3.1.1. Huấn luyện YOLO một lớp và báo cáo thực nghiệm

Trong bài toán này, YOLO chỉ có một lớp `object` để tìm vùng hộp, vỉ hoặc
viên thuốc. YOLO không phân loại ba loại này; bước phân loại crop là một module
khác và không được đưa vào các chỉ số của detector.

Tạo `data/yolo/data.yaml` với `names: [object]`, sau đó chạy:

```powershell
python scripts/train_yolo.py `
  --data data/yolo/data.yaml `
  --model yolov9s.pt `
  --epochs 50 `
  --imgsz 640 `
  --batch 16 `
  --seeds 0
```

Có thể chạy nhiều seed để báo cáo độ ổn định:

```powershell
python scripts/train_yolo.py --data data/yolo/data.yaml --seeds 0,1,2
```

Mỗi run sinh `metrics.json`, `metrics.csv`, `hyperparameters.csv`, trọng số
`best.pt`, cùng các biểu đồ của Ultralytics trong `runs/yolo/`. Bảng metrics
gồm mAP@50, mAP@75, mAP@[50:95], thời gian huấn luyện, thời gian suy luận
trên ảnh, FPS và số tham số. `confusion_matrix.png` là ma trận của detector
giữa `object` và `background`; ô `object → object` là dự đoán đúng, không phải
ma trận phân loại `hop/vi/thuoc`.

## 3.2. Branch YOLO + MobileNet phân loại loại vật thể

Branch triển khai phần này là:

```text
feature/yolo-mobilenet-type-classification
```

Luồng của branch:

```text
POST /api/v1/predict
    -> YOLO phát hiện từng vùng
    -> lưu crop
    -> MobileNetV3 phân loại mỗi crop: hop, vi hoặc thuoc
    -> trả bounding box + hai confidence
```

Code nằm ở:

```text
app/services/detection.py       # YOLO
app/services/classification.py  # MobileNetV3 3 lớp
app/services/pipeline.py        # nối hai model
app/api/routes.py               # endpoint upload ảnh
```

MobileNet hiện có trong thư mục cũ (`mobilenet_thuoc_best*.pth`) là model phân
loại **tên thuốc**, không phải model 3 lớp `hop/vi/thuoc`, nên không được đổi
tên rồi dùng nhầm vào branch này. Cần đặt checkpoint đã train đúng 3 lớp tại:

```text
models/mobilenet/type_classifier.pth
```

Endpoint trả dạng:

```json
{
  "image_id": "generated-id",
  "detections": [
    {
      "label": "hop",
      "confidence": 0.92,
      "bounding_box": [100, 80, 350, 500],
      "crop_path": "data/processed/generated-id_crop_0.jpg",
      "classification": {
        "label": "hop",
        "confidence": 0.96
      }
    }
  ]
}
```

Nếu chưa có `type_classifier.pth`, endpoint sẽ không được xem là đã hoàn
thiện; hãy train model 3 lớp trước khi tạo PR merge vào `main`.

## 4. Cấu trúc thư mục

```text
app/
  api/
    routes.py          # Nhận request và trả response HTTP
  core/                # Cấu hình, logging và hạ tầng dùng chung
  schemas/
    prediction.py      # Format input/output chuẩn
  services/
    pipeline.py        # Interface điều phối pipeline
    detection.py       # Sẽ chứa adapter YOLO
    classification.py  # Sẽ chứa adapter MobileNet
    retrieval.py       # Sẽ chứa adapter FAISS
    vlm.py             # Sẽ chứa adapter VLM
  main.py              # Khởi tạo FastAPI

data/
  raw/                 # Ảnh gốc; không commit ảnh thật
  processed/           # Ảnh crop và kết quả tiền xử lý
  annotations/         # Nhãn YOLO và metadata

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

MobileNet nhận một crop và trả:

```python
{
    "label": "paracetamol",
    "confidence": 0.91,
}
```

FAISS nhận embedding của crop và trả các kết quả gần nhất kèm metadata. VLM
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

### Nhánh MobileNet/API

```text
app/services/classification.py
app/services/pipeline.py
app/api/routes.py
tests/
```

Phụ trách phân loại crop, nối detector với classifier và kiểm thử API. Hai bên
chỉ cần thống nhất schema/output, hạn chế sửa cùng một file.

### Các bước tiếp theo

1. Hoàn thiện dataset và YOLO.
2. Hoàn thiện MobileNet trên một crop.
3. Nối YOLO và MobileNet trong `pipeline.py`.
4. Thêm embedding few-shot và FAISS.
5. Thêm VLM sau khi retrieval đã trả metadata đáng tin cậy.

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
