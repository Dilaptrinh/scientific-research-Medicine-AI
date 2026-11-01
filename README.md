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
pip install -e ".[retrieval]" # thêm FAISS để tìm kiếm vector
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
    pill_embedding.py   # Encoder few-shot cho ảnh viên thuốc
    retrieval.py       # Adapter FAISS và metadata nguồn
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

Các script đọc ảnh bằng OpenCV, vì vậy embedding model nhận ảnh màu dạng
`numpy.ndarray` theo thứ tự kênh BGR và trả vector chuẩn hóa. FAISS nhận vector đó và trả
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
app/services/pill_embedding.py
app/services/retrieval.py
scripts/build_pill_index.py
scripts/search_pill.py
scripts/run_pill_search.py
tests/test_retrieval.py
data/pill_dataset/
data/pill_queries/
data/pill_index/
```

Phụ trách train embedding model cho ảnh viên, xây FAISS index và ánh xạ kết quả
tới metadata thuốc. `PillReference` là metadata tối thiểu cho mỗi ảnh mẫu;
thông tin hoạt chất, hàm lượng và nguồn chính thức sẽ bổ sung theo
`medicine_id`, không ghi trực tiếp vào FAISS.

Có thể test ngay bằng MobileNetV3 pretrained, chưa cần checkpoint đã fine-tune:

```powershell
pip install -e ".[ml,retrieval,dev]"
python scripts/build_pill_index.py
python scripts/search_pill.py data/pill_queries/query.jpg
```

Hoặc chạy một lệnh duy nhất để tự build index nếu cần và nhập ảnh local hoặc
URL trực tiếp:

```powershell
python scripts/run_pill_search.py "https://example.com/pill.jpg"
python scripts/run_pill_search.py "data/pill_queries/query.jpg"
```

Để đánh giá độ chính xác, đặt ảnh chưa dùng để train theo cấu trúc:

```text
data/pill_queries/
├── Vien_DauBaoTu_Glomoti-M_013/
│   ├── test_01.jpg
│   └── test_02.jpg
└── Vien_DauBaoTu_Kremil-S_014/
    └── test_01.jpg
```

Tên thư mục phải trùng với `medicine_id` trong dataset. Chạy:

```powershell
python scripts/evaluate_pill_retrieval.py --queries data/pill_queries
```

Script sẽ in kết quả từng ảnh và tổng hợp Top-1, Top-3. Các ảnh đánh giá phải
được chụp hoặc tách riêng khỏi ảnh dùng để xây FAISS index; nếu dùng lại ảnh
trong `data/pill_dataset/`, kết quả sẽ bị lạc quan giả tạo.

`build_pill_index.py` đọc đệ quy ảnh từ `data/pill_dataset/<medicine_id>/`
(hoặc `data/pill_dataset/VIEN/<medicine_id>/`), tự tạo
metadata tối thiểu từ tên folder và ghi index vào `data/pill_index/`. Nếu bạn đặt
checkpoint local ở `models/mobilenet/pill_mobilenet_embedding.pt`, script sẽ tự
load file đó mà không cần `--checkpoint`. Lần chạy đầu MobileNetV3 pretrained có
thể cần Internet để tải weights. Khi đã fine-tune embedding model, bạn vẫn có
thể truyền checkpoint bằng `--checkpoint` để override.

### RAG ảnh trên crop đã có (`feature/few-shot-rag-img`)

Luồng RAG không phụ thuộc YOLO: nhận một ảnh crop, tạo embedding, tìm các ảnh
gần nhất bằng FAISS, gom các ảnh theo `medicine_id`, áp dụng ngưỡng tin cậy và
tra cứu knowledge base.

Branch này chỉ xử lý ảnh viên thuốc đã được crop sẵn. YOLO vẫn là một pipeline
độc lập và không được gọi bởi CLI hoặc API RAG crop-only.

Index hiện có được xây từ:

```powershell
python scripts/build_pill_index.py `
  --dataset data/pill_dataset/VIEN `
  --output data/pill_index
```

Tạo knowledge base template từ 47 thuốc trong metadata:

```powershell
python scripts/build_medicine_knowledge_template.py
```

Có thể nhập bệnh và nguồn từ file Excel thống kê thuốc:

```powershell
python scripts/import_medicine_knowledge_xlsx.py `
  "D:\Dowloads\CentOS\Thống kê thuốc.xlsx"
```

File `data/medicine_knowledge/medicines.json` hiện có 47 thuốc, 47 bệnh,
47 URL và 9 mô tả nguồn từ Excel. Thông tin web đã được bổ sung cho 36 thuốc:
36 hoạt chất, 32 hàm lượng và 32 cảnh báo. Một số trường vẫn để trống khi
trang nguồn không nêu rõ; không suy đoán các trường này.

Chạy RAG cho một crop đã được YOLO cắt:

```powershell
python scripts/search_pill_rag.py data/processed/crop_001.jpg
```

Mỗi ảnh query được trả về danh sách thuốc Top-k khác nhau. `--top-k` điều khiển
số candidate, còn `--threshold` điều khiển việc trả `match` hay `unknown`:

```powershell
python scripts/search_pill_rag.py `
  "data/pill_queries/Vien_Gout_Coxileb200_034/Coxileb.png" `
  --top-k 3 --threshold 0.75
```

Mặc định similarity dưới `0.75` trả trạng thái `unknown`. Ngưỡng này chỉ là
giá trị ban đầu và cần điều chỉnh bằng tập query độc lập:

```powershell
python scripts/search_pill_rag.py data/processed/crop_001.jpg `
  --threshold 0.80 --top-k 5
```

API crop-only là `POST /api/v1/rag/predict`; endpoint này không chạy YOLO:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/rag/predict `
  -F "image=@data/processed/crop_001.jpg"
```

Response gồm `status`, `answer`, danh sách candidate đã gom theo thuốc,
similarity cao nhất, similarity trung bình, số ảnh mẫu khớp và nguồn.

Để đo accuracy, mỗi ảnh phải nằm trong thư mục mang đúng `medicine_id`; tên
file ảnh không được dùng làm nhãn:

```text
data/pill_queries/
├── Vien_Gout_Coxileb200_034/
│   └── Coxileb.png
└── Vien_DauRang_HapacolCaplet500_026/
    └── Hapacol.png
```

Chạy đánh giá:

```powershell
python scripts/evaluate_pill_retrieval.py `
  --queries data/pill_queries `
  --top-k 3
```

`Top-1` kiểm tra thuốc đúng có đứng đầu hay không; `Top-3` kiểm tra thuốc đúng
có nằm trong ba thuốc khác nhau đầu tiên hay không. Ảnh đặt trực tiếp ngay dưới
`data/pill_queries/` sẽ bị từ chối vì không có nhãn ground truth.

### Các bước còn cần dữ liệu

1. Tách query khỏi ảnh dùng build index và chạy đánh giá Top-1/Top-3.
2. Điền thông tin y khoa và nguồn chính thức cho từng `medicine_id`.
3. Chọn threshold dựa trên validation set, không dùng mặc định 0.75 như
   một kết luận độ tin cậy.

VLM/LLM chưa cần ở giai đoạn này: template answer hiện tại đã trả kết quả
retrieval có nguồn và cảnh báo, tránh để model tự đoán thông tin thuốc.

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
