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

Mỗi thành viên cần tự tạo `.venv` trên máy của mình. Không commit thư mục này
vào Git; `.gitignore` đã bỏ qua nó. Khi làm việc, kiểm tra interpreter đang dùng:

```powershell
python -c "import sys; print(sys.executable)"
```

Kết quả phải trỏ tới `D:\My_File\code\NCKH\.venv\Scripts\python.exe` (hoặc
đường dẫn project tương ứng trên máy thành viên).

Dependency dùng chung được khai báo trong `pyproject.toml`:

```powershell
pip install -e ".[dev]"       # test và phát triển API
pip install -e ".[ml,dev]"    # thêm YOLO/OpenCV/PyTorch
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
trên ảnh, FPS và số tham số. Khi chạy nhiều seed, `summary_statistics.csv`
thêm trung bình và độ lệch chuẩn cho các chỉ số này. `confusion_matrix.png`
là ma trận của detector
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

Phụ trách fine-tune embedding model cho ảnh viên thuốc, xây FAISS index và ánh
xạ kết quả tới metadata thuốc. Nhánh này không sửa phần MobileNet phân loại
`hop/vi/thuoc` của nhánh khác.

Có thể test ngay bằng MobileNetV3 pretrained, chưa cần checkpoint đã fine-tune:

1. Hoàn thiện dataset và YOLO.
2. Hoàn thiện MobileNet phân loại `hop/vi/thuoc`.
3. Train embedding model cho ảnh viên.
4. Xây FAISS index và metadata.
5. Thêm VLM sau khi retrieval đã trả metadata đáng tin cậy.

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

## 8. Chạy Text RAG và Full RAG local

### Hỏi trực tiếp trong terminal VS Code

Không cần chạy HTTP hoặc mở Swagger:

```powershell
.\.venv\Scripts\python.exe -m app.main
```

Sau đó nhập câu hỏi tại:

```text
Câu hỏi: Becacold-E có thành phần chính là gì?
```

Nhập `thoat` để kết thúc. Index phải được tạo trước bằng
`scripts/build_text_index.py`.

### Chạy API (tùy chọn)

Khởi động API bằng PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Mở <http://127.0.0.1:8000/docs>. Text RAG có thể hỏi trực tiếp bằng
`POST /api/v1/text-rag/query`:

```json
{
  "question": "Becacold-E có thành phần chính là gì?"
}
```

Tạo index Text RAG local một lần (lần đầu model embedding có thể cần tải về):

```powershell
.\.venv\Scripts\python.exe scripts\build_text_index.py
```

Lệnh này đọc `docs/rag-text/thuoc_48.csv`, tạo embedding TF-IDF hoàn toàn
offline và tạo:

```text
data/text_index/text.index
data/text_index/text.json
data/text_index/text_model.json
```

Sau đó API chỉ đọc FAISS local, không gọi Gemini, Hugging Face hay dịch vụ
online nào.

Khi đã có FAISS image index, dùng `POST /api/v1/full-rag/predict` với:

- `image`: ảnh crop viên thuốc;
- `question`: câu hỏi, ví dụ `Thuốc này dùng để điều trị bệnh gì?`.

Endpoint sẽ nhận diện thuốc bằng Image RAG trước, sau đó truyền tên thuốc vào
Text RAG để trả lời. Image index cần được tạo trước bằng
`scripts/build_pill_index.py`; nếu chưa có `data/pill_index/pills.index` và
`pills.json`, endpoint sẽ trả HTTP 503 thay vì giả kết quả.

### Gemini Vision (tùy chọn)

Gemini Vision là luồng online riêng: upload ảnh trực tiếp cho Gemini để nhận
câu trả lời, không thay thế FAISS local. Không ghi API key vào code hoặc Git.
Sau khi revoke key đã lộ, tạo key mới và đặt trong terminal:

```powershell
$env:GEMINI_API_KEY = "KEY_MOI"
```

Chạy hỏi ảnh trong terminal VS Code:

```powershell
.\.venv\Scripts\python.exe -m app.main --gemini
```

Hoặc dùng Swagger với `POST /api/v1/gemini/vision`, gồm hai trường `image` và
`question`. Model mặc định là `gemini-2.0-flash`; có thể đổi bằng
`$env:GEMINI_MODEL`.

`main` chỉ chứa code đã review. Mỗi task làm trên một branch riêng:

```text
main
├── feature/data-preparation
├── feature/yolo
├── feature/mobilenet-classification
├── feature/few-shot-rag-img
└── feature/vlm-answering
```

Quy ước hiện tại:

- `main`: code đã review và tài liệu/quy ước dùng chung; không code trực tiếp.
- `feature/yolo`: YOLO detection và crop, không bao gồm MobileNet phân loại.
- `feature/mobilenet-classification`: MobileNet phân loại `hop`, `vi`, `thuoc`.
- `feature/few-shot-rag-img`: embedding ảnh viên, FAISS và metadata nguồn.
- `feature/vlm-answering`: tổng hợp câu trả lời từ ảnh, retrieval và tài liệu.

Tên branch dùng chữ thường, dấu gạch ngang và mô tả đúng một task. Không dùng
chung một branch cho nhiều task không liên quan. Nếu branch cần code từ branch
khác, ghi rõ dependency trong PR và thống nhất schema giữa hai bên.

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

Trước khi tạo PR, chạy tối thiểu:

```powershell
python -m pytest
git diff --check
git status --short
```

Không commit secrets, ảnh người dùng, weights lớn hoặc dữ liệu thật vào
repository. Không code trực tiếp trên `main`.
