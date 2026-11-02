# Medicine AI Research

Khung project chung cho các nhánh nghiên cứu nhận diện thuốc. `main` không
chứa model, checkpoint, index hay adapter của một nhánh cụ thể.

## Cấu trúc

```text
app/
  api/                    # API contract chung
  core/                   # Hạ tầng dùng chung
  schemas/                # Format request/response chung
  services/
    pipeline.py           # Interface pipeline mở rộng

data/
  raw/                    # Dữ liệu gốc của từng nhánh
  annotations/            # Annotation của từng nhánh
  processed/              # Output trung gian

models/
  mobilenet/              # Chỗ dành cho branch MobileNet

tests/                    # Test chung và test theo branch
pyproject.toml            # Dependency nền tảng
```

Các branch tự thêm thư mục, dependency, code và model cần thiết. Ví dụ:

```text
feature/yolo              -> adapter YOLO và best.pt
feature/few-shot-rag-img -> embedding, FAISS và pill index
feature/rag-text         -> Text RAG và text index
```

## Cài đặt

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Chạy skeleton

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Mở tài liệu API tại <http://127.0.0.1:8000/docs>.

## Quy ước

- Không commit model hoặc index của branch tính năng vào `main`.
- Dùng Git LFS khi branch cần lưu file lớn.
- Không để API giả tạo kết quả khi module/model chưa được nối.
- Kết quả nhận diện thuốc chỉ mang tính dự đoán, không thay thế tư vấn y tế.
