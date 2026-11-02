# Chọn tổ hợp thuốc bằng đồ thị Neo4j + PyKEEN

Chương trình xây hai quan hệ **HỖ TRỢ** và **XUNG ĐỘT** giữa các thuốc trong Neo4j, huấn luyện
Knowledge Graph Embedding bằng PyKEEN, rồi xếp hạng các tổ hợp viên thuốc:

- Mỗi viên thuốc được nhập là **chắc chắn** (1 tên) hoặc **nghi ngờ** (2–3 tên cách nhau bằng `/`).
- Chương trình thử mọi kịch bản có thể, **loại tổ hợp có cặp xung đột đã khai báo**, **cộng điểm thưởng
  cho cặp hỗ trợ đã khai báo**, và in ra **top K** tổ hợp điểm cao nhất.

> ⚠️ Đây là công cụ nghiên cứu. Điểm số từ mô hình embedding **không phải** chỉ định hay tư vấn y tế.

## Cấu trúc

```text
.
├── main.py            # điểm vào: build -> train -> nhập thuốc -> in top K
├── config.py          # cấu hình, đọc thông tin Neo4j từ biến môi trường / .env
├── neo4j_graph.py     # nạp CSV vào Neo4j và đọc lại cặp hỗ trợ / xung đột / triple
├── embedding.py       # train PyKEEN và tính điểm một cặp thuốc
├── scoring.py         # loại xung đột, thưởng hỗ trợ, xếp hạng tổ hợp
├── data/              # đặt support.csv và conflict.csv (xem data/README.md)
├── requirements.txt
├── .env.example
└── .gitignore
```

Phân chia từ notebook gốc:

| Mục trong notebook | Nằm ở file |
|---|---|
| 1. Cài thư viện | `requirements.txt` |
| 3. Cấu hình | `config.py` (+ `.env`) |
| 4. Xây đồ thị Neo4j | `neo4j_graph.py` |
| 4. Train PyKEEN, tính score | `embedding.py` |
| 5. Chấm điểm tổ hợp | `scoring.py` |
| 6–7. Chạy và nhập thuốc | `main.py` |

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env      # rồi điền NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
```

Tạo database miễn phí tại [Neo4j Aura](https://neo4j.com/cloud/aura/) để lấy URI, user và password.

## Chạy

```bash
# Nhập thuốc từ bàn phím
python main.py

# Không tương tác: mỗi viên là một chuỗi, viên nghi ngờ cách nhau bằng '/'
python main.py --drugs "Paracetamol" "Aspirin / Ibuprofen / Naproxen"

# Đổi tham số
python main.py --epochs 200 --top-k 5 --bonus 0.8 --file-ho-tro path/support.csv --file-xung-dot path/conflict.csv
```

Xem tất cả tuỳ chọn: `python main.py --help`.

## Chạy trên Google Colab

```python
!git clone https://github.com/<ban>/<repo>.git
%cd <repo>
!pip install -q -r requirements.txt

import os
from google.colab import drive, userdata        # lưu bí mật trong Colab: biểu tượng chìa khoá ở thanh trái
drive.mount('/content/drive')
for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"):
    os.environ[k] = userdata.get(k)

# Dùng %run (không phải !python) để gõ được thuốc bằng bàn phím
%run main.py --file-ho-tro /content/drive/MyDrive/NCKH/support.csv --file-xung-dot /content/drive/MyDrive/NCKH/conflict.csv
```

## Lưu ý

- **Mỗi lần chạy sẽ xoá toàn bộ node `:Thuoc` trong database đang kết nối** rồi nạp lại từ CSV. Chỉ dùng
  database dành riêng cho dự án này.
- Mô hình PyKEEN được train lại mỗi lần chạy `main.py`. Chế độ tương tác cho phép thử nhiều danh sách mà không phải train lại.
- Thuốc chưa có trong đồ thị sẽ có điểm PyKEEN bằng 0 (chương trình có cảnh báo khi nhập).
- Số kịch bản = tích số lượng tên của từng viên; vượt `TOI_DA_KICH_BAN` (mặc định 5000) thì chương trình từ chối chạy.

## Bảo mật khi đưa lên GitHub

- **Không** ghi URI/password vào code hoặc notebook. Dùng `.env` (đã nằm trong `.gitignore`) hoặc Colab Secrets.
- Nếu từng commit mật khẩu, **hãy đổi mật khẩu ngay** trong Neo4j Aura; xoá file đi vẫn không đủ vì lịch sử Git còn lưu.
- Cân nhắc có nên đưa 2 file CSV dữ liệu lên repo công khai hay không (xem comment trong `.gitignore`).
