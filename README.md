# RAG hỏi đáp thông tin thuốc

Hệ thống hỏi đáp tiếng Việt về 48 loại thuốc, dùng **Retrieval-Augmented Generation**:

1. Thông tin mỗi thuốc được cắt thành các đoạn theo mục (thông tin chung, chỉ định, liều dùng, chống chỉ định,
   tác dụng phụ, lưu ý) và đánh chỉ mục bằng embedding `BAAI/bge-m3` + FAISS.
2. Các cặp hỏi-đáp mẫu được đánh chỉ mục theo câu hỏi.
3. Khi có câu hỏi:
   - nếu **khớp một câu hỏi mẫu** (vượt ngưỡng): trả lời bằng câu trả lời mẫu, kèm 2–3 câu bổ sung từ tài liệu;
   - nếu không: mô hình `Qwen2.5-3B-Instruct` viết câu trả lời **chỉ từ tài liệu**, bắt chước giọng văn các câu mẫu,
     kèm nguồn.
   - nếu câu hỏi có nhắc tên thuốc, chỉ tìm trong thông tin của thuốc đó.

> ⚠️ Đây là sản phẩm nghiên cứu, **không thay thế tư vấn của bác sĩ hoặc dược sĩ**. Câu trả lời do mô hình sinh
> có thể sai hoặc thiếu; không dùng để tự quyết định dùng thuốc, đổi liều hay ngừng thuốc.

## Cấu trúc

```text
.
├── main.py          # dòng lệnh: chat | eval-retrieval | eval-answer
├── config.py        # cấu hình mặc định (đường dẫn, mô hình, ngưỡng, top-k)
├── data.py          # đọc CSV, cắt đoạn theo mục, kiểm tra tên thuốc khớp giữa 2 file
├── retriever.py     # embedding + FAISS, lọc theo tên thuốc
├── generator.py     # tải LLM và sinh câu trả lời
├── rag.py           # luồng trả lời (câu mẫu + bổ sung / viết từ tài liệu)
├── evaluate.py      # đo độ chính xác truy xuất và chất lượng câu trả lời
├── data/            # đặt 2 file CSV (xem data/README.md)
├── requirements.txt
└── .gitignore
```

| Cell trong notebook | File |
|---|---|
| 1. Cài thư viện | `requirements.txt` |
| 3. Cấu hình | `config.py` |
| 4. Đọc CSV, cắt đoạn | `data.py` |
| 4. Embedding + FAISS; 7. Hàm tìm kiếm | `retriever.py` |
| 5. Kiểm tra tên thuốc | `data.kiem_tra_ten_khop` (tự chạy khi khởi động) |
| 6. Tải LLM; hàm `sinh` | `generator.py` |
| 7. Hàm `hoi` | `rag.py` |
| 8. Vòng lặp hỏi đáp | `main.py chat` |
| 9–10. Đo độ chính xác | `evaluate.py` (`main.py eval-retrieval`, `eval-answer`) |

## Cài đặt

Yêu cầu Python ≥ 3.9. Nên có GPU (T4 trở lên); CPU vẫn chạy nhưng rất chậm với mô hình LLM.

```bash
pip install -r requirements.txt
```

Lần chạy đầu sẽ tải mô hình từ Hugging Face (bge-m3 và Qwen2.5-3B, tổng vài GB).

## Chạy

```bash
# Hỏi đáp tương tác (gõ 'thoat' để dừng)
python main.py chat

# Hỏi một câu
python main.py chat -q "Liều dùng của Becacold-E là bao nhiêu?"

# Đo top-1 / top-k accuracy của truy xuất (không cần tải LLM, nhanh)
python main.py eval-retrieval

# Đo độ giống ngữ nghĩa giữa câu trả lời của hệ thống và đáp án gốc (lưu results/ket_qua_danh_gia.csv)
python main.py eval-answer --so-cau 10
```

Một số tuỳ chọn: `--file-thuoc`, `--file-qa`, `--embed-model`, `--llm-model`, `--top-k-doan`, `--top-k-qa`,
`--no-qa`, `--no-bo-sung`. Xem tất cả: `python main.py chat --help`.

Dùng mô hình lớn hơn (cần bật nén 4-bit và cài `bitsandbytes`):

```bash
python main.py chat --llm-model Qwen/Qwen2.5-7B-Instruct --load-in-4bit
```

Các ngưỡng khớp câu mẫu (`nguong_co_ten = 0.78`, `nguong_khong_ten = 0.88`), số đoạn bổ sung và độ dài câu trả lời
chỉnh trong `config.py`.

## Chạy trên Google Colab

Chọn **Runtime → Change runtime type → T4 GPU**, rồi:

```python
!git clone https://github.com/<ban>/<repo>.git
%cd <repo>
!pip install -q -r requirements.txt

from google.colab import drive
drive.mount('/content/drive')

# Dùng %run (không phải !python) để gõ được câu hỏi bằng bàn phím
%run main.py chat --file-thuoc /content/drive/MyDrive/NCKH/thuoc_48.csv --file-qa /content/drive/MyDrive/NCKH/bo_tra_loi_cau_hoi.csv
```

## Về hai phép đo

- `eval-retrieval` tắt cả lọc theo tên thuốc lẫn cặp hỏi-đáp mẫu, để đo thuần khả năng tìm kiếm của embedding.
- `eval-answer` tắt cặp hỏi-đáp mẫu để hệ thống không chép lại đáp án. Độ giống ngữ nghĩa cao chưa chắc đồng nghĩa
  với đúng về chuyên môn; nên có người có chuyên môn xem xét lại các câu trả lời.
- Phép đo lấy câu hỏi từ chính file hỏi-đáp mẫu. Nếu có thể, hãy tạo thêm bộ câu hỏi kiểm tra riêng, chưa xuất hiện
  trong bộ mẫu, để số liệu phản ánh đúng khả năng với câu hỏi mới.

## Giấy phép dữ liệu và mô hình

Trước khi công khai repo, hãy kiểm tra quyền sử dụng/đăng tải nội dung trong 2 file CSV (đặc biệt nếu sao chép từ
tờ hướng dẫn sử dụng hoặc tài liệu có bản quyền), và điều khoản giấy phép của các mô hình
`BAAI/bge-m3` và `Qwen/Qwen2.5-3B-Instruct`.
