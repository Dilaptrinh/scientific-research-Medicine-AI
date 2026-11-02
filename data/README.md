# Dữ liệu đầu vào

Đặt 2 file CSV vào thư mục này (hoặc dùng `--file-thuoc` / `--file-qa`, hoặc biến môi trường
`FILE_THUOC` / `FILE_QA`). File nên lưu UTF-8; chương trình tự thử thêm cp1258, cp1252, latin-1.

## `thuoc_48.csv` – thông tin thuốc

Mỗi dòng là một thuốc. Cột bắt buộc: `ten_thuoc`. Các cột còn lại dùng để cắt thành các đoạn;
ô trống hoặc ô bắt đầu bằng `<` (ví dụ `<chưa có>`) được bỏ qua.

| Cột | Đưa vào đoạn |
|---|---|
| `ten_thuoc` | tên thuốc (bắt buộc), ghi ở đầu mỗi đoạn |
| `hoat_chat`, `dang_bao_che` | Thông tin chung |
| `chi_dinh` | Chỉ định |
| `lieu_dung` | Liều dùng |
| `chong_chi_dinh` | Chống chỉ định |
| `tac_dung_phu` | Tác dụng phụ |
| `luu_y` | Lưu ý |
| `nguon` | nguồn tham khảo (in ra khi trả lời) |

## `bo_tra_loi_cau_hoi.csv` – cặp hỏi-đáp mẫu

Cần đủ 3 cột: `ten_thuoc`, `cau_hoi`, `cau_tra_loi`.
Giá trị `ten_thuoc` phải **trùng khít** với tên trong `thuoc_48.csv` (chương trình có kiểm tra khi khởi động).
