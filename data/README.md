# Dữ liệu đầu vào

Đặt 2 file CSV (UTF-8) vào thư mục này, hoặc chỉ đường dẫn khác bằng `--file-ho-tro` / `--file-xung-dot`
(hoặc biến môi trường `FILE_HO_TRO` / `FILE_XUNG_DOT`).

**`support.csv`** – quan hệ HỖ TRỢ, cần đủ 3 cột:

| Tên thuốc | Thuốc hỗ trợ | Cách thức hỗ trợ cụ thể |
|---|---|---|

**`conflict.csv`** – quan hệ XUNG ĐỘT, cần đủ 3 cột:

| Thuốc chính | Thuốc xung đột | Lý do xung đột |
|---|---|---|

Tên cột phải khớp chính xác (có dấu tiếng Việt). Mỗi cặp thuốc chỉ được nạp một lần, không phân biệt thứ tự (A,B và B,A là một).
