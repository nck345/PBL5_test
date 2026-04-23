# PBL5_test

## Cài đặt môi trường

Để chạy được dự án, bạn cần cài đặt các thư viện liên quan. Vui lòng mở Terminal (hoặc Command Prompt) tại thư mục gốc của dự án và chạy lệnh sau:

pip install -r requirements.txt

## Kiến trúc các mô hình AI (Models Architecture)

Dự án hiện tại hỗ trợ 3 nhóm mô hình AI chính phục vụ bài toán Phát hiện té ngã (Fall Detection) dựa trên dữ liệu Time-series.

### 1. LSTM Thuần (Vanilla LSTM)
- **Cấu trúc:** Nhận toàn bộ dữ liệu thô (7 kênh báo cáo đa cảm biến) đi thẳng vào 1 lớp LSTM 64 units duy nhất, đi qua 1 lớp đánh giá xác suất. Thiết kế tối giản, không Dropout.
- **Ưu điểm:** Áp dụng giao thoa đặc trưng sớm (Early Fusion). Mô hình tự động học được sự tương tác chéo giữa Gia tốc (Accelerometer) và Cảm biến góc (Gyroscope) ngay từ tín hiệu thô tại mỗi time-step. Hội tụ cực kỳ nhanh, tối ưu cho baseline.
- **Phù hợp:** Chạy trên thiết bị cấu hình nhẹ, phản hồi trơn tru.

### 2. Stacked LSTM (Multi-Branch)
- **Cấu trúc:** Sử dụng kiến trúc Late-Fusion. Tách biệt nhánh dữ liệu Gia tốc và nhánh Góc xoay để chạy độc lập trên 2 cụm mạng LSTM (gồm 30 neurons, 2 tầng xếp chồng, Dropout 0.1). Đối với tập dữ liệu mất cảm biến Gyroscope (như tập archive 3), có cơ chế Gating tự động vô hiệu hóa nhánh nhiễu số 0 trước khi hoà trộn.
- **Ưu điểm:** Tách bạch tính năng của từng Cảm biến và loại trừ xuất sắc tín hiệu hỏng từ các nguồn ghi dữ liệu không hoàn hảo. Chống Overfitting.
- **Phù hợp:** Đòi hỏi học các đặc tính dữ liệu độc lập một cách chắc chắn.

### 3. Ensemble LSTM (Heterogeneous Mix)
- **Cấu trúc:** Cỗ máy tối thượng của dự án, áp dụng kỹ thuật học thích ứng (Stacking & Bagging). 
  - **Base Models (Tầng cơ sở):** Tập hợp liên minh 10 mô hình mạnh mẽ cấu trúc đan xen nhau liên tục đa nền tảng gồm: `1D-CNN` chuyên mổ xẻ đặc tả không gian biên độ sóng, `GRU` tốc độ bắt nhịp cực nhanh và `MultiBranchLSTM` ghi nhớ chuỗi dài hạn. Mỗi thẻ model tự Random Sampling ngẫu nhiên 90% dữ liệu để tăng mức độ đa tạp.
  - **Meta-classifier (Tầng quyết định):** Mạng học sâu thứ cấp đánh giá 10 báo cáo xác suất của tầng Base, có tích hợp kỹ thuật Dropout 0.2 tự chặn Overfitting, từ đó tổng hợp phán quyết chốt hạ siêu chính xác.
- **Ưu điểm:** Triệt tiêu hoàn toàn điểm yếu và thiên kiến của 1 thuật toán đơn vì các mạng Neural đã bù trừ cho nhau. Xử lý xuất sắc các trường hợp "Edge Cases" - tình huống cận biên cực khó phân biệt. Độc cô cầu bại ở tập dữ liệu thực.

## Hướng dẫn sử dụng bằng lệnh Python

### 1. Huấn luyện mô hình (Train)

Huấn luyện 1 mô hình (thay đổi model, số epochs, ép chạy bỏ qua Early Stopping nếu cần):
python train.py --model lstm --epochs 50 --no-es

Benchmark 3 mô hình cùng lúc (ép chạy bỏ qua Early Stopping nếu cần):
python train_all.py --epochs 30 --no-es

**Trong lúc train, nếu muốn early stopping thì nhấn Ctrl + C**

### 2. Đánh giá mô hình (Test)

Đánh giá 1 mô hình đã huấn luyện:
python test.py --model lstm

### 3. Dự đoán (Predict)

**Ghi chú:** Khi dùng dự đoán bằng lệnh python, bạn luôn phải đưa đường dẫn dữ liệu thu từ con chip IOT (định dạng txt, csv) vào sau đối số `--input`. Các tập Raw cũ trên máy đã bị dọn dẹp nên hãy trỏ tới file của riêng bạn nhé!

Quét 1 file và tự động lưu báo cáo mặc định:
python predict.py --input "duong_dan_file_arduino_cua_ban.txt"

Quét 1 file, lưu báo cáo với tên file chỉ định:
python predict.py --input "duong_dan_file_arduino_cua_ban.txt" bao_cao_cua_toi.txt

Quét Streaming 1 file, tự động lưu nhật ký mặc định (đuôi json):
python predict.py --input "duong_dan_file_arduino_cua_ban.txt" --stream

Quét Streaming 1 file, lưu nhật ký thành tên json chỉ định:
python predict.py --input "duong_dan_file_arduino_cua_ban.txt" --stream nhat_ky_trinh_dien.json

Quét TOÀN BỘ THƯ MỤC và tự động lưu báo cáo mặc định:
python predict.py --input "thu_muc_chua_nhieu_file_txt"

Quét TOÀN BỘ THƯ MỤC, lưu báo cáo thành tên txt chỉ định:
python predict.py --input "thu_muc_chua_nhieu_file_txt" bao_cao_tong_ket.txt