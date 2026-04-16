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

## Tiêu chuẩn Dữ liệu (Standardized Dataset Form)

Để mô hình AI hoạt động trơn tru, đồng thời đạt độ chính xác cao nhất và tiết kiệm pin nhất trên các bộ chip mạch nhúng (Edge AI, IoT) trong tương lai, toàn bộ dữ liệu dự kiến đi vào dự án (kể cả tự thu thập thêm) **BẮT BUỘC** phải tuân theo hệ quy chiếu vật lý và toán học sau:

### 1. Cấu trúc Cảm biến (Sensor Channels)
Mạng Nơ-ron đã được lập trình để tiêu thụ mảng **6 kênh tín hiệu** liên tục, sắp xếp theo bề ngang đúng thứ tự sau:
1. Nhánh 1: `Acc_X`, `Acc_Y`, `Acc_Z` (Ký hiệu gia tốc kế 3 trục)
2. Nhánh 2: `Gyro_X`, `Gyro_Y`, `Gyro_Z` (Ký hiệu con quay hồi chuyển 3 trục)

*Lưu ý:* Các kênh cảm biến phụ khác (như Phương vị góc Orient, lực Từ kế...) phải được lọc bỏ đi tại vi mạch gốc để tránh làm vỡ ma trận chiều của Input Layer.

### 2. Thông số Trục Thời gian (Temporal Setup)
- **Tần số lấy mẫu (Sampling Rate):** `50Hz`. Đây là băng tần vàng tối ưu nhất cho Fall Detection. Vừa dư sức để bắt trọn 100% các xung lực va đập chớp nhoáng khi ngã, vừa giúp tiết kiệm gấp 4 lần lượng điện năng, bộ nhớ RAM và băng thông truyền tải IOT so với mức xung 200Hz.
- **Kích thước Cửa sổ (Window Size / Timesteps):** `100 timesteps`. Ở mức xung 50Hz, 100 mốc thời gian sẽ bằng cực kì chuẩn xác `2.0 giây`. Giai đoạn té ngã vật lý thường chỉ diễn ra chưa tới 1 giây.
- **Hệ quả:** Mỗi mẫu học (Data Sample) đưa vào mạng Nơ-ron sẽ được đúc vĩnh viễn thành một khối Rubik có hình dạng: `[100, 6]`.

### 3. Đơn vị Đo lường Chuẩn (Unified Physical Units)
Để việc chuẩn hóa (Z-Score) và tính toán khoảng cách vector không bị sai lệch, nghiêm cấm trộn lẫn các hệ đo lường có tầm phủ (Scale) khác nhau. Đơn vị chuẩn được thống nhất cho toàn dự án:
- **Gia tốc kế (Accelerometer):** Thu nhận bằng hệ `g` (Gia tốc trọng trường, với $1g \approx 9.81 \text{m/s}^2$). Không được nạp trực tiếp giá trị theo $\text{m/s}^2$. 
- **Con quay hồi chuyển (Gyroscope):** Thu nhận bằng hệ `Độ/giây (deg/s)`. Tuyệt đối không nạp hệ `Radian/s`.

*(Chỉ cần áp chặt dữ liệu vi mạch Arduino vào "khuôn đúc chuẩn" ở trên, bạn có thể tự do gộp chung tập dữ liệu tự thu của thiết bị vào cùng túi chung với Sisfall mà mô hình sẽ hoàn toàn không bị "tẩu hỏa nhập ma"!*