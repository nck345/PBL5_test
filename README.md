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
- **Cấu trúc:** Sử dụng kiến trúc Late-Fusion. Tách biệt nhánh dữ liệu Gia tốc và nhánh Góc xoay để chạy độc lập trên 2 cụm mạng LSTM (gồm 64 neurons, 2 tầng xếp chồng, Dropout 0.1). Đối với tập dữ liệu mất cảm biến Gyroscope (như tập archive 3), có cơ chế Gating tự động vô hiệu hóa nhánh nhiễu số 0 trước khi hoà trộn.
- **Ưu điểm:** Tách bạch tính năng của từng Cảm biến và loại trừ xuất sắc tín hiệu hỏng từ các nguồn ghi dữ liệu không hoàn hảo. Chống Overfitting.
- **Phù hợp:** Đòi hỏi học các đặc tính dữ liệu độc lập một cách chắc chắn.

### 3. Ensemble LSTM (Heterogeneous Mix)
- **Cấu trúc:** Cỗ máy tối thượng của dự án, áp dụng kỹ thuật học thích ứng (Stacking & Bagging). 
  - **Base Models (Tầng cơ sở):** Tập hợp liên minh 10 mô hình mạnh mẽ cấu trúc đan xen nhau liên tục đa nền tảng gồm: `1D-CNN` chuyên mổ xẻ đặc tả không gian biên độ sóng, `GRU` tốc độ bắt nhịp cực nhanh và `MultiBranchLSTM` ghi nhớ chuỗi dài hạn. Mỗi thẻ model tự Random Sampling ngẫu nhiên 90% dữ liệu để tăng mức độ đa tạp.
  - **Meta-classifier (Tầng quyết định):** Mạng học sâu thứ cấp đánh giá 10 báo cáo xác suất của tầng Base, có tích hợp kỹ thuật Dropout 0.2 tự chặn Overfitting, từ đó tổng hợp phán quyết chốt hạ siêu chính xác.
- **Ưu điểm:** Triệt tiêu hoàn toàn điểm yếu và thiên kiến của 1 thuật toán đơn vì các mạng Neural đã bù trừ cho nhau. Xử lý xuất sắc các trường hợp "Edge Cases" - tình huống cận biên cực khó phân biệt. Độc cô cầu bại ở tập dữ liệu thực.

## Hướng dẫn sử dụng bằng lệnh Python

### 1. Huấn luyện mô hình (Train)

**Trong lúc train, nếu muốn ngắt giữa chừng (Early Stopping) thì nhấn tổ hợp phím Ctrl + C**

#### 1.1. Huấn luyện từ đầu (Huấn luyện Scratch trên các tập dữ liệu công khai)
Huấn luyện 1 mô hình đơn lẻ:
```bash
python train.py --model lstm --epochs 50 --no-es
```

Huấn luyện benchmark cả 3 mô hình cùng lúc:
```bash
python train_all.py --epochs 30 --no-es
```
*(Lưu ý: Mặc định khi không truyền cờ `--fine_tuning`, hệ thống sẽ tự động lưu các mô hình và bộ chuẩn hóa dữ liệu vào thư mục `models/final_model/scratch`)*

#### 1.2. Huấn luyện chuyển vị / Fine-tuning (Huấn luyện tiếp tục trên tập dữ liệu tự thu ESP32)
Nạp mô hình gốc pre-trained từ thư mục `scratch`, huấn luyện tiếp tục trên tập dữ liệu tự thu ESP32 và lưu kết quả vào thư mục `fine-tuning`:

Fine-tuning 1 mô hình đơn lẻ:
```bash
python train.py --model lstm --fine_tuning --epochs 20
```

Fine-tuning đồng loạt cả 3 mô hình:
```bash
python train_all.py --fine_tuning --epochs 20
```
*(Cờ `--fine_tuning` sẽ tự động thiết lập `--pretrained_dir models/final_model/scratch`, `--final_model_dir models/final_model/fine-tuning` và nạp dữ liệu từ `dataset/esp32_processed`)*

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

---

## Bảng tra cứu các tham số dòng lệnh (Command Line Arguments)

Dưới đây là chi tiết tất cả các tham số có tên (`--`) hỗ trợ trong các script Python để bạn tiện tra cứu khi sử dụng.

### 1. Script Huấn luyện Mô hình (`train.py`)
Dùng để huấn luyện một mô hình cụ thể.

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `configs/config.yaml` | Đường dẫn tới file cấu hình YAML của dự án. |
| `--model` | `str` | `None` | Loại mô hình huấn luyện. Lựa chọn: `lstm`, `stacked_lstm`, `ensemble` (ghi đè config). |
| `--epochs` | `int` | `None` | Số epoch chạy huấn luyện tối đa (ghi đè config). |
| `--batch_size` | `int` | `None` | Kích thước batch khi chia dữ liệu huấn luyện (ghi đè config). |
| `--lr` | `float` | `None` | Tốc độ học (learning rate) ban đầu của optimizer (ghi đè config). |
| `--datasets` | `str` (nhiều) | `None` | Chỉ định cụ thể các thư mục chứa dữ liệu để huấn luyện (ví dụ: `--datasets dataset/esp32_processed`). |
| `--device` | `str` | `None` | Thiết bị chạy huấn luyện. Lựa chọn: `auto`, `cpu`, `cuda`. |
| `--no-es` | *Flag (Boolean)* | `False` | Nếu bật, sẽ tắt Early Stopping và ép mô hình chạy hết số epochs đã định cấu hình. |
| `--final_model_dir` | `str` | `None` | Thư mục lưu file trọng số mô hình cuối cùng và scaler tương ứng (ví dụ: `models/final_model/scratch`). |
| `--pretrained_dir` | `str` | `None` | Thư mục chứa mô hình và scaler pre-trained để nạp trước khi fine-tuning. |
| `--fine_tuning` | *Flag (Boolean)* | `False` | Kích hoạt chế độ fine-tuning: Tự động trỏ `pretrained_dir` về `scratch`, `final_model_dir` về `fine-tuning` và lấy tập dữ liệu từ `dataset/esp32_processed`. |

### 2. Script Huấn luyện Đồng thời / Benchmark (`train_all.py`)
Dùng để huấn luyện tuần tự cả 3 mô hình (`lstm`, `stacked_lstm`, `ensemble`) và xuất biểu đồ so sánh chung.

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `configs/config.yaml` | Đường dẫn tới file cấu hình YAML của dự án. |
| `--epochs` | `int` | `None` | Số epoch chạy huấn luyện tối đa cho mỗi mô hình (ghi đè config). |
| `--batch_size` | `int` | `None` | Kích thước batch khi chia dữ liệu huấn luyện (ghi đè config). |
| `--lr` | `float` | `None` | Tốc độ học (learning rate) ban đầu của optimizer (ghi đè config). |
| `--datasets` | `str` (nhiều) | `None` | Chỉ định cụ thể các thư mục chứa dữ liệu để huấn luyện benchmark. |
| `--device` | `str` | `None` | Thiết bị chạy huấn luyện. Lựa chọn: `auto`, `cpu`, `cuda`. |
| `--no-es` | *Flag (Boolean)* | `False` | Tắt Early Stopping, ép chạy hết số epochs cho tất cả các mô hình. |
| `--final_model_dir` | `str` | `None` | Thư mục lưu file trọng số cuối cùng của 3 mô hình và scaler. |
| `--pretrained_dir` | `str` | `None` | Thư mục chứa 3 mô hình và scaler pre-trained để thực hiện fine-tuning đồng loạt. |
| `--fine_tuning` | *Flag (Boolean)* | `False` | Kích hoạt chế độ fine-tuning cho cả 3 mô hình: Tự động thiết lập đường dẫn pre-trained, đầu ra và dữ liệu tương tự `train.py`. |

### 3. Script Đánh giá Ngoại tuyến (`test.py`)
Dùng để đánh giá hiệu năng của mô hình trên tập kiểm thử (Test set).

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `configs/config.yaml` | Đường dẫn tới file cấu hình YAML. |
| `--model_path` | `str` | `models/final_model/fall_detection_model.pt` | Đường dẫn tới file mô hình đã huấn luyện cần đánh giá. |
| `--device` | `str` | `None` | Thiết bị chạy đánh giá. Lựa chọn: `auto`, `cpu`, `cuda`. |
| `--save_dir` | `str` | `logs/test_results` | Thư mục lưu kết quả và biểu đồ đánh giá (ROC, Confusion Matrix). |
| `--threshold-mode` | `str` | `None` | Chế độ chọn ngưỡng phân loại. Lựa chọn: `fixed` (cố định), `val_calibrated` (tối ưu theo Val set). |
| `--threshold` | `float` | `None` | Ngưỡng phân loại cố định (chỉ có tác dụng khi `--threshold-mode` là `fixed`). |

### 4. Script Dự đoán Thực tế / Mô phỏng dòng dữ liệu (`predict.py`)
Dùng để quét file dữ liệu cảm biến thô để dự đoán té ngã thời gian thực hoặc dự đoán theo lô.

| Tham số | Kiểu dữ liệu | Mặc định | Mô tả |
| :--- | :--- | :--- | :--- |
| `--config` | `str` | `configs/config.yaml` | Đường dẫn tới file cấu hình YAML. |
| `--model` | `str` | `stacked_lstm` | Loại mô hình dùng để dự đoán. Lựa chọn: `lstm`, `stacked_lstm`, `ensemble`, `fall_detection_model`. |
| `--input` | `str` | *(Bắt buộc)* | Đường dẫn tới file cảm biến thô hoặc thư mục chứa nhiều file cần quét. |
| `--stream` | *Flag (Boolean)* | `False` | Nếu bật, sẽ kích hoạt chế độ mô phỏng luồng dữ liệu thời gian thực (in kết quả sau mỗi cửa sổ trượt). |
| `--threshold` | `float` | `0.5` | Ngưỡng xác suất phân loại té ngã (từ 0.0 đến 1.0). |
| `--device` | `str` | `None` | Thiết bị chạy dự đoán. Lựa chọn: `auto`, `cpu`, `cuda`. |
| `--save_report` | `str` | `None` | Đường dẫn cụ thể để xuất file báo cáo dự đoán. |
| `--final_model_dir` | `str` | `None` | Chỉ định thư mục chứa cặp model + scaler tương ứng cần dùng (ví dụ: `models/final_model/fine-tuning`). |