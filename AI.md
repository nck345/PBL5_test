# Tài liệu nội bộ dự án - Hệ thống AI phát hiện té ngã cho người cao tuổi (PBL5)

File này đóng vai trò cung cấp cho AI cái nhìn tổng quan về kiến trúc và cấu trúc thư mục của dự án, nhằm hỗ trợ việc đọc hiểu, debug, và mở rộng tính năng thuận lợi hơn.

## 1. Tổng quan dự án

Dự án này tập trung xây dựng một hệ thống IoT thông minh giúp phát hiện và cảnh báo té ngã ở người cao tuổi theo thời gian thực.
Các điểm cốt lõi của dự án:
- **Thiết bị đeo (Wearable)**: Sử dụng cảm biến gia tốc 3 trục để thu thập dữ liệu chuyển động của người dùng.
- **Trí tuệ nhân tạo (AI)**: Áp dụng mô hình học sâu **LSTM** (Long Short-Term Memory) để phân tích dữ liệu và phân loại hành vi té ngã với độ chính xác đạt tới 95.87%.
- **Hạ tầng kết nối**: Sử dụng các giao thức năng lượng thấp (6LowPAN, CoAP) và thiết bị Gateway (Raspberry Pi) để truyền tải dữ liệu một cách hiệu quả và tiết kiệm điện năng.

## 2. Cấu trúc thư mục nền tảng

Cấu trúc thư mục tổng thể của dự án được tổ chức như sau nhằm phân rõ các lớp quản lý gồm: lớp dữ liệu, lớp nhúng (thực thi trên thiết bị), lớp kết nối (Gateway) và lớp triển khai AI.

```text
PBL5_test/
├── dataset/                    # Quản lý dữ liệu (đã được thu thập và cung cấp sẵn)
│   ├── Raw Data/               # Dữ liệu thô ban đầu
│   ├── Annotated Data/         # Dữ liệu đã được gán nhãn
│   └── Readme.txt              # Thông tin về bộ dữ liệu
├── firmware/                   # <--- Cho Wearable Node (NUCLEO-L152RE) [cite: 282]
│   ├── main.c                  # Cấu hình cảm biến LSM6DS0 & SPI [cite: 285]
│   ├── coap-server.c           # Triển khai CoAP Server trên node [cite: 293]
│   └── project-conf.h          # Cấu hình 6LowPAN & IPv6 [cite: 288]
├── gateway/                    # <--- Cho Smart IoT Gateway (Raspberry Pi) [cite: 301]
│   ├── tunslip6.c              # Tạo tunnel IPv6/IPv4 
│   ├── coap_collector.py       # Thu thập dữ liệu qua CoAP GET [cite: 294, 511]
│   └── mqtt_client.py          # Gửi cảnh báo GPS qua MQTT QoS 2 [cite: 497, 500]
├── models/                     # Lưu trữ Model AI
│   ├── checkpoints/            # Các phiên bản trong lúc train
│   └── final_model/            # Model Stacked LSTM/CNN hoàn thiện 
├── notebooks/                  # EDA & Thử nghiệm mô hình (Jupyter)
├── src/                        # Mã nguồn chính của AI
│   ├── __init__.py             # Khởi tạo package
│   ├── architecture.py         # Định nghĩa Stacked LSTM, 1D CNN và Ensemble
│   ├── dataset.py              # Xử lý windowing, overlap 15% & PyTorch DataLoader
│   ├── trainer.py              # Logic huấn luyện mô hình (Trainer class)
│   ├── evaluator.py            # Tính Accuracy, Precision, Recall, F1-Score
│   └── utils.py                # Bộ lọc IIR Low-pass filter, logging, utils
├── configs/
│   └── config.yaml             # Tham số cấu hình model, data, hyperparameters
├── logs/                       # Lưu trữ file báo cáo (predict_reports, test_results)
├── train.py                    # Script chạy huấn luyện mô hình
├── test.py                     # Script chạy đánh giá (Offline) trên tập Test
├── predict.py                  # Script nhận diện (Online/Offline) trên file hoặc stream
├── run.bat                     # Script hỗ trợ chạy lệnh nhanh trên HĐH Windows
├── requirements.txt            # Thư viện yêu cầu: torch, numpy, pandas, scikit-learn, v.v.
└── AI.md                       # File tài liệu mô tả cho AI (File này)
```

## 3. Các Task cần thực hiện

### Lập trình chi tiết các module AI
- [x] **`src/architecture.py`**: Định nghĩa kiến trúc mô hình học sâu (ví dụ: Stacked LSTM / 1D CNN).
- [x] **`src/dataset.py`**: Xây dựng logic tải, tiền xử lý dữ liệu (windowing, overlap định mức 15%), phân chia dữ liệu thành Train/Validation/Test và tạo DataLoader.
- [x] **`src/trainer.py`**: Xây dựng pipeline huấn luyện mô hình (bao gồm vòng lặp Train, đánh giá Validation, Loss, Optimizer) và lưu lại checkpoints.
- [x] **`src/evaluator.py`**: Cài đặt các hàm đánh giá hiệu năng mô hình (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
- [x] **`src/utils.py`**: Triển khai các hàm tiện ích như bộ lọc IIR Low-pass, chuẩn hóa dữ liệu, và visualize kết quả.
- [x] **`configs/config.yaml`**: Trình bày và thiết lập các thông số hyperparameters (e.g. freqs: 50Hz, neurons: 30, window_size).
- [x] **`train.py`**: Script kết nối Datasets, Architecture, Trainer để tiến hành huấn luyện từ dòng lệnh (giám sát metrics của cả Train và Validation).
- [x] **`test.py`**: Script load model đã train để chạy kiểm thử offline trên tập Test.
- [x] **`predict.py`**: Script cho luồng Online Inference (dự đoán thời gian thực) nhận dữ liệu từ Gateway hoặc luồng stream trực tiếp.
