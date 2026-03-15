# PBL5_test

## Cài đặt môi trường

Để chạy được dự án, bạn cần cài đặt các thư viện liên quan. Vui lòng mở Terminal (hoặc Command Prompt) tại thư mục gốc của dự án và chạy lệnh sau:

pip install -r requirements.txt

## Hướng dẫn sử dụng bằng lệnh Python

### 1. Huấn luyện mô hình (Train)

Huấn luyện 1 mô hình (thay đổi model, số epochs, ép chạy bỏ qua Early Stopping nếu cần):
python train.py --model lstm --epochs 50 --no-es

Benchmark 3 mô hình cùng lúc (ép chạy bỏ qua Early Stopping nếu cần):
python train_all.py --epochs 30 --no-es

### 2. Đánh giá mô hình (Test)

Đánh giá 1 mô hình đã huấn luyện:
python test.py --model lstm

### 3. Dự đoán (Predict)

**Ghi chú:** Khi dùng dự đoán bằng lệnh python, bạn luôn phải đưa đường dẫn vào sau đối số `--input`.

Quét 1 file và tự động lưu báo cáo mặc định:
python predict.py --input "dataset/Raw Data/SCH/SCH_acc_10_2.txt"

Quét 1 file, lưu báo cáo với tên file chỉ định:
python predict.py --input "dataset/Raw Data/SCH/SCH_acc_10_2.txt" bao_cao_cua_toi.txt

Quét Streaming 1 file, tự động lưu nhật ký mặc định (đuôi json):
python predict.py --input "dataset/Raw Data/SCH/SCH_acc_10_2.txt" --stream

Quét Streaming 1 file, lưu nhật ký thành tên json chỉ định:
python predict.py --input "dataset/Raw Data/SCH/SCH_acc_10_2.txt" --stream nhat_ky_trinh_dien.json

Quét TOÀN BỘ THƯ MỤC và tự động lưu báo cáo mặc định:
python predict.py --input "dataset/Raw Data/SCH"

Quét TOÀN BỘ THƯ MỤC, lưu báo cáo thành tên txt chỉ định:
python predict.py --input "dataset/Raw Data/SCH" bao_cao_tong_ket.txt