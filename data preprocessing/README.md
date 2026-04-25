# Tiêu Chuẩn Tiền Xử Lý Dữ Liệu (Data Preprocessing Standard)

Tài liệu này định nghĩa cấu trúc dữ liệu đầu ra và quy trình tiền xử lý chung bắt buộc cho mọi tập dữ liệu (như SisFall, MobiAct, v.v.) trong dự án phát hiện té ngã (Fall Detection). Mục tiêu là tối ưu hóa không gian lưu trữ, loại bỏ các file tạm không cần thiết và đảm bảo tất cả các model có thể sử dụng dữ liệu với cùng một định dạng (Form).

## 1. Định Dạng Đầu Ra (Standard Form)

Thư mục đầu ra của mỗi tập dữ liệu (ví dụ: `dataset/sisfall_processed/` hoặc `dataset/MobiAct_Processed/`) **CHỈ ĐƯỢC PHÉP** chứa đúng 6 file sau:

1. **`X_train.npy`**: Dữ liệu huấn luyện CHƯA SCALE, tensor Numpy 3D `[N_train, 128, 9]`.
2. **`y_train.npy`**: Nhãn tương ứng cho tập huấn luyện, mảng 1D `[N_train]`.
3. **`X_val.npy`**: Dữ liệu validation CHƯA SCALE, tensor 3D `[N_val, 128, 9]`.
4. **`y_val.npy`**: Nhãn validation, mảng 1D `[N_val]`.
5. **`X_test.npy`**: Dữ liệu kiểm thử CHƯA SCALE, tensor 3D `[N_test, 128, 9]`.
6. **`y_test.npy`**: Nhãn kiểm thử, mảng 1D `[N_test]`.

> [!WARNING]
> Bắt buộc loại bỏ việc dùng `StandardScaler` tĩnh sinh ra file `scaler.pkl` tại đây. Data được nạp động và gộp lại trong `src/dataset.py`, sau đó hệ thống mới dùng chung 1 `scaler_global.pkl` lúc Train.

**Chi tiết Tensor `X`:**
- **Đơn vị vật lý chuẩn SI (Bắt buộc):** 
  - Gia tốc: $m/s^2$ (Hệ số $g$ phải nhân với $9.80665$).
  - Góc quay: $rad/s$ (Hệ số $°/s$ phải nhân với $\pi/180$).
- **Hệ tọa độ không gian (Bắt buộc):** Trọng trường khi đứng yên phải hướng theo chiều âm của cảm biến, tức là đo được **gia tốc xấp xỉ +9.8 m/s² trên trục Y**. Bất kỳ dataset nào có trục trỏ khác (như SisFall đo ra -9.8) đều **phải được xoay hệ tọa độ** (vd: nhân -1 cho X và Y tương đương xoay 180 độ quanh trục Z) để đồng bộ hoàn toàn với MobiAct.
- **Time steps (Cửa sổ):** Luôn là `128` mẫu (tương đương 2.56 giây ở tần số 50Hz).
- **Channels:** Luôn là **9 Kênh**. Xuất phát từ đúng 2 cảm biến thực tế là **1 Gia Tốc + 1 Con quay hồi chuyển (6 kênh)**, sau đó chạy qua hàm Feature Engineering để sinh thêm 3 kênh: **SMA**, **SMV** và **Jerk**.
- **Dtype:** `float32`.

**Chi tiết Labels `y`:**
- `0`: ADL (Hoạt động sống hàng ngày).
- `1`: FALL (Té ngã).
- **Dtype:** `int32` hoặc `int64`.

---

## 2. Quy Trình Tiền Xử Lý Chuẩn (Standard Pipeline)

Mỗi file script (ví dụ: `process_sisfall.py`) cần tuân thủ thứ tự xử lý sau:

1. **Data Cleaning (Làm sạch):**
   - Đọc dữ liệu raw từ Google Drive / thư mục local.
   - Loại bỏ các dòng chứa NaN hoặc Infinity.
   - Loại bỏ các nhãn dư thừa hoặc không rõ ràng (ví dụ: nhãn 'LYI' trong MobiAct).
2. **Unit Conversion (Đồng bộ đơn vị SI):**
   - Chuyển đổi dữ liệu thô (bits, g, °/s) về đúng $m/s^2$ và $rad/s$.
3. **Low-pass Filtering (Lọc nhiễu):**
   - Sử dụng bộ lọc Butterworth Low-pass (Cutoff = 20Hz, Order = 4).
   - Dùng hàm `scipy.signal.filtfilt` để tránh lệch pha.
4. **Downsampling (Giảm tần số lấy mẫu):**
   - Đưa tất cả dữ liệu về tần số chuẩn **50Hz** (mức lấy mẫu của thiết bị thực tế ESP32).
5. **Sliding Window & Feature Engineering:**
   - Cắt dữ liệu thành các đoạn dài `128` mẫu (Overlap 75% cho FALL, 50% cho ADL).
   - Biến đổi 6 kênh raw thành 9 kênh bằng cách tính thêm SMA, SMV, Jerk.
6. **Train / Validation / Test Split (Chia tập):**
   - Khuyến nghị chia theo **Subject-wise** (Người dùng độc lập) thay vì Random Split.
7. **Export (Lưu trữ):**
   - Lưu thẳng 6 file NumPy `.npy` CHƯA SCALE ra thư mục dataset đích. Đảm bảo dữ liệu giữ nguyên độ lớn vật lý thực để gộp với các tập khác.

> [!TIP]
> Bằng cách tuân thủ hoàn toàn form này, pipeline AI từ bước Training, Inference đến Deployment sẽ hoàn toàn trong suốt, sạch sẽ và tối ưu hóa tối đa tốc độ đọc/ghi đĩa.
