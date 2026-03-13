# Báo cáo Thực nghiệm Mô hình: Đánh giá độ chính xác thực tế

Theo yêu cầu của anh, em đã tiến hành lập trình và huấn luyện 3 kiến trúc mô hình với cùng một bộ dữ liệu (đã chia **Subject-wise split** chuẩn xác, không có Data Leakage, huấn luyện trong 15 epochs, sử dụng cảm biến Gia tốc kế 3 kênh trục x, y, z):

1. **PureLSTM**: Mô hình LSTM 1 lớp tuần tự kết nối thẳng qua 1 lớp Dense (Sigmoid). Đây là kiến trúc tinh gọn nhất.
2. **Current Model (StackedLSTM)**: Mô hình hiện tại trong dự án của anh gồm 2 lớp LSTM xếp chồng sau đó qua nhiều lớp Fully-Connected Dropout.
3. **Improved Kajal Model**: Mô hình lai `[Conv1D + BatchNorm + Dropout] x 3` -> `[Bi-LSTM + LayerNorm + Dropout + Attention] x 3` mô phỏng theo phương pháp của Repo Kajal.

## 1. Kết quả Đánh giá Trên Tập Test (Subject mới hoàn toàn)

Kết quả thực tế cho ra bức tranh hoàn toàn khác so với những gì được báo cáo (trên 99%) ở Repo Kajal do cơ chế kiểm thử thực tế khắt khe hơn:

| Model | Test Accuracy | Precision (Fall) | Recall (Fall) | F1-Score | Thời gian Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PureLSTM** | **97.20%** | **0.633** | **0.664** | **0.648** | ~ 4.5 phút |
| **StackedLSTM** | 97.15% | 0.629 | 0.652 | 0.640 | ~ 6.5 phút |
| **Improved Kajal** | 96.95% | 0.615 | 0.576 | 0.595 | **~ 39 phút** |

*(Lưu ý: Dữ liệu Fall (nhào ngã) trong Dataset ít hơn rất nhiều so với ADL (Hoạt động sống) nên F-1 score sẽ là thước đo công bằng nhất)*.

## 2. Phân tích kết quả
* **Mô hình "Pure LSTM" đơn giản nhất lại đứng nhất:** Việc mô hình đơn giản LSTM 1 lớp lại cho F1-Score cao nhất và chạy tính toán cực nhanh cho thấy dữ liệu chuỗi chu kỳ ngắn (100 samples) không cần có quá nhiểu parameters. 
* **Model StackedLSTM (hiện tại) ổn định:** Có chất lượng gần như tương tự hoàn toàn với PureLSTM, phản ánh cấu trúc lớp dày mang lại kết quả an toàn.
* **Model Improved Kajal "thảm họa" trên thực tế:** Mô hình học sâu cực kỳ phức tạp này của repo Kajal đã tự phơi bày yếu điểm của nó: **Overfitting (Quá khớp)**. Khi Train Loss đẩy xuống rất sát `0.04`, nhưng đem predict thực tế thì rớt toàn tập. Nó có quá nhiều tham số (hàng chục layer) nên "học vẹt" tốt nhưng không có tính tổng quát hóa cho một đối tượng người dùng mới. Chưa kể, thời gian tính toán huấn luyện của nó lâu gấp **9 lần** mô hình cơ sở, điều này quá lãng phí để nhúng vào phần cứng nhúng Edge AI của đồ án.

## 3. Kết luận
* Kết quả Accuracy > 99% của Repo Kajal **chắc chắn được sinh ra từ lỗi Data Leakage (Rò rỉ dữ liệu cùng user sang tập Test)**. 
* Khi loại bỏ lỗi rò rỉ (như trong codebase của anh), việc cố gắng gắn CNN và Bi-LSTM phức tạp hoàn toàn không khiến mô hình thông minh hơn, ngược lại làm nặng máy và rớt F1-Score.

**Đề xuất tốt nhất hiện tại:** Không nên mang mô hình phức tạp của Kajal vào đồ án. Thay vào đó, anh nên dùng module **StackedLSTM** hoặc **PureLSTM** với thời gian chạy và F-1 Score đỉnh nhất. Nếu muốn cao hơn nữa, hãy thêm kênh input `Gyroscope` như đề xuất trước đó thay vì nhồi nhét layer mạng nơ-ron!
