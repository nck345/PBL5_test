"""
PBL5 - Fall Detection AI System
Module: architecture.py
Định nghĩa kiến trúc mô hình: Stacked LSTM, 1D CNN, và Ensemble.
"""

import torch
import torch.nn as nn


# ============================================================
# 1. LSTM (Tham khảo từ Kajal)
# ============================================================

class LSTM(nn.Module):
    """
    Mô hình LSTM thuần tuý (tham khảo).
    """
    def __init__(self, input_size: int = 3, hidden_size: int = 64, num_classes: int = 1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(-1)


# ============================================================
# 2. Stacked LSTM (Hiện tại)
# ============================================================

class StackedLSTM(nn.Module):
    """
    Mô hình Stacked LSTM cho phát hiện té ngã.

    Kiến trúc:
        Input (window_size, 3) → LSTM Layer 1 → LSTM Layer 2
        → FC → Dropout → Sigmoid → Output (1)
    """

    def __init__(self, input_size=3, hidden_size=30, num_layers=2, dropout=0.3, bidirectional=False, num_classes=1):
        super(StackedLSTM, self).__init__()
        
        # Mô hình base (thường là 2-layer LSTM)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.num_directions = 2 if bidirectional else 1
        
        # Lớp Fully-Connected
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch_size, window_size, input_size)
        """
        # H_n trả về hidden states tại time_step cuối cho mỗi layer (num_layers*num_dirs, batch, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Chỉ lấy hidden state của layer cuối cùng
        if self.lstm.bidirectional:
            # Gộp output của 2 chiều forward và backward
            out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            out = h_n[-1,:,:]
            
        return self.fc(out).squeeze(-1) # -> (batch_size,)

# ============================================================
# 3. Multi-branch LSTM (Cách 3)
# ============================================================

class MultiBranchLSTM(nn.Module):
    """
    Kiến trúc Đa nhánh (Multi-branch) để xử lý dữ liệu khi có/không có Gyroscope.
    - Nhánh 1: Xử lý Gia tốc (luôn chạy)
    - Nhánh 2: Xử lý Gyroscope (có cổng Gating)
    """
    def __init__(self, hidden_size: int = 30, num_layers: int = 2, 
                 dropout: float = 0.3, bidirectional: bool = False, 
                 num_classes: int = 1):
        super(MultiBranchLSTM, self).__init__()
        
        self.num_directions = 2 if bidirectional else 1
        
        # Nhánh 1: Accelerometer (luôn là 3 kênh)
        self.acc_lstm = nn.LSTM(
            input_size=3, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Nhánh 2: Gyroscope (luôn là 3 kênh)
        self.gyro_lstm = nn.LSTM(
            input_size=3, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Gộp 2 nhánh (Concat)
        fc_input_size = hidden_size * self.num_directions * 2
        
        # Lớp Phân loại
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x có shape: (batch, window_size, 7) nếu dùng đa cảm biến
        """
        if x.shape[2] == 3:
            # Tương thích ngược: Nếu input lỡ vào là 3 kênh thuần túy
            acc_x = x
            has_gyro_mask = torch.zeros(x.shape[0], 1, device=x.device)
            gyro_x = torch.zeros_like(x)
        else:
            acc_x = x[:, :, 0:3]
            gyro_x = x[:, :, 3:6]
            has_gyro_mask = x[:, 0, 6].unsqueeze(-1) # (batch, 1)

        # Xử lý nhánh Acc
        _, (h_n_a, _) = self.acc_lstm(acc_x)
        if self.acc_lstm.bidirectional:
            acc_feat = torch.cat([h_n_a[-2], h_n_a[-1]], dim=1)
        else:
            acc_feat = h_n_a[-1]

        # Xử lý output Gyro
        _, (h_n_g, _) = self.gyro_lstm(gyro_x)
        if self.gyro_lstm.bidirectional:
            gyro_feat = torch.cat([h_n_g[-2], h_n_g[-1]], dim=1)
        else:
            gyro_feat = h_n_g[-1]
            
        # GATING: Xóa tín hiệu nhiễu với những dữ liệu không có Gyro (cờ = 0)
        gyro_feat = gyro_feat * has_gyro_mask

        # Fusion & Classification
        fused = torch.cat([acc_feat, gyro_feat], dim=1)
        out = self.fc(fused)
        
        return out.squeeze(-1)

# ============================================================
# 4. CNN-1D Model (Heterogeneous Mix)
# ============================================================

class CNN1DModel(nn.Module):
    """
    Mô hình CNN 1 chiều để học các mẫu không gian/tần số cao.
    """
    def __init__(self, input_size=3, hidden_size=30, num_layers=2, dropout=0.3, num_classes=1):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, window_size, input_size) -> (batch_size, channels, length) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        # Global Average Pooling theo chiều length (dim=2)
        x = torch.mean(x, dim=2)
        return self.fc(x).squeeze(-1)

# ============================================================
# 5. GRU Model (Heterogeneous Mix)
# ============================================================

class GRUModel(nn.Module):
    """
    Mô hình GRU để chống bão hòa thay cho LSTM, học nhanh hơn.
    """
    def __init__(self, input_size=3, hidden_size=30, num_layers=2, dropout=0.3, bidirectional=False, num_classes=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        if self.gru.bidirectional:
            out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            out = h_n[-1,:,:]
        return self.fc(out).squeeze(-1)

# ============================================================
# 6. Ensemble LSTM (Stacking) / Heterogeneous
# ============================================================

class EnsembleLSTM(nn.Module):
    """
    Mô hình Heterogeneous Ensemble sử dụng kiến trúc Mix.
    Luân phiên giữa CNN, GRU và MultiBranchLSTM để tạo mảng học đa dạng.
    Sử dụng Meta-Classifier ở cuối để tổng hợp (Stacking).
    """
    def __init__(self, num_models=3, input_size=3, hidden_size=30, num_layers=2, dropout=0.3, bidirectional=False, num_classes=1):
        super(EnsembleLSTM, self).__init__()
        self.num_models = num_models
        
        # Tạo danh sách các mô hình cơ sở Đa dạng hóa (Mix)
        # Sử dụng chu kỳ vòng lặp: CNN -> GRU -> LSTM
        self.base_models = nn.ModuleList()
        for i in range(num_models):
            if i % 3 == 0:
                self.base_models.append(CNN1DModel(input_size, hidden_size, num_layers, dropout, num_classes))
            elif i % 3 == 1:
                self.base_models.append(GRUModel(input_size, hidden_size, num_layers, dropout, bidirectional, num_classes))
            else:
                if input_size >= 7:
                    self.base_models.append(MultiBranchLSTM(hidden_size, num_layers, dropout, bidirectional, num_classes))
                else:
                    self.base_models.append(StackedLSTM(input_size, hidden_size, num_layers, dropout, bidirectional, num_classes))
        
        # Meta-classifier: học cách kết hợp kết quả của các model cơ sở
        # Thêm Dropout 0.2 để tránh Overfit do mô hình được training trên 100% data
        meta_input_size = num_models * num_classes
        self.meta_classifier = nn.Sequential(
            nn.Linear(meta_input_size, max(num_models, 4)), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(max(num_models, 4), num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass toàn bộ Ensemble (Dùng lúc Test/Inference).
        """
        base_outputs = []
        for model in self.base_models:
            # Output model(x) dạng (batch,)
            out = model(x) 
            base_outputs.append(out.unsqueeze(1))
            
        # Ghép kết quả: (batch, num_models)
        meta_input = torch.cat(base_outputs, dim=1)
        
        # Phân loại cuối cùng
        meta_out = self.meta_classifier(meta_input)
        return meta_out.squeeze(-1)

# ============================================================
# 5. Factory Function
# ============================================================

def build_model(config: dict) -> nn.Module:
    """
    Tạo model dựa trên cấu hình.

    Args:
        config: Dict cấu hình từ config.yaml
    Returns:
        nn.Module model
    """
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('type', 'stacked_lstm')
    num_classes = model_cfg.get('num_classes', 1)
    window_size = config.get('data', {}).get('window_size', 100)
    
    # Tính toán input_size. NẾU đa cảm biến, dùng 7 kênh (kênh thứ 7 làm Cờ hiệu Gating)
    sensors = config.get('data', {}).get('sensors', ['acc'])
    input_size = 7 if len(sensors) > 1 else 3

    if model_type == 'stacked_lstm':
        lstm_cfg = model_cfg.get('lstm', {})
        if input_size >= 7:
            model = MultiBranchLSTM(
                hidden_size=lstm_cfg.get('hidden_size', 30),
                num_layers=lstm_cfg.get('num_layers', 2),
                dropout=lstm_cfg.get('dropout', 0.3),
                bidirectional=lstm_cfg.get('bidirectional', False),
                num_classes=num_classes
            )
        else:
            model = StackedLSTM(
                input_size=input_size,
                hidden_size=lstm_cfg.get('hidden_size', 30),
                num_layers=lstm_cfg.get('num_layers', 2),
                dropout=lstm_cfg.get('dropout', 0.3),
                bidirectional=lstm_cfg.get('bidirectional', False),
                num_classes=num_classes
            )

    elif model_type == 'lstm':
        model = LSTM(
            input_size=input_size,
            hidden_size=64, # match the 64 in kajal's code
            num_classes=num_classes
        )

    elif model_type == 'ensemble_lstm':
        ensemble_cfg = model_cfg.get('ensemble', {})
        lstm_cfg = model_cfg.get('lstm', {})
        model = EnsembleLSTM(
            num_models=ensemble_cfg.get('num_models', 3),
            input_size=input_size,
            hidden_size=lstm_cfg.get('hidden_size', 30),
            num_layers=lstm_cfg.get('num_layers', 2),
            dropout=lstm_cfg.get('dropout', 0.3),
            bidirectional=lstm_cfg.get('bidirectional', False),
            num_classes=num_classes
        )

    else:
        raise ValueError(f"Loại model không hợp lệ: {model_type}. "
                         f"Chọn: lstm, stacked_lstm, ensemble_lstm")

    return model

