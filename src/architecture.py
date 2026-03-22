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

    def __init__(self, input_size: int = 3, hidden_size: int = 30,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False, num_classes: int = 1):
        """
        Args:
            input_size: Số kênh đầu vào (3 cho x, y, z)
            hidden_size: Số neurons LSTM mỗi tầng
            num_layers: Số tầng LSTM xếp chồng
            dropout: Tỷ lệ dropout
            bidirectional: Sử dụng bidirectional LSTM
            num_classes: Số lớp đầu ra (1 cho binary)
        """
        super(StackedLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Fully connected layers
        fc_input_size = hidden_size * self.num_directions
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
        Forward pass.

        Args:
            x: Input tensor (batch, window_size, input_size)
        Returns:
            Output tensor (batch, num_classes)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Lấy output tại timestep cuối cùng
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            out = h_n[-1]

        # Fully connected
        out = self.fc(out)
        return out.squeeze(-1)



# ============================================================
# 4. Ensemble LSTM (Stacking)
# ============================================================

class EnsembleLSTM(nn.Module):
    """
    Mô hình Stacked Ensemble LSTM dựa theo kỹ thuật Heap Strategy.
    Kết hợp dự đoán của nhiều mô hình StackedLSTM cơ sở qua một Meta-Classifier.
    """
    def __init__(self, num_models=3, input_size=3, hidden_size=30, num_layers=2, dropout=0.3, bidirectional=False, num_classes=1):
        super(EnsembleLSTM, self).__init__()
        self.num_models = num_models
        
        # Tạo danh sách các mô hình cơ sở độc lập
        self.base_models = nn.ModuleList([
            StackedLSTM(input_size, hidden_size, num_layers, dropout, bidirectional, num_classes)
            for _ in range(num_models)
        ])
        
        # Meta-classifier: học cách kết hợp kết quả của các model cơ sở
        # Đầu vào là output probability của N models
        meta_input_size = num_models * num_classes
        self.meta_classifier = nn.Sequential(
            nn.Linear(meta_input_size, max(num_models, 4)), 
            nn.ReLU(),
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
    input_size = model_cfg.get('input_size', 3)
    num_classes = model_cfg.get('num_classes', 1)
    window_size = config.get('data', {}).get('window_size', 100)

    if model_type == 'stacked_lstm':
        lstm_cfg = model_cfg.get('lstm', {})
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

