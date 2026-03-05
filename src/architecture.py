"""
PBL5 - Fall Detection AI System
Module: architecture.py
Định nghĩa kiến trúc mô hình: Stacked LSTM, 1D CNN, và Ensemble.
"""

import torch
import torch.nn as nn


# ============================================================
# 1. Stacked LSTM
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
# 2. 1D CNN
# ============================================================

class FallDetectionCNN(nn.Module):
    """
    Mô hình 1D CNN cho phát hiện té ngã.

    Kiến trúc:
        Input (window_size, 3) → Conv1D blocks → Global Average Pooling
        → FC → Sigmoid → Output (1)
    """

    def __init__(self, input_size: int = 3, window_size: int = 100,
                 filters: list = None, kernel_size: int = 3,
                 pool_size: int = 2, dropout: float = 0.3,
                 num_classes: int = 1):
        """
        Args:
            input_size: Số kênh đầu vào (3)
            window_size: Kích thước cửa sổ
            filters: Danh sách số filter cho mỗi lớp Conv1D
            kernel_size: Kích thước kernel
            pool_size: Kích thước pooling
            dropout: Tỷ lệ dropout
            num_classes: Số lớp đầu ra
        """
        super(FallDetectionCNN, self).__init__()

        if filters is None:
            filters = [64, 128]

        layers = []
        in_channels = input_size

        for out_channels in filters:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(filters[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
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
        # Conv1d cần format: (batch, channels, length)
        x = x.permute(0, 2, 1)

        # Convolutional layers
        x = self.conv_layers(x)

        # Global Average Pooling
        x = self.global_pool(x).squeeze(-1)

        # Fully connected
        x = self.fc(x)
        return x.squeeze(-1)


# ============================================================
# 3. Ensemble Model
# ============================================================

class EnsembleModel(nn.Module):
    """
    Mô hình Ensemble kết hợp LSTM và CNN.
    Dự đoán cuối cùng = weighted average của 2 mô hình.
    """

    def __init__(self, lstm_model: StackedLSTM, cnn_model: FallDetectionCNN,
                 lstm_weight: float = 0.6, cnn_weight: float = 0.4):
        """
        Args:
            lstm_model: Mô hình LSTM đã khởi tạo
            cnn_model: Mô hình CNN đã khởi tạo
            lstm_weight: Trọng số cho LSTM
            cnn_weight: Trọng số cho CNN
        """
        super(EnsembleModel, self).__init__()
        self.lstm = lstm_model
        self.cnn = cnn_model
        self.lstm_weight = lstm_weight
        self.cnn_weight = cnn_weight

    def forward(self, x):
        """
        Forward pass.
        """
        lstm_out = self.lstm(x)
        cnn_out = self.cnn(x)
        return self.lstm_weight * lstm_out + self.cnn_weight * cnn_out


# ============================================================
# 4. Factory Function
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

    elif model_type == 'cnn_1d':
        cnn_cfg = model_cfg.get('cnn', {})
        model = FallDetectionCNN(
            input_size=input_size,
            window_size=window_size,
            filters=cnn_cfg.get('filters', [64, 128]),
            kernel_size=cnn_cfg.get('kernel_size', 3),
            pool_size=cnn_cfg.get('pool_size', 2),
            dropout=cnn_cfg.get('dropout', 0.3),
            num_classes=num_classes
        )

    elif model_type == 'ensemble':
        lstm_cfg = model_cfg.get('lstm', {})
        cnn_cfg = model_cfg.get('cnn', {})
        ensemble_cfg = model_cfg.get('ensemble', {})

        lstm_model = StackedLSTM(
            input_size=input_size,
            hidden_size=lstm_cfg.get('hidden_size', 30),
            num_layers=lstm_cfg.get('num_layers', 2),
            dropout=lstm_cfg.get('dropout', 0.3),
            bidirectional=lstm_cfg.get('bidirectional', False),
            num_classes=num_classes
        )
        cnn_model = FallDetectionCNN(
            input_size=input_size,
            window_size=window_size,
            filters=cnn_cfg.get('filters', [64, 128]),
            kernel_size=cnn_cfg.get('kernel_size', 3),
            pool_size=cnn_cfg.get('pool_size', 2),
            dropout=cnn_cfg.get('dropout', 0.3),
            num_classes=num_classes
        )
        model = EnsembleModel(
            lstm_model, cnn_model,
            lstm_weight=ensemble_cfg.get('lstm_weight', 0.6),
            cnn_weight=ensemble_cfg.get('cnn_weight', 0.4)
        )

    else:
        raise ValueError(f"Loại model không hợp lệ: {model_type}. "
                         f"Chọn: stacked_lstm, cnn_1d, ensemble")

    return model
