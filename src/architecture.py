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
# 3. BiLSTM_CNN (Cải tiến dựa trên Kaggle/Tham khảo)
# ============================================================

class Attention(nn.Module):
    def __init__(self, hidden_size, return_sequences=True):
        super(Attention, self).__init__()
        self.return_sequences = return_sequences
        self.W = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x):
        e = torch.tanh(self.W(x))
        a = torch.softmax(e, dim=1)
        output = x * a
        if self.return_sequences:
            return output
        return torch.sum(output, dim=1)

class BiLSTM_CNN(nn.Module):
    """
    Mô hình lai CNN + BiLSTM + Attention lấy cảm hứng từ repo Kajal.
    """
    def __init__(self, input_size: int = 3, window_size: int = 100, num_classes: int = 1):
        super(BiLSTM_CNN, self).__init__()
        
        # CNN block
        self.conv1 = nn.Conv1d(input_size, 9, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(9)
        self.drop1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(9, 18, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(18)
        self.drop2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv1d(18, 36, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(36)
        self.drop3 = nn.Dropout(0.2)
        
        # Bi-LSTM 1
        self.lstm1 = nn.LSTM(36, 18, bidirectional=True, batch_first=True)
        self.ln1 = nn.LayerNorm(36)
        self.drop_lstm1 = nn.Dropout(0.2)
        self.att1 = Attention(36, return_sequences=True)
        
        # Bi-LSTM 2
        self.lstm2 = nn.LSTM(36, 36, bidirectional=True, batch_first=True)
        self.ln2 = nn.LayerNorm(72)
        self.drop_lstm2 = nn.Dropout(0.2)
        self.att2 = Attention(72, return_sequences=True)
        
        # Bi-LSTM 3
        self.lstm3 = nn.LSTM(72, 72, bidirectional=True, batch_first=True)
        self.ln3 = nn.LayerNorm(144)
        self.drop_lstm3 = nn.Dropout(0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(144, 72),
            nn.ReLU(),
            nn.Linear(72, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, channels, window)
        
        x = self.drop1(self.bn1(torch.relu(self.conv1(x))))
        x = self.drop2(self.bn2(torch.relu(self.conv2(x))))
        x = self.drop3(self.bn3(torch.relu(self.conv3(x))))
        
        x = x.permute(0, 2, 1) # (batch, window, channels)
        
        x, _ = self.lstm1(x)
        x = self.ln1(x)
        x = self.drop_lstm1(x)
        x = self.att1(x)
        
        x, _ = self.lstm2(x)
        x = self.ln2(x)
        x = self.drop_lstm2(x)
        x = self.att2(x)
        
        _, (h_n, _) = self.lstm3(x)
        x = torch.cat([h_n[-2], h_n[-1]], dim=1) # Lấy output của step cuối
        x = self.ln3(x)
        x = self.drop_lstm3(x)
        
        x = self.fc(x)
        return x.squeeze(-1)


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

    elif model_type == 'lstm':
        model = LSTM(
            input_size=input_size,
            hidden_size=64, # match the 64 in kajal's code
            num_classes=num_classes
        )

    elif model_type == 'bilstm_cnn':
        model = BiLSTM_CNN(
            input_size=input_size,
            window_size=window_size,
            num_classes=num_classes
        )

    else:
        raise ValueError(f"Loại model không hợp lệ: {model_type}. "
                         f"Chọn: lstm, stacked_lstm, bilstm_cnn")

    return model
