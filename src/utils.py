"""
PBL5 - Fall Detection AI System
Module: utils.py
Các hàm tiện ích: bộ lọc IIR Low-pass, chuẩn hóa dữ liệu, visualization.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
import torch
import yaml


# ============================================================
# 1. Seed & Device
# ============================================================

def set_seed(seed: int = 42):
    """Thiết lập seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """Tự động chọn device (CPU/GPU)."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


# ============================================================
# 2. Config Loader
# ============================================================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load file cấu hình YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================
# 3. IIR Low-pass Filter (Butterworth)
# ============================================================

def butter_lowpass(cutoff: float, fs: float, order: int = 4):
    """
    Tạo hệ số bộ lọc Butterworth low-pass.

    Args:
        cutoff: Tần số cắt (Hz)
        fs: Tần số lấy mẫu (Hz)
        order: Bậc bộ lọc
    Returns:
        b, a: Hệ số bộ lọc
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass_filter(data: np.ndarray, cutoff: float = 20.0,
                         fs: float = 50.0, order: int = 4) -> np.ndarray:
    """
    Áp dụng bộ lọc IIR Butterworth low-pass lên dữ liệu.

    Args:
        data: Mảng dữ liệu (n_samples, n_channels) hoặc (n_samples,)
        cutoff: Tần số cắt (Hz)
        fs: Tần số lấy mẫu (Hz)
        order: Bậc bộ lọc
    Returns:
        Dữ liệu đã lọc
    """
    b, a = butter_lowpass(cutoff, fs, order)

    if data.ndim == 1:
        # Chỉ lọc nếu đủ dài
        if len(data) <= 3 * max(len(b), len(a)):
            return data
        return filtfilt(b, a, data)
    else:
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):
            col = data[:, i]
            if len(col) <= 3 * max(len(b), len(a)):
                filtered[:, i] = col
            else:
                filtered[:, i] = filtfilt(b, a, col)
        return filtered


# ============================================================
# 4. Chuẩn hóa dữ liệu
# ============================================================

def get_scaler(method: str = "standard"):
    """
    Tạo scaler dựa trên phương pháp chuẩn hóa.

    Args:
        method: "standard" (z-score), "minmax", "none"
    Returns:
        Scaler object hoặc None
    """
    if method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "none":
        return None
    else:
        raise ValueError(f"Phương pháp chuẩn hóa không hợp lệ: {method}")


def normalize_data(data: np.ndarray, scaler=None, fit: bool = True) -> tuple:
    """
    Chuẩn hóa dữ liệu.

    Args:
        data: Dữ liệu cần chuẩn hóa (n_samples, n_features)
        scaler: Scaler object, nếu None thì trả về data gốc
        fit: Nếu True thì fit scaler, nếu False thì chỉ transform
    Returns:
        (data_normalized, scaler)
    """
    if scaler is None:
        return data, None

    if fit:
        data_normalized = scaler.fit_transform(data)
    else:
        data_normalized = scaler.transform(data)

    return data_normalized, scaler


# ============================================================
# 5. Visualization
# ============================================================

def plot_signals(data: np.ndarray, title: str = "Accelerometer Signal",
                 labels: list = None, fs: float = 50.0,
                 save_path: str = None):
    """
    Vẽ biểu đồ tín hiệu gia tốc.

    Args:
        data: Dữ liệu (n_samples, n_channels)
        title: Tiêu đề biểu đồ
        labels: Nhãn cho mỗi kênh
        fs: Tần số lấy mẫu
        save_path: Đường dẫn lưu ảnh (nếu có)
    """
    if labels is None:
        labels = ['X', 'Y', 'Z']

    time = np.arange(len(data)) / fs

    fig, axes = plt.subplots(len(labels), 1, figsize=(12, 3 * len(labels)),
                             sharex=True)
    if len(labels) == 1:
        axes = [axes]

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(time, data[:, i], linewidth=0.5, color=f'C{i}')
        ax.set_ylabel(f'{label} (m/s²)')
        ax.set_title(f'{title} - {label}')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names: list = None,
                          title: str = "Confusion Matrix",
                          save_path: str = None):
    """
    Vẽ confusion matrix.

    Args:
        y_true: Nhãn thực
        y_pred: Nhãn dự đoán
        class_names: Tên các lớp
        title: Tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    if class_names is None:
        class_names = ['ADL', 'Fall']

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_losses: list, val_losses: list,
                         train_accs: list = None, val_accs: list = None,
                         title: str = "Training Curves",
                         save_path: str = None):
    """
    Vẽ biểu đồ loss và accuracy qua các epoch.

    Args:
        train_losses: Loss huấn luyện qua các epoch
        val_losses: Loss validation qua các epoch
        train_accs: Accuracy huấn luyện (tùy chọn)
        val_accs: Accuracy validation (tùy chọn)
        title: Tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    n_plots = 2 if train_accs is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if train_accs is not None:
        axes[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=1.5)
        axes[1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=1.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title(f'{title} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score: float,
                   title: str = "ROC Curve",
                   save_path: str = None):
    """
    Vẽ đường cong ROC.

    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        auc_score: Giá trị AUC
        title: Tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_training_curves(all_histories: dict, 
                                  title: str = "Combined Training Curves",
                                  save_path: str = None):
    """
    Vẽ biểu đồ chung cho nhiều mô hình (Train/Val Loss và Train/Val Accuracy).

    Args:
        all_histories: Dict chứa lịch sử huấn luyện dạng {model_name: history_dict}
        title: Tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']

    for i, (model_name, history) in enumerate(all_histories.items()):
        color = colors[i % len(colors)]
        epochs = range(1, len(history['val_loss']) + 1)
        
        # Plot Loss
        axes[0].plot(epochs, history['train_loss'], f'{color}--', 
                     label=f'{model_name} Train Loss', linewidth=1.5, alpha=0.6)
        axes[0].plot(epochs, history['val_loss'], f'{color}-', 
                     label=f'{model_name} Val Loss', linewidth=2.0)
        
        # Plot Acc
        if 'val_acc' in history and len(history['val_acc']) > 0:
            axes[1].plot(epochs, history['train_acc'], f'{color}--', 
                         label=f'{model_name} Train Acc', linewidth=1.5, alpha=0.6)
            axes[1].plot(epochs, history['val_acc'], f'{color}-', 
                         label=f'{model_name} Val Acc', linewidth=2.0)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend(fontsize='small')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend(fontsize='small')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_roc_curves(all_roc_data: dict, 
                             title: str = "Combined ROC Curves",
                             save_path: str = None):
    """
    Vẽ biểu đồ ROC chung cho nhiều mô hình.

    Args:
        all_roc_data: Dict dạng {model_name: {'fpr': fpr, 'tpr': tpr, 'auc': auc}}
        title: Tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    plt.figure(figsize=(9, 7))
    
    colors = ['darkorange', 'green', 'blue', 'red', 'purple', 'brown']
    
    for i, (model_name, data) in enumerate(all_roc_data.items()):
        color = colors[i % len(colors)]
        plt.plot(data['fpr'], data['tpr'], color=color, lw=2,
                 label=f'{model_name} (AUC = {data["auc"]:.4f})')
                 
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_confusion_matrices(all_metrics: dict, class_names: list = None,
                                     title: str = "Combined Confusion Matrices",
                                     save_path: str = None):
    """
    Vẽ biểu đồ Confusion Matrix chung cho nhiều mô hình trên cùng một mảng (1 x n_models).

    Args:
        all_metrics: Dict dạng {model_name: {'confusion_matrix': cm}}
        class_names: Tên các lớp
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu ảnh
    """
    if class_names is None:
        class_names = ['ADL', 'Fall']

    model_names = list(all_metrics.keys())
    n_models = len(model_names)
    
    if n_models == 0:
        return

    # Tính toán global_vmax để dải màu thống nhất
    all_cm = [all_metrics[m].get('confusion_matrix') for m in model_names if 'confusion_matrix' in all_metrics[m]]
    if not all_cm:
        return
    global_vmax = max(cm.max() for cm in all_cm)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models + 1, 5))
    if n_models == 1:
        axes = [axes]

    for i, m_name in enumerate(model_names):
        cm = all_metrics[m_name].get('confusion_matrix')
        if cm is None:
            continue
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', vmin=0, vmax=global_vmax,
                    xticklabels=class_names, yticklabels=class_names,
                    square=True, linewidths=0.5, ax=axes[i], cbar=(i == n_models - 1))
        axes[i].set_xlabel('Predicted')
        if i == 0:
            axes[i].set_ylabel('Actual')
        axes[i].set_title(f"{m_name.upper()}")

    plt.suptitle(title, fontsize=14, y=1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



# ============================================================
# 6. Miscellaneous
# ============================================================

def ensure_dir(path: str):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def plot_combined_metrics_comparison(all_metrics: dict, 
                                     title: str = "Model Metrics Comparison",
                                     save_path: str = None):
    """
    Vẽ biểu đồ cột so sánh các metrics (Precision, Recall, F1) giữa các mô hình.

    Args:
        all_metrics: Dict dạng {model_name: {'precision': p, 'recall': r, 'f1_score': f1}}
        title: Tiêu đề
        save_path: Đường dẫn lưu ảnh
    """
    model_names = list(all_metrics.keys())
    metrics_names = ['precision', 'recall', 'f1_score']
    
    # Chuẩn bị dữ liệu cho bar chart
    data = []
    for m_name in model_names:
        row = [all_metrics[m_name].get(met, 0) for met in metrics_names]
        data.append(row)
    
    data = np.array(data)
    
    x = np.arange(len(metrics_names))  # vị trí của các nhóm metrics
    width = 0.8 / len(model_names)      # độ rộng của mỗi cột
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, m_name in enumerate(model_names):
        offset = (i - (len(model_names) - 1) / 2) * width
        rects = ax.bar(x + offset, data[i], width, label=m_name)
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=9)

    ax.set_ylabel('Score (0-1)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics_names])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def count_parameters(model) -> int:
    """Đếm số tham số huấn luyện được của model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
