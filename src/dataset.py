"""
PBL5 - Fall Detection AI System
Module: dataset.py
Xử lý tải dữ liệu, tiền xử lý, windowing, và tạo DataLoader.
"""

import os
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import apply_lowpass_filter, get_scaler, normalize_data, load_config


# ============================================================
# 1. Đọc dữ liệu thô MobiAct
# ============================================================

def parse_mobiact_file(filepath: str) -> dict:
    """
    Đọc một file dữ liệu thô MobiAct (.txt).

    File format:
        - Header lines bắt đầu bằng '#'
        - Dòng '@DATA' đánh dấu bắt đầu dữ liệu
        - Dữ liệu: timestamp(ns), x, y, z

    Args:
        filepath: Đường dẫn file .txt
    Returns:
        dict chứa metadata và data (numpy array)
    """
    metadata = {}
    data_lines = []
    data_started = False

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line == '@DATA':
                data_started = True
                continue

            if not data_started:
                # Parse metadata từ header
                if line.startswith('#Activity:'):
                    parts = line.replace('#Activity:', '').strip().split(' - ')
                    if len(parts) >= 2:
                        metadata['activity_label'] = parts[1].strip()
                elif line.startswith('#Subject ID:'):
                    metadata['subject_id'] = int(line.replace('#Subject ID:', '').strip())
                continue

            # Parse data lines
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    values = [float(p.strip()) for p in parts[:4]]
                    data_lines.append(values)
                except ValueError:
                    continue

    if data_lines:
        data = np.array(data_lines)
        metadata['data'] = data[:, 1:]       # x, y, z (bỏ timestamp)
        metadata['timestamps'] = data[:, 0]  # timestamp
    else:
        metadata['data'] = np.array([]).reshape(0, 3)
        metadata['timestamps'] = np.array([])

    # Trích xuất activity code từ tên file
    basename = os.path.basename(filepath)
    parts = basename.replace('.txt', '').split('_')
    if len(parts) >= 1:
        metadata['activity_code'] = parts[0]
    if 'subject_id' not in metadata and len(parts) >= 3:
        try:
            metadata['subject_id'] = int(parts[2])
        except ValueError:
            metadata['subject_id'] = -1

    return metadata











# ============================================================
# 2. Windowing (Sliding Window)
# ============================================================

def create_windows(data: np.ndarray, window_size: int = 100,
                   overlap: float = 0.15) -> np.ndarray:
    """
    Chia dữ liệu thành các cửa sổ trượt (sliding window).

    Args:
        data: Dữ liệu 1 recording (n_samples, n_channels)
        window_size: Kích thước cửa sổ (số samples)
        overlap: Tỷ lệ overlap (0.0 - 1.0), mặc định 15%
    Returns:
        Mảng windows (n_windows, window_size, n_channels)
    """
    if len(data) < window_size:
        return np.array([]).reshape(0, window_size, data.shape[1])

    step = int(window_size * (1 - overlap))
    step = max(step, 1)  # Đảm bảo step >= 1

    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        window = data[start:start + window_size]
        windows.append(window)

    if not windows:
        return np.array([]).reshape(0, window_size, data.shape[1])

    return np.array(windows)


# ============================================================
# 3. Preprocessing Pipeline
# ============================================================



# ============================================================
# 4. Phân chia dữ liệu theo Subject ID
# ============================================================



# ============================================================
# 5. PyTorch Dataset & DataLoader
# ============================================================

class FallDetectionDataset(Dataset):
    """
    PyTorch Dataset cho bài toán phát hiện té ngã.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 scaler=None, fit_scaler: bool = False):
        """
        Args:
            X: Dữ liệu windows (n_windows, window_size, n_channels)
            y: Nhãn (n_windows,)
            scaler: Scaler object cho chuẩn hóa
            fit_scaler: Nếu True, fit scaler trên dữ liệu này
        """
        # Chuẩn hóa: reshape sang 2D → normalize → reshape lại
        n_windows, win_size, n_channels = X.shape

        if scaler is not None:
            X_flat = X.reshape(-1, n_channels)
            if fit_scaler:
                X_flat, scaler = normalize_data(X_flat, scaler, fit=True)
            else:
                X_flat, scaler = normalize_data(X_flat, scaler, fit=False)
            X = X_flat.reshape(n_windows, win_size, n_channels)

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.scaler = scaler

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




# ============================================================
# 6. Convenience function: Full Pipeline
# ============================================================

def prepare_data(config: dict, verbose: bool = True) -> dict:
    """
    Nạp dữ liệu từ nhiều dataset (MobiAct, SisFall, etc.), gộp lại và chuẩn hóa chung.
    """
    import pickle
    from sklearn.preprocessing import StandardScaler

    data_config = config.get('data', {})
    
    # Hỗ trợ cả 'data_dirs' (list) và 'data_dir' (chuỗi) để tương thích ngược
    if 'data_dirs' in data_config:
        data_dirs = data_config['data_dirs']
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
    else:
        data_dirs = [data_config.get('data_dir', 'all')]

    # Tính năng tự động quét tất cả dataset nếu data_dirs có chứa 'all'
    if 'all' in data_dirs:
        import glob
        data_dirs = []
        base_dataset_path = 'dataset'
        if os.path.exists(base_dataset_path):
            for d in os.listdir(base_dataset_path):
                full_path = os.path.join(base_dataset_path, d)
                if os.path.isdir(full_path) and os.path.exists(os.path.join(full_path, 'X_train.npy')):
                    data_dirs.append(full_path)
        if not data_dirs:
            print("⚠️ Cảnh báo: Chế độ 'all' không tìm thấy thư mục dataset hợp lệ nào trong 'dataset/'.")

    if verbose:
        print("=" * 50)
        print("🚀 BƯỚC 1: Tải và Nạp Động dữ liệu (Dynamic Loading)...")
        
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for data_dir in tqdm(data_dirs, desc="Nạp dữ liệu từ các thư mục", disable=not verbose):
        # Load các file chưa scale
        try:
            X_train_list.append(np.load(os.path.join(data_dir, 'X_train.npy')))
            y_train_list.append(np.load(os.path.join(data_dir, 'y_train.npy')))
            X_val_list.append(np.load(os.path.join(data_dir, 'X_val.npy')))
            y_val_list.append(np.load(os.path.join(data_dir, 'y_val.npy')))
            X_test_list.append(np.load(os.path.join(data_dir, 'X_test.npy')))
            y_test_list.append(np.load(os.path.join(data_dir, 'y_test.npy')))
        except FileNotFoundError as e:
            print(f"  ⚠️ Lỗi: Không tìm thấy file trong {data_dir}. {e}")
            continue

    if not X_train_list:
        raise ValueError("Không nạp được dataset nào! Kiểm tra lại config 'data_dirs'.")

    # Gộp toàn bộ dữ liệu
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_val   = np.concatenate(X_val_list, axis=0)
    y_val   = np.concatenate(y_val_list, axis=0)
    X_test  = np.concatenate(X_test_list, axis=0)
    y_test  = np.concatenate(y_test_list, axis=0)

    if verbose:
        print("\n📊 Tổng hợp dữ liệu (Combined):")
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    # Chuẩn hóa chung bằng StandardScaler
    if verbose:
        print("\n⚙️ Đang fit StandardScaler trên toàn bộ tập Train kết hợp...")
    n, T, C = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, C))

    # Save global scaler to models dir
    models_dir = 'models/final_model'
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'scaler_global.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    if verbose:
        print(f"✅ Đã lưu Global Scaler tại: {scaler_path}")

    # Truyền scaler vào Dataset (Dataset sẽ tự gọi scaler.transform())
    train_dataset = FallDetectionDataset(X_train, y_train, scaler=scaler, fit_scaler=False)
    val_dataset   = FallDetectionDataset(X_val, y_val, scaler=scaler, fit_scaler=False)
    test_dataset  = FallDetectionDataset(X_test, y_test, scaler=scaler, fit_scaler=False)

    batch_size = config.get('training', {}).get('batch_size', 64)
    use_sampler = config.get('training', {}).get('use_sampler', True)
    
    if use_sampler:
        class_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
        total_samples = len(y_train)
        class_weights = [total_samples / c if c > 0 else 0 for c in class_counts]
        sample_weights = [class_weights[int(label)] for label in y_train]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=total_samples, replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=False, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
    }



# ============================================================
# 7. Phân chia dữ liệu Ensemble (Dành cho Stacking)
# ============================================================

def get_ensemble_subsets(dataset: Dataset, num_models: int, overlap_ratio: float = 0.15) -> list:
    """
    Chia dataset gốc (train_dataset) thành `num_models` tập con (Subsets).
    Sử dụng chiến lược Bagging (Random Sampling).
    Mỗi model cơ sở sẽ lấy ngẫu nhiên 90% dữ liệu gốc (10% bị drop ngẫu nhiên).
    Việc này giúp các model cơ sở đủ sức mạnh học được quy luật chung giống LSTM thuần,
    nhưng vẫn đảm bảo tính đa dạng (Diversity) để Meta-classifier kết hợp hiệu quả.
    """
    import random
    
    total_len = len(dataset)
    if num_models <= 1:
        return [dataset]
        
    subsets = []
    # Tỉ lệ lấy mẫu: mỗi base model học trên 90% tổng số liệu
    sample_size = int(total_len * 0.9)
    if sample_size == 0:
        sample_size = total_len
        
    all_indices = list(range(total_len))
    
    for i in range(num_models):
        # random.sample lấy ngẫu nhiên KHÔNG hoàn lại từ dataset
        indices = random.sample(all_indices, sample_size)
        subsets.append(Subset(dataset, indices))
        
    return subsets
