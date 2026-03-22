"""
PBL5 - Fall Detection AI System
Module: dataset.py
Xử lý tải dữ liệu, tiền xử lý, windowing, và tạo DataLoader.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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


def load_all_data(raw_data_dir: str, sensor_type: str = "acc",
                  fall_labels: list = None, adl_labels: list = None,
                  verbose: bool = True) -> tuple:
    """
    Tải tất cả dữ liệu từ thư mục Raw Data.

    Args:
        raw_data_dir: Đường dẫn thư mục Raw Data
        sensor_type: Loại cảm biến ("acc", "gyro", "ori")
        fall_labels: Danh sách nhãn fall
        adl_labels: Danh sách nhãn ADL
        verbose: In thông tin quá trình tải

    Returns:
        (all_segments, all_labels, all_subjects):
            - all_segments: list of np.ndarray, mỗi phần tử là dữ liệu 1 recording
            - all_labels: list of int (0=ADL, 1=Fall)
            - all_subjects: list of int (subject IDs)
    """
    if fall_labels is None:
        fall_labels = ["FOL", "FKL", "BSC", "SDL"]
    if adl_labels is None:
        adl_labels = ["STD", "WAL", "JOG", "JUM", "STU", "STN",
                       "SCH", "SIT", "CHU", "CSI", "CSO", "LYI"]

    all_segments = []
    all_labels = []
    all_subjects = []

    activity_dirs = sorted([d for d in os.listdir(raw_data_dir)
                           if os.path.isdir(os.path.join(raw_data_dir, d))])

    for activity_dir in activity_dirs:
        activity_code = activity_dir.upper()
        activity_path = os.path.join(raw_data_dir, activity_dir)

        # Xác định label: Fall (1) hoặc ADL (0)
        if activity_code in fall_labels:
            label = 1
        elif activity_code in adl_labels:
            label = 0
        else:
            # Scenario dirs (SLH, SBW, SLW, SBE, SRH) — bỏ qua
            if verbose:
                print(f"  Bỏ qua thư mục scenario: {activity_dir}")
            continue

        # Tìm các file sensor phù hợp
        pattern = os.path.join(activity_path, f"{activity_code}_{sensor_type}_*.txt")
        files = sorted(glob.glob(pattern))

        if not files and verbose:
            print(f"  Không tìm thấy file {sensor_type} cho {activity_code}")
            continue

        for fpath in files:
            result = parse_mobiact_file(fpath)
            data = result.get('data', np.array([]).reshape(0, 3))
            subject_id = result.get('subject_id', -1)

            if data.shape[0] < 10:  # Bỏ qua file quá ngắn
                continue

            all_segments.append(data)
            all_labels.append(label)
            all_subjects.append(subject_id)

    if verbose:
        n_fall = sum(1 for l in all_labels if l == 1)
        n_adl = sum(1 for l in all_labels if l == 0)
        print(f"\nTotal: {len(all_segments)} recordings "
              f"({n_adl} ADL, {n_fall} Fall)")

    return all_segments, all_labels, all_subjects


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

def preprocess_data(segments: list, labels: list, subjects: list,
                    config: dict) -> tuple:
    """
    Pipeline tiền xử lý hoàn chỉnh:
    1. Low-pass filter
    2. Windowing
    3. Gom tất cả windows

    Args:
        segments: Danh sách dữ liệu thô (mỗi phần tử là 1 recording)
        labels: Danh sách nhãn tương ứng
        subjects: Danh sách subject IDs
        config: Dictionary chứa cấu hình

    Returns:
        (X, y, subject_ids):
            - X: (n_total_windows, window_size, n_channels)
            - y: (n_total_windows,)
            - subject_ids: (n_total_windows,)
    """
    preproc_cfg = config.get('preprocessing', {})
    data_cfg = config.get('data', {})

    window_size = data_cfg.get('window_size', 100)
    overlap = data_cfg.get('overlap', 0.15)
    fs = data_cfg.get('sampling_rate', 50)

    # Bộ lọc low-pass
    filter_cfg = preproc_cfg.get('lowpass_filter', {})
    use_filter = filter_cfg.get('enabled', True)
    cutoff = filter_cfg.get('cutoff_freq', 20)
    order = filter_cfg.get('order', 4)

    all_windows = []
    all_window_labels = []
    all_window_subjects = []

    for seg, lbl, subj in zip(segments, labels, subjects):
        # Bước 1: Lọc low-pass
        if use_filter and seg.shape[0] > 20:
            seg_filtered = apply_lowpass_filter(seg, cutoff=cutoff,
                                                fs=fs, order=order)
        else:
            seg_filtered = seg

        # Bước 2: Windowing
        windows = create_windows(seg_filtered, window_size=window_size,
                                 overlap=overlap)

        if windows.shape[0] > 0:
            all_windows.append(windows)
            all_window_labels.extend([lbl] * windows.shape[0])
            all_window_subjects.extend([subj] * windows.shape[0])

    if not all_windows:
        return (np.array([]).reshape(0, window_size, 3),
                np.array([]),
                np.array([]))

    X = np.concatenate(all_windows, axis=0)
    y = np.array(all_window_labels)
    subject_ids = np.array(all_window_subjects)

    return X, y, subject_ids


# ============================================================
# 4. Phân chia dữ liệu theo Subject ID
# ============================================================

def split_by_subject(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
                     train_ratio: float = 0.70, val_ratio: float = 0.15,
                     test_ratio: float = 0.15, seed: int = 42) -> dict:
    """
    Phân chia dữ liệu thành Train/Validation/Test theo subject ID
    để tránh data leakage (cùng 1 subject không xuất hiện ở 2 tập).

    Args:
        X: Dữ liệu windows
        y: Nhãn
        subject_ids: Subject IDs cho mỗi window
        train_ratio, val_ratio, test_ratio: Tỷ lệ phân chia
        seed: Random seed

    Returns:
        dict chứa X_train, y_train, X_val, y_val, X_test, y_test
    """
    unique_subjects = np.unique(subject_ids)
    np.random.seed(seed)
    np.random.shuffle(unique_subjects)

    n_subjects = len(unique_subjects)
    n_train = max(1, int(n_subjects * train_ratio))
    n_val = max(1, int(n_subjects * val_ratio))

    train_subjects = set(unique_subjects[:n_train])
    val_subjects = set(unique_subjects[n_train:n_train + n_val])
    test_subjects = set(unique_subjects[n_train + n_val:])

    # Nếu test_subjects rỗng, lấy từ val
    if not test_subjects:
        test_subjects = val_subjects

    train_mask = np.array([s in train_subjects for s in subject_ids])
    val_mask = np.array([s in val_subjects for s in subject_ids])
    test_mask = np.array([s in test_subjects for s in subject_ids])

    splits = {
        'X_train': X[train_mask], 'y_train': y[train_mask],
        'X_val': X[val_mask], 'y_val': y[val_mask],
        'X_test': X[test_mask], 'y_test': y[test_mask],
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
    }

    return splits


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


def create_dataloaders(splits: dict, config: dict, scaler=None) -> dict:
    """
    Tạo DataLoaders từ dữ liệu đã phân chia.

    Args:
        splits: Dict từ split_by_subject()
        config: Dict cấu hình
        scaler: Scaler object (nếu None sẽ tạo mới)

    Returns:
        dict chứa train_loader, val_loader, test_loader, scaler
    """
    batch_size = config.get('training', {}).get('batch_size', 64)
    norm_method = config.get('preprocessing', {}).get('normalization', 'standard')

    if scaler is None and norm_method != 'none':
        scaler = get_scaler(norm_method)

    # Tạo datasets
    train_dataset = FallDetectionDataset(
        splits['X_train'], splits['y_train'],
        scaler=scaler, fit_scaler=True
    )
    # Dùng scaler đã fit từ train cho val và test
    fitted_scaler = train_dataset.scaler

    val_dataset = FallDetectionDataset(
        splits['X_val'], splits['y_val'],
        scaler=fitted_scaler, fit_scaler=False
    )
    test_dataset = FallDetectionDataset(
        splits['X_test'], splits['y_test'],
        scaler=fitted_scaler, fit_scaler=False
    )

    # Tạo DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False, num_workers=0)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': fitted_scaler,
    }


# ============================================================
# 6. Convenience function: Full Pipeline
# ============================================================

def prepare_data(config: dict, verbose: bool = True) -> dict:
    """
    Pipeline đầy đủ: Load → Preprocess → Split → DataLoader.

    Args:
        config: Dict cấu hình từ config.yaml
        verbose: In thông tin

    Returns:
        dict chứa dataloaders, scaler, splits info
    """
    data_cfg = config['data']

    # Bước 1: Tải dữ liệu
    if verbose:
        print("=" * 50)
        print("BƯỚC 1: Tải dữ liệu MobiAct...")
    segments, labels, subjects = load_all_data(
        raw_data_dir=data_cfg['raw_data_dir'],
        sensor_type=data_cfg.get('sensor_type', 'acc'),
        fall_labels=data_cfg.get('fall_labels'),
        adl_labels=data_cfg.get('adl_labels'),
        verbose=verbose
    )

    if not segments:
        raise ValueError("Không tìm thấy dữ liệu nào! "
                         "Kiểm tra đường dẫn raw_data_dir trong config.")

    # Bước 2: Tiền xử lý + Windowing
    if verbose:
        print("\nBƯỚC 2: Tiền xử lý & Windowing...")
    X, y, subject_ids = preprocess_data(segments, labels, subjects, config)

    if verbose:
        print(f"  Tổng windows: {X.shape[0]}")
        print(f"  Window shape: ({X.shape[1]}, {X.shape[2]})")
        print(f"  Phân bố nhãn: ADL={np.sum(y == 0)}, Fall={np.sum(y == 1)}")

    # Bước 3: Phân chia dữ liệu theo Subject
    if verbose:
        print("\nBƯỚC 3: Phân chia dữ liệu...")
    splits = split_by_subject(
        X, y, subject_ids,
        train_ratio=data_cfg.get('train_ratio', 0.70),
        val_ratio=data_cfg.get('val_ratio', 0.15),
        test_ratio=data_cfg.get('test_ratio', 0.15),
        seed=config.get('seed', 42)
    )

    if verbose:
        print(f"  Train: {splits['X_train'].shape[0]} windows")
        print(f"  Val:   {splits['X_val'].shape[0]} windows")
        print(f"  Test:  {splits['X_test'].shape[0]} windows")

    # Bước 4: Tạo DataLoaders
    if verbose:
        print("\nBƯỚC 4: Tạo DataLoaders...")
    loaders = create_dataloaders(splits, config)

    if verbose:
        print("  ✓ DataLoaders đã sẵn sàng!")
        print("=" * 50)

    return {
        'train_loader': loaders['train_loader'],
        'val_loader': loaders['val_loader'],
        'test_loader': loaders['test_loader'],
        'scaler': loaders['scaler'],
        'splits': splits,
    }
