import os
import glob
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split

# Reconfigure stdout to use UTF-8 to prevent encoding errors on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_PATH = os.path.join(PROJECT_ROOT, 'raw data', 'esp32')
SAVE_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'esp32_processed')

FS = 50       # sampling frequency (already 50Hz)
WINDOW = 128   # window size
STEP_FALL = 32 # overlap 75%
STEP_ADL = 64  # overlap 50%

os.makedirs(SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def clean_dataframe(df):
    feature_cols = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
    df = df.dropna(subset=feature_cols)
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)

def butterworth_filter(data, cutoff=20, fs=50, order=4):
    if len(data) <= order * 3:
        # Avoid SciPy padlen errors for very short files
        return data
    nyq = 0.5 * fs
    norm = cutoff / nyq
    b, a = scipy_signal.butter(order, norm, btype='low')
    return scipy_signal.filtfilt(b, a, data, axis=0)

def create_windows_and_labels(df, trial_id, is_fall, impact_ms, step):
    feature_cols = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
    data = df[feature_cols].values
    timestamps = df['elapsed_ms'].values
    
    X_list, y_list = [], []
    for s in range(0, len(data) - WINDOW + 1, step):
        window = data[s : s+WINDOW]
        start_ms = timestamps[s]
        end_ms = timestamps[s + WINDOW - 1]
        
        # Gán nhãn cho từng cửa sổ
        if is_fall:
            # Chỉ gán nhãn 1 nếu cửa sổ chứa hoặc rất gần điểm va chạm (impact_ms)
            # Dung sai 100ms ở 2 đầu
            if (start_ms - 100) <= impact_ms <= (end_ms + 100):
                label = 1
            else:
                label = 0
        else:
            label = 0
            
        X_list.append(window)
        y_list.append(label)
        
    if not X_list:
        return np.array([]).reshape(0, WINDOW, len(feature_cols)), np.array([])
    return np.array(X_list), np.array(y_list)

def feature_engineering(window):
    """
    Thêm 3 kênh đặc trưng: SMA, SMV, Jerk (Tổng cộng 9 kênh)
    Input: (WINDOW, 6) -> Output: (WINDOW, 9)
    """
    acc  = window[:, :3]
    gyro = window[:, 3:]
    sma  = np.sqrt(np.sum(acc**2, axis=1, keepdims=True))
    smv  = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
    jerk = np.diff(sma[:, 0], prepend=sma[0, 0]).reshape(-1, 1)
    return np.concatenate([window, sma, smv, jerk], axis=1)

def process_file(filepath, impact_lookup):
    try:
        df = pd.read_csv(filepath)
        df = clean_dataframe(df)
        if len(df) < WINDOW:
            return np.array([]), np.array([]), None
            
        filename = os.path.basename(filepath)
        filename_no_ext = os.path.splitext(filename)[0]
        subject_id = os.path.basename(os.path.dirname(filepath))
        trial_id = f"{subject_id}_{filename_no_ext}"
        
        # Xác định có phải ngã không và bước trượt
        activity = df['activity'].iloc[0]
        is_fall = 1 if activity.startswith('FALL') else 0
        step = STEP_FALL if is_fall else STEP_ADL
        
        # Tra cứu impact_ms từ manifest
        impact_ms = impact_lookup.get(trial_id, None)
        if is_fall and impact_ms is None:
            # Fallback nếu không có trong manifest: lấy thời điểm độ lớn gia tốc lớn nhất
            acc_mag = np.sqrt(df['accX']**2 + df['accY']**2 + df['accZ']**2)
            max_idx = acc_mag.idxmax()
            impact_ms = float(df['elapsed_ms'].iloc[max_idx])
            
        # Áp dụng bộ lọc Butterworth Low-pass 20Hz
        feature_cols = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']
        filtered_data = butterworth_filter(df[feature_cols].values, cutoff=20, fs=FS)
        df[feature_cols] = filtered_data
        
        # Tạo các cửa sổ và nhãn tương ứng
        X_w, y_w = create_windows_and_labels(df, trial_id, is_fall, impact_ms, step)
        
        if len(X_w) == 0:
            return np.array([]), np.array([]), None
            
        # Trích xuất đặc trưng (6 kênh -> 9 kênh)
        X_eng = np.array([feature_engineering(w) for w in X_w])
        
        return X_eng, y_w, activity
    except Exception as e:
        print(f"Lỗi khi xử lý file {filepath}: {e}")
        return np.array([]), np.array([]), None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Scanning raw ESP32 data folder at: {BASE_PATH}")
    manifest_path = os.path.join(BASE_PATH, 'manifest.csv')
    if not os.path.exists(manifest_path):
        print(f"Error: manifest.csv not found at {manifest_path}!")
        exit(1)
        
    manifest_df = pd.read_csv(manifest_path)
    impact_lookup = {}
    for _, row in manifest_df.iterrows():
        t_id = row['trial_id']
        imp_ms = row['impact_ms']
        if pd.notna(imp_ms):
            impact_lookup[t_id] = float(imp_ms)
            
    all_files = glob.glob(os.path.join(BASE_PATH, '**', '*.csv'), recursive=True)
    all_files = [f for f in all_files if os.path.basename(f) != 'manifest.csv']
    print(f"Found {len(all_files)} raw CSV files to process.")
    
    trials_data = []
    
    for fp in all_files:
        X_w, y_w, activity = process_file(fp, impact_lookup)
        if len(X_w) > 0:
            filename = os.path.basename(fp)
            filename_no_ext = os.path.splitext(filename)[0]
            subject_id = os.path.basename(os.path.dirname(fp))
            trial_id = f"{subject_id}_{filename_no_ext}"
            
            trials_data.append({
                'trial_id': trial_id,
                'activity': activity,
                'X': X_w,
                'y': y_w
            })
            
    if not trials_data:
        print("Error: No trials processed successfully!")
        exit(1)
        
    print(f"Extracted data for {len(trials_data)} trials successfully.")
    
    # Prepare labels for stratified trial-wise split
    trial_labels = [t['activity'] for t in trials_data]
    stratify_labels = ['FALL' if act.startswith('FALL') else 'ADL' for act in trial_labels]
    
    # Trial-wise Split
    # 70% Train, 30% Temporary
    train_idx, temp_idx = train_test_split(
        list(range(len(trials_data))),
        test_size=0.30,
        random_state=42,
        stratify=stratify_labels
    )
    
    # 15% Validation, 15% Test
    temp_stratify = [stratify_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=42,
        stratify=temp_stratify
    )
    
    # Concatenate windows
    X_train = np.concatenate([trials_data[i]['X'] for i in train_idx], axis=0)
    y_train = np.concatenate([trials_data[i]['y'] for i in train_idx], axis=0)
    
    X_val = np.concatenate([trials_data[i]['X'] for i in val_idx], axis=0)
    y_val = np.concatenate([trials_data[i]['y'] for i in val_idx], axis=0)
    
    X_test = np.concatenate([trials_data[i]['X'] for i in test_idx], axis=0)
    y_test = np.concatenate([trials_data[i]['y'] for i in test_idx], axis=0)
    
    print("\nDataset split sizes (Trial-wise):")
    print(f"  Train: X={X_train.shape}, y={y_train.shape} (Fall ratio: {np.mean(y_train)*100:.2f}%)")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape} (Fall ratio: {np.mean(y_val)*100:.2f}%)")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape} (Fall ratio: {np.mean(y_test)*100:.2f}%)")
    
    # Save files
    print(f"\nSaving preprocessed dataset to: {SAVE_PATH}")
    np.save(os.path.join(SAVE_PATH, 'X_train.npy'), X_train)
    np.save(os.path.join(SAVE_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(SAVE_PATH, 'X_val.npy'), X_val)
    np.save(os.path.join(SAVE_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(SAVE_PATH, 'X_test.npy'), X_test)
    np.save(os.path.join(SAVE_PATH, 'y_test.npy'), y_test)
    
    print("ESP32 data preprocessing complete!")
