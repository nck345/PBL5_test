import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from io import StringIO

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = '../raw data/SisFall'  # Path to raw SisFall data
SAVE_PATH = '../dataset/sisfall_processed'

FS_ORIG   = 200   # Original Hz
FS_TARGET = 50    # Target Hz
DS_FACTOR = 4     # 200/50
WINDOW    = 128   # 2.56 seconds
STEP_FALL = 32    # overlap 75%
STEP_ADL  = 64    # overlap 50%

# Conversion factors
ADXL345_SCALE  = 32.0 / 8192.0     # 0.00390625
ITG3200_SCALE  = 4000.0 / 65536.0  # 0.0610352
MMA8451Q_SCALE = 16.0 / 16384.0    # 0.000977

os.makedirs(SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def load_sisfall_file(filepath):
    """Load text file and convert to numpy array"""
    with open(filepath, 'r') as f:
        cleaned = [line.strip().rstrip(';') for line in f if line.strip()]
    data = np.loadtxt(StringIO('\n'.join(cleaned)), delimiter=',')
    return data.astype(np.float32)

def convert_units_and_extract_6_channels(samples):
    """
    Extracts ADXL345 (accel) and ITG3200 (gyro)
    Converts ADXL345 to m/s^2.
    Converts ITG3200 to rad/s.
    
    Coordinate Alignment:
    SisFall standing Y-axis gravity is ~ -9.8 m/s^2.
    MobiAct standing Y-axis gravity is ~ +9.8 m/s^2.
    Apply a 180-degree rotation around the Z-axis to align them:
    acc_x -> -acc_x, acc_y -> -acc_y
    gyro_x -> -gyro_x, gyro_y -> -gyro_y
    """
    s = samples.copy()
    
    # Apply scales
    acc = s[:, 0:3] * ADXL345_SCALE * 9.80665       # ADXL345 (m/s^2)
    gyro = s[:, 3:6] * ITG3200_SCALE * (np.pi / 180.0) # ITG3200 (rad/s)
    
    # 180-deg rotation around Z-axis to align with MobiAct
    acc[:, 0] = -acc[:, 0]
    acc[:, 1] = -acc[:, 1]
    gyro[:, 0] = -gyro[:, 0]
    gyro[:, 1] = -gyro[:, 1]
    
    s[:, 0:3] = acc
    s[:, 3:6] = gyro
    return s[:, 0:6]  # Chỉ trả về 6 cột đầu tiên (Gia tốc 1 + Gyro)

def clean_data(data):
    """Basic cleaning, returning None if invalid"""
    if np.isnan(data).any() or np.isinf(data).any():
        return None
    return data

def butterworth_filter(data, cutoff=20, fs=200, order=4):
    nyq  = 0.5 * fs
    norm = cutoff / nyq
    b, a = scipy_signal.butter(order, norm, btype='low')
    return scipy_signal.filtfilt(b, a, data, axis=0)

def downsample(data, factor=4):
    return data[::factor, :]

def create_windows(data, label, step, threshold=0.05):
    X_list, y_list = [], []
    for s in range(0, len(data) - WINDOW + 1, step):
        window = data[s : s+WINDOW]
        X_list.append(window)
        y_list.append(label)
    if not X_list:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)

def feature_engineering(window):
    """
    Giống hệ thống MobiAct: Thêm 3 kênh SMA, SMV, Jerk
    Input: (128, 6) -> Output: (128, 9)
    """
    acc  = window[:, :3]
    gyro = window[:, 3:]
    sma  = np.sqrt(np.sum(acc**2, axis=1, keepdims=True))
    smv  = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
    jerk = np.diff(sma[:, 0], prepend=sma[0, 0]).reshape(-1, 1)
    return np.concatenate([window, sma, smv, jerk], axis=1)

def process_file(filepath, activity_code):
    try:
        # Load and clean
        data = load_sisfall_file(filepath)
        data = clean_data(data)
        if data is None or len(data) < WINDOW * DS_FACTOR:
            return np.array([]), np.array([])
        
        # Unit conversion & Drop the redundant sensor (9 channels -> 6 channels)
        data = convert_units_and_extract_6_channels(data)
        
        # Filtering
        data = butterworth_filter(data, fs=FS_ORIG)
        
        # Downsampling
        data = downsample(data, factor=DS_FACTOR)
        
        # Determine label and step
        is_fall = 1 if activity_code.startswith('F') else 0
        step = STEP_FALL if is_fall else STEP_ADL
        
        X_w, y_w = create_windows(data, label=is_fall, step=step)
        
        if len(X_w) == 0:
            return np.array([]), np.array([])
            
        # Feature Engineering (6 channels -> 9 channels)
        X_eng = np.array([feature_engineering(w) for w in X_w])
        
        return X_eng, y_w
    except Exception as e:
        return np.array([]), np.array([])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Scanning for SisFall dataset at: {BASE_PATH}")
    if not os.path.exists(BASE_PATH):
        print(f"Error: Directory {BASE_PATH} not found. Please place raw SisFall data there.")
        exit(1)

    all_files = glob.glob(os.path.join(BASE_PATH, '**', '*.txt'), recursive=True)
    print(f"Found {len(all_files)} TXT files to process.")

    X_fall_all, X_adl_all = [], []
    y_fall_all, y_adl_all = [], []

    for i, fp in enumerate(all_files):
        filename = os.path.basename(fp)
        activity_code = filename.split('_')[0]
        
        X_w, y_w = process_file(fp, activity_code)
        
        if len(X_w) > 0:
            if activity_code.startswith('F'):
                X_fall_all.append(X_w)
                y_fall_all.append(y_w)
            else:
                X_adl_all.append(X_w)
                y_adl_all.append(y_w)
                
        if (i+1) % 500 == 0:
            print(f"Processed {i+1}/{len(all_files)} files...")

    X_fall = np.concatenate(X_fall_all, axis=0) if X_fall_all else np.array([])
    y_fall = np.concatenate(y_fall_all, axis=0) if y_fall_all else np.array([])
    X_adl  = np.concatenate(X_adl_all,  axis=0) if X_adl_all else np.array([])
    y_adl  = np.concatenate(y_adl_all,  axis=0) if y_adl_all else np.array([])

    if len(X_fall) == 0 or len(X_adl) == 0:
        print("Error: No valid data processed.")
        exit(1)

    print(f"Extracted: {len(X_fall)} Fall windows, {len(X_adl)} ADL windows.")

    # Balance classes (1 Fall : 4 ADL)
    n_fall = len(X_fall)
    n_adl_keep = min(len(X_adl), n_fall * 4)
    
    np.random.seed(42)
    idx_adl = np.random.choice(len(X_adl), n_adl_keep, replace=False)
    X_adl_sub = X_adl[idx_adl]
    y_adl_sub = y_adl[idx_adl]

    X_all = np.concatenate([X_fall, X_adl_sub])
    y_all = np.concatenate([y_fall, y_adl_sub])

    # Shuffle
    idx = np.random.permutation(len(X_all))
    X_all = X_all[idx]
    y_all = y_all[idx]

    # Split 70/15/15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_all, y_all, test_size=0.30, random_state=42, stratify=y_all)
    X_vl, X_te, y_vl, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    # Standardize (REMOVED)
    print("Skipping StandardScaler (will be done dynamically in dataset.py)...")

    # Save outputs
    print(f"Saving unscaled final dataset to {SAVE_PATH}...")
    np.save(os.path.join(SAVE_PATH, 'X_train.npy'), X_tr)
    np.save(os.path.join(SAVE_PATH, 'y_train.npy'), y_tr)
    np.save(os.path.join(SAVE_PATH, 'X_val.npy'), X_vl)
    np.save(os.path.join(SAVE_PATH, 'y_val.npy'), y_vl)
    np.save(os.path.join(SAVE_PATH, 'X_test.npy'), X_te)
    np.save(os.path.join(SAVE_PATH, 'y_test.npy'), y_te)

    print("SisFall processing complete. Target generated files:")
    print(" - X_train.npy, y_train.npy")
    print(" - X_val.npy, y_val.npy")
    print(" - X_test.npy, y_test.npy")
