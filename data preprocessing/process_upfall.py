import os
import warnings
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = '../raw data/UpFall/CompleteDataSet.csv'
SAVE_PATH = '../dataset/upfall_processed'

FS_TARGET = 50    # Target Hz
WINDOW    = 128   # 2.56 seconds at 50Hz
STEP_FALL = 32    # overlap 75%
STEP_ADL  = 64    # overlap 50%

os.makedirs(SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(data):
    """Basic cleaning, returning None if invalid"""
    if np.isnan(data).any() or np.isinf(data).any():
        return None
    return data

def butterworth_filter(data, cutoff=20, fs=50, order=4):
    nyq  = 0.5 * fs
    norm = cutoff / nyq
    b, a = scipy_signal.butter(order, norm, btype='low')
    return scipy_signal.filtfilt(b, a, data, axis=0)

def create_windows(data, label, step):
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
    Input: (128, 6) -> Output: (128, 9)
    Adds SMA, SMV, Jerk
    """
    acc  = window[:, :3]
    gyro = window[:, 3:6]
    sma  = np.sqrt(np.sum(acc**2, axis=1, keepdims=True))
    smv  = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
    jerk = np.diff(sma[:, 0], prepend=sma[0, 0]).reshape(-1, 1)
    return np.concatenate([window, sma, smv, jerk], axis=1)

def process_trial_group(df_group):
    """Process a single continuous trial (Subject, Activity, Trial)"""
    # Columns in UpFall:
    # 0: TimeStamps
    # 15,16,17: BeltAcc X, Y, Z (g)
    # 18,19,20: BeltGyro X, Y, Z (deg/s)
    # 44: Activity
    
    # Sort by time just in case
    df_group = df_group.sort_values(by=0).reset_index(drop=True)
    
    # Convert timestamps to relative seconds
    t_obj = pd.to_datetime(df_group[0])
    t_sec = (t_obj - t_obj.iloc[0]).dt.total_seconds().values
    
    # Extract 6 channels
    # Acc (g -> m/s^2)
    acc = df_group[[15, 16, 17]].values.astype(np.float32) * 9.80665
    # Gyro (deg/s -> rad/s)
    gyro = df_group[[18, 19, 20]].values.astype(np.float32) * (np.pi / 180.0)
    
    data_raw = np.concatenate([acc, gyro], axis=1)
    data_raw = clean_data(data_raw)
    
    if data_raw is None or len(t_sec) < 2:
        return np.array([]), np.array([])
    
    # Check if there are duplicate timestamps and remove them
    unique_t, unique_indices = np.unique(t_sec, return_index=True)
    if len(unique_t) < 2:
        return np.array([]), np.array([])
    
    t_sec = t_sec[unique_indices]
    data_raw = data_raw[unique_indices]
    
    # Interpolate to exactly 50Hz
    duration = t_sec[-1]
    # We need at least enough data for 1 window
    min_duration = (WINDOW / FS_TARGET)
    if duration < min_duration:
        return np.array([]), np.array([])
        
    t_target = np.arange(0, duration, 1.0 / FS_TARGET)
    
    interpolator = interp1d(t_sec, data_raw, axis=0, kind='linear', fill_value="extrapolate")
    data_interp = interpolator(t_target)
    
    # Butterworth Filter
    data_filtered = butterworth_filter(data_interp, fs=FS_TARGET)
    
    # Labels and Steps
    activity_code = df_group[44].iloc[0]
    is_fall = 1 if activity_code <= 5 else 0
    step = STEP_FALL if is_fall else STEP_ADL
    
    X_w, y_w = create_windows(data_filtered, label=is_fall, step=step)
    
    if len(X_w) == 0:
        return np.array([]), np.array([])
        
    # Feature Engineering
    X_eng = np.array([feature_engineering(w) for w in X_w])
    
    return X_eng, y_w

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Scanning for UpFall dataset at: {BASE_PATH}")
    if not os.path.exists(BASE_PATH):
        print(f"Error: {BASE_PATH} not found.")
        exit(1)

    print("Loading large CSV (this may take a minute)...")
    # Read CSV, skip header lines, ignore bad lines just in case
    # UpFall dataset format: 
    # Row 1 is column names, Row 2 is units, data starts at Row 3 (skiprows=2)
    # Using specific columns to save memory:
    # 0: Timestamp
    # 15,16,17: BeltAcc
    # 18,19,20: BeltGyro
    # 43: Subject, 44: Activity, 45: Trial
    usecols = [0, 15, 16, 17, 18, 19, 20, 43, 44, 45]
    df = pd.read_csv(BASE_PATH, skiprows=2, header=None, usecols=usecols, on_bad_lines='skip')
    
    # Drop rows where Activity is 0 (unlabeled/transitional)
    df = df[df[44] != 0].dropna()
    
    print(f"Loaded {len(df)} rows of active data.")
    
    # Group by Subject (43), Activity (44), Trial (45)
    groups = df.groupby([43, 44, 45])
    
    X_fall_all, X_adl_all = [], []
    y_fall_all, y_adl_all = [], []

    total_groups = len(groups)
    print(f"Found {total_groups} trial groups to process.")

    for i, (name, group) in enumerate(groups):
        X_w, y_w = process_trial_group(group)
        
        if len(X_w) > 0:
            if y_w[0] == 1:
                X_fall_all.append(X_w)
                y_fall_all.append(y_w)
            else:
                X_adl_all.append(X_w)
                y_adl_all.append(y_w)
                
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{total_groups} groups...")

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

    # Save outputs
    print(f"Saving final unscaled dataset to {SAVE_PATH}...")
    np.save(os.path.join(SAVE_PATH, 'X_train.npy'), X_tr)
    np.save(os.path.join(SAVE_PATH, 'y_train.npy'), y_tr)
    np.save(os.path.join(SAVE_PATH, 'X_val.npy'), X_vl)
    np.save(os.path.join(SAVE_PATH, 'y_val.npy'), y_vl)
    np.save(os.path.join(SAVE_PATH, 'X_test.npy'), X_te)
    np.save(os.path.join(SAVE_PATH, 'y_test.npy'), y_te)

    print("UpFall processing complete.")
