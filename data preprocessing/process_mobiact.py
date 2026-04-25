import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE_PATH = '../dataset/raw data/MobiAct'  # Path to raw CSV files
SAVE_PATH = '../dataset/MobiAct_Processed'

FALL_ACTS     = ['FOL', 'FKL', 'BSC', 'SDL']
ADL_ACTS      = ['STD', 'WAL', 'JOG', 'JUM', 'STU',
                 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO',
                 'SBE', 'SBW', 'SLH', 'SLW', 'SRH']
IGNORE_LABELS = ['LYI']
FEATURE_COLS  = ['acc_x', 'acc_y', 'acc_z',
                 'gyro_x', 'gyro_y', 'gyro_z']

FS_ORIG   = 200   # Original Hz
FS_TARGET = 50    # Target Hz
DS_FACTOR = 4     # 200/50
WINDOW    = 128   # 2.56 seconds
STEP_FALL = 32    # overlap 75%
STEP_ADL  = 64    # overlap 50%

os.makedirs(SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def clean_dataframe(df):
    df = df[~df['label'].isin(IGNORE_LABELS)].copy()
    df = df.dropna(subset=FEATURE_COLS)
    for col in FEATURE_COLS:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)

def butterworth_filter(data, cutoff=20, fs=200, order=4):
    nyq  = 0.5 * fs
    norm = cutoff / nyq
    b, a = scipy_signal.butter(order, norm, btype='low')
    return scipy_signal.filtfilt(b, a, data)

def downsample(df, factor=4):
    return df.iloc[::factor].reset_index(drop=True)

def create_windows(data, labels, step, threshold=0.05):
    X_list, y_list = [], []
    for s in range(0, len(data)-WINDOW+1, step):
        window = data[s : s+WINDOW]
        lbl_w  = labels[s : s+WINDOW]
        X_list.append(window)
        y_list.append(1 if lbl_w.mean() > threshold else 0)
    if not X_list:
        return np.array([]), np.array([])
    return np.array(X_list), np.array(y_list)

def feature_engineering(window):
    """Adds 3 channels: SMA, SMV, Jerk (Total 9 channels)"""
    acc  = window[:, :3]
    gyro = window[:, 3:]
    sma  = np.sqrt(np.sum(acc**2, axis=1, keepdims=True))
    smv  = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
    jerk = np.diff(sma[:, 0], prepend=sma[0, 0]).reshape(-1, 1)
    return np.concatenate([window, sma, smv, jerk], axis=1)

def process_fall_file(filepath):
    try:
        df = pd.read_csv(filepath)
        df = clean_dataframe(df)
        if len(df) < WINDOW * DS_FACTOR:
            return np.array([]), np.array([])

        for col in FEATURE_COLS:
            df[col] = butterworth_filter(df[col].values, fs=FS_ORIG)
        df = downsample(df, factor=DS_FACTOR)
        df['y'] = df['label'].apply(lambda x: 1 if x in FALL_ACTS else 0)

        idx_fall = np.where(df['y'].values == 1)[0]
        if len(idx_fall) == 0:
            return np.array([]), np.array([])

        BUFFER = 50
        i_start = max(0, idx_fall[0]  - BUFFER)
        i_end   = min(len(df), idx_fall[-1] + BUFFER)
        df_region = df.iloc[i_start:i_end].reset_index(drop=True)

        if len(df_region) < WINDOW:
            df_region = df.reset_index(drop=True)

        data_np = df_region[FEATURE_COLS].values
        lbl_np  = df_region['y'].values
        X_w, y_w = create_windows(data_np, lbl_np, step=STEP_FALL, threshold=0.05)

        if len(X_w) == 0:
            return np.array([]), np.array([])

        mask = y_w == 1
        X_fall_only = X_w[mask]
        if len(X_fall_only) == 0:
            return np.array([]), np.array([])

        X_eng = np.array([feature_engineering(w) for w in X_fall_only])
        return X_eng, np.ones(len(X_eng), dtype=int)
    except Exception as e:
        return np.array([]), np.array([])

def process_adl_file(filepath):
    try:
        df = pd.read_csv(filepath)
        df = clean_dataframe(df)
        if len(df) < WINDOW * DS_FACTOR:
            return np.array([]), np.array([])

        for col in FEATURE_COLS:
            df[col] = butterworth_filter(df[col].values, fs=FS_ORIG)
        df = downsample(df, factor=DS_FACTOR)

        data_np = df[FEATURE_COLS].values
        lbl_np  = np.zeros(len(df), dtype=int)

        X_w, y_w = create_windows(data_np, lbl_np, step=STEP_ADL, threshold=0.5)

        if len(X_w) == 0:
            return np.array([]), np.array([])

        X_eng = np.array([feature_engineering(w) for w in X_w])
        return X_eng, np.zeros(len(X_eng), dtype=int)
    except Exception as e:
        return np.array([]), np.array([])

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Scanning for MobiAct dataset at: {BASE_PATH}")
    if not os.path.exists(BASE_PATH):
        print(f"Error: Directory {BASE_PATH} not found. Please place raw MobiAct data there.")
        exit(1)

    all_files = glob.glob(os.path.join(BASE_PATH, '**', '*.csv'), recursive=True)
    print(f"Found {len(all_files)} CSV files to process.")

    X_fall_all, X_adl_all = [], []
    y_fall_all, y_adl_all = [], []

    for i, fp in enumerate(all_files):
        act = os.path.basename(fp).split('_')[0]
        if act in FALL_ACTS:
            X_w, y_w = process_fall_file(fp)
            if len(X_w) > 0:
                X_fall_all.append(X_w)
                y_fall_all.append(y_w)
        elif act in ADL_ACTS:
            X_w, y_w = process_adl_file(fp)
            if len(X_w) > 0:
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

    print("MobiAct processing complete. Target generated files:")
    print(" - X_train.npy, y_train.npy")
    print(" - X_val.npy, y_val.npy")
    print(" - X_test.npy, y_test.npy")
