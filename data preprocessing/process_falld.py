import os
import pickle
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
BASE_PATH = '../raw data/FallAllD/FallAllD.pkl'
SAVE_PATH = '../dataset/falld_processed'

FS_ORIG   = 238   # Original Hz for FallAllD
FS_TARGET = 50    # Target Hz
WINDOW    = 128   # 2.56 seconds
STEP_FALL = 32    # overlap 75%
STEP_ADL  = 64    # overlap 50%

os.makedirs(SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def butterworth_filter(data, cutoff=20, fs=238, order=4):
    nyq  = 0.5 * fs
    norm = cutoff / nyq
    b, a = scipy_signal.butter(order, norm, btype='low')
    return scipy_signal.filtfilt(b, a, data, axis=0)

def feature_engineering(window):
    """
    Same as other datasets: Add 3 channels SMA, SMV, Jerk
    Input: (128, 6) -> Output: (128, 9)
    """
    acc  = window[:, :3]
    gyro = window[:, 3:]
    sma  = np.sqrt(np.sum(acc**2, axis=1, keepdims=True))
    smv  = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
    jerk = np.diff(sma[:, 0], prepend=sma[0, 0]).reshape(-1, 1)
    return np.concatenate([window, sma, smv, jerk], axis=1)

def process_row(row):
    try:
        acc_raw = row['Acc']
        gyr_raw = row['Gyr']
        activity_id = row['ActivityID']
        
        # 1. Unit conversion (to m/s^2 and rad/s)
        # LSM9DS1 Acc range ±8g -> 1/4096 LSB/g
        acc = acc_raw.astype(np.float32) / 4096.0 * 9.80665
        # LSM9DS1 Gyr range ±2000 dps -> 70 mdps/LSB = 0.07 dps/LSB
        gyro = gyr_raw.astype(np.float32) * 0.07 * (np.pi / 180.0)
        
        # 2. Coordinate Alignment: 90-degree counterclockwise rotation around Z-axis
        # x_aligned = -y_raw
        # y_aligned = x_raw
        # z_aligned = z_raw
        acc_aligned = np.zeros_like(acc)
        acc_aligned[:, 0] = -acc[:, 1]
        acc_aligned[:, 1] = acc[:, 0]
        acc_aligned[:, 2] = acc[:, 2]
        
        gyro_aligned = np.zeros_like(gyro)
        gyro_aligned[:, 0] = -gyro[:, 1]
        gyro_aligned[:, 1] = gyro[:, 0]
        gyro_aligned[:, 2] = gyro[:, 2]
        
        data_raw = np.concatenate([acc_aligned, gyro_aligned], axis=1)
        
        if np.isnan(data_raw).any() or np.isinf(data_raw).any():
            return np.array([]), np.array([])
            
        # 3. Butterworth Low-pass Filter at 238 Hz
        data_filtered = butterworth_filter(data_raw, cutoff=20, fs=FS_ORIG)
        
        # 4. Resample to 50 Hz using 1D linear interpolation
        n_samples = len(data_filtered)
        duration = n_samples / FS_ORIG
        t_orig = np.arange(n_samples) / FS_ORIG
        t_target = np.arange(0, duration, 1.0 / FS_TARGET)
        
        if len(t_target) < WINDOW:
            return np.array([]), np.array([])
            
        interpolator = interp1d(t_orig, data_filtered, axis=0, kind='linear', fill_value='extrapolate')
        data_resampled = interpolator(t_target)
        
        # 5. Windowing
        is_fall = 1 if activity_id >= 101 else 0
        step = STEP_FALL if is_fall else STEP_ADL
        
        X_list, y_list = [], []
        for s in range(0, len(data_resampled) - WINDOW + 1, step):
            window = data_resampled[s : s+WINDOW]
            window_eng = feature_engineering(window)
            X_list.append(window_eng)
            y_list.append(is_fall)
            
        if not X_list:
            return np.array([]), np.array([])
            
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)
    except Exception as e:
        return np.array([]), np.array([])

def balance_and_shuffle(X_fall, y_fall, X_lying, y_lying, X_other_adl, y_other_adl):
    if len(X_fall) == 0:
        return np.array([]).reshape(0, WINDOW, 9), np.array([])
        
    # Oversample lying down ADLs 5x to combat class imbalance
    if len(X_lying) > 0:
        X_lying = np.repeat(X_lying, 5, axis=0)
        y_lying = np.repeat(y_lying, 5, axis=0)
        
    # Balance classes (1 Fall : 4 ADL)
    n_fall = len(X_fall)
    n_adl_keep = n_fall * 4
    
    n_lying_keep = min(len(X_lying), n_adl_keep)
    n_other_keep = n_adl_keep - n_lying_keep
    
    np.random.seed(42)
    
    if n_lying_keep < len(X_lying):
        idx_lying = np.random.choice(len(X_lying), n_lying_keep, replace=False)
        X_lying_sub = X_lying[idx_lying]
        y_lying_sub = y_lying[idx_lying]
    else:
        X_lying_sub = X_lying
        y_lying_sub = y_lying
        
    if n_other_keep > 0 and len(X_other_adl) > 0:
        idx_other = np.random.choice(len(X_other_adl), min(len(X_other_adl), n_other_keep), replace=False)
        X_other_sub = X_other_adl[idx_other]
        y_other_sub = y_other_adl[idx_other]
        
        X_adl_sub = np.concatenate([X_lying_sub, X_other_sub], axis=0)
        y_adl_sub = np.concatenate([y_lying_sub, y_other_sub], axis=0)
    else:
        X_adl_sub = X_lying_sub
        y_adl_sub = y_lying_sub
        
    X_all = np.concatenate([X_fall, X_adl_sub], axis=0)
    y_all = np.concatenate([y_fall, y_adl_sub], axis=0)
    
    # Shuffle
    idx = np.random.permutation(len(X_all))
    X_all = X_all[idx]
    y_all = y_all[idx]
    
    return X_all, y_all

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Scanning for FallAllD dataset at: {BASE_PATH}")
    if not os.path.exists(BASE_PATH):
        print(f"Error: File {BASE_PATH} not found.")
        exit(1)

    print("Loading FallAllD pickle DataFrame...")
    df = pd.read_pickle(BASE_PATH)
    print(f"Loaded DataFrame with {len(df)} rows.")

    # Filter Waist device and exclude unknown activities (IDs 45-100)
    df_waist = df[(df['Device'] == 'Waist') & (~df['ActivityID'].isin(range(45, 101)))]
    print(f"Filtered Waist rows (excluding unknown activities): {len(df_waist)}")

    # Subject-wise groupings
    train_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    val_subjects   = [12, 13]
    test_subjects  = [14, 15]

    # Lists to collect windows before balancing
    splits = {
        'train': {'fall': [], 'lying': [], 'other': []},
        'val':   {'fall': [], 'lying': [], 'other': []},
        'test':  {'fall': [], 'lying': [], 'other': []}
    }

    for i, (_, row) in enumerate(df_waist.iterrows()):
        subject = int(row['SubjectID'])
        activity = int(row['ActivityID'])
        
        # Process the single trial
        X_w, y_w = process_row(row)
        
        if len(X_w) == 0:
            continue
            
        # Determine the split group
        if subject in train_subjects:
            group = 'train'
        elif subject in val_subjects:
            group = 'val'
        elif subject in test_subjects:
            group = 'test'
        else:
            continue
            
        # Classify windows
        if activity >= 101:
            splits[group]['fall'].append(X_w)
        elif activity in [16, 17, 18]: # Lying down, Turning while lying, Rising up
            splits[group]['lying'].append(X_w)
        else:
            splits[group]['other'].append(X_w)

        if (i+1) % 400 == 0:
            print(f"Processed {i+1}/{len(df_waist)} trials...")

    # Balance and shuffle each split group
    final_data = {}
    for group in ['train', 'val', 'test']:
        print(f"\nBalancing and shuffling {group} set...")
        fall_list = splits[group]['fall']
        lying_list = splits[group]['lying']
        other_list = splits[group]['other']
        
        X_f = np.concatenate(fall_list, axis=0) if fall_list else np.array([]).reshape(0, WINDOW, 9)
        y_f = np.concatenate([np.ones(len(w), dtype=np.int32) for w in fall_list]) if fall_list else np.array([])
        
        X_ly = np.concatenate(lying_list, axis=0) if lying_list else np.array([]).reshape(0, WINDOW, 9)
        y_ly = np.concatenate([np.zeros(len(w), dtype=np.int32) for w in lying_list]) if lying_list else np.array([])
        
        X_ot = np.concatenate(other_list, axis=0) if other_list else np.array([]).reshape(0, WINDOW, 9)
        y_ot = np.concatenate([np.zeros(len(w), dtype=np.int32) for w in other_list]) if other_list else np.array([])
        
        print(f"  Raw: Fall={len(X_f)}, Lying ADL={len(X_ly)}, Other ADL={len(X_ot)}")
        
        X_balanced, y_balanced = balance_and_shuffle(X_f, y_f, X_ly, y_ly, X_ot, y_ot)
        final_data[f'X_{group}'] = X_balanced
        final_data[f'y_{group}'] = y_balanced
        
        print(f"  Balanced {group} shape: X={X_balanced.shape}, y={y_balanced.shape}")
        if len(y_balanced) > 0:
            print(f"  Class distribution: ADL (0) = {np.sum(y_balanced == 0)}, Fall (1) = {np.sum(y_balanced == 1)}")

    # Save outputs
    print(f"\nSaving unscaled final datasets to {SAVE_PATH}...")
    np.save(os.path.join(SAVE_PATH, 'X_train.npy'), final_data['X_train'])
    np.save(os.path.join(SAVE_PATH, 'y_train.npy'), final_data['y_train'])
    np.save(os.path.join(SAVE_PATH, 'X_val.npy'), final_data['X_val'])
    np.save(os.path.join(SAVE_PATH, 'y_val.npy'), final_data['y_val'])
    np.save(os.path.join(SAVE_PATH, 'X_test.npy'), final_data['X_test'])
    np.save(os.path.join(SAVE_PATH, 'y_test.npy'), final_data['y_test'])

    print("FallAllD processing complete. Target generated files:")
    print(" - X_train.npy, y_train.npy")
    print(" - X_val.npy, y_val.npy")
    print(" - X_test.npy, y_test.npy")
