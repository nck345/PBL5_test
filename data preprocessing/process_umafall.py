import os
import glob
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
BASE_PATH = '../raw data/UMAFall/UMAFall_Dataset_corrected_version'
SAVE_PATH = '../dataset/umafall_processed'

FS_TARGET = 50    # Target Hz
WINDOW    = 128   # 2.56 seconds at 50Hz
STEP_FALL = 32    # overlap 75%
STEP_ADL  = 64    # overlap 50%

os.makedirs(SAVE_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2"""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    c = np.dot(a, b)
    
    if c > 0.9999:
        return np.eye(3)
    if c < -0.9999:
        # Rotate 180 degrees around Z axis as fallback
        return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def clean_data(data):
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
    acc  = window[:, :3]
    gyro = window[:, 3:6]
    sma  = np.sqrt(np.sum(acc**2, axis=1, keepdims=True))
    smv  = np.sqrt(np.sum(gyro**2, axis=1, keepdims=True))
    jerk = np.diff(sma[:, 0], prepend=sma[0, 0]).reshape(-1, 1)
    return np.concatenate([window, sma, smv, jerk], axis=1)

def process_file(filepath):
    """Process a single UMAFall CSV file."""
    filename = os.path.basename(filepath)
    is_fall = 1 if '_Fall_' in filename or '_FALL_' in filename else 0
    
    # Load data
    usecols = [0, 2, 3, 4, 5, 6]
    try:
        df = pd.read_csv(filepath, sep=';', comment='%', header=None, 
                         names=['TimeStamp', 'X', 'Y', 'Z', 'SensorType', 'SensorID'], 
                         usecols=usecols, on_bad_lines='skip').dropna()
    except Exception as e:
        return np.array([]), np.array([])
        
    # Extract Waist sensor (SensorID == 2)
    df_waist = df[df['SensorID'] == 2]
    if len(df_waist) < 10:
        return np.array([]), np.array([])
        
    df_acc = df_waist[df_waist['SensorType'] == 0].sort_values(by='TimeStamp').reset_index(drop=True)
    df_gyro = df_waist[df_waist['SensorType'] == 1].sort_values(by='TimeStamp').reset_index(drop=True)
    
    if len(df_acc) < 10 or len(df_gyro) < 10:
        return np.array([]), np.array([])
        
    # Convert timestamps to seconds relative to start
    start_time = min(df_acc['TimeStamp'].min(), df_gyro['TimeStamp'].min())
    t_acc = (df_acc['TimeStamp'] - start_time).values / 1000.0
    t_gyro = (df_gyro['TimeStamp'] - start_time).values / 1000.0
    
    acc_data = df_acc[['X', 'Y', 'Z']].values.astype(np.float32) * 9.80665
    gyro_data = df_gyro[['X', 'Y', 'Z']].values.astype(np.float32) * (np.pi / 180.0)
    
    # Remove duplicate timestamps
    unique_t_acc, idx_acc = np.unique(t_acc, return_index=True)
    acc_data = acc_data[idx_acc]
    
    unique_t_gyro, idx_gyro = np.unique(t_gyro, return_index=True)
    gyro_data = gyro_data[idx_gyro]
    
    if len(unique_t_acc) < 2 or len(unique_t_gyro) < 2:
        return np.array([]), np.array([])
        
    # Common time base for interpolation
    min_t = max(unique_t_acc[0], unique_t_gyro[0])
    max_t = min(unique_t_acc[-1], unique_t_gyro[-1])
    
    if max_t - min_t < (WINDOW / FS_TARGET):
        return np.array([]), np.array([])
        
    t_target = np.arange(min_t, max_t, 1.0 / FS_TARGET)
    
    # Interpolate
    interp_acc = interp1d(unique_t_acc, acc_data, axis=0, kind='linear', fill_value="extrapolate")
    interp_gyro = interp1d(unique_t_gyro, gyro_data, axis=0, kind='linear', fill_value="extrapolate")
    
    acc_resampled = interp_acc(t_target)
    gyro_resampled = interp_gyro(t_target)
    
    # Gravity Alignment (MobiAct expects gravity to be ~ +9.8 on Y axis)
    # Estimate gravity from the first few seconds of acceleration (assuming mostly stationary/slow movement)
    gravity_vec = np.mean(acc_resampled[:FS_TARGET*2], axis=0) # Mean of first 2 seconds
    if np.linalg.norm(gravity_vec) > 1.0: # Ensure valid vector
        rot_mat = rotation_matrix_from_vectors(gravity_vec, np.array([0.0, 9.80665, 0.0]))
        acc_resampled = np.dot(acc_resampled, rot_mat.T)
        gyro_resampled = np.dot(gyro_resampled, rot_mat.T)
        
    data_raw = np.concatenate([acc_resampled, gyro_resampled], axis=1)
    data_raw = clean_data(data_raw)
    
    if data_raw is None:
        return np.array([]), np.array([])
        
    # Filtering
    data_filtered = butterworth_filter(data_raw, fs=FS_TARGET)
    
    # Windowing
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
    print(f"Scanning for UMAFall dataset at: {BASE_PATH}")
    if not os.path.exists(BASE_PATH):
        print(f"Error: Directory {BASE_PATH} not found.")
        exit(1)

    all_files = glob.glob(os.path.join(BASE_PATH, '**', '*.csv'), recursive=True)
    print(f"Found {len(all_files)} CSV files to process.")

    X_fall_all, X_adl_all = [], []
    y_fall_all, y_adl_all = [], []

    for i, fp in enumerate(all_files):
        X_w, y_w = process_file(fp)
        
        if len(X_w) > 0:
            if y_w[0] == 1:
                X_fall_all.append(X_w)
                y_fall_all.append(y_w)
            else:
                X_adl_all.append(X_w)
                y_adl_all.append(y_w)
                
        if (i+1) % 100 == 0:
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
    if n_fall == 0:
        print("No Fall windows found! Cannot balance.")
        exit(1)
        
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

    print("UMAFall processing complete.")
