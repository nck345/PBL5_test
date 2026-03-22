import os
import yaml
import numpy as np
import sys

# Thêm đường dẫn để import src
sys.path.append(os.getcwd())

from src.utils import load_config
from src.dataset import load_all_data, preprocess_data

def analyze_distribution():
    config = load_config('configs/config.yaml')
    data_cfg = config['data']
    
    print("--- Analyze MobiAct Data Distribution ---")
    segments, labels, subjects = load_all_data(
        raw_data_dir=data_cfg['raw_data_dir'],
        sensor_type=data_cfg.get('sensor_type', 'acc'),
        fall_labels=data_cfg.get('fall_labels'),
        adl_labels=data_cfg.get('adl_labels'),
        verbose=True
    )
    
    X, y, subject_ids = preprocess_data(segments, labels, subjects, config)
    
    print(f"\nAfter Windowing (window_size={data_cfg['window_size']}, overlap={data_cfg['overlap']}):")
    n_adl = np.sum(y == 0)
    n_fall = np.sum(y == 1)
    total = len(y)
    
    print(f"  ADL (0): {n_adl} ({n_adl/total*100:.2f}%)")
    print(f"  Fall (1): {n_fall} ({n_fall/total*100:.2f}%)")
    print(f"  Ratio ADL/Fall: {n_adl/n_fall:.2f}")

if __name__ == "__main__":
    analyze_distribution()
