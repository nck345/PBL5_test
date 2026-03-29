import os
import yaml
import numpy as np
import sys

# Thêm đường dẫn để import src
sys.path.append(os.getcwd())

from src.utils import load_config
from src.dataset import load_mobiact_data, preprocess_data, load_archive3_data

def analyze_distribution():
    config = load_config('configs/config.yaml')
    data_cfg = config['data']
    
    print("--- Analyze MobiAct Data Distribution ---")
    # B1: Tải dữ liệu MobiAct
    print("Loading MobiAct dataset...")
    segments, labels, subjects = load_mobiact_data(
        raw_data_dir=data_cfg['raw_data_dir'],
        sensor_type=data_cfg.get('sensor_type', 'acc'),
        sensors=data_cfg.get('sensors', ['acc']),
        fall_labels=data_cfg.get('fall_labels'),
        adl_labels=data_cfg.get('adl_labels'),
        verbose=True
    )
    
    # Tải dữ liệu Archive (3)
    archive_dir = os.path.join(os.path.dirname(data_cfg['raw_data_dir']), "archive (3)")
    a3_s, a3_l, a3_sub = load_archive3_data(archive_dir, data_cfg.get('sensors', ['acc']))
    segments.extend(a3_s)
    labels.extend(a3_l)
    subjects.extend(a3_sub)
    
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
