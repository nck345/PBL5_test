import os
from src.utils import load_config
from src.dataset import prepare_data
import numpy as np

def main():
    print("Loading config...")
    config = load_config("configs/config.yaml")
    
    # We set verbose=False to avoid massive stdout, but some prints might still happen
    print("\n--- PREPARING DATA ---")
    data_info = prepare_data(config, verbose=False)
    
    splits = data_info['splits']
    train_loader = data_info['train_loader']
    
    print("\n--- 1. KIỂM TRA ĐỘ CÂN BẰNG TỔNG THỂ (RAW DATA) ---")
    for phase in ['train', 'val', 'test']:
        y = splits[f'y_{phase}']
        n_adl = np.sum(y == 0)
        n_fall = np.sum(y == 1)
        total = len(y)
        print(f"[{phase.upper()}] Tổng số cửa sổ (windows): {total}")
        print(f"      ADL (Nhãn 0)  : {n_adl} ({n_adl/total*100:.2f}%)")
        print(f"      Fall (Nhãn 1) : {n_fall} ({n_fall/total*100:.2f}%)")
        
    print("\n--- 2. KIỂM TRA ĐỘ CÂN BẰNG TRONG BATCH (SAU KHI ĐƯA VÀO DATALOADER) ---")
    print(f"Lưu ý: training/use_sampler={config['training'].get('use_sampler', True)} trong config.")
    
    total_adl_batch = 0
    total_fall_batch = 0
    num_batches_to_check = min(10, len(train_loader))
    
    for i, (X_batch, y_batch) in enumerate(train_loader):
        if i >= num_batches_to_check:
            break
        adl = torch.sum(y_batch == 0).item()
        fall = torch.sum(y_batch == 1).item()
        total_adl_batch += adl
        total_fall_batch += fall
        
    total_samples = total_adl_batch + total_fall_batch
    print(f"Over {num_batches_to_check} batches ({total_samples} samples):")
    print(f"      ADL (Nhãn 0)  : {total_adl_batch} ({total_adl_batch/total_samples*100:.2f}%)")
    print(f"      Fall (Nhãn 1) : {total_fall_batch} ({total_fall_batch/total_samples*100:.2f}%)")
    
    print("\n--- 3. KIỂM TRA ĐỊNH DẠNG DATA (SHAPE & VALUES) MÀ AI SẼ HỌC ---")
    # Lấy 1 batch từ DataLoader
    import torch
    X_batch, y_batch = next(iter(train_loader))
    
    print(f"Shape của X_batch (đầu vào): {X_batch.shape} -> (Batch Size, Window Size, Features)")
    print(f"Shape của y_batch (đầu ra) : {y_batch.shape} -> (Batch Size,)")
    
    # Kiểm tra Features (num_channels)
    print(f"Số lượng đặc trưng (Features): {X_batch.shape[2]}")
    expected_features = 7 if len(config['data'].get('sensors', ['acc'])) > 1 else 3
    print(f" -> Các Features bao gồm: acc_x, acc_y, acc_z" + 
          (", gyro_x, gyro_y, gyro_z, Flag_gyro" if X_batch.shape[2] == 7 else ""))
    
    print(f"\nKiểm tra dữ liệu (1 cửa sổ ngẫu nhiên):")
    print(f"- Max value : {torch.max(X_batch).item():.4f}")
    print(f"- Min value : {torch.min(X_batch).item():.4f}")
    print(f"- Mean value: {torch.mean(X_batch).item():.4f}")
    print("- Có giá trị NaN/Inf không?", torch.isnan(X_batch).any().item() or torch.isinf(X_batch).any().item())
    
if __name__ == "__main__":
    import torch
    main()
