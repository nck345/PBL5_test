"""
PBL5 - Fall Detection AI System
Script: predict.py
Online Inference - Dự đoán thời gian thực phát hiện té ngã.

Usage:
    # Dự đoán từ file
    python predict.py --input dataset/Raw\ Data/WAL/WAL_acc_1_1.txt

    # Dự đoán từ thư mục (nhiều file)
    python predict.py --input dataset/Raw\ Data/WAL/

    # Chế độ streaming simulation
    python predict.py --input dataset/Raw\ Data/WAL/WAL_acc_1_1.txt --stream
"""

import argparse
import os
import sys
import time
import glob
import numpy as np
import torch

from src.utils import load_config, set_seed, get_device, apply_lowpass_filter, \
    get_scaler, normalize_data
from src.dataset import parse_mobiact_file, create_windows
from src.architecture import build_model
from src.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='PBL5 - Online Inference phát hiện té ngã'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Đường dẫn file cấu hình YAML')
    parser.add_argument('--model_path', type=str,
                        default='models/final_model/fall_detection_model.pt',
                        help='Đường dẫn model đã train')
    parser.add_argument('--input', type=str, required=True,
                        help='Đường dẫn file hoặc thư mục dữ liệu đầu vào')
    parser.add_argument('--stream', action='store_true',
                        help='Chế độ streaming simulation')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Ngưỡng phân loại (mặc định: 0.5)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: auto, cpu, cuda')
    parser.add_argument('--save_report', type=str, default=None, nargs='?', const='',
                        help='Đường dẫn file lưu báo cáo. Nếu không truyền sẽ tự tạo default.')
    
    # Allow extra unparsed positional arguments from batch script (e.g. stream flag, report name)
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, 
                        help='Các tham số truyền tự do (ví dụ: --stream, tên_file)')
    return parser.parse_args()


def preprocess_signal(data: np.ndarray, config: dict,
                      scaler=None) -> np.ndarray:
    """
    Tiền xử lý tín hiệu đầu vào.

    Args:
        data: Dữ liệu thô (n_samples, 3)
        config: Cấu hình
        scaler: Scaler đã fit (nếu có)
    Returns:
        Dữ liệu đã xử lý
    """
    preproc_cfg = config.get('preprocessing', {})

    # Low-pass filter
    filter_cfg = preproc_cfg.get('lowpass_filter', {})
    if filter_cfg.get('enabled', True) and len(data) > 20:
        fs = config['data'].get('sampling_rate', 50)
        cutoff = filter_cfg.get('cutoff_freq', 20)
        order = filter_cfg.get('order', 4)
        data = apply_lowpass_filter(data, cutoff=cutoff, fs=fs, order=order)

    # Normalize
    if scaler is not None:
        data, _ = normalize_data(data, scaler, fit=False)

    return data


def predict_file(filepath: str, model: torch.nn.Module, config: dict,
                 device: torch.device, threshold: float = 0.5,
                 scaler=None) -> dict:
    """
    Dự đoán trên 1 file dữ liệu.

    Returns:
        dict chứa kết quả dự đoán
    """
    # Parse file
    result = parse_mobiact_file(filepath)
    data = result.get('data', np.array([]).reshape(0, 3))

    if data.shape[0] < 10:
        return {
            'file': os.path.basename(filepath),
            'status': 'SKIPPED',
            'reason': 'Dữ liệu quá ngắn',
        }

    # Tiền xử lý
    data_processed = preprocess_signal(data, config, scaler)

    # Windowing
    window_size = config['data'].get('window_size', 100)
    overlap = config['data'].get('overlap', 0.15)
    windows = create_windows(data_processed, window_size=window_size,
                             overlap=overlap)

    if windows.shape[0] == 0:
        return {
            'file': os.path.basename(filepath),
            'status': 'SKIPPED',
            'reason': 'Không đủ dữ liệu cho windowing',
        }

    # Dự đoán
    model.eval()
    with torch.no_grad():
        X = torch.FloatTensor(windows).to(device)
        proba = model(X).cpu().numpy()

    # Tổng hợp kết quả
    mean_proba = float(np.mean(proba))
    max_proba = float(np.max(proba))
    fall_windows = int(np.sum(proba >= threshold))
    total_windows = len(proba)
    fall_ratio = fall_windows / total_windows

    # Quyết định cuối cùng
    is_fall = mean_proba >= threshold

    return {
        'file': os.path.basename(filepath),
        'activity_code': result.get('activity_code', 'N/A'),
        'subject_id': result.get('subject_id', -1),
        'status': '🚨 FALL DETECTED' if is_fall else '✅ NORMAL (ADL)',
        'is_fall': is_fall,
        'mean_probability': mean_proba,
        'max_probability': max_proba,
        'fall_windows': fall_windows,
        'total_windows': total_windows,
        'fall_ratio': fall_ratio,
        'data_length': data.shape[0],
    }


def stream_predict(filepath: str, model: torch.nn.Module, config: dict,
                   device: torch.device, threshold: float = 0.5,
                   scaler=None):
    """
    Mô phỏng streaming inference — xử lý dữ liệu theo từng cửa sổ.
    """
    result = parse_mobiact_file(filepath)
    data = result.get('data', np.array([]).reshape(0, 3))

    if data.shape[0] < 10:
        print("⚠ Dữ liệu quá ngắn để stream.")
        return

    window_size = config['data'].get('window_size', 100)
    fs = config['data'].get('sampling_rate', 50)

    print(f"\n🔴 STREAMING MODE - {os.path.basename(filepath)}")
    print(f"   Window: {window_size} samples ({window_size/fs:.1f}s)")
    print(f"   Threshold: {threshold}")
    print(f"{'='*55}")

    model.eval()
    buffer = []

    import json
    from datetime import datetime
    
    report_data = []

    for i, sample in enumerate(data):
        buffer.append(sample)

        if len(buffer) >= window_size:
            # Lấy cửa sổ gần nhất
            window_data = np.array(buffer[-window_size:])

            # Tiền xử lý
            window_processed = preprocess_signal(window_data, config, scaler)

            # Dự đoán
            with torch.no_grad():
                X = torch.FloatTensor(window_processed).unsqueeze(0).to(device)
                proba = model(X).cpu().item()

            timestamp = i / fs
            status = "🚨 FALL!" if proba >= threshold else "✅ OK"
            is_fall = bool(proba >= threshold)

            print(f"  t={timestamp:7.2f}s | prob={proba:.4f} | {status}")
            
            report_data.append({
                "timestamp_s": round(timestamp, 2),
                "probability": round(proba, 4),
                "is_fall": is_fall
            })

            # Simulate real-time delay
            time.sleep(0.01)

            # Shift buffer
            step = int(window_size * (1 - config['data'].get('overlap', 0.15)))
            buffer = buffer[step:]

    print(f"{'='*55}")
    print("🔴 STREAMING COMPLETED")
    
    # Save report if requested
    save_path = config.get('save_report_path')
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        report = {
            "file": os.path.basename(filepath),
            "date_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": threshold,
            "predictions": report_data
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"\n📁 Báo cáo streaming đã lưu tại: {save_path}")


def main():
    args = parse_args()

    # Xử lý các tham số phụ (từ file batch hoặc gõ tự do)
    if hasattr(args, 'extra_args') and args.extra_args:
        for arg in args.extra_args:
            if arg.lower() == '--stream':
                args.stream = True
            elif arg.strip() != "":
                # Nếu không phải --stream thì là tên file báo cáo tự chọn
                args.save_report = arg.strip()

    # ========================================
    # 1. Setup
    # ========================================
    print("📁 Đang tải cấu hình...")
    config = load_config(args.config)
    if args.device:
        config['device'] = args.device

    set_seed(config.get('seed', 42))
    device = get_device(config.get('device', 'auto'))

    # Default to saving a report if not specified
    save_path = args.save_report
    
    if save_path is None or not os.path.dirname(save_path):
        default_dir = os.path.join("logs", "predict_reports")
        
        # Determine default filename
        if not save_path:
            input_name = os.path.basename(os.path.normpath(args.input))
            if args.stream:
                save_path = f"{input_name}_stream_report.json"
            elif os.path.isdir(args.input):
                save_path = f"{input_name}_folder_report.txt"
            else:
                save_path = f"{input_name}_report.txt"
        
        save_path = os.path.join(default_dir, save_path)
        
    config['save_report_path'] = save_path

    # ========================================
    # 2. Load model
    # ========================================
    if not os.path.exists(args.model_path):
        print(f"\n❌ Không tìm thấy model tại: {args.model_path}")
        print("   Hãy chạy 'python train.py' trước.")
        sys.exit(1)

    print(f"🏗️ Đang tải model từ: {args.model_path}")
    model = build_model(config)
    model, _, _ = Trainer.load_model(model, args.model_path, device)
    print("  ✓ Model đã sẵn sàng!")

    # ========================================
    # 3. Xác định input files
    # ========================================
    input_path = args.input
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, '*.txt')))
    else:
        print(f"\n❌ Không tìm thấy input: {input_path}")
        sys.exit(1)

    if not files:
        print(f"\n❌ Không tìm thấy file .txt trong: {input_path}")
        sys.exit(1)

    print(f"\n📂 Tìm thấy {len(files)} file(s)")

    # ========================================
    # 4. Dự đoán
    # ========================================
    if args.stream and len(files) == 1:
        # Streaming mode
        stream_predict(files[0], model, config, device,
                       threshold=args.threshold)
    else:
        # Batch prediction
        print(f"\n{'='*70}")
        print(f"{'File':>30} | {'Activity':>8} | {'Status':>20} | "
              f"{'Prob':>6} | {'Fall%':>6}")
        print(f"{'='*70}")

        fall_count = 0
        total_count = 0
        results_list = []

        for fpath in files:
            result = predict_file(fpath, model, config, device,
                                  threshold=args.threshold)

            if result.get('status') == 'SKIPPED':
                continue

            total_count += 1
            if result.get('is_fall', False):
                fall_count += 1

            results_list.append(result)

            print(f"{result['file']:>30} | "
                  f"{result.get('activity_code', 'N/A'):>8} | "
                  f"{result['status']:>20} | "
                  f"{result.get('mean_probability', 0):.4f} | "
                  f"{result.get('fall_ratio', 0)*100:.1f}%")

        print(f"{'='*70}")
        print(f"\n📋 Tóm tắt: {fall_count}/{total_count} files "
              f"phát hiện té ngã")
              
        # Save TXT report if requested
        save_file = config.get('save_report_path')
        if save_file and results_list:
            os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write("======================================================================\n")
                f.write(f"{'File':>30} | {'Activity':>8} | {'Status':>20} | {'Prob':>6} | {'Fall%':>6}\n")
                f.write("======================================================================\n")
                for r in results_list:
                    f.write(f"{r['file']:>30} | "
                            f"{r.get('activity_code', 'N/A'):>8} | "
                            f"{r['status']:>20} | "
                            f"{r.get('mean_probability', 0):.4f} | "
                            f"{r.get('fall_ratio', 0)*100:.1f}%\n")
                f.write("======================================================================\n")
                f.write(f"Tổng kết: {fall_count}/{total_count} files phát hiện té ngã\n")
            print(f"\n📁 Báo cáo dự đoán hàng loạt (TXT) đã lưu tại: {save_file}")

    print("\n✅ Hoàn thành!")


if __name__ == '__main__':
    main()
