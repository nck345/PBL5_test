"""
PBL5 - Fall Detection AI System
Script: test.py
Chạy đánh giá mô hình đã huấn luyện trên tập Test (Offline).

Usage:
    python test.py
    python test.py --model_path models/final_model/fall_detection_model.pt
    python test.py --config configs/config.yaml
"""

import argparse
import sys
import os

from src.utils import load_config, set_seed, get_device
from src.dataset import prepare_data
from src.architecture import build_model
from src.trainer import Trainer
from src.evaluator import quick_evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description='PBL5 - Đánh giá mô hình phát hiện té ngã (Offline)'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Đường dẫn file cấu hình YAML')
    parser.add_argument('--model_path', type=str,
                        default='models/final_model/fall_detection_model.pt',
                        help='Đường dẫn model đã train')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: auto, cpu, cuda')
    parser.add_argument('--save_dir', type=str, default='logs/test_results',
                        help='Thư mục lưu kết quả')
    parser.add_argument('--threshold-mode', type=str, default=None,
                        choices=['fixed', 'val_calibrated'],
                        help='Che do threshold: fixed hoac val_calibrated')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold co dinh khi threshold-mode=fixed')
    return parser.parse_args()


def main():
    args = parse_args()

    # ========================================
    # 1. Load config
    # ========================================
    print("📁 Đang tải cấu hình...")
    config = load_config(args.config)

    if args.device:
        config['device'] = args.device

    seed = config.get('seed', 42)
    set_seed(seed)
    device = get_device(config.get('device', 'auto'))

    print(f"🔧 Device: {device}")

    # ========================================
    # 2. Kiểm tra model file
    # ========================================
    if not os.path.exists(args.model_path):
        print(f"\n❌ Không tìm thấy model tại: {args.model_path}")
        print("   Hãy chạy 'python train.py' trước để huấn luyện model.")
        sys.exit(1)

    # ========================================
    # 3. Chuẩn bị dữ liệu
    # ========================================
    print("\n📊 Đang chuẩn bị dữ liệu...")
    try:
        data = prepare_data(config, verbose=True)
    except ValueError as e:
        print(f"\n❌ Lỗi: {e}")
        sys.exit(1)

    val_loader = data['val_loader']
    test_loader = data['test_loader']

    # ========================================
    # 4. Load model
    # ========================================
    print("\n🏗️ Đang tải model...")
    model = build_model(config)
    model, loaded_config, history = Trainer.load_model(
        model, args.model_path, device
    )
    print(f"  ✓ Model đã tải từ: {args.model_path}")

    # ========================================
    # 5. Đánh giá
    # ========================================
    print("\n🧪 Đánh giá trên tập Test...")
    eval_cfg = config.get('evaluation', {})
    threshold_mode = args.threshold_mode or eval_cfg.get('threshold_mode', 'fixed')
    fixed_threshold = args.threshold
    if fixed_threshold is None:
        fixed_threshold = eval_cfg.get('threshold', config['model'].get('threshold', 0.5))

    if threshold_mode == 'val_calibrated':
        print("  -> Threshold mode: val_calibrated")
        metrics = quick_evaluate(
            model, test_loader, device,
            verbose=True, save_dir=args.save_dir,
            optimize_threshold=False,
            threshold=fixed_threshold,
            calibration_loader=val_loader
        )
    else:
        print(f"  -> Threshold mode: fixed ({fixed_threshold:.2f})")
        metrics = quick_evaluate(
            model, test_loader, device,
            verbose=True, save_dir=args.save_dir,
            optimize_threshold=False,
            threshold=fixed_threshold
        )

    # ========================================
    # 6. Tóm tắt
    # ========================================
    print(f"\n📋 Tóm tắt kết quả:")
    print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {metrics['precision']*100:.2f}%")
    print(f"   Recall:    {metrics['recall']*100:.2f}%")
    print(f"   F1-Score:  {metrics['f1_score']*100:.2f}%")
    print(f"   AUC:       {metrics['auc']:.4f}")

    print(f"\n📊 Biểu đồ đã lưu tại: {args.save_dir}")
    print("\n✅ Hoàn thành!")


if __name__ == '__main__':
    main()
