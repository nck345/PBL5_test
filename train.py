"""
PBL5 - Fall Detection AI System
Script: train.py
Chạy huấn luyện mô hình phát hiện té ngã từ dòng lệnh.

Usage:
    python train.py
    python train.py --config configs/config.yaml
    python train.py --epochs 10 --model stacked_lstm
"""

import argparse
import sys
import os

from src.utils import load_config, set_seed, get_device, count_parameters, \
    plot_training_curves, ensure_dir
from src.dataset import prepare_data
from src.architecture import build_model
from src.trainer import Trainer
from src.evaluator import quick_evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description='PBL5 - Huấn luyện mô hình phát hiện té ngã'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Đường dẫn file cấu hình YAML')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Số epochs (ghi đè config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (ghi đè config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (ghi đè config)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['stacked_lstm', 'cnn_1d', 'ensemble'],
                        help='Loại model (ghi đè config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: auto, cpu, cuda')
    return parser.parse_args()


def main():
    args = parse_args()

    # ========================================
    # 1. Load config
    # ========================================
    print("📁 Đang tải cấu hình...")
    config = load_config(args.config)

    # Ghi đè config bằng command-line args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.model is not None:
        config['model']['type'] = args.model
    if args.device is not None:
        config['device'] = args.device

    # ========================================
    # 2. Setup
    # ========================================
    seed = config.get('seed', 42)
    set_seed(seed)
    device = get_device(config.get('device', 'auto'))

    print(f"🔧 Device: {device}")
    print(f"🔧 Seed: {seed}")
    print(f"🔧 Model: {config['model']['type']}")

    # ========================================
    # 3. Chuẩn bị dữ liệu
    # ========================================
    print("\n📊 Đang chuẩn bị dữ liệu...")
    try:
        data = prepare_data(config, verbose=True)
    except ValueError as e:
        print(f"\n❌ Lỗi: {e}")
        sys.exit(1)

    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']

    # ========================================
    # 4. Xây dựng model
    # ========================================
    print("\n🏗️ Đang xây dựng model...")
    model = build_model(config)
    n_params = count_parameters(model)
    print(f"  Model: {config['model']['type']}")
    print(f"  Tổng tham số: {n_params:,}")
    print(f"  Kiến trúc:\n{model}")

    # ========================================
    # 5. Huấn luyện
    # ========================================
    trainer = Trainer(model, config, device)
    history = trainer.train(train_loader, val_loader, verbose=True)

    # ========================================
    # 6. Đánh giá trên tập Test
    # ========================================
    print("\n🧪 Đánh giá trên tập Test...")
    results_dir = os.path.join(config['paths'].get('log_dir', 'logs'), 'results')
    metrics = quick_evaluate(model, test_loader, device,
                             verbose=True, save_dir=results_dir)

    # ========================================
    # 7. Vẽ biểu đồ training
    # ========================================
    ensure_dir(results_dir)
    plot_training_curves(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        title='Fall Detection Training',
        save_path=os.path.join(results_dir, 'training_curves.png')
    )
    print(f"\n📊 Biểu đồ huấn luyện đã lưu tại: {results_dir}")

    print("\n✅ Hoàn thành!")


if __name__ == '__main__':
    main()
