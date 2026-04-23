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

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import os

from src.utils import load_config, set_seed, get_device, count_parameters, \
    plot_training_curves, ensure_dir
from src.dataset import prepare_data
from src.architecture import build_model
from src.trainer import Trainer
from src.ensemble_trainer import EnsembleTrainer
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
                        choices=['lstm', 'stacked_lstm', 'ensemble'],
                        help='Loại model (ghi đè config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: auto, cpu, cuda')
    parser.add_argument('--no-es', action='store_true',
                        help='Tắt tính năng Early Stopping (cho chạy hết epochs)')
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
    if args.no_es:
        if 'training' in config and 'early_stopping' in config['training']:
            config['training']['early_stopping']['enabled'] = False

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
    if config['model']['type'] == 'ensemble':
        trainer = EnsembleTrainer(model, config, device)
    else:
        trainer = Trainer(model, config, device)
    history = trainer.train(train_loader, val_loader, verbose=True)

    # ========================================
    # 6. Đánh giá trên tập Test
    # ========================================
    print("\n🧪 Đánh giá trên tập Test...")
    results_dir = os.path.join(config['paths'].get('log_dir', 'logs'), 'results')
    eval_cfg = config.get('evaluation', {})
    threshold_mode = eval_cfg.get('threshold_mode', 'fixed')
    fixed_threshold = eval_cfg.get('threshold', config['model'].get('threshold', 0.5))
    calibration_loader = val_loader if threshold_mode == 'val_calibrated' else None
    metrics = quick_evaluate(
        model, test_loader, device,
        verbose=True, save_dir=results_dir,
        optimize_threshold=False,
        threshold=fixed_threshold,
        calibration_loader=calibration_loader
    )

    # ========================================
    # 7. Vẽ biểu đồ training riêng biệt
    # ========================================
    ensure_dir(results_dir)
    model_name = config['model']['type']
    
    plot_training_curves(
        history['train_loss'], history['val_loss'],
        history['train_acc'], history['val_acc'],
        title=f'{model_name.upper()} Training',
        save_path=os.path.join(results_dir, f'{model_name}_training_curves.png')
    )
    print(f"\n📊 Biểu đồ huấn luyện riêng biệt đã lưu tại: {results_dir}")

    # ========================================
    # 8. Cập nhật và vẽ biểu đồ so sánh chung
    # ========================================
    print("\n🔄 Đang cập nhật biểu đồ so sánh chung...")
    import pickle
    from src.utils import plot_combined_training_curves, plot_combined_roc_curves, plot_combined_metrics_comparison
    
    combined_data_path = os.path.join(results_dir, 'combined_benchmark_data.pkl')
    
    if os.path.exists(combined_data_path):
        with open(combined_data_path, 'rb') as f:
            all_data = pickle.load(f)
    else:
        all_data = {'histories': {}, 'roc_data': {}, 'metrics': {}}
        
    all_data['histories'][model_name] = history
    all_data['roc_data'][model_name] = {
        'fpr': metrics.get('fpr', []),
        'tpr': metrics.get('tpr', []),
        'auc': metrics.get('auc', 0.0)
    }
    all_data['metrics'][model_name] = {
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'f1_score': metrics.get('f1_score', 0.0)
    }
    
    with open(combined_data_path, 'wb') as f:
        pickle.dump(all_data, f)
        
    plot_combined_training_curves(all_data['histories'], title="Training Comparison", save_path=os.path.join(results_dir, 'combined_training_curves.png'))
    plot_combined_roc_curves(all_data['roc_data'], title="ROC/AUC Comparison", save_path=os.path.join(results_dir, 'combined_roc_curves.png'))
    plot_combined_metrics_comparison(all_data['metrics'], title="Metrics Comparison", save_path=os.path.join(results_dir, 'model_metrics_comparison.png'))
    
    print("✅ Biểu đồ so sánh chung đã được cập nhật thành công!")
    print("\n✅ Hoàn thành!")


if __name__ == '__main__':
    main()
