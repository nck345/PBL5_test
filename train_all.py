"""
PBL5 - Fall Detection AI System
Script: train_all.py
Chạy huấn luyện tuần tự tất cả các mô hình và xuất biểu đồ so sánh chung.

Usage:
    python train_all.py
    python train_all.py --epochs 15
"""

import argparse
import sys
import os
import copy
import torch

from src.utils import load_config, set_seed, get_device, count_parameters, \
    plot_combined_training_curves, plot_combined_roc_curves, \
    plot_combined_metrics_comparison, ensure_dir
from src.dataset import prepare_data
from src.architecture import build_model
from src.trainer import Trainer
from src.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description='PBL5 - Huấn luyện benchmark tất cả các mô hình'
    )
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Đường dẫn file cấu hình YAML')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Số epochs (ghi đè config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (ghi đè config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (ghi đè config)')
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
    base_config = load_config(args.config)

    # Ghi đè config bằng command-line args
    if args.epochs is not None:
        base_config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        base_config['training']['batch_size'] = args.batch_size
    if args.lr is not None:
        base_config['training']['learning_rate'] = args.lr
    if args.device is not None:
        base_config['device'] = args.device
    if args.no_es:
        if 'training' in base_config and 'early_stopping' in base_config['training']:
            base_config['training']['early_stopping']['enabled'] = False

    # ========================================
    # 2. Setup
    # ========================================
    seed = base_config.get('seed', 42)
    set_seed(seed)
    device = get_device(base_config.get('device', 'auto'))

    print(f"🔧 Device: {device}")
    print(f"🔧 Seed: {seed}")

    # ========================================
    # 3. Chuẩn bị dữ liệu chung
    # ========================================
    print("\n📊 Đang chuẩn bị dữ liệu chung cho tất cả model...")
    try:
        data = prepare_data(base_config, verbose=False)
    except ValueError as e:
        print(f"\n❌ Lỗi: {e}")
        sys.exit(1)

    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    
    # Chuẩn bị biến chứa kết quả gộp
    all_histories = {}
    all_roc_data = {}
    all_metrics = {}
    
    models_to_train = ['lstm', 'stacked_lstm', 'bilstm_cnn']
    
    print(f"\n🚀 BẮT ĐẦU CHUỖI HUẤN LUYỆN: {models_to_train}\n")
    print("="*60)

    # ========================================
    # 4. Vòng lặp Train từng Model
    # ========================================
    results_dir = os.path.join(base_config['paths'].get('log_dir', 'logs'), 'results')
    ensure_dir(results_dir)
    final_model_dir = base_config['paths'].get('final_model_dir', 'models/final_model')
    ensure_dir(final_model_dir)

    for current_model in models_to_train:
        print(f"\n▶️ ĐANG HUẤN LUYỆN MODEL: {current_model.upper()}")
        
        # Sửa cấu hình để Load đúng Model
        config = copy.deepcopy(base_config)
        config['model']['type'] = current_model
        
        # Build Model
        model = build_model(config)
        n_params = count_parameters(model)
        print(f"  Tổng tham số: {n_params:,}")
        
        # Train
        trainer = Trainer(model, config, device)
        history = trainer.train(train_loader, val_loader, verbose=True)
        all_histories[current_model] = history
        
        # Riêng lưu weights thì phải custom tên
        model_save_path = os.path.join(final_model_dir, f"{current_model}.pt")
        # Do trainer.train tự lưu vào fall_detection_model.pt cuối cùng, ta cần rename 
        temp_save_path = os.path.join(final_model_dir, "fall_detection_model.pt")
        if os.path.exists(temp_save_path):
            if os.path.exists(model_save_path):
                os.remove(model_save_path)
            os.rename(temp_save_path, model_save_path)

        # Cập nhật state_dict tốt nhất vào model để Test
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Đánh giá Model trên tập Test để lấy ROC curve
        print(f"\n🧪 Đánh giá {current_model.upper()} trên tập Test...")
        evaluator = Evaluator(model, device)
        y_true, y_pred, y_proba = evaluator.predict(test_loader)
        metrics = evaluator._compute_metrics(y_true, y_pred, y_proba)
        
        # Log nhanh kết quả Test
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Test F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Test AUC:      {metrics['auc']:.4f}")
        
        # Lưu lại params ROC
        all_roc_data[current_model] = {
            'fpr': metrics['fpr'],
            'tpr': metrics['tpr'],
            'auc': metrics['auc']
        }
        
        # Lưu lại các metrics khác để vẽ Bar Chart
        all_metrics[current_model] = {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }
        
        print("="*60)
        
    # ========================================
    # 5. Vẽ biểu đồ so sánh gộp
    # ========================================
    print("\n📈 Đang vẽ biểu đồ so sánh chung...")
    
    plot_combined_training_curves(
        all_histories,
        title="So sánh Huấn luyện (Validation)",
        save_path=os.path.join(results_dir, 'combined_training_curves.png')
    )
    
    plot_combined_roc_curves(
        all_roc_data,
        title="So sánh ROC/AUC trên tập Test",
        save_path=os.path.join(results_dir, 'combined_roc_curves.png')
    )
    
    plot_combined_metrics_comparison(
        all_metrics,
        title="So sánh Precision, Recall, F1-Score",
        save_path=os.path.join(results_dir, 'model_metrics_comparison.png')
    )
    
    print(f"\n✅ ĐÃ HOÀN TẤT CHUỖI BENCHMARK.")
    print(f"📁 Biểu đồ so sánh   : {results_dir}")
    print(f"📁 Các file trọng số : {final_model_dir}")


if __name__ == '__main__':
    main()
