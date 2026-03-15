"""
PBL5 - Fall Detection AI System
Script: generate_comparison_chart.py
Tạo biểu đồ cột so sánh các mô hình đã huấn luyện.
"""

import os
import torch
import copy
from src.utils import load_config, get_device, plot_combined_metrics_comparison, ensure_dir
from src.dataset import prepare_data
from src.architecture import build_model
from src.evaluator import Evaluator

def main():
    # 1. Load config
    config = load_config('configs/config.yaml')
    device = get_device('cpu') # Chạy trên CPU cho nhẹ
    
    # 2. Chuẩn bị dữ liệu
    print("📊 Đang chuẩn bị dữ liệu Test...")
    data = prepare_data(config, verbose=False)
    test_loader = data['test_loader']
    
    # 3. Danh sách model cần kiểm tra
    models_to_compare = ['lstm', 'stacked_lstm', 'bilstm_cnn']
    model_dir = config['paths'].get('final_model_dir', 'models/final_model')
    results_dir = os.path.join(config['paths'].get('log_dir', 'logs'), 'results')
    ensure_dir(results_dir)
    
    all_metrics = {}
    
    print(f"\n🚀 Đang đánh giá các model trong: {model_dir}")
    print("-" * 50)
    
    for m_type in models_to_compare:
        model_path = os.path.join(model_dir, f"{m_type}.pt")
        
        if not os.path.exists(model_path):
            print(f"⚠️ Không tìm thấy file trọng số cho: {m_type}")
            continue
            
        print(f"▶️ Đang đánh giá: {m_type.upper()}")
        
        # Build model
        temp_config = copy.deepcopy(config)
        temp_config['model']['type'] = m_type
        model = build_model(temp_config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Evaluate
        evaluator = Evaluator(model, device)
        y_true, y_pred, y_proba = evaluator.predict(test_loader)
        metrics = evaluator._compute_metrics(y_true, y_pred, y_proba)
        
        all_metrics[m_type] = {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }
        
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    
    # 4. Vẽ biểu đồ
    if all_metrics:
        save_path = os.path.join(results_dir, 'model_metrics_comparison_current.png')
        plot_combined_metrics_comparison(
            all_metrics,
            title="So sánh Precision, Recall, F1-Score (Tất cả Model)",
            save_path=save_path
        )
        print("-" * 50)
        print(f"✅ Đã tạo xong biểu đồ cột so sánh!")
        print(f"📁 Đường dẫn: {save_path}")
    else:
        print("\n❌ Không có model nào được đánh giá thành công.")

if __name__ == '__main__':
    main()
