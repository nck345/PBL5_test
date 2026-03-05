"""
PBL5 - Fall Detection AI System
Module: evaluator.py
Đánh giá hiệu năng mô hình: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from .utils import plot_confusion_matrix, plot_roc_curve


# ============================================================
# 1. Evaluator Class
# ============================================================

class Evaluator:
    """
    Lớp đánh giá hiệu năng mô hình phát hiện té ngã.
    """

    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Args:
            model: Mô hình đã huấn luyện
            device: Thiết bị tính toán
        """
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, data_loader) -> tuple:
        """
        Dự đoán trên toàn bộ DataLoader.

        Args:
            data_loader: DataLoader cần dự đoán

        Returns:
            (y_true, y_pred, y_proba):
                - y_true: Nhãn thực (numpy)
                - y_pred: Nhãn dự đoán (0/1, numpy)
                - y_proba: Xác suất dự đoán (numpy)
        """
        all_true = []
        all_proba = []

        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(self.device)
            outputs = self.model(X_batch)

            all_true.append(y_batch.numpy())
            all_proba.append(outputs.cpu().numpy())

        y_true = np.concatenate(all_true)
        y_proba = np.concatenate(all_proba)
        y_pred = (y_proba >= 0.5).astype(int)

        return y_true, y_pred, y_proba

    def evaluate(self, data_loader, verbose: bool = True,
                 save_dir: str = None) -> dict:
        """
        Đánh giá đầy đủ trên một tập dữ liệu.

        Args:
            data_loader: DataLoader tập đánh giá
            verbose: In kết quả
            save_dir: Thư mục lưu biểu đồ (nếu có)

        Returns:
            dict chứa các metrics
        """
        y_true, y_pred, y_proba = self.predict(data_loader)

        # Tính metrics
        metrics = self._compute_metrics(y_true, y_pred, y_proba)

        if verbose:
            self._print_report(metrics, y_true, y_pred)

        # Vẽ biểu đồ
        if save_dir:
            self._save_plots(y_true, y_pred, y_proba, metrics, save_dir)

        return metrics

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray) -> dict:
        """Tính tất cả metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # ROC & AUC
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)
            metrics['auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
            metrics['thresholds'] = thresholds
        except Exception:
            metrics['auc'] = 0.0
            metrics['fpr'] = np.array([0, 1])
            metrics['tpr'] = np.array([0, 1])

        # Specificity (True Negative Rate)
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel() \
            if metrics['confusion_matrix'].size == 4 \
            else (0, 0, 0, 0)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Sample counts
        metrics['total_samples'] = len(y_true)
        metrics['n_positive'] = int(np.sum(y_true == 1))
        metrics['n_negative'] = int(np.sum(y_true == 0))

        return metrics

    def _print_report(self, metrics: dict, y_true: np.ndarray,
                      y_pred: np.ndarray):
        """In báo cáo đánh giá."""
        print("\n" + "=" * 55)
        print("           KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
        print("=" * 55)

        print(f"\n📊 Tổng mẫu: {metrics['total_samples']}")
        print(f"   ADL (0): {metrics['n_negative']} | "
              f"Fall (1): {metrics['n_positive']}")

        print(f"\n📈 Các Metrics:")
        print(f"   Accuracy:    {metrics['accuracy']:.4f} "
              f"({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision:   {metrics['precision']:.4f}")
        print(f"   Recall:      {metrics['recall']:.4f}")
        print(f"   F1-Score:    {metrics['f1_score']:.4f}")
        print(f"   Specificity: {metrics['specificity']:.4f}")
        print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"   AUC:         {metrics['auc']:.4f}")

        print(f"\n📋 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"   {'':>12} Predicted")
        print(f"   {'':>12} {'ADL':>7} {'Fall':>7}")
        if cm.size == 4:
            print(f"   Actual ADL  {cm[0][0]:>7} {cm[0][1]:>7}")
            print(f"   Actual Fall {cm[1][0]:>7} {cm[1][1]:>7}")
        else:
            print(f"   {cm}")

        print(f"\n📄 Classification Report:")
        class_names = ['ADL', 'Fall']
        # Chỉ hiểm report nếu có đủ 2 class
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        if len(unique_labels) > 1:
            report = classification_report(y_true, y_pred,
                                           target_names=class_names,
                                           zero_division=0)
        else:
            report = classification_report(y_true, y_pred,
                                           zero_division=0)
        print(report)
        print("=" * 55)

    def _save_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray, metrics: dict, save_dir: str):
        """Lưu biểu đồ đánh giá."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Confusion Matrix
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=['ADL', 'Fall'],
            title='Confusion Matrix - Fall Detection',
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )

        # ROC Curve
        if 'fpr' in metrics and 'tpr' in metrics:
            plot_roc_curve(
                metrics['fpr'], metrics['tpr'], metrics['auc'],
                title='ROC Curve - Fall Detection',
                save_path=os.path.join(save_dir, 'roc_curve.png')
            )

        print(f"  📊 Biểu đồ đã lưu tại: {save_dir}")


# ============================================================
# 2. Quick Evaluation Function
# ============================================================

def quick_evaluate(model: nn.Module, data_loader,
                   device: torch.device = None,
                   verbose: bool = True, save_dir: str = None) -> dict:
    """
    Hàm đánh giá nhanh.

    Args:
        model: Mô hình đã huấn luyện
        data_loader: DataLoader
        device: Device
        verbose: In kết quả
        save_dir: Lưu biểu đồ

    Returns:
        dict metrics
    """
    evaluator = Evaluator(model, device)
    return evaluator.evaluate(data_loader, verbose=verbose, save_dir=save_dir)
