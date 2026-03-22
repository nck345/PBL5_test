"""
PBL5 - Fall Detection AI System
Module: trainer.py
Pipeline huấn luyện mô hình: training loop, validation, checkpoints, early stopping.
"""

import sys
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from tqdm import tqdm

from .utils import ensure_dir


# ============================================================
# 1. Trainer Class
# ============================================================

class Trainer:
    """
    Lớp quản lý toàn bộ quá trình huấn luyện mô hình.
    """

    def __init__(self, model: nn.Module, config: dict,
                 device: torch.device = None):
        """
        Args:
            model: Mô hình PyTorch
            config: Dict cấu hình
            device: Thiết bị tính toán
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cpu')

        # Di chuyển model sang device
        self.model.to(self.device)

        # Training config
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 50)
        self.lr = train_cfg.get('learning_rate', 0.001)
        self.weight_decay = train_cfg.get('weight_decay', 0.0001)

        # Tính toán pos_weight cho BCELoss (hoặc tự nhân manual trong vòng lặp)
        # Vì model đầu ra có Sigmoid, ta vẫn dụng BCELoss.
        # Nhưng thay vì xài parameter weight không có của BCELoss, ta dùng thuộc tính nội bộ.
        # Ở đây, BCELoss truyền thống cũng có param "weight" cho từng minibatch element.
        # Hoặc dùng weight cố định cho class = 1 (mặc dù nn.BCELoss không có pos_weight như BCEWithLogitsLoss)
        # Giải pháp tốt nhất: WeightedRandomSampler đã đổi data, loss ko nhất thiết cần pos_weight nữa.
        # Để an toàn, cứ giữ BCELoss nguyên bản vì sampler đã lo vụ balance 50/50 rồi.
        self.criterion = nn.BCELoss()

        # Optimizer
        optimizer_name = train_cfg.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            self.optimizer = Adam(model.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=self.lr,
                                momentum=0.9, weight_decay=self.weight_decay)
        else:
            self.optimizer = Adam(model.parameters(), lr=self.lr)

        # Learning rate scheduler
        sched_cfg = train_cfg.get('scheduler', {})
        sched_type = sched_cfg.get('type', 'reduce_on_plateau')
        if sched_type == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min',
                patience=sched_cfg.get('patience', 5),
                factor=sched_cfg.get('factor', 0.5)
            )
        elif sched_type == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif sched_type == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        else:
            self.scheduler = None

        # Early stopping
        es_cfg = train_cfg.get('early_stopping', {})
        self.early_stopping_enabled = es_cfg.get('enabled', True)
        self.es_patience = es_cfg.get('patience', 10)
        self.es_min_delta = es_cfg.get('min_delta', 0.001)

        # Paths
        paths_cfg = config.get('paths', {})
        self.checkpoint_dir = paths_cfg.get('checkpoint_dir', 'models/checkpoints')
        self.final_model_dir = paths_cfg.get('final_model_dir', 'models/final_model')

        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }

    def _train_one_epoch(self, train_loader, epoch: int = 0, verbose: bool = False) -> tuple:
        """Huấn luyện 1 epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Thay vì chèn đè lên tất cả, ta cho pbar chạy động và xóa khi xong
        pbar = tqdm(train_loader, desc=f"Train [{epoch}/{self.epochs}]", leave=False, dynamic_ncols=True) if verbose else train_loader

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * X_batch.size(0)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f"{running_loss/total:.3f}", 'acc': f"{100.0*correct/total:.1f}%"})

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        if verbose and isinstance(pbar, tqdm):
            pbar.close()
            
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def _validate(self, val_loader, epoch: int = 0, verbose: bool = False) -> tuple:
        """Đánh giá trên tập validation."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(val_loader, desc=f"Val   [{epoch}/{self.epochs}]", leave=False, dynamic_ncols=True) if verbose else val_loader

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            
            if verbose and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f"{running_loss/total:.3f}", 'acc': f"{100.0*correct/total:.1f}%"})

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        if verbose and isinstance(pbar, tqdm):
            pbar.close()
            
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, verbose: bool = True) -> dict:
        """
        Huấn luyện mô hình đầy đủ.

        Args:
            train_loader: DataLoader tập train
            val_loader: DataLoader tập validation
            verbose: In thông tin mỗi epoch

        Returns:
            dict history chứa loss và accuracy qua các epoch
        """
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.final_model_dir)

        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        start_time = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting training on {self.device}")
            print(f"Epochs: {self.epochs} | LR: {self.lr} | "
                  f"Early Stopping: {self.early_stopping_enabled}")
            print(f"{'='*60}\n")

        try:
            for epoch in range(1, self.epochs + 1):
                epoch_start = time.time()

                # Train
                train_loss, train_acc = self._train_one_epoch(train_loader, epoch, verbose)

                # Validate
                val_loss, val_acc = self._validate(val_loader, epoch, verbose)

                # Lưu history
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                self.history['lr'].append(current_lr)

                epoch_time = time.time() - epoch_start
                
                # Khóa tự in dòng mới liên tục, dùng carriage return (\r) thay thế
                if verbose:
                    # Xóa dòng cũ nếu có
                    sys.stdout.write('\033[K') 
                    print(f"Epoch [{epoch:3d}/{self.epochs}] "
                          f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
                          f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% "
                          f"| LR: {current_lr:.6f} "
                          f"| Time: {epoch_time:.1f}s")

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Save best model
                if val_loss < best_val_loss - self.es_min_delta:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0

                    # Save checkpoint
                    model_type = self.config.get('model', {}).get('type', 'model')
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f"{model_type}_epoch{epoch}.pt"
                    )
                    self._save_checkpoint(epoch, val_loss, val_acc, checkpoint_path)

                    if verbose:
                        print(f"  -> Saved best checkpoint (val_loss: {val_loss:.4f})")
                else:
                    patience_counter += 1

                # Early stopping
                if self.early_stopping_enabled and patience_counter >= self.es_patience:
                    if verbose:
                        print(f"\n! Early stopping at epoch {epoch} "
                              f"(patience={self.es_patience})")
                    break
        except KeyboardInterrupt:
            if verbose:
                print("\n! Training interrupted by user (KeyboardInterrupt). Evaluating best model so far...")

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Save final model
        final_path = os.path.join(self.final_model_dir, "fall_detection_model.pt")
        self._save_final_model(final_path)

        total_time = time.time() - start_time
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"Total time: {total_time:.1f}s")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            print(f"Model saved to: {final_path}")
            print(f"{'='*60}")

        return self.history

    def _save_checkpoint(self, epoch: int, val_loss: float,
                         val_acc: float, path: str):
        """Lưu checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'config': self.config,
        }, path)

    def _save_final_model(self, path: str):
        """Lưu model cuối cùng."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
        }, path)

    @staticmethod
    def load_model(model: nn.Module, path: str,
                   device: torch.device = None) -> tuple:
        """
        Tải model từ file.

        Args:
            model: Model architecture (chưa có weights)
            path: Đường dẫn file .pt
            device: Device

        Returns:
            (model, config, history)
        """
        if device is None:
            device = torch.device('cpu')

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        config = checkpoint.get('config', {})
        history = checkpoint.get('history', {})

        return model, config, history
