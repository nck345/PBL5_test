import os
import time
import copy
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

from .dataset import get_ensemble_subsets

class EnsembleTrainer:
    """
    Huấn luyện Stacked Ensemble LSTM theo phương pháp Heap Strategy (Reference Paper):
    1. Huấn luyện từng Base LSTM độc lập trên các subset (có sự giao nhau/overlap).
    2. Đóng băng trọng số Base LSTM.
    3. Huấn luyện Meta-Classifier trên toàn bộ tập train.
    """
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 50)
        self.lr = train_cfg.get('learning_rate', 0.001)
        self.batch_size = train_cfg.get('batch_size', 64)
        self.weight_decay = train_cfg.get('weight_decay', 0.0001)
        
        self.num_models = config.get('model', {}).get('ensemble', {}).get('num_models', 3)
        self.overlap_ratio = config.get('model', {}).get('ensemble', {}).get('overlap_ratio', 0.15)
        self.meta_train_ratio = config.get('model', {}).get('ensemble', {}).get('meta_train_ratio', 0.25)
        self.meta_epochs = config.get('model', {}).get('ensemble', {}).get('meta_epochs', self.epochs)
        self.meta_lr = config.get('model', {}).get('ensemble', {}).get('meta_learning_rate', 0.0005)
        
        # Paths
        paths_cfg = config.get('paths', {})
        self.checkpoint_dir = paths_cfg.get('checkpoint_dir', 'models/checkpoints')
        self.final_model_dir = paths_cfg.get('final_model_dir', 'models/final_model')
        
        # Optimizer selection logic
        self.optimizer_name = train_cfg.get('optimizer', 'adam')

        # Early stopping logic
        es_cfg = train_cfg.get('early_stopping', {})
        self.early_stopping_enabled = es_cfg.get('enabled', True)
        self.es_patience = es_cfg.get('patience', 10)
        self.es_min_delta = es_cfg.get('min_delta', 0.001)

        # Scheduler logic
        sched_cfg = train_cfg.get('scheduler', {})
        self.sched_type = sched_cfg.get('type', 'reduce_on_plateau')
        self.sched_patience = sched_cfg.get('patience', 5)
        self.sched_factor = sched_cfg.get('factor', 0.5)
        
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }

    def _create_optimizer_and_scheduler(self, parameters, is_meta=False):
        lr = self.meta_lr if is_meta else self.lr
        epochs = self.meta_epochs if is_meta else self.epochs
        
        if self.optimizer_name == 'adam':
            optimizer = Adam(parameters, lr=lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = SGD(parameters, lr=lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            optimizer = Adam(parameters, lr=lr)

        if self.sched_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer, mode='min',
                patience=self.sched_patience,
                factor=self.sched_factor
            )
        elif self.sched_type == 'step':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        elif self.sched_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None
        
        return optimizer, scheduler

    def train(self, global_train_loader, val_loader, verbose=True):
        train_dataset = global_train_loader.dataset
        
        # Tách tập train_dataset thành 2 tập: base_train_dataset (để huấn luyện Base Models) 
        # và meta_train_dataset (để huấn luyện Meta-Classifier). Khắc phục Data Leakage.
        total_len = len(train_dataset)
        indices = np.arange(total_len)
        np.random.seed(self.config.get('seed', 42))
        np.random.shuffle(indices)
        
        base_train_len = int((1.0 - self.meta_train_ratio) * total_len)
        base_indices = indices[:base_train_len]
        meta_indices = indices[base_train_len:]
        
        from torch.utils.data import Subset
        base_train_dataset = Subset(train_dataset, base_indices)
        meta_train_dataset = Subset(train_dataset, meta_indices)

        subsets = get_ensemble_subsets(base_train_dataset, self.num_models, self.overlap_ratio)
        
        # 1. Huấn luyện từng Base Model
        if verbose:
            print(f"\\n--- STEP 1: TRAINING {self.num_models} BASE MODELS (WITH {self.overlap_ratio*100}% OVERLAP) ---")
            print(f"Data split: Base Models ({len(base_indices)} samples), Meta-Classifier ({len(meta_indices)} samples)")
            
        for i, base_model in enumerate(self.model.base_models):
            if verbose:
                print(f"[Base Model {i+1}/{self.num_models}] Data size: {len(subsets[i])}")
                
            use_sampler = self.config.get('training', {}).get('use_sampler', True)
            if use_sampler:
                subset_indices = subsets[i].indices
                original_indices = [base_indices[idx] for idx in subset_indices]
                subset_y = train_dataset.y[original_indices].numpy()
                class_counts = [np.sum(subset_y == 0), np.sum(subset_y == 1)]
                total_samples = len(subset_y)
                class_weights = [total_samples / c if c > 0 else 0.0 for c in class_counts]
                sample_weights = [class_weights[int(label)] for label in subset_y]
                
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)
                loader = DataLoader(subsets[i], batch_size=self.batch_size, sampler=sampler)
            else:
                loader = DataLoader(subsets[i], batch_size=self.batch_size, shuffle=True)
                
            optimizer, _ = self._create_optimizer_and_scheduler(base_model.parameters())
            criterion = nn.BCELoss()
            
            try:
                for epoch in range(self.epochs):
                    base_model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    pbar = tqdm(loader, desc=f"Base {i+1} [{epoch+1}/{self.epochs}]", leave=False, dynamic_ncols=True) if verbose else loader
                    for X_batch, y_batch in pbar:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        optimizer.zero_grad()
                        out = base_model(X_batch)
                        loss = criterion(out, y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        running_loss += loss.item() * X_batch.size(0)
                        predicted = (out >= 0.5).float()
                        correct += (predicted == y_batch).sum().item()
                        total += y_batch.size(0)
                        
                        if verbose and isinstance(pbar, tqdm):
                            pbar.set_postfix({'loss': f"{running_loss/total:.3f}", 'acc': f"{100.0*correct/total:.1f}%"})
                    
                    if verbose and isinstance(pbar, tqdm):
                        pbar.close()
            except KeyboardInterrupt:
                if verbose:
                    print(f"\n! Canceled Base Model {i+1} training (KeyboardInterrupt).")
                    
        # 2. Đóng băng Base Models
        for param in self.model.base_models.parameters():
            param.requires_grad = False
            
        self.model.base_models.eval()
        
        # 3. Huấn luyện Meta Classifier
        if verbose:
            print(f"\n--- STEP 2: TRAINING META-CLASSIFIER ---")
            
        meta_optimizer, meta_scheduler = self._create_optimizer_and_scheduler(self.model.meta_classifier.parameters(), is_meta=True)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Sử dụng meta_train_dataset đã tách từ trước để huấn luyện Meta-Classifier
        # Tránh tình trạng meta-classifier bị "mù quáng" tin tưởng base models 
        use_sampler = self.config.get('training', {}).get('use_sampler', True)
        if use_sampler:
            meta_y = train_dataset.y[meta_indices].numpy()
            class_counts = [np.sum(meta_y == 0), np.sum(meta_y == 1)]
            total_samples = len(meta_y)
            class_weights = [total_samples / c if c > 0 else 0.0 for c in class_counts]
            sample_weights = [class_weights[int(label)] for label in meta_y]
            
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)
            meta_train_loader = DataLoader(meta_train_dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            meta_train_loader = DataLoader(meta_train_dataset, batch_size=self.batch_size, shuffle=True)
        
        start_time = time.time()
        try:
            for epoch in range(self.meta_epochs):
                # Train
                self.model.train() 
                self.model.base_models.eval() # Force base to stay eval
                running_loss, correct, total = 0.0, 0, 0
                
                pbar = tqdm(meta_train_loader, desc=f"Meta [{epoch+1}/{self.meta_epochs}]", leave=False, dynamic_ncols=True) if verbose else meta_train_loader
                for X_batch, y_batch in pbar:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    meta_optimizer.zero_grad()
                    out = self.model(X_batch)
                    loss = criterion(out, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.meta_classifier.parameters(), max_norm=1.0)
                    meta_optimizer.step()
                    
                    running_loss += loss.item() * X_batch.size(0)
                    pred = (out >= 0.5).float()
                    correct += (pred == y_batch).sum().item()
                    total += y_batch.size(0)
                    
                    if verbose and isinstance(pbar, tqdm):
                        pbar.set_postfix({'loss': f"{running_loss/total:.3f}", 'acc': f"{100.0*correct/total:.1f}%"})
                
                if verbose and isinstance(pbar, tqdm):
                    pbar.close()
                    
                train_loss = running_loss / total
                train_acc = 100.0 * correct / total
                
                # Val
                self.model.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                
                val_pbar = tqdm(val_loader, desc=f"Val [{epoch+1}/{self.meta_epochs}]", leave=False, dynamic_ncols=True) if verbose else val_loader
                with torch.no_grad():
                    for X_batch, y_batch in val_pbar:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        out = self.model(X_batch)
                        loss = criterion(out, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                        pred = (out >= 0.5).float()
                        val_correct += (pred == y_batch).sum().item()
                        val_total += y_batch.size(0)
                        
                        if verbose and isinstance(val_pbar, tqdm):
                            val_pbar.set_postfix({'loss': f"{val_loss/val_total:.3f}", 'acc': f"{100.0*val_correct/val_total:.1f}%"})
                            
                if verbose and isinstance(val_pbar, tqdm):
                    val_pbar.close()
                    
                val_loss /= val_total
                val_acc = 100.0 * val_correct / val_total
                
                current_lr = meta_optimizer.param_groups[0]['lr']
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['lr'].append(current_lr)
                
                if verbose:
                    import sys
                    sys.stdout.write('\033[K') 
                    print(f"Meta-Epoch [{epoch+1:2d}/{self.meta_epochs}] "
                          f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
                          f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% "
                          f"| LR: {current_lr:.6f}")
                          
                if meta_scheduler is not None:
                    if isinstance(meta_scheduler, ReduceLROnPlateau):
                        meta_scheduler.step(val_loss)
                    else:
                        meta_scheduler.step()

                if val_loss < best_val_loss - self.es_min_delta:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0

                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                    model_type = self.config.get('model', {}).get('type', 'model')
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"{model_type}_epoch{epoch+1}.pt")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': meta_optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'config': self.config,
                    }, checkpoint_path)
                    
                    if verbose:
                        print(f"  -> Saved best checkpoint (val_loss: {val_loss:.4f})")
                else:
                    patience_counter += 1

                # Early stopping
                if self.early_stopping_enabled and patience_counter >= self.es_patience:
                    if verbose:
                        print(f"\n! Early stopping at epoch {epoch+1} "
                              f"(patience={self.es_patience})")
                    break

        except KeyboardInterrupt:
            if verbose:
                print("\n! Meta-classifier training interrupted (KeyboardInterrupt). Evaluating best model so far...")
                
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Save final model
        model_name = self.config.get('model', {}).get('type', 'model')
        final_path = os.path.join(self.final_model_dir, f"{model_name}.pth")
        if not os.path.exists(self.final_model_dir):
            os.makedirs(self.final_model_dir)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history,
        }, final_path)

        if verbose:
            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"Total time: {total_time:.1f}s")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            print(f"Model saved to: {final_path}")
            print(f"{'='*60}")
            
        return self.history
