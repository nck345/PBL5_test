import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Binary Focal Loss
    Tập trung huấn luyện các mẫu khó và giải quyết mất cân bằng phân lớp.
    Thay vì BCE phạt đều, Focal loss giảm nhẹ hình phạt cho các mẫu đã được 
    phân loại đúng một cách chắc chắn, ép mô hình học cách phân biệt các mẫu khó.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Trọng số ưu tiên cho class Positive (Fall = 1). Phân phối Fall hiếm nên gán alpha cao.
                   (Với nhãn ADL=0, trọng số là 1-alpha).
            gamma: Mức độ "ép" tập trung vào class khó. gamma=0 sẽ tương đương BCE thông thường.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Giới hạn inputs để tránh tính log(0)
        epsilon = 1e-7
        inputs = torch.clamp(inputs, epsilon, 1.0 - epsilon)
        
        # BCE thông thường
        bce = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        
        # Xác suất dự đoán đúng thực tế p_t
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        
        # Tính trọng số alpha_t cho focal loss
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Công thức Focal Loss
        focal_loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
