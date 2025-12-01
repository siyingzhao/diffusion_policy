import torch
import torch.nn as nn

class REPAProjectionHead(nn.Module):
    """
    REPA-style projection head for alignment loss.
    Projects mid-level features to displacement space.
    """
    def __init__(
        self, 
        input_dim: int,          # UNet mid_feature 的通道数 (e.g., 1024)
        output_dim: int,         # displacement 维度 (e.g., action_dim)
        hidden_dims: list = [512, 256], # MLP 隐藏层维度
        activation: str = 'relu'
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]  # 默认两层隐藏层
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'mish':
                layers.append(nn.Mish())
            layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        # 最后一层不加激活函数
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, mid_feature):
        """
        Args:
            mid_feature: (B, C, T) - UNet 中间特征
        Returns:
            projected: (B, output_dim) - 投影到 displacement 空间
        """
        # Global pooling: (B, C, T) -> (B, C)
        pooled = mid_feature.mean(dim=-1)  # 时间维度平均池化
        
        # MLP projection: (B, C) -> (B, output_dim)
        projected = self.mlp(pooled)
        
        return projected