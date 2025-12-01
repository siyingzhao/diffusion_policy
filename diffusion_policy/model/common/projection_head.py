import torch
import torch.nn as nn

class REPAProjectionHead(nn.Module):
    """
    REPA-style projection head for alignment loss.
    Projects displacement to latent space (reverse direction).
    """
    def __init__(
        self, 
        input_dim: int,          # displacement 维度 (action_dim)
        output_dim: int,         # UNet mid_feature 的通道数 (e.g., 1024)
        hidden_dims: list = [256, 512], # MLP 隐藏层维度
        activation: str = 'relu'
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512]  # 默认两层隐藏层
        
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
    
    def forward(self, displacement):
        """
        Args:
            displacement: (B, action_dim) - 动作位移向量
        Returns:
            projected_latent: (B, output_dim) - 投影到 latent space
        """
        # MLP projection: (B, action_dim) -> (B, latent_dim)
        projected_latent = self.mlp(displacement)
        
        return projected_latent