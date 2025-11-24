import torch
import torch.nn as nn

class REPAProjectionHead(nn.Module):
    """
    REPA-style projection head for alignment loss.
    Projects mid-level features to displacement space.
    """
    def __init__(
        self, 
        input_dim: int,          # displacement 维度 (e.g., action_dim)
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
            
            # 建议顺序: Linear -> LayerNorm -> Activation (Pre-Activation 风格)
            # 或者: Linear -> Activation -> LayerNorm (这也是常见的)
            # 这里保持你原本的风格，但建议把 LN 放在激活函数后面更稳定
            layers.append(nn.LayerNorm(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'mish':
                layers.append(nn.Mish())
            
            prev_dim = hidden_dim
        
        # 最后一层投射到目标特征维度
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 初始化权重 (可选，有助于收敛)
        self.apply(self._init_weights)
        self.mlp = nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, displacement):
        """
        Args:
            displacement: (B, action_dim) - 真实的位移向量
        Returns:
            projected: (B, mid_channels) - 投影后的特征向量，用于和 UNet 特征做 Loss
        """
        # 输入已经是向量，直接过 MLP
        projected = self.mlp(displacement)
        return projected