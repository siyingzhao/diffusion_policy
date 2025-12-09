import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention1D(nn.Module):
    """
    Cross Attention 模块，用于将外部条件（如 displacement）注入 UNet 特征
    
    Query: 来自 UNet 的特征 (B, C, T)
    Key/Value: 来自外部条件（如 displacement）(B, cond_dim)
    """
    def __init__(
        self,
        query_dim: int,          # UNet 特征的通道数
        cond_dim: int,           # 条件的维度（如 displacement 的维度）
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Query 来自 UNet 特征
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # Key, Value 来自条件（displacement）
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)
        
        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, cond):
        """
        Args:
            x: UNet 特征 (B, C, T) - Query
            cond: 条件向量 (B, cond_dim) - Key/Value
        Returns:
            out: 融合后的特征 (B, C, T)
        """
        B, C, T = x.shape
        
        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        
        # 扩展 cond 为序列形式 (B, cond_dim) -> (B, 1, cond_dim)
        cond = cond.unsqueeze(1)
        
        # 计算 Q, K, V
        q = self.to_q(x)  # (B, T, inner_dim)
        k = self.to_k(cond)  # (B, 1, inner_dim)
        v = self.to_v(cond)  # (B, 1, inner_dim)
        
        # 多头注意力: reshape to (B, num_heads, seq_len, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, T, 1)
        attn = F.softmax(attn, dim=-1)
        
        # 加权求和
        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)
        
        # 合并多头
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1)  # (B, T, inner_dim)
        
        # 输出投影
        out = self.to_out(out)  # (B, T, C)
        
        # (B, T, C) -> (B, C, T)
        out = out.permute(0, 2, 1)
        
        return out


class CrossAttentionBlock1D(nn.Module):
    """
    带残差连接和 LayerNorm 的 Cross Attention 块
    """
    def __init__(
        self,
        query_dim: int,
        cond_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm = nn.GroupNorm(8, query_dim)
        self.cross_attn = CrossAttention1D(
            query_dim=query_dim,
            cond_dim=cond_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
    def forward(self, x, cond):
        """
        Args:
            x: (B, C, T)
            cond: (B, cond_dim)
        Returns:
            out: (B, C, T)
        """
        # 残差连接
        residual = x
        x = self.norm(x)
        x = self.cross_attn(x, cond)
        return x + residual
