"""
基础分类器模型
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe, persistent=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        if x.size(1) > self.pe.size(1):
            length, width = x.size(1), x.size(2)
            position = torch.arange(length, device=x.device, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, width, 2, device=x.device).float() * (-math.log(10000.0) / width))
            pe = torch.zeros(length, width, device=x.device)
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)[:, :width // 2]
            self.pe = pe.unsqueeze(0)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BaseClassifier(nn.Module, ABC):
    """基础分类器抽象类"""
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        pf_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pf_dim = pf_dim
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.device = device
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len, dropout)
        
        # 输入投影（如果 embed_dim != hidden_dim）
        if embed_dim != hidden_dim:
            self.input_projection = nn.Linear(embed_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    @abstractmethod
    def encode(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码输入序列
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            encoded: [batch_size, seq_len, hidden_dim]
            kl_div: KL 散度损失（如果适用）
        """
        pass
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            logits: [batch_size, num_classes]
            kl_div: KL 散度损失
        """
        # 编码
        encoded, kl_div = self.encode(input_ids, attention_mask)
        
        # 池化（使用 [CLS] token 或平均池化）
        if attention_mask is not None:
            # 使用 attention mask 进行平均池化
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = encoded.mean(dim=1)
        
        # 分类
        logits = self.classifier(pooled)
        
        return logits, kl_div
    
    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionwiseFeedforward(nn.Module):
    """位置前馈网络"""
    
    def __init__(self, hidden_dim: int, pf_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(hidden_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, hidden_dim]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

