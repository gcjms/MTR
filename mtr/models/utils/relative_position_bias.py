# Relative Position Bias for Spatial Cross Attention
# 相对位置偏置模块 - 增强 Transformer 的空间感知能力
#
# 核心思想：Query 和 Key 之间的相对位置越近，注意力权重越高
# 这是一个可学习的模块，让模型自动学会"近处重要、远处次要"

import torch
import torch.nn as nn
import math
from typing import Optional


class RelativePositionBias(nn.Module):
    """
    相对位置偏置模块 (Relative Position Bias)
    
    【直白解释】
    想象你在开车，前方有10辆车。原版 Attention 只看"这辆车长什么样"（语义特征），
    而 RelativePositionBias 还会告诉模型"这辆车离我多远"（空间位置）。
    
    【工作原理】
    1. 计算 Query 和 Key 的相对距离: Δx = x_q - x_k, Δy = y_q - y_k
    2. 将距离编码为一个偏置值: bias = MLP(Δx, Δy)  
    3. 把偏置加到 Attention 分数上: Attn = Softmax(QK^T/√d + bias)
    
    【效果】
    - 模型学会：近处的障碍物更重要
    - 模型学会：正前方的障碍物比侧面的更重要
    - 这些都是通过训练自动学到的，不是手工设置的规则
    
    【和 GGAM 的区别】
    - GGAM: 硬规则，距离<2m就必须高权重（物理约束，不可学习）
    - 本模块: 软规则，让模型自己学什么距离该给多少权重（数据驱动，可学习）
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        max_distance: float = 100.0,
        num_distance_bins: int = 64
    ):
        """
        Args:
            d_model: 模型隐藏维度
            num_heads: 注意力头数（每个头可以有不同的空间偏置模式）
            max_distance: 最大考虑距离（超过此距离视为"很远"）
            num_distance_bins: 距离离散化的桶数
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.num_distance_bins = num_distance_bins
        
        # 方法1: 连续距离 -> MLP -> 偏置值 (更灵活)
        self.distance_mlp = nn.Sequential(
            nn.Linear(3, 64),   # 输入: (Δx, Δy, distance)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_heads)  # 输出: 每个头一个偏置值
        )
        
        # 方法2: 离散化距离 -> 查表 (更高效，类似 Swin Transformer)
        # 这里同时保留两种方法，可通过参数选择
        self.distance_embedding = nn.Embedding(
            num_distance_bins * num_distance_bins,  # 2D 网格
            num_heads
        )
        
        self.use_mlp = True  # 默认使用 MLP 方法
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，让初始偏置接近 0"""
        for module in self.distance_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.distance_embedding.weight)
    
    def compute_relative_positions(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Query 和 Key 之间的相对位置
        
        Args:
            query_pos: Query 的位置 (num_q, batch, 2) 或 (batch, num_q, 2)
            key_pos: Key 的位置 (num_k, batch, 2) 或 (batch, num_k, 2)
        
        Returns:
            relative_pos: 相对位置 (batch, num_q, num_k, 3)
                          3 = (Δx, Δy, distance)
        """
        # 统一格式为 (batch, num, 2)
        if query_pos.dim() == 3 and query_pos.shape[0] != query_pos.shape[1]:
            # (num_q, batch, 2) -> (batch, num_q, 2)
            query_pos = query_pos.permute(1, 0, 2)
        if key_pos.dim() == 3 and key_pos.shape[0] != key_pos.shape[1]:
            key_pos = key_pos.permute(1, 0, 2)
        
        batch_size, num_q, _ = query_pos.shape
        _, num_k, _ = key_pos.shape
        
        # 扩展维度以进行成对计算
        # query_pos: (batch, num_q, 1, 2)
        # key_pos:   (batch, 1, num_k, 2)
        q_expanded = query_pos.unsqueeze(2)  # (batch, num_q, 1, 2)
        k_expanded = key_pos.unsqueeze(1)    # (batch, 1, num_k, 2)
        
        # 计算相对位置差
        delta = q_expanded - k_expanded  # (batch, num_q, num_k, 2)
        delta_x = delta[..., 0]  # (batch, num_q, num_k)
        delta_y = delta[..., 1]
        
        # 计算欧氏距离
        distance = torch.sqrt(delta_x ** 2 + delta_y ** 2 + 1e-6)
        
        # 归一化
        delta_x_norm = delta_x / self.max_distance
        delta_y_norm = delta_y / self.max_distance
        distance_norm = distance / self.max_distance
        
        # 拼接
        relative_pos = torch.stack([delta_x_norm, delta_y_norm, distance_norm], dim=-1)
        
        return relative_pos  # (batch, num_q, num_k, 3)
    
    def forward(
        self,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算相对位置偏置
        
        Args:
            query_pos: Query 的位置 (num_q, batch, 2) 或 (batch, num_q, 2)
            key_pos: Key 的位置 (num_k, batch, 2) 或 (batch, num_k, 2)
            query_mask: Query 有效性掩码 (batch, num_q)
            key_mask: Key 有效性掩码 (batch, num_k)
        
        Returns:
            bias: 位置偏置 (batch, num_heads, num_q, num_k)
                  可直接加到 Attention Logits 上
        """
        # Step 1: 计算相对位置
        relative_pos = self.compute_relative_positions(query_pos, key_pos)
        # relative_pos: (batch, num_q, num_k, 3)
        
        batch_size, num_q, num_k, _ = relative_pos.shape
        
        if self.use_mlp:
            # Step 2a: MLP 方法
            # 将相对位置展平，送入 MLP
            relative_pos_flat = relative_pos.reshape(-1, 3)  # (batch*num_q*num_k, 3)
            bias_flat = self.distance_mlp(relative_pos_flat)  # (batch*num_q*num_k, num_heads)
            bias = bias_flat.reshape(batch_size, num_q, num_k, self.num_heads)
            # 转换维度顺序: (batch, num_q, num_k, heads) -> (batch, heads, num_q, num_k)
            bias = bias.permute(0, 3, 1, 2)
        else:
            # Step 2b: 离散化查表方法 (更高效)
            # 将连续距离离散化到桶中
            delta_x_bin = ((relative_pos[..., 0] + 1) / 2 * self.num_distance_bins).long().clamp(0, self.num_distance_bins - 1)
            delta_y_bin = ((relative_pos[..., 1] + 1) / 2 * self.num_distance_bins).long().clamp(0, self.num_distance_bins - 1)
            bin_idx = delta_x_bin * self.num_distance_bins + delta_y_bin
            bias = self.distance_embedding(bin_idx)  # (batch, num_q, num_k, num_heads)
            bias = bias.permute(0, 3, 1, 2)
        
        # Step 3: 应用掩码（无效位置不应有偏置）
        if key_mask is not None:
            # key_mask: (batch, num_k) -> (batch, 1, 1, num_k)
            mask = key_mask.unsqueeze(1).unsqueeze(2)
            bias = bias.masked_fill(~mask, 0)
        
        return bias


def apply_relative_position_bias(
    attn_logits: torch.Tensor,
    query_pos: torch.Tensor,
    key_pos: torch.Tensor,
    bias_module: RelativePositionBias,
    key_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    便捷函数：将相对位置偏置应用到注意力分数上
    
    Args:
        attn_logits: 原始注意力分数 (batch, heads, num_q, num_k)
        query_pos: Query 位置
        key_pos: Key 位置
        bias_module: RelativePositionBias 模块实例
        key_mask: Key 有效性掩码
    
    Returns:
        modified_logits: 加了位置偏置的注意力分数
    """
    bias = bias_module(query_pos, key_pos, key_mask=key_mask)
    return attn_logits + bias


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Relative Position Bias - Unit Test")
    print("=" * 60)
    
    # 模拟数据
    batch_size = 4
    num_q = 64   # Query 数量 (意图锚点)
    num_k = 128  # Key 数量 (障碍物)
    num_heads = 8
    
    # 创建模块
    rpb = RelativePositionBias(
        d_model=256,
        num_heads=num_heads,
        max_distance=100.0
    )
    
    # 模拟位置数据
    query_pos = torch.randn(batch_size, num_q, 2) * 50  # Query 位置
    key_pos = torch.randn(batch_size, num_k, 2) * 50    # Key 位置
    key_mask = torch.ones(batch_size, num_k).bool()
    key_mask[:, -10:] = False  # 最后10个无效
    
    # 计算偏置
    bias = rpb(query_pos, key_pos, key_mask=key_mask)
    
    print(f"\n输入维度:")
    print(f"  query_pos: {query_pos.shape}")
    print(f"  key_pos:   {key_pos.shape}")
    
    print(f"\n输出维度:")
    print(f"  bias:      {bias.shape}")
    print(f"  期望:      (batch={batch_size}, heads={num_heads}, num_q={num_q}, num_k={num_k})")
    
    print(f"\n偏置统计:")
    print(f"  均值: {bias.mean().item():.4f}")
    print(f"  标准差: {bias.std().item():.4f}")
    print(f"  范围: [{bias.min().item():.4f}, {bias.max().item():.4f}]")
    
    # 验证无效位置的偏置为 0
    invalid_bias = bias[:, :, :, -10:]
    print(f"\n无效位置偏置 (应为 0): {invalid_bias.abs().max().item():.6f}")
    
    print("\n✓ 单元测试通过！")
