# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Extended for Causal-Aware Conditional Planning
# Original by Shaoshuai Shi, Extended for Conditional Query Injection

"""
ConditionEncoder 模块

【作用】
将自车 (Ego) 的 K 条候选轨迹编码为高维条件向量 (Condition Vector)。
这个向量将被注入到 MTRDecoder 中，使得模型能够进行条件预测：
"如果我（自车）这样走，周围的障碍物会怎么反应？"

【为什么需要这个模块】
原版 MTR 是一个"旁观者"模型：它预测障碍物的轨迹，但不考虑自车的行为会影响别人。
引入 ConditionEncoder 后，我们可以：
1. 输入 K 条自车候选轨迹（如：急加速、匀速、刹车）
2. 让模型分别预测在这 K 种情况下，障碍物会如何反应
3. 选择对自车最有利的那条轨迹执行

【技术实现】
- 输入: (batch_size, K, num_future_frames, 2) 的轨迹张量
        K = 候选轨迹数量（如 5 条）
        num_future_frames = 未来时间步数（如 80 帧 = 8 秒）
        2 = (x, y) 坐标
- 输出: (batch_size, K, d_model) 的条件向量
        d_model = Transformer 隐层维度（如 256）

【编码策略】
采用 PointNet 风格的编码：
1. 先对每个时间步的 (x, y) 进行逐点 MLP 编码
2. 再对整条轨迹进行 Max Pooling，提取全局特征
这种方式对轨迹的长度不敏感，且能捕捉轨迹的整体形状。
"""

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    条件编码器：将自车候选轨迹编码为条件向量
    
    Args:
        d_model (int): 输出特征维度，需与 Decoder 的隐层维度一致
        num_future_frames (int): 未来轨迹的时间步数
        hidden_dim (int): MLP 隐层维度
        num_layers (int): MLP 层数
    """
    
    def __init__(self, d_model, num_future_frames, hidden_dim=128, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.num_future_frames = num_future_frames
        
        # === 轨迹点编码器 (Per-Point Encoder) ===
        # 将每个 (x, y) 点编码为高维特征
        # 输入: 2 (x, y)
        # 输出: hidden_dim
        layers = []
        in_dim = 2  # (x, y) 坐标
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
        self.point_encoder = nn.Sequential(*layers)
        
        # === 轨迹聚合器 (Trajectory Aggregator) ===
        # 将整条轨迹的特征聚合为单个向量
        # 采用 MLP + Max Pooling 的方式
        self.traj_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        # === 输出投影层 ===
        # 最终将聚合后的特征投影到 d_model 维度
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, ego_future_candidates):
        """
        前向传播
        
        Args:
            ego_future_candidates: (batch_size, K, num_future_frames, 2)
                K 条自车候选轨迹，每条包含 num_future_frames 个 (x, y) 点
        
        Returns:
            condition_vector: (batch_size, K, d_model)
                编码后的条件向量，可直接用于 Decoder 的 Cross-Attention
        """
        batch_size, K, T, _ = ego_future_candidates.shape
        
        # Step 1: 展平 batch 和 K 维度，方便并行处理
        # (batch_size, K, T, 2) -> (batch_size * K, T, 2)
        traj_flat = ego_future_candidates.view(batch_size * K, T, 2)
        
        # Step 2: 对每个点进行编码
        # (batch_size * K, T, 2) -> (batch_size * K, T, hidden_dim)
        point_features = self.point_encoder(traj_flat)
        
        # Step 3: 对整条轨迹进行 Max Pooling (沿时间维度)
        # 这样可以捕捉轨迹中最显著的特征（如最大速度、最大位移）
        # (batch_size * K, T, hidden_dim) -> (batch_size * K, hidden_dim)
        traj_features, _ = point_features.max(dim=1)
        
        # Step 4: 通过聚合器生成轨迹级特征
        # (batch_size * K, hidden_dim) -> (batch_size * K, d_model)
        traj_embedding = self.traj_aggregator(traj_features)
        
        # Step 5: 最终投影
        # (batch_size * K, d_model) -> (batch_size * K, d_model)
        condition_vector = self.output_proj(traj_embedding)
        
        # Step 6: 恢复 batch 和 K 维度
        # (batch_size * K, d_model) -> (batch_size, K, d_model)
        condition_vector = condition_vector.view(batch_size, K, self.d_model)
        
        return condition_vector


def build_condition_encoder(d_model, num_future_frames, hidden_dim=128):
    """
    构建 ConditionEncoder 的工厂函数
    
    Args:
        d_model: Transformer 隐层维度
        num_future_frames: 未来轨迹时间步数
        hidden_dim: MLP 隐层维度
    
    Returns:
        ConditionEncoder 实例
    """
    return ConditionEncoder(
        d_model=d_model,
        num_future_frames=num_future_frames,
        hidden_dim=hidden_dim
    )
