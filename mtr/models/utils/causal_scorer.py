#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Causal Scorer 模块

【目的】
为 K 个平行世界打分，选择最安全/最优的自车决策。

【设计思路】
使用 Attention 机制让自车特征"观察"每个平行世界中的邻居预测，
然后输出一个标量分数表示这个世界的"好坏"。

【输入】
- ego_feat: 自车在 K 种决策下的特征
- neighbor_pred_feat: K 个平行世界中邻居的预测轨迹特征 (因果变化的核心)
- neighbor_mask: 邻居有效性 mask

【输出】
- scores: [B, K] 每个平行世界的分数，越高越好
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalScorer(nn.Module):
    """
    因果打分器：评估 K 个平行世界的优劣
    
    核心思想：
    - 不同的自车决策会导致不同的邻居反应 (因果性)
    - 打分器学习判断哪种决策下的邻居反应对自车更有利
    """
    
    def __init__(self, d_model=256, num_heads=4, hidden_dim=128):
        super().__init__()
        self.d_model = d_model
        
        # 1. 轨迹编码器：将轨迹 (T, 2) 编码为特征向量 (D,)
        # 用于将自车候选轨迹和邻居预测轨迹都编码成可比较的特征
        self.traj_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 每个点 (x, y) -> hidden_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        # 2. 交互注意力层
        # 让自车特征 (Query) 去观察邻居特征 (Key/Value)
        # 不同平行世界的邻居特征不同，attention 结果就不同
        self.interaction_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # 3. 场景融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 4. 最终打分头
        # 输入融合后的特征，输出一个标量分数
        self.score_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # 输出标量分数
        )
        
    def encode_trajectory(self, traj):
        """
        将轨迹编码为特征向量
        
        Args:
            traj: (*, T, 2) 轨迹点序列
            
        Returns:
            feat: (*, D) 轨迹特征向量 (通过 max pooling 聚合)
        """
        # (*, T, 2) -> (*, T, D)
        point_feat = self.traj_encoder(traj)
        # Max pooling over time: (*, T, D) -> (*, D)
        traj_feat = point_feat.max(dim=-2)[0]
        return traj_feat
        
    def forward(self, ego_trajs, neighbor_pred_trajs, neighbor_mask):
        """
        为 K 个平行世界打分
        
        Args:
            ego_trajs: (B, K, T, 2) K 条自车候选轨迹
            neighbor_pred_trajs: (B, K, N, T, 2) K 个平行世界中邻居的预测轨迹
            neighbor_mask: (B, N) 邻居有效性 mask (True = 有效)
            
        Returns:
            scores: (B, K) 每个平行世界的分数
        """
        B, K, T, _ = ego_trajs.shape
        N = neighbor_pred_trajs.shape[2]
        
        # 1. 编码自车轨迹: (B, K, T, 2) -> (B, K, D)
        ego_feat = self.encode_trajectory(ego_trajs)
        
        # 2. 编码邻居轨迹: (B, K, N, T, 2) -> (B, K, N, D)
        neighbor_feat = self.encode_trajectory(neighbor_pred_trajs)
        
        # 3. 展平 Batch 和 K 维度，方便并行计算
        # ego: (B*K, 1, D) - 作为 Query
        ego_flat = ego_feat.reshape(B * K, 1, self.d_model)
        # neighbor: (B*K, N, D) - 作为 Key/Value
        neighbor_flat = neighbor_feat.reshape(B * K, N, self.d_model)
        
        # 4. 准备 Attention Mask
        # 扩展 mask: (B, N) -> (B, K, N) -> (B*K, N)
        mask_expanded = neighbor_mask.unsqueeze(1).expand(-1, K, -1)
        mask_flat = mask_expanded.reshape(B * K, N)
        
        # 5. 交互注意力
        # 自车去"看"这个平行世界里的邻居
        # 因为邻居预测是因果变化的，不同 K 得到不同的 attn_out
        attn_out, attn_weights = self.interaction_attn(
            query=ego_flat,      # (B*K, 1, D)
            key=neighbor_flat,   # (B*K, N, D)
            value=neighbor_flat, # (B*K, N, D)
            key_padding_mask=~mask_flat  # True 表示忽略
        )
        # attn_out: (B*K, 1, D)
        
        # 6. 特征融合 (残差 + concat)
        # 拼接自车特征和交互后特征
        ego_flat_squeezed = ego_flat.squeeze(1)  # (B*K, D)
        attn_out_squeezed = attn_out.squeeze(1)  # (B*K, D)
        fused_feat = self.fusion_layer(
            torch.cat([ego_flat_squeezed, attn_out_squeezed], dim=-1)
        )  # (B*K, D)
        
        # 7. 打分
        scores = self.score_head(fused_feat)  # (B*K, 1)
        scores = scores.reshape(B, K)  # (B, K)
        
        return scores


def compute_collision_cost(ego_trajs, neighbor_pred_trajs, neighbor_mask, 
                           safety_threshold=1.5):
    """
    计算碰撞代价 (用于推理时选择最安全的决策)
    
    Args:
        ego_trajs: (B, K, T, 2) K 条自车候选轨迹
        neighbor_pred_trajs: (B, K, N, T, 2) K 个平行世界中邻居的预测轨迹
        neighbor_mask: (B, N) 邻居有效性 mask
        safety_threshold: 安全距离阈值 (米)
        
    Returns:
        collision_cost: (B, K) 碰撞代价，越小越安全
        min_distances: (B, K) 每个世界的最小距离
    """
    B, K, N, T, _ = neighbor_pred_trajs.shape
    
    # 1. 计算自车与所有邻居在所有时刻的距离
    # ego: (B, K, 1, T, 2)
    # neighbor: (B, K, N, T, 2)
    # dist: (B, K, N, T)
    ego_expanded = ego_trajs.unsqueeze(2)  # (B, K, 1, T, 2)
    dist_matrix = torch.norm(ego_expanded - neighbor_pred_trajs, dim=-1)  # (B, K, N, T)
    
    # 2. 处理无效邻居
    # mask: (B, N) -> (B, 1, N, 1)
    mask_expanded = neighbor_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
    mask_expanded = mask_expanded.expand(-1, K, -1, T)  # (B, K, N, T)
    
    # 将无效邻居的距离设为无穷大
    dist_matrix = dist_matrix.masked_fill(~mask_expanded, float('inf'))
    
    # 3. 找到每个平行世界里的最小距离
    # min over time -> (B, K, N)
    min_dist_over_time = dist_matrix.min(dim=-1)[0]
    # min over neighbors -> (B, K)
    min_distances = min_dist_over_time.min(dim=-1)[0]
    
    # 4. 计算碰撞代价
    # 如果距离 < 阈值，产生代价
    # ReLU(1.5 - 0.5) = 1.0 (太近了，有代价)
    # ReLU(1.5 - 5.0) = 0.0 (安全，无代价)
    collision_cost = F.relu(safety_threshold - min_distances)
    
    return collision_cost, min_distances


def causal_planning_loss(pred_scores, ego_trajs, neighbor_pred_trajs, 
                         neighbor_mask, gt_trajectory=None,
                         safety_threshold=1.5, imitation_weight=1.0, 
                         safety_weight=10.0):
    """
    因果规划损失函数
    
    【设计思路】
    1. 模仿损失 (Imitation Loss): 
       - 找到离 GT 最近的那个 K，强迫模型给它打最高分
       - 作用：让模型学会模仿人类驾驶员的决策
       
    2. 安全损失 (Safety Loss):
       - 检查 K 个平行世界里是否有碰撞风险
       - 如果模型给危险的世界打了高分，就惩罚它
       - 作用：让模型学会避开危险决策
    
    Args:
        pred_scores: (B, K) 模型预测的分数
        ego_trajs: (B, K, T, 2) K 条自车候选轨迹
        neighbor_pred_trajs: (B, K, N, T, 2) K 个平行世界中邻居的预测轨迹
        neighbor_mask: (B, N) 邻居有效性 mask
        gt_trajectory: (B, T, 2) 自车真值轨迹 (可选，用于模仿损失)
        safety_threshold: 安全距离阈值
        imitation_weight: 模仿损失权重
        safety_weight: 安全损失权重
        
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
    """
    B, K, T, _ = ego_trajs.shape
    device = ego_trajs.device
    
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)
    
    # ========== 1. 模仿损失 (Imitation Loss) ==========
    if gt_trajectory is not None:
        # 计算 K 条候选轨迹与 GT 的距离 (ADE)
        # gt: (B, T, 2) -> (B, 1, T, 2)
        gt_expanded = gt_trajectory.unsqueeze(1)
        # dist: (B, K, T, 2) -> (B, K)
        dist_to_gt = torch.norm(ego_trajs - gt_expanded, dim=-1).mean(dim=-1)  # ADE
        
        # 找到离 GT 最近的那条轨迹作为正样本
        best_k_idx = torch.argmin(dist_to_gt, dim=-1)  # (B,)
        
        # 使用 Cross Entropy Loss
        # 强迫模型给 best_k_idx 打最高分
        loss_imitation = F.cross_entropy(pred_scores, best_k_idx)
        
        total_loss = total_loss + imitation_weight * loss_imitation
        loss_dict['loss_imitation'] = loss_imitation.item()
        loss_dict['best_k_dist'] = dist_to_gt[torch.arange(B), best_k_idx].mean().item()
    
    # ========== 2. 安全损失 (Safety Loss) ==========
    # 计算碰撞代价
    collision_cost, min_distances = compute_collision_cost(
        ego_trajs, neighbor_pred_trajs, neighbor_mask, safety_threshold
    )
    
    # 获取模型给每个世界的概率
    probs = F.softmax(pred_scores, dim=-1)  # (B, K)
    
    # 安全损失：如果模型给危险世界打了高分，就惩罚
    # probs * collision_cost: 加权惩罚
    # 概率越高、碰撞代价越大，惩罚越严重
    loss_safety = (probs * collision_cost).sum(dim=-1).mean()
    
    total_loss = total_loss + safety_weight * loss_safety
    loss_dict['loss_safety'] = loss_safety.item()
    loss_dict['collision_cost_mean'] = collision_cost.mean().item()
    loss_dict['min_dist_mean'] = min_distances.mean().item()
    
    return total_loss, loss_dict
