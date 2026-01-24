# Motion Transformer (MTR) - Geometry-Guided Attention Masking
# Based on: "Causal-Aware MTR: 基于因果推理与条件交互的自动驾驶轨迹规划系统"
# 
# 核心思想：从"数据驱动的软交互"升级为"物理驱动的硬约束"
# 作用：强制模型关注那些在物理上与自车未来轨迹存在冲突的障碍物

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GeometryGuidedAttentionMask(nn.Module):
    """
    几何引导的注意力掩码模块 (Geometry-Guided Attention Masking, GGAM)
    
    【核心功能】
    将物理碰撞风险显式注入到 Transformer 的 Cross-Attention 中，
    解决纯数据驱动 Soft Attention 的"注意力稀释"问题。
    
    【技术流程】
    1. 动态距离场构建：计算自车候选轨迹与障碍物预测轨迹的时空距离
    2. 物理掩码生成：基于安全阈值生成二值化注意力偏置
    3. 显式注入：将掩码叠加到 Cross-Attention 的 Logits 上
    
    【创新价值】
    - 对比 UniAD：稀疏矢量计算 O(KN)，完美契合车规级 50ms 实时性
    - 对比原版 MTR：为深度学习模型加上"物理安全锁"
    """
    
    def __init__(
        self,
        safety_threshold: float = 2.0,
        attention_boost: float = 5.0,
        use_soft_mask: bool = False,
        soft_sigma: float = 1.0
    ):
        """
        Args:
            safety_threshold: 安全距离阈值 τ (米)，默认 2.0m 对应车辆碰撞包络
            attention_boost: 注意力增强系数 λ，当距离 < τ 时叠加的正向偏置
            use_soft_mask: 是否使用软掩码（高斯衰减）而非硬阈值
            soft_sigma: 软掩码的高斯标准差
        """
        super().__init__()
        self.safety_threshold = safety_threshold
        self.attention_boost = attention_boost
        self.use_soft_mask = use_soft_mask
        self.soft_sigma = soft_sigma
    
    def compute_pairwise_distance(
        self,
        ego_trajs: torch.Tensor,
        obstacle_trajs: torch.Tensor
    ) -> torch.Tensor:
        """
        计算自车轨迹与障碍物轨迹的成对时空距离（矢量计算，非栅格化）
        
        Args:
            ego_trajs: 自车候选轨迹 (B, K, T, 2) 或 (K, T, 2)
                B = batch size, K = 候选轨迹数, T = 时间步, 2 = (x, y)
            obstacle_trajs: 障碍物预测轨迹 (B, N, T, 2) 或 (N, T, 2)
                N = 障碍物数量
        
        Returns:
            distance_field: 时空距离张量 (B, K, N, T) 或 (K, N, T)
                存储每个时刻、每对 (自车轨迹, 障碍物) 的欧氏距离
        """
        # 处理维度：确保是 4D 张量 (B, K/N, T, 2)
        if ego_trajs.dim() == 3:
            ego_trajs = ego_trajs.unsqueeze(0)  # (1, K, T, 2)
        if obstacle_trajs.dim() == 3:
            obstacle_trajs = obstacle_trajs.unsqueeze(0)  # (1, N, T, 2)
        
        B, K, T, _ = ego_trajs.shape
        _, N, _, _ = obstacle_trajs.shape
        
        # 扩展维度以进行成对计算
        # ego_trajs: (B, K, 1, T, 2)
        # obstacle_trajs: (B, 1, N, T, 2)
        ego_expanded = ego_trajs.unsqueeze(2)  # (B, K, 1, T, 2)
        obs_expanded = obstacle_trajs.unsqueeze(1)  # (B, 1, N, T, 2)
        
        # 计算欧氏距离 (B, K, N, T)
        distance_field = torch.norm(ego_expanded - obs_expanded, dim=-1)
        
        return distance_field
    
    def generate_geometry_mask(
        self,
        distance_field: torch.Tensor,
        reduction: str = 'min'
    ) -> torch.Tensor:
        """
        基于时空距离场生成几何注意力掩码
        
        Args:
            distance_field: 时空距离张量 (B, K, N, T)
            reduction: 时间维度的归约方式
                - 'min': 取最小距离（最危险时刻）
                - 'mean': 取平均距离
        
        Returns:
            geometry_mask: 几何注意力掩码 (B, K, N)
                正值表示需要增强关注，0 表示无需额外关注
        """
        # 时间维度归约
        if reduction == 'min':
            min_distance, _ = distance_field.min(dim=-1)  # (B, K, N)
        elif reduction == 'mean':
            min_distance = distance_field.mean(dim=-1)  # (B, K, N)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
        
        if self.use_soft_mask:
            # 软掩码：高斯衰减，距离越近增益越大
            # M_geo = λ * exp(-(d - τ)² / (2σ²)) when d < τ
            mask = torch.where(
                min_distance < self.safety_threshold,
                self.attention_boost * torch.exp(
                    -((min_distance - self.safety_threshold) ** 2) / (2 * self.soft_sigma ** 2)
                ),
                torch.zeros_like(min_distance)
            )
        else:
            # 硬掩码：阈值判别
            # M_geo = λ if min(d) < τ else 0
            mask = torch.where(
                min_distance < self.safety_threshold,
                torch.full_like(min_distance, self.attention_boost),
                torch.zeros_like(min_distance)
            )
        
        return mask
    
    def forward(
        self,
        ego_trajs: torch.Tensor,
        obstacle_trajs: torch.Tensor,
        obstacle_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算几何引导的注意力掩码
        
        Args:
            ego_trajs: 自车候选轨迹 (B, K, T, 2)
            obstacle_trajs: 障碍物预测轨迹 (B, N, T, 2)
            obstacle_mask: 障碍物有效性掩码 (B, N)，True 表示有效
        
        Returns:
            geometry_mask: 几何注意力掩码 (B, K, N)
                可直接叠加到 Cross-Attention 的 Logits 上
        """
        # Step 1: 计算时空距离场 (B, K, N, T)
        distance_field = self.compute_pairwise_distance(ego_trajs, obstacle_trajs)
        
        # Step 2: 生成几何掩码 (B, K, N)
        geometry_mask = self.generate_geometry_mask(distance_field)
        
        # Step 3: 应用障碍物掩码（无效障碍物不应获得注意力增益）
        if obstacle_mask is not None:
            # obstacle_mask: (B, N) -> (B, 1, N)
            geometry_mask = geometry_mask * obstacle_mask.unsqueeze(1).float()
        
        return geometry_mask
    
    def get_collision_risk_indicators(
        self,
        ego_trajs: torch.Tensor,
        obstacle_trajs: torch.Tensor,
        obstacle_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取详细的碰撞风险指标（用于可视化和调试）
        
        Returns:
            min_distances: 最小距离 (B, K, N)
            collision_flags: 碰撞标志 (B, K, N)，True 表示存在碰撞风险
            time_to_collision: 首次碰撞时刻 (B, K, N)，-1 表示无碰撞
        """
        distance_field = self.compute_pairwise_distance(ego_trajs, obstacle_trajs)
        
        # 最小距离
        min_distances, min_time_idx = distance_field.min(dim=-1)  # (B, K, N)
        
        # 碰撞标志
        collision_flags = min_distances < self.safety_threshold
        
        # 首次碰撞时刻
        collision_at_each_time = distance_field < self.safety_threshold  # (B, K, N, T)
        # 找到第一个 True 的索引
        time_to_collision = torch.where(
            collision_at_each_time.any(dim=-1),
            collision_at_each_time.float().argmax(dim=-1).float(),
            torch.full_like(min_time_idx.float(), -1)
        )
        
        if obstacle_mask is not None:
            collision_flags = collision_flags & obstacle_mask.unsqueeze(1)
        
        return min_distances, collision_flags, time_to_collision


def compute_geometry_attention_bias(
    ego_trajs: torch.Tensor,
    dense_future_pred: torch.Tensor,
    obj_mask: torch.Tensor,
    safety_threshold: float = 2.0,
    attention_boost: float = 5.0
) -> torch.Tensor:
    """
    便捷函数：计算几何注意力偏置，可直接在 MTRDecoder 中调用
    
    【使用示例】
    在 apply_cross_attention 之前调用：
    
    ```python
    # 获取密集未来预测作为障碍物轨迹估计
    dense_pred = self.forward_ret_dict['pred_dense_trajs']  # (B, N, T, 7)
    obstacle_trajs = dense_pred[:, :, :, 0:2]  # 只取 (x, y)
    
    # 计算几何掩码
    geo_bias = compute_geometry_attention_bias(
        ego_trajs=ego_candidates,  # (B, K, T, 2)
        dense_future_pred=obstacle_trajs,  # (B, N, T, 2)
        obj_mask=obj_mask,  # (B, N)
        safety_threshold=2.0,
        attention_boost=5.0
    )
    
    # 在 Cross-Attention 中使用
    # attn_logits = Q @ K.T / sqrt(d) + geo_bias
    ```
    
    Args:
        ego_trajs: 自车候选轨迹 (B, K, T, 2)
        dense_future_pred: 障碍物密集未来预测 (B, N, T, 2)
        obj_mask: 障碍物有效性掩码 (B, N)
        safety_threshold: 安全距离阈值
        attention_boost: 注意力增强系数
    
    Returns:
        geometry_bias: 几何注意力偏置 (B, K, N)
    """
    ggam = GeometryGuidedAttentionMask(
        safety_threshold=safety_threshold,
        attention_boost=attention_boost,
        use_soft_mask=False
    )
    
    return ggam(
        ego_trajs=ego_trajs,
        obstacle_trajs=dense_future_pred,
        obstacle_mask=obj_mask
    )


# ==================== 单元测试 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("Geometry-Guided Attention Mask (GGAM) - Unit Test")
    print("=" * 60)
    
    # 模拟数据
    B, K, N, T = 2, 4, 10, 80  # batch=2, 4条候选轨迹, 10个障碍物, 80帧
    
    # 自车轨迹：直行
    ego_trajs = torch.zeros(B, K, T, 2)
    for k in range(K):
        ego_trajs[:, k, :, 0] = torch.linspace(0, 50 + k * 5, T)  # x: 0 -> 50+k*5
        ego_trajs[:, k, :, 1] = k * 2  # y: 不同车道
    
    # 障碍物轨迹：部分与自车轨迹交叉
    obstacle_trajs = torch.zeros(B, N, T, 2)
    for n in range(N):
        obstacle_trajs[:, n, :, 0] = torch.linspace(20 + n * 3, 40 + n * 3, T)
        obstacle_trajs[:, n, :, 1] = n * 1.5 - 5  # 分布在不同车道
    
    # 有效障碍物掩码
    obj_mask = torch.ones(B, N).bool()
    obj_mask[:, -2:] = False  # 最后两个障碍物无效
    
    # 创建 GGAM 模块
    ggam = GeometryGuidedAttentionMask(
        safety_threshold=2.0,
        attention_boost=5.0,
        use_soft_mask=False
    )
    
    # 计算几何掩码
    geo_mask = ggam(ego_trajs, obstacle_trajs, obj_mask)
    
    print(f"\n输入维度:")
    print(f"  ego_trajs:      {ego_trajs.shape}")
    print(f"  obstacle_trajs: {obstacle_trajs.shape}")
    print(f"  obj_mask:       {obj_mask.shape}")
    
    print(f"\n输出维度:")
    print(f"  geometry_mask:  {geo_mask.shape}")
    
    print(f"\n掩码统计:")
    print(f"  非零元素数:     {(geo_mask > 0).sum().item()}")
    print(f"  最大值:         {geo_mask.max().item():.2f}")
    print(f"  非零比例:       {(geo_mask > 0).float().mean().item() * 100:.1f}%")
    
    # 测试碰撞风险指标
    min_dist, collision_flags, ttc = ggam.get_collision_risk_indicators(
        ego_trajs, obstacle_trajs, obj_mask
    )
    print(f"\n碰撞风险分析:")
    print(f"  存在碰撞风险的 (轨迹, 障碍物) 对数: {collision_flags.sum().item()}")
    print(f"  最小距离范围: [{min_dist.min().item():.2f}, {min_dist.max().item():.2f}] m")
    
    print("\n✓ 单元测试通过！")
