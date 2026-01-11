# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Extended for Causal-Aware Conditional Planning
# Frenet Frame Trajectory Sampler

"""
FrenetSampler 模块

【作用】
在 Frenet 坐标系中为自车 (Ego) 生成 K 条候选轨迹。
Frenet 坐标系将轨迹分解为：
  - 纵向 (s轴): 沿参考线方向的位移（控制速度/加速度）
  - 横向 (l轴): 垂直于参考线的偏移（控制车道变换）

【为什么使用 Frenet 坐标系】
1. 解耦纵向与横向运动：可以独立控制"走多快"和"往哪偏"
2. 道路对齐：s轴自动沿着道路中心线，无需复杂的坐标变换
3. 轨迹平滑：通过多项式拟合保证轨迹的连续性和舒适性

【采样策略】
纵向 (s轴) 行为模式：
  - 急加速 (Aggressive Acceleration): 适用于超车/汇入
  - 匀速 (Constant Velocity): 正常巡航
  - 缓行 (Deceleration): 谨慎跟车
  - 急刹车 (Emergency Braking): 避险

横向 (l轴) 暂时固定为 0（车道保持），后续可扩展为变道采样。

【输入/输出】
输入:
  - current_state: (v, a) 当前速度和加速度
  - reference_line: 参考轨迹线（可选，默认直行）
  - time_horizon: 预测时间窗口（如 8 秒）
  - dt: 时间步长（如 0.1 秒）
  
输出:
  - candidates: (K, T, 2) 的轨迹张量
    K = 候选轨迹数量
    T = 时间步数 = time_horizon / dt
    2 = (x, y) 坐标
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class FrenetSampler:
    """
    Frenet 坐标系轨迹采样器
    
    生成 K 条自车候选轨迹，覆盖不同的驾驶意图：
    - 激进加速、平稳巡航、谨慎减速、紧急制动
    
    Args:
        time_horizon (float): 预测时间窗口，单位秒（默认 8.0）
        dt (float): 时间步长，单位秒（默认 0.1）
        num_lateral_samples (int): 横向采样数量（默认 1，即车道保持）
        max_acceleration (float): 最大加速度 m/s² (默认 3.0)
        max_deceleration (float): 最大减速度 m/s² (默认 6.0，负值）
        max_velocity (float): 最大速度 m/s (默认 30.0，约 108 km/h)
    """
    
    def __init__(
        self,
        time_horizon: float = 8.0,
        dt: float = 0.1,
        num_lateral_samples: int = 1,
        max_acceleration: float = 3.0,
        max_deceleration: float = 6.0,
        max_velocity: float = 30.0,
    ):
        self.time_horizon = time_horizon
        self.dt = dt
        self.num_steps = int(time_horizon / dt)  # 例如 80 步
        self.num_lateral_samples = num_lateral_samples
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.max_velocity = max_velocity
        
        # 定义纵向行为模式的加速度方案
        # 格式: (name, acceleration_ratio)
        # acceleration_ratio: 相对于 max_accel 或 max_decel 的比例
        self.longitudinal_modes = [
            ("aggressive_accel", 1.0),    # 急加速: 100% max_acceleration
            ("mild_accel", 0.5),          # 温和加速: 50% max_acceleration
            ("constant", 0.0),            # 匀速: 0 加速度
            ("mild_decel", -0.3),         # 温和减速: 30% max_deceleration
            ("emergency_brake", -1.0),    # 急刹车: 100% max_deceleration
        ]
    
    def sample_longitudinal_trajectory(
        self,
        current_velocity: float,
        current_acceleration: float,
        acceleration_ratio: float,
    ) -> np.ndarray:
        """
        根据加速度策略生成纵向轨迹 (s 轴位移序列)
        
        使用恒定加速度模型:
            s(t) = v0 * t + 0.5 * a * t^2
            v(t) = v0 + a * t
        
        Args:
            current_velocity: 当前速度 m/s
            current_acceleration: 当前加速度 m/s² (用于平滑过渡，暂未使用)
            acceleration_ratio: 加速度比例 [-1, 1]
                正值: 使用 max_acceleration * ratio
                负值: 使用 max_deceleration * abs(ratio)
        
        Returns:
            positions: (T, 2) 的位置序列，其中 [:, 0] 是 s 坐标，[:, 1] 是 l 坐标（初始为0）
        """
        # 计算目标加速度
        if acceleration_ratio >= 0:
            target_accel = self.max_acceleration * acceleration_ratio
        else:
            target_accel = -self.max_deceleration * abs(acceleration_ratio)
        
        positions = np.zeros((self.num_steps, 2))
        
        v = current_velocity
        s = 0.0  # 假设起点为原点
        
        for t_idx in range(self.num_steps):
            t = t_idx * self.dt
            
            # 更新速度 (限制在 [0, max_velocity])
            v_new = v + target_accel * self.dt
            v_new = np.clip(v_new, 0.0, self.max_velocity)
            
            # 更新位置
            s += (v + v_new) / 2 * self.dt  # 梯形积分
            
            v = v_new
            
            # 记录位置 (s, l)，此处 l = 0 (车道保持)
            positions[t_idx, 0] = s
            positions[t_idx, 1] = 0.0
        
        return positions
    
    def frenet_to_cartesian(
        self,
        frenet_traj: np.ndarray,
        reference_line: Optional[np.ndarray] = None,
        start_heading: float = 0.0,
    ) -> np.ndarray:
        """
        将 Frenet 坐标转换为 Cartesian 坐标
        
        简化版本: 假设参考线是直线 (沿 heading 方向)
        
        Args:
            frenet_traj: (T, 2) Frenet 坐标，[:, 0]=s, [:, 1]=l
            reference_line: 参考线点序列 (可选)
            start_heading: 起始航向角 (弧度)
        
        Returns:
            cartesian_traj: (T, 2) Cartesian 坐标
        """
        if reference_line is not None:
            # TODO: 实现沿任意曲线的 Frenet 转换
            # 当前简化为直线
            pass
        
        # 简化版本: 参考线为直线，沿 heading 方向
        cos_h = np.cos(start_heading)
        sin_h = np.sin(start_heading)
        
        s = frenet_traj[:, 0]
        l = frenet_traj[:, 1]
        
        # 旋转变换
        x = s * cos_h - l * sin_h
        y = s * sin_h + l * cos_h
        
        cartesian_traj = np.stack([x, y], axis=-1)
        return cartesian_traj
    
    def sample(
        self,
        current_velocity: float = 10.0,
        current_acceleration: float = 0.0,
        start_heading: float = 0.0,
        reference_line: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        生成 K 条候选轨迹
        
        Args:
            current_velocity: 当前速度 m/s (默认 10.0，约 36 km/h)
            current_acceleration: 当前加速度 m/s²
            start_heading: 起始航向角 (弧度，0 = 沿 x 轴正方向)
            reference_line: 参考线 (可选)
        
        Returns:
            candidates: (K, T, 2) 候选轨迹张量
                K = len(longitudinal_modes) * num_lateral_samples
                T = num_steps
                2 = (x, y)
        """
        trajectories = []
        
        for mode_name, accel_ratio in self.longitudinal_modes:
            # 1. 生成 Frenet 坐标轨迹
            frenet_traj = self.sample_longitudinal_trajectory(
                current_velocity=current_velocity,
                current_acceleration=current_acceleration,
                acceleration_ratio=accel_ratio,
            )
            
            # 2. 转换为 Cartesian 坐标
            cartesian_traj = self.frenet_to_cartesian(
                frenet_traj=frenet_traj,
                reference_line=reference_line,
                start_heading=start_heading,
            )
            
            trajectories.append(cartesian_traj)
        
        # (K, T, 2)
        candidates = np.stack(trajectories, axis=0)
        return candidates
    
    def sample_batch(
        self,
        batch_current_velocity: np.ndarray,
        batch_current_acceleration: Optional[np.ndarray] = None,
        batch_start_heading: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        批量生成候选轨迹
        
        Args:
            batch_current_velocity: (B,) 每个样本的当前速度
            batch_current_acceleration: (B,) 每个样本的当前加速度，默认为 0
            batch_start_heading: (B,) 每个样本的起始航向，默认为 0
        
        Returns:
            candidates: (B, K, T, 2) 候选轨迹张量
        """
        batch_size = len(batch_current_velocity)
        
        if batch_current_acceleration is None:
            batch_current_acceleration = np.zeros(batch_size)
        if batch_start_heading is None:
            batch_start_heading = np.zeros(batch_size)
        
        all_candidates = []
        for i in range(batch_size):
            candidates = self.sample(
                current_velocity=batch_current_velocity[i],
                current_acceleration=batch_current_acceleration[i],
                start_heading=batch_start_heading[i],
            )
            all_candidates.append(candidates)
        
        # (B, K, T, 2)
        batch_candidates = np.stack(all_candidates, axis=0)
        return batch_candidates
    
    def sample_torch(
        self,
        current_velocity: torch.Tensor,
        current_acceleration: Optional[torch.Tensor] = None,
        start_heading: Optional[torch.Tensor] = None,
        device: str = 'cuda',
    ) -> torch.Tensor:
        """
        PyTorch 版本的批量采样，返回 GPU Tensor
        
        Args:
            current_velocity: (B,) 当前速度
            current_acceleration: (B,) 当前加速度
            start_heading: (B,) 起始航向
            device: 目标设备
        
        Returns:
            candidates: (B, K, T, 2) torch.Tensor
        """
        # 转换为 numpy 进行采样
        vel_np = current_velocity.detach().cpu().numpy()
        accel_np = current_acceleration.detach().cpu().numpy() if current_acceleration is not None else None
        heading_np = start_heading.detach().cpu().numpy() if start_heading is not None else None
        
        # 批量采样
        candidates_np = self.sample_batch(vel_np, accel_np, heading_np)
        
        # 转换回 torch tensor
        candidates = torch.from_numpy(candidates_np).float().to(device)
        return candidates


def build_frenet_sampler(config: dict = None) -> FrenetSampler:
    """
    构建 FrenetSampler 的工厂函数
    
    Args:
        config: 配置字典，可包含:
            - time_horizon: 预测时间窗口 (默认 8.0)
            - dt: 时间步长 (默认 0.1)
            - max_acceleration: 最大加速度 (默认 3.0)
            - max_deceleration: 最大减速度 (默认 6.0)
            - max_velocity: 最大速度 (默认 30.0)
    
    Returns:
        FrenetSampler 实例
    """
    if config is None:
        config = {}
    
    return FrenetSampler(
        time_horizon=config.get('time_horizon', 8.0),
        dt=config.get('dt', 0.1),
        num_lateral_samples=config.get('num_lateral_samples', 1),
        max_acceleration=config.get('max_acceleration', 3.0),
        max_deceleration=config.get('max_deceleration', 6.0),
        max_velocity=config.get('max_velocity', 30.0),
    )


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FrenetSampler 测试")
    print("=" * 60)
    
    # 创建采样器
    sampler = FrenetSampler(
        time_horizon=8.0,
        dt=0.1,
        max_acceleration=3.0,
        max_deceleration=6.0,
        max_velocity=30.0,
    )
    
    # 单个样本采样
    print("\n1. 单个样本采样:")
    candidates = sampler.sample(
        current_velocity=10.0,  # 36 km/h
        current_acceleration=0.0,
        start_heading=0.0,
    )
    print(f"   输出形状: {candidates.shape}")  # 期望 (5, 80, 2)
    print(f"   候选轨迹数量: {candidates.shape[0]}")
    print(f"   时间步数: {candidates.shape[1]}")
    
    # 打印每条轨迹的终点
    print("\n   各轨迹终点位置:")
    for i, (mode_name, _) in enumerate(sampler.longitudinal_modes):
        end_pos = candidates[i, -1]
        print(f"   - {mode_name}: ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
    
    # 批量采样
    print("\n2. 批量采样:")
    batch_vel = np.array([5.0, 10.0, 15.0, 20.0])  # 4 个不同速度
    batch_candidates = sampler.sample_batch(batch_vel)
    print(f"   输入批量大小: {len(batch_vel)}")
    print(f"   输出形状: {batch_candidates.shape}")  # 期望 (4, 5, 80, 2)
    
    # PyTorch 版本
    print("\n3. PyTorch 版本:")
    vel_torch = torch.tensor([10.0, 15.0])
    candidates_torch = sampler.sample_torch(vel_torch, device='cpu')
    print(f"   输入类型: {type(vel_torch)}")
    print(f"   输出类型: {type(candidates_torch)}")
    print(f"   输出形状: {candidates_torch.shape}")  # 期望 (2, 5, 80, 2)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
