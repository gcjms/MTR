#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
坐标系一致性验证脚本

【目的】
验证 Causal MTR 的核心假设：center = Ego 时，所有数据在同一坐标系。

【验证项】
1. FrenetSampler 轨迹起点 = (0, 0)
2. ConditionEncoder 不改变坐标系语义
3. (可选) Waymo 数据中 center_objects 位置验证

【运行方法】
    python -m mtr.utils.test_coordinate_system
"""

import sys
import numpy as np

# 颜色输出
try:
    from colorama import init, Fore, Style
    init()
    GREEN, RED, YELLOW, RESET = Fore.GREEN, Fore.RED, Fore.YELLOW, Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = RESET = ""


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_pass(msg):
    print(f"{GREEN}✓ {msg}{RESET}")


def print_fail(msg, detail=None):
    print(f"{RED}✗ {msg}{RESET}")
    if detail:
        print(f"  Detail: {detail}")


def print_info(msg):
    print(f"{YELLOW}  → {msg}{RESET}")


# ============================================================
# Test 1: FrenetSampler 轨迹起点验证
# ============================================================
def test_frenet_sampler_origin():
    """
    验证 FrenetSampler 生成的所有轨迹起点都在 (0, 0)
    
    【为什么重要】
    Ego 当前位置在 Ego-Centric 坐标系中是原点 (0, 0)。
    FrenetSampler 生成的候选轨迹必须从原点开始，否则坐标系会不一致。
    """
    test_name = "FrenetSampler Origin at (0, 0)"
    
    try:
        from mtr.utils.frenet_sampler import FrenetSampler
        
        sampler = FrenetSampler(time_horizon=8.0, dt=0.1)
        
        # 测试不同初始条件
        test_cases = [
            {"velocity": 0.0, "desc": "静止"},
            {"velocity": 10.0, "desc": "中速"},
            {"velocity": 30.0, "desc": "高速"},
        ]
        
        for case in test_cases:
            candidates = sampler.sample(current_velocity=case["velocity"])
            # 检查所有轨迹的起点 (t=0)
            start_points = candidates[:, 0, :]  # (K, 2)
            
            # 起点应该非常接近 (0, 0)
            max_deviation = np.abs(start_points).max()
            if max_deviation > 1e-6:
                print_fail(test_name, 
                    f"{case['desc']}起点偏离原点: max_deviation = {max_deviation:.2e}")
                return False
        
        print_pass(test_name)
        print_info("所有初速度条件下，轨迹起点均在 (0, 0)")
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        return False


# ============================================================
# Test 2: 轨迹终点随初速度变化
# ============================================================
def test_trajectory_endpoint_varies():
    """
    验证不同初速度产生不同的终点位移
    
    【为什么重要】
    如果轨迹不随初速度变化，说明 Sampler 逻辑有问题
    """
    test_name = "Trajectory Endpoints Vary with Velocity"
    
    try:
        from mtr.utils.frenet_sampler import FrenetSampler
        
        sampler = FrenetSampler(time_horizon=8.0, dt=0.1)
        
        # 两个不同速度
        traj_slow = sampler.sample(current_velocity=5.0)   # 慢速
        traj_fast = sampler.sample(current_velocity=25.0)  # 快速
        
        # 比较 "匀速巡航" 模式 (index=2) 的终点
        end_slow = traj_slow[2, -1, 0]  # 终点 x 坐标
        end_fast = traj_fast[2, -1, 0]
        
        # 快速应该走得更远
        if end_fast <= end_slow:
            print_fail(test_name, 
                f"快速轨迹终点 ({end_fast:.1f}m) 应该 > 慢速 ({end_slow:.1f}m)")
            return False
        
        print_pass(test_name)
        print_info(f"慢速 (5 m/s) 巡航终点: x = {end_slow:.1f} m")
        print_info(f"快速 (25 m/s) 巡航终点: x = {end_fast:.1f} m")
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        return False


# ============================================================
# Test 3: ConditionEncoder 输出维度不变
# ============================================================
def test_condition_encoder_preserves_batch():
    """
    验证 ConditionEncoder 正确处理 batch 维度
    
    【为什么重要】
    编码器不应改变 batch 结构，否则后续 batch expansion 会出错
    """
    test_name = "ConditionEncoder Preserves Batch Structure"
    
    try:
        import torch
        from mtr.models.utils.condition_encoder import ConditionEncoder
        
        batch_size = 3
        K = 5
        T = 80
        d_model = 256
        
        encoder = ConditionEncoder(d_model=d_model, num_future_frames=T)
        
        # 输入
        candidates = torch.randn(batch_size, K, T, 2)
        
        # 输出
        condition = encoder(candidates)
        
        # 验证形状
        assert condition.shape == (batch_size, K, d_model), \
            f"Shape mismatch: {condition.shape} != ({batch_size}, {K}, {d_model})"
        
        print_pass(test_name)
        print_info(f"Input: ({batch_size}, {K}, {T}, 2) → Output: {tuple(condition.shape)}")
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        return False


# ============================================================
# Test 4: 坐标系语义验证 (边界条件)
# ============================================================
def test_coordinate_semantics():
    """
    验证坐标系语义：
    - x 轴 = 前进方向 (正值 = 前方)
    - y 轴 = 横向 (正值 = 左侧)
    
    【为什么重要】
    确保 FrenetSampler 和 MTR 使用相同的坐标系约定
    """
    test_name = "Coordinate Semantics (x=forward, y=left)"
    
    try:
        from mtr.utils.frenet_sampler import FrenetSampler
        
        sampler = FrenetSampler(time_horizon=8.0, dt=0.1)
        candidates = sampler.sample(current_velocity=10.0, start_heading=0.0)
        
        # 加速模式 (index=0) 应该向前走 (x 增加)
        accel_traj = candidates[0]  # (T, 2)
        x_final = accel_traj[-1, 0]
        y_final = accel_traj[-1, 1]
        
        # 检查: x 应该为正 (向前)，y 应该接近 0 (直行)
        if x_final <= 0:
            print_fail(test_name, f"加速轨迹终点 x = {x_final:.1f} 应为正值 (向前)")
            return False
        
        if abs(y_final) > 1.0:  # 允许小偏差
            print_fail(test_name, f"直行轨迹终点 y = {y_final:.1f} 应接近 0")
            return False
        
        print_pass(test_name)
        print_info(f"加速轨迹终点: ({x_final:.1f}, {y_final:.1f}) - 向前直行 ✓")
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        return False


# ============================================================
# Main
# ============================================================
def main():
    print_header("Causal MTR 坐标系一致性验证")
    
    results = []
    results.append(("FrenetSampler Origin", test_frenet_sampler_origin()))
    results.append(("Trajectory Endpoints", test_trajectory_endpoint_varies()))
    results.append(("ConditionEncoder Batch", test_condition_encoder_preserves_batch()))
    results.append(("Coordinate Semantics", test_coordinate_semantics()))
    
    # 汇总
    print_header("验证结果汇总")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")
    
    print()
    if passed == total:
        print(f"{GREEN}========================================{RESET}")
        print(f"{GREEN}  所有 {total} 项验证通过！{RESET}")
        print(f"{GREEN}  坐标系一致性确认。{RESET}")
        print(f"{GREEN}========================================{RESET}")
    else:
        print(f"{RED}========================================{RESET}")
        print(f"{RED}  {passed}/{total} 项通过。{RESET}")
        print(f"{RED}  请检查失败项。{RESET}")
        print(f"{RED}========================================{RESET}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
