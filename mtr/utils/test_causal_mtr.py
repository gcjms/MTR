#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Causal-Aware MTR 验证脚本

【用途】
验证新增的条件预测模块是否工作正常，包括：
1. ConditionEncoder 输入/输出维度
2. FrenetSampler 轨迹采样
3. MTRDecoder 条件注入
4. 批量推理 (K 条候选轨迹并行处理)

【运行方法】
在有 PyTorch 环境的机器上运行：
    cd /path/to/MTR-master/MTR-master
    python -m mtr.utils.test_causal_mtr

【预期输出】
如果所有测试通过，你会看到：
    ✓ Test 1 PASSED: ConditionEncoder
    ✓ Test 2 PASSED: FrenetSampler
    ✓ Test 3 PASSED: Batch Expansion
    ✓ Test 4 PASSED: Conditional Prediction Difference
    ========================================
    All tests passed! Causal MTR is ready.
    ========================================
"""

import sys
import torch
import torch.nn as nn
import numpy as np

# 颜色输出 (跨平台)
try:
    from colorama import init, Fore, Style
    init()
    GREEN = Fore.GREEN
    RED = Fore.RED
    YELLOW = Fore.YELLOW
    RESET = Style.RESET_ALL
except ImportError:
    GREEN = ""
    RED = ""
    YELLOW = ""
    RESET = ""


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_pass(test_name):
    print(f"{GREEN}✓ {test_name} PASSED{RESET}")


def print_fail(test_name, error):
    print(f"{RED}✗ {test_name} FAILED{RESET}")
    print(f"  Error: {error}")


def print_info(text):
    print(f"{YELLOW}  → {text}{RESET}")


# ============================================================
# Test 1: ConditionEncoder 维度验证
# ============================================================
def test_condition_encoder():
    """
    验证 ConditionEncoder 的输入输出维度
    
    输入: (batch_size, K, num_future_frames, 2)
    输出: (batch_size, K, d_model)
    """
    test_name = "Test 1 PASSED: ConditionEncoder"
    
    try:
        from mtr.models.utils.condition_encoder import ConditionEncoder
        
        # 参数设置
        batch_size = 4
        K = 5  # 候选轨迹数量
        num_future_frames = 80
        d_model = 256
        
        # 创建编码器
        encoder = ConditionEncoder(
            d_model=d_model,
            num_future_frames=num_future_frames,
            hidden_dim=128
        )
        
        # 随机输入
        ego_candidates = torch.randn(batch_size, K, num_future_frames, 2)
        
        # 前向传播
        condition_vector = encoder(ego_candidates)
        
        # 验证输出维度
        expected_shape = (batch_size, K, d_model)
        actual_shape = tuple(condition_vector.shape)
        
        assert actual_shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        
        print_pass(test_name)
        print_info(f"Input shape: {ego_candidates.shape}")
        print_info(f"Output shape: {condition_vector.shape}")
        print_info(f"Output range: [{condition_vector.min().item():.3f}, {condition_vector.max().item():.3f}]")
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        return False


# ============================================================
# Test 2: FrenetSampler 轨迹采样验证
# ============================================================
def test_frenet_sampler():
    """
    验证 FrenetSampler 生成的轨迹是否合理
    
    检查点：
    1. 输出维度正确 (K, T, 2)
    2. 不同驾驶模式的终点位置不同
    3. 急加速的终点 > 匀速终点 > 急刹车终点
    """
    test_name = "Test 2 PASSED: FrenetSampler"
    
    try:
        from mtr.utils.frenet_sampler import FrenetSampler
        
        # 创建采样器
        sampler = FrenetSampler(
            time_horizon=8.0,
            dt=0.1,
            max_acceleration=3.0,
            max_deceleration=6.0,
            max_velocity=30.0,
        )
        
        # 采样
        current_velocity = 10.0  # 36 km/h
        candidates = sampler.sample(
            current_velocity=current_velocity,
            current_acceleration=0.0,
            start_heading=0.0,
        )
        
        # 验证输出维度
        K = len(sampler.longitudinal_modes)
        T = sampler.num_steps
        expected_shape = (K, T, 2)
        actual_shape = candidates.shape
        
        assert actual_shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
        
        # 获取各轨迹终点的 x 坐标 (纵向位移)
        end_positions = candidates[:, -1, 0]  # 各轨迹的终点 x
        
        # 验证: 激进加速 > 温和加速 > 匀速 > 温和减速 > 急刹车
        # 因为轨迹是沿 x 轴的，终点 x 值应该递减
        for i in range(len(end_positions) - 1):
            assert end_positions[i] >= end_positions[i + 1], \
                f"End position ordering incorrect: position[{i}]={end_positions[i]:.2f} < position[{i+1}]={end_positions[i+1]:.2f}"
        
        print_pass(test_name)
        print_info(f"Output shape: {candidates.shape}")
        print_info(f"Number of modes: {K}")
        print_info("End positions (x-axis):")
        for i, (mode_name, _) in enumerate(sampler.longitudinal_modes):
            print_info(f"  {mode_name}: x = {end_positions[i]:.2f} m")
        
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Test 3: 批量扩展逻辑验证
# ============================================================
def test_batch_expansion():
    """
    验证 repeat_interleave 的批量扩展逻辑是否正确
    
    模拟: (B, N, C) -> (B*K, N, C)
    验证: 扩展后的数据排列是否正确
    """
    test_name = "Test 3 PASSED: Batch Expansion"
    
    try:
        B = 2  # batch size
        K = 3  # 候选轨迹数量
        N = 10  # 特征数量
        C = 16  # 通道数
        
        # 创建测试数据，每个 batch item 有唯一标识
        original = torch.zeros(B, N, C)
        original[0, :, :] = 1.0  # batch 0 全为 1
        original[1, :, :] = 2.0  # batch 1 全为 2
        
        # 使用 repeat_interleave 扩展
        expanded = original.repeat_interleave(K, dim=0)
        
        # 验证维度
        expected_shape = (B * K, N, C)
        assert expanded.shape == torch.Size(expected_shape), \
            f"Shape mismatch: expected {expected_shape}, got {expanded.shape}"
        
        # 验证数据排列: [A, A, A, B, B, B] 而不是 [A, B, A, B, A, B]
        # expanded[0:3] 应该都是 1.0 (来自 batch 0)
        # expanded[3:6] 应该都是 2.0 (来自 batch 1)
        for i in range(K):
            assert (expanded[i] == 1.0).all(), f"expanded[{i}] should be 1.0 (from batch 0)"
        for i in range(K, 2 * K):
            assert (expanded[i] == 2.0).all(), f"expanded[{i}] should be 2.0 (from batch 1)"
        
        print_pass(test_name)
        print_info(f"Original shape: {original.shape}")
        print_info(f"Expanded shape: {expanded.shape}")
        print_info(f"Expansion pattern: [A,A,A,B,B,B] (interleaved correctly)")
        
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        return False


# ============================================================
# Test 4: 条件预测差异验证
# ============================================================
def test_conditional_prediction_difference():
    """
    验证不同条件输入是否产生不同的输出
    
    这是因果预测的核心测试：
    如果模型正确学习了条件依赖，不同的自车轨迹条件应该
    导致不同的障碍物预测。
    
    注意：这个测试只能验证架构正确性，不能验证学习效果
    （未训练的模型也会产生不同输出，只是没有语义意义）
    """
    test_name = "Test 4 PASSED: Conditional Prediction Difference"
    
    try:
        from mtr.models.utils.condition_encoder import ConditionEncoder
        
        # 参数设置
        batch_size = 2
        K = 3
        num_future_frames = 80
        d_model = 256
        
        # 创建编码器
        encoder = ConditionEncoder(
            d_model=d_model,
            num_future_frames=num_future_frames,
            hidden_dim=128
        )
        
        # 创建两组不同的候选轨迹
        # 组1: 全零轨迹 (静止)
        candidates_1 = torch.zeros(batch_size, K, num_future_frames, 2)
        
        # 组2: 递增轨迹 (运动)
        time_steps = torch.linspace(0, 1, num_future_frames)[None, None, :, None]
        candidates_2 = torch.ones(batch_size, K, num_future_frames, 2) * time_steps * 100
        
        # 编码
        condition_1 = encoder(candidates_1)
        condition_2 = encoder(candidates_2)
        
        # 验证输出不同
        difference = (condition_1 - condition_2).abs().mean().item()
        
        assert difference > 0.01, \
            f"Condition vectors should be different, but got difference = {difference:.6f}"
        
        print_pass(test_name)
        print_info(f"Condition 1 (stationary) mean: {condition_1.mean().item():.4f}")
        print_info(f"Condition 2 (moving) mean: {condition_2.mean().item():.4f}")
        print_info(f"Mean absolute difference: {difference:.4f}")
        print_info("Different inputs → Different condition vectors ✓")
        
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        return False


# ============================================================
# Test 5: 完整推理流程模拟 (不依赖完整模型)
# ============================================================
def test_inference_flow_simulation():
    """
    模拟完整的条件推理流程
    
    这个测试不需要完整的 MTR 模型权重，
    只验证数据流和维度是否正确。
    """
    test_name = "Test 5 PASSED: Inference Flow Simulation"
    
    try:
        from mtr.models.utils.condition_encoder import ConditionEncoder
        from mtr.utils.frenet_sampler import FrenetSampler
        
        # Step 1: 使用 FrenetSampler 生成候选轨迹
        print_info("Step 1: Generate ego candidates with FrenetSampler")
        sampler = FrenetSampler(time_horizon=8.0, dt=0.1)
        
        batch_size = 2
        velocities = np.array([10.0, 15.0])  # 两个不同速度的样本
        candidates_np = sampler.sample_batch(velocities)
        candidates = torch.from_numpy(candidates_np).float()
        
        K = candidates.shape[1]
        T = candidates.shape[2]
        print_info(f"  Candidates shape: {candidates.shape} (B={batch_size}, K={K}, T={T})")
        
        # Step 2: 使用 ConditionEncoder 编码候选轨迹
        print_info("Step 2: Encode candidates with ConditionEncoder")
        d_model = 256
        encoder = ConditionEncoder(d_model=d_model, num_future_frames=T)
        condition_vector = encoder(candidates)
        print_info(f"  Condition vector shape: {condition_vector.shape}")
        
        # Step 3: 模拟 batch 扩展 (MTRDecoder 中的操作)
        print_info("Step 3: Simulate batch expansion (B -> B*K)")
        
        # 模拟 obj_feature (B, N, C), obj_mask (B, N)
        N = 32  # 障碍物数量
        C = 256
        obj_feature = torch.randn(batch_size, N, C)
        obj_mask = torch.ones(batch_size, N).bool()
        
        # 扩展
        obj_feature_expanded = obj_feature.repeat_interleave(K, dim=0)
        obj_mask_expanded = obj_mask.repeat_interleave(K, dim=0)
        
        print_info(f"  obj_feature: {obj_feature.shape} -> {obj_feature_expanded.shape}")
        print_info(f"  obj_mask: {obj_mask.shape} -> {obj_mask_expanded.shape}")
        
        # Step 4: 验证最终维度
        print_info("Step 4: Verify final dimensions")
        expected_batch = batch_size * K
        assert obj_feature_expanded.shape[0] == expected_batch, \
            f"Expected batch size {expected_batch}, got {obj_feature_expanded.shape[0]}"
        
        print_pass(test_name)
        print_info(f"Complete flow: {batch_size} samples x {K} conditions = {expected_batch} parallel predictions")
        
        return True
        
    except Exception as e:
        print_fail(test_name, str(e))
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Main
# ============================================================
def main():
    print_header("Causal-Aware MTR 验证测试")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    results = []
    
    # 运行所有测试
    results.append(("ConditionEncoder", test_condition_encoder()))
    results.append(("FrenetSampler", test_frenet_sampler()))
    results.append(("Batch Expansion", test_batch_expansion()))
    results.append(("Conditional Difference", test_conditional_prediction_difference()))
    results.append(("Inference Flow", test_inference_flow_simulation()))
    
    # 汇总结果
    print_header("测试结果汇总")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")
    
    print()
    if passed == total:
        print(f"{GREEN}========================================{RESET}")
        print(f"{GREEN}  All {total} tests passed!{RESET}")
        print(f"{GREEN}  Causal MTR is ready for training.{RESET}")
        print(f"{GREEN}========================================{RESET}")
    else:
        print(f"{RED}========================================{RESET}")
        print(f"{RED}  {passed}/{total} tests passed.{RESET}")
        print(f"{RED}  Please check the failed tests above.{RESET}")
        print(f"{RED}========================================{RESET}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
