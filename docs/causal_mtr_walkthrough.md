# Causal MTR 完整实现 Walkthrough

## 1. 项目背景与目标

### 原始问题
原版 MTR 是一个 **"旁观者模式"** 的预测模型：它预测周围障碍物的轨迹，但 **不考虑自车的行为会如何影响他人**。

### 改造目标
将 MTR 改造为 **"参与者模式"**：预测 "如果我（自车）这样走，障碍物会怎么反应？"

### 核心价值
- **规划决策**: 自动驾驶规划器可以用 K 条候选轨迹分别查询模型，获得 K 组预测，选择最安全的方案
- **因果推理**: 模型不再只是"看"，而是能"想象"不同行动的后果

---

## 2. 实现模块一览

| 模块 | 文件 | 功能 |
|------|------|------|
| **ConditionEncoder** | [condition_encoder.py](file:///c:/Users/12629/Desktop/MTR-master/MTR-master/mtr/models/utils/condition_encoder.py) | 将 K 条自车候选轨迹编码为向量 |
| **FrenetSampler** | [frenet_sampler.py](file:///c:/Users/12629/Desktop/MTR-master/MTR-master/mtr/utils/frenet_sampler.py) | 生成 K 条不同驾驶模式的候选轨迹 |
| **CausalScorer** | [causal_scorer.py](file:///c:/Users/12629/Desktop/MTR-master/MTR-master/mtr/models/utils/causal_scorer.py) | 为 K 个平行世界打分，选择最安全的决策 |
| **MTRDecoder** | [mtr_decoder.py](file:///c:/Users/12629/Desktop/MTR-master/MTR-master/mtr/models/motion_decoder/mtr_decoder.py) | 主模型，集成所有模块 |

---

## 3. 核心改动详解

### 3.1 CausalScorer 打分网络

**为什么需要这个模块**:
我们有 K 个平行世界（对应 K 种自车决策），需要选择最优的一个。

**架构设计**:
```
自车轨迹 (B, K, T, 2)
    ↓ encode_trajectory
自车特征 (B, K, D)
    ↓
    └──────────────────────────┐
                               ↓
邻居预测 (B, K, N, T, 2)     Attention
    ↓ encode_trajectory           ↓
邻居特征 (B, K, N, D) ──────→ 交互特征
                               ↓
                          score_head
                               ↓
                         分数 (B, K)
```

**关键代码**:
```python
# 自车去"看"邻居预测，判断这个世界对自己有利吗
attn_out, _ = self.interaction_attn(
    query=ego_feat,      # 自车特征
    key=neighbor_feat,   # 邻居特征
    value=neighbor_feat
)
scores = self.score_head(fused_feat)  # 输出分数
```

---

### 3.2 因果规划损失 (causal_planning_loss)

**两部分组成**:

| 损失 | 公式 | 作用 |
|------|------|------|
| **模仿损失** | `CrossEntropy(scores, best_k)` | 让模型给接近 GT 的轨迹打高分 |
| **安全损失** | `probs * collision_cost` | 让模型给危险轨迹打低分 |

**模仿损失**:
```python
# 找到离 GT 最近的那条轨迹
dist_to_gt = ||ego_trajs - gt_trajectory||
best_k_idx = argmin(dist_to_gt)
loss_imitation = CrossEntropy(pred_scores, best_k_idx)
```

**安全损失**:
```python
# 计算碰撞代价
min_dist = min(||ego - neighbor||)  # 最近距离
collision_cost = ReLU(1.5m - min_dist)  # 距离小于1.5m就有代价

# 概率加权惩罚
probs = softmax(pred_scores)
loss_safety = sum(probs * collision_cost)
```

---

### 3.3 推理时的 K 世界选择

**流程**:
```
1. 输入 K 条自车候选轨迹
    ↓
2. 模型预测 K 组邻居轨迹 (B*K 并行)
    ↓
3. 计算每个世界的:
   - collision_cost (碰撞代价)
   - causal_scores (打分网络分数，可选)
    ↓
4. final_score = causal_scores - 10 * collision_cost
    ↓
5. best_k = argmax(final_score)
    ↓
6. 输出 best_k 对应的预测
```

**输出内容**:
```python
batch_dict['pred_trajs']           # 最优世界的预测轨迹
batch_dict['all_world_pred_trajs'] # 所有 K 个世界的预测
batch_dict['selected_world_idx']   # 被选中的 K 索引
batch_dict['collision_costs']      # 每个世界的碰撞代价
batch_dict['min_distances']        # 每个世界的最小距离
```

---

## 4. 配置选项

在 YAML 配置文件中添加以下选项：

```yaml
MODEL:
  MOTION_DECODER:
    # === 基础条件预测 ===
    USE_CONTRASTIVE_LOSS: True      # 对比损失 (鼓励不同条件→不同预测)
    CONTRASTIVE_MARGIN: 2.0         # 对比损失的 margin (米)
    
    # === 因果打分网络 ===
    USE_CAUSAL_SCORER: True         # 启用打分网络
    SCORER_NUM_HEADS: 4             # Attention 头数
    SCORER_HIDDEN_DIM: 128          # 隐藏层维度
    
    # === 因果规划损失 ===
    USE_CAUSAL_PLANNING_LOSS: True  # 启用模仿+安全损失
    SAFETY_THRESHOLD: 1.5           # 安全距离阈值 (米)
    
    LOSS_WEIGHTS:
      contrastive: 0.1              # 对比损失权重
      causal_imitation: 1.0         # 模仿损失权重
      causal_safety: 10.0           # 安全损失权重
```

---

## 5. 使用方法

### 训练
```bash
python tools/train.py --cfg_file cfgs/waymo/mtr_causal.yaml --grad_accum_nums 4
```

### 推理
```python
# 生成 K 条候选轨迹
from mtr.utils.frenet_sampler import FrenetSampler
sampler = FrenetSampler(time_horizon=8.0, dt=0.1)
ego_candidates = sampler.sample(current_velocity=10.0)  # (K, T, 2)

# 放入 batch_dict
batch_dict['input_dict']['ego_future_candidates'] = ego_candidates.unsqueeze(0)  # (1, K, T, 2)

# 推理
output = model(batch_dict)

# 结果
print(f"选择了第 {output['selected_world_idx']} 个世界")
print(f"碰撞代价: {output['collision_costs']}")
```

---

## 6. 验证方法

### 功能模块测试
```bash
cd c:\Users\12629\Desktop\MTR-master\MTR-master
python -m mtr.utils.test_causal_mtr
python -m mtr.utils.test_coordinate_system
```

### 新增验证点
- [ ] CausalScorer 输入输出维度
- [ ] causal_planning_loss 梯度流通
- [ ] 推理时 K 世界选择逻辑
- [ ] collision_cost 计算正确性

---

## 7. 数据流总结

```
输入:
├── obj_feature (邻居历史)
├── map_feature (地图)
└── ego_future_candidates (K 条自车候选) 
    ├── 训练时: 自动生成 (GT + 噪声) [NEW]
    └── 推理时: 自动生成 (FrenetSampler) [NEW]

Encoder (运行 1 次):
└── context_features

Decoder (批量运行 B*K 次并行):
├── condition_vector ← ConditionEncoder(ego_candidates)
├── intention_query + condition_fusion
└── pred_trajs (B*K, 64, T, 7)

后处理:
├── 训练: causal_planning_loss (模仿 + 安全)
└── 推理: K 世界选择 (CausalScorer + collision_cost)

输出:
├── pred_trajs (最优世界的预测)
├── selected_world_idx (选择的 K)
└── all_world_pred_trajs (所有 K 个世界的预测)
```
