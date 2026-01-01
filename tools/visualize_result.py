import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys
import logging
import torch
from collections import defaultdict

# 确保能找到 mtr 库
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mtr.datasets.waymo.waymo_dataset import WaymoDataset
from mtr.config import cfg, cfg_from_yaml_file

def create_logger():
    logger = logging.getLogger("MTR_Visualizer")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def get_safe_item(idx):
    try:
        if isinstance(idx, torch.Tensor): idx = idx.cpu().numpy()
        idx_arr = np.array(idx)
        if idx_arr.ndim == 0: return int(idx_arr)
        if idx_arr.size > 0: return int(idx_arr.flatten()[0])
        return 0
    except: return 0

def draw_single_agent(ax, data_dict, pred_traj, agent_id_label=None):
    cx, cy = 0, 0
    target_idx = get_safe_item(data_dict.get('track_index_to_predict', 0))

    # ==========================================
    # 1. 准备数据
    # ==========================================
    obj_trajs = None
    if 'obj_trajs' in data_dict:
        obj_trajs = data_dict['obj_trajs']
        if isinstance(obj_trajs, torch.Tensor): obj_trajs = obj_trajs.cpu().numpy()
        if obj_trajs.ndim == 4: obj_trajs = obj_trajs[0]

    # 获取全局中心点
    center_world = np.zeros(2)
    if 'center_objects_world' in data_dict:
        c = data_dict['center_objects_world']
        if isinstance(c, torch.Tensor): c = c.cpu().numpy()
        center_world = c.flatten()[:2]

    # ==========================================
    # 2. 绘制基础元素 (Box & GT)
    # ==========================================
    if obj_trajs is not None and obj_trajs.shape[0] > target_idx:
        current_idx = 10 
        
        # 2.1 绘制车框 (Green Box)
        if obj_trajs.shape[1] > current_idx:
            agent_state = obj_trajs[target_idx, current_idx]
            if len(agent_state) >= 7:
                cx, cy, _, l, w, _, yaw = agent_state[:7]
                cx, cy, l, w, yaw = float(cx), float(cy), float(l), float(w), float(yaw)

                rect = Rectangle((cx - l/2, cy - w/2), l, w, color='#32CD32', alpha=1.0, zorder=30)
                import matplotlib.transforms as transforms
                t = transforms.Affine2D().rotate_around(cx, cy, yaw) + ax.transData
                rect.set_transform(t)
                ax.add_patch(rect)
                
                if agent_id_label is not None:
                    ax.text(cx, cy, str(agent_id_label), color='white', fontsize=9, fontweight='bold', zorder=35, ha='center', va='center')

        # 2.2 绘制真值 (Blue GT) - 修复版
        valid_gt = None
        
        # 优先尝试从 future_state 获取 (这是专门存未来的字段)
        if 'obj_trajs_future_state' in data_dict:
            future_state = data_dict['obj_trajs_future_state']
            if isinstance(future_state, torch.Tensor): future_state = future_state.cpu().numpy()
            if future_state.ndim == 4: future_state = future_state[0] # Handle batch
            
            # future_state 通常是 (N, 80, 4) -> xy, vxy
            if future_state.shape[0] > target_idx:
                gt_pts = future_state[target_idx, :, :2]
                if np.abs(gt_pts).sum() > 0.1: # 确保不是全0
                    valid_gt = gt_pts

        # 如果没有 future_state，再尝试从 obj_trajs 切片 (兜底)
        if valid_gt is None and obj_trajs.shape[1] > 11:
             gt_pts = obj_trajs[target_idx, 11:, :2]
             if np.abs(gt_pts).sum() > 0.1:
                 valid_gt = gt_pts
        
        # 开始画
        if valid_gt is not None:
            ax.plot(valid_gt[:, 0], valid_gt[:, 1], color='#0000FF', linewidth=2, zorder=25, alpha=0.8, label='GT')
            ax.scatter(valid_gt[-1, 0], valid_gt[-1, 1], color='#0000FF', s=20, zorder=25, marker='x')

    # ==========================================
    # 3. 绘制预测 (Prediction)
    # ==========================================
    if pred_traj is not None:
        if isinstance(pred_traj, torch.Tensor): pred_traj = pred_traj.cpu().numpy()
        if pred_traj.shape[-1] > 2: pred_traj = pred_traj[..., :2]
        if pred_traj.ndim == 2: pred_traj = pred_traj[None, ...]

        # 强力对齐
        if np.abs(pred_traj).mean() > 500:
            pred_traj = pred_traj.copy()
            pred_traj[..., :2] -= center_world

        if cx != 0 and cy != 0:
            pred_start = pred_traj[:, 0, :2]
            offset = pred_start[0] - np.array([cx, cy])
            if np.linalg.norm(offset) > 2.0:
                pred_traj[..., :2] -= offset

        K, T, _ = pred_traj.shape
        cmap = plt.get_cmap('autumn_r')
        for k in range(K):
            traj = pred_traj[k]
            colors = [cmap(i/T) for i in range(T)]
            ax.scatter(traj[:, 0], traj[:, 1], c=colors, s=12, alpha=0.8, zorder=20, edgecolors='none')

    return cx, cy
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='../data/waymo')
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='vis_output')
    parser.add_argument('--scenario_id', type=str, default=None)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.DATA_CONFIG.DATA_ROOT = args.data_path
    if not hasattr(cfg.DATA_CONFIG, 'CLASS_NAMES'):
        cfg.DATA_CONFIG.CLASS_NAMES = ['Vehicle', 'Pedestrian', 'Cyclist']

    logger = create_logger()
    try:
        dataset = WaymoDataset(dataset_cfg=cfg.DATA_CONFIG, training=False, logger=logger)
    except: return

    print(f"Loading results from {args.result_path}...")
    with open(args.result_path, 'rb') as f:
        raw_results = pickle.load(f)
    
    results = []
    if isinstance(raw_results, list) and len(raw_results) > 0:
        if isinstance(raw_results[0], list):
            for batch in raw_results: results.extend(batch)
        else: results = raw_results

    print("Grouping results by Scenario ID...")
    scene_groups = defaultdict(list)
    for res in results:
        s_id = res.get('scenario_id', res.get('scene_id'))
        if s_id: scene_groups[s_id].append(res)
    
    print(f"Total Scenarios: {len(scene_groups)}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.scenario_id:
        target_ids = [args.scenario_id] if args.scenario_id in scene_groups else []
    else:
        target_ids = list(scene_groups.keys())[:20]

    for s_id in target_ids:
        preds_list = scene_groups[s_id]
        print(f"Drawing {s_id} ({len(preds_list)} agents)...")
        
        data_dict = None
        for i in range(len(dataset)):
            try:
                info = dataset.infos[i]
                curr_id = info.get('scenario_id', info.get('scene_id'))
            except: continue
            if curr_id == s_id:
                data_dict = dataset[i]
                break
        if data_dict is None: continue

        fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
        
        # 1. 画地图 (绝对不转！)
        if 'map_polylines' in data_dict:
            polys = data_dict['map_polylines']
            mask = data_dict.get('map_polylines_mask', None)
            if isinstance(polys, torch.Tensor): polys = polys.cpu().numpy()
            if isinstance(mask, torch.Tensor): mask = mask.cpu().numpy()
            if polys.ndim == 4: polys = polys[0]
            if mask is not None and mask.ndim == 3: mask = mask[0]

            for i in range(len(polys)):
                poly = polys[i]
                if mask is not None:
                    if i < len(mask):
                        valid = mask[i].astype(bool)
                        if valid.sum() < 2: continue
                        poly = poly[valid]
                ax.plot(poly[:, 0], poly[:, 1], color='gray', alpha=0.3, linewidth=0.5, zorder=0)

        # 2. 循环画车 (带 Blue GT)
        center_x_sum, center_y_sum = 0, 0
        valid_agents_count = 0

        for idx, pred_item in enumerate(preds_list):
            current_data_dict = data_dict.copy()
            if 'track_index_to_predict' in pred_item:
                current_data_dict['track_index_to_predict'] = pred_item['track_index_to_predict']
            
            cx, cy = draw_single_agent(ax, current_data_dict, pred_item['pred_trajs'], agent_id_label=idx+1)
            
            if cx != 0 or cy != 0:
                center_x_sum += cx
                center_y_sum += cy
                valid_agents_count += 1

        # 3. 设置视野
        if valid_agents_count > 0:
            avg_cx = center_x_sum / valid_agents_count
            avg_cy = center_y_sum / valid_agents_count
            ax.set_xlim(avg_cx - 60, avg_cx + 60)
            ax.set_ylim(avg_cy - 60, avg_cy + 60)

        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        save_file = os.path.join(args.save_dir, f'{s_id}_fixed.png')
        plt.savefig(save_file, bbox_inches='tight', dpi=150)
        print(f"Saved: {save_file}")
        plt.close()

if __name__ == '__main__':
    main()