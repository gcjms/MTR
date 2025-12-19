# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508

# Published at NeurIPS 2022

# Written by Shaoshuai Shi

# All Rights Reserved

import copy

import pickle

import torch

import torch.nn as nn

import torch.nn.functional as F

from mtr.models.utils.transformer import transformer_decoder_layer

from mtr.models.utils.transformer import position_encoding_utils

from mtr.models.utils import common_layers

from mtr.utils import common_utils, loss_utils, motion_utils

from mtr.config import cfg

class MTRDecoder(nn.Module):

    def __init__(self, in_channels, config):

        super().__init__()

        self.model_cfg = config

        self.object_type = self.model_cfg.OBJECT_TYPE

        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES

        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES

        self.use_place_holder = self.model_cfg.get('USE_PLACE_HOLDER', False)

        self.d_model = self.model_cfg.D_MODEL

        self.num_decoder_layers = self.model_cfg.NUM_DECODER_LAYERS

        # define the cross-attn layers

        self.in_proj_center_obj = nn.Sequential(

            nn.Linear(in_channels, self.d_model),

            nn.ReLU(),

            nn.Linear(self.d_model, self.d_model),

        )

        self.in_proj_obj, self.obj_decoder_layers = self.build_transformer_decoder(

            in_channels=in_channels,

            d_model=self.d_model,

            nhead=self.model_cfg.NUM_ATTN_HEAD,

            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),

            num_decoder_layers=self.num_decoder_layers,

            use_local_attn=False

        )

        map_d_model = self.model_cfg.get('MAP_D_MODEL', self.d_model)

        self.in_proj_map, self.map_decoder_layers = self.build_transformer_decoder(

            in_channels=in_channels,

            d_model=map_d_model,

            nhead=self.model_cfg.NUM_ATTN_HEAD,

            dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),

            num_decoder_layers=self.num_decoder_layers,

            use_local_attn=True

        )

        if map_d_model != self.d_model:

            temp_layer = nn.Linear(self.d_model, map_d_model)

            self.map_query_content_mlps = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

            self.map_query_embed_mlps = nn.Linear(self.d_model, map_d_model)

        else:

            self.map_query_content_mlps = self.map_query_embed_mlps = None

        # define the dense future prediction layers

        self.build_dense_future_prediction_layers(

            hidden_dim=self.d_model, num_future_frames=self.num_future_frames

        )

        # define the motion query

        self.intention_points, self.intention_query, self.intention_query_mlps = self.build_motion_query(

            self.d_model, use_place_holder=self.use_place_holder

        )

        # define the motion head

        temp_layer = common_layers.build_mlps(c_in=self.d_model * 2 + map_d_model, mlp_channels=[self.d_model, self.d_model], ret_before_act=True)

        self.query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(self.num_decoder_layers)])

        self.motion_reg_heads, self.motion_cls_heads, self.motion_vel_heads = self.build_motion_head(

            in_channels=self.d_model, hidden_size=self.d_model, num_decoder_layers=self.num_decoder_layers

        )

        self.forward_ret_dict = {}

    def build_dense_future_prediction_layers(self, hidden_dim, num_future_frames):

        self.obj_pos_encoding_layer = common_layers.build_mlps(

            c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True

        )

        self.dense_future_head = common_layers.build_mlps(

            c_in=hidden_dim * 2,

            mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True

        )

        self.future_traj_mlps = common_layers.build_mlps(

            c_in=4 * self.num_future_frames, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True

        )

        self.traj_fusion_mlps = common_layers.build_mlps(

            c_in=hidden_dim * 2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True

        )

    def build_transformer_decoder(self, in_channels, d_model, nhead, dropout=0.1, num_decoder_layers=1, use_local_attn=False):

        in_proj_layer = nn.Sequential(

            nn.Linear(in_channels, d_model),

            nn.ReLU(),

            nn.Linear(d_model, d_model),

        )

        decoder_layer = transformer_decoder_layer.TransformerDecoderLayer(

            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout,

            activation="relu", normalize_before=False, keep_query_pos=False,

            rm_self_attn_decoder=False, use_local_attn=use_local_attn

        )

        decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])

        return in_proj_layer, decoder_layers

    def build_motion_query(self, d_model, use_place_holder=False):

        intention_points = intention_query = intention_query_mlps = None

        if use_place_holder:

            raise NotImplementedError

        else:

            intention_points_file = cfg.ROOT_DIR / self.model_cfg.INTENTION_POINTS_FILE

            with open(intention_points_file, 'rb') as f:

                intention_points_dict = pickle.load(f)

            intention_points = {}

            for cur_type in self.object_type:

                cur_intention_points = intention_points_dict[cur_type]

                cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2).cuda()

                intention_points[cur_type] = cur_intention_points

            intention_query_mlps = common_layers.build_mlps(

                c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True

            )

        return intention_points, intention_query, intention_query_mlps

    def build_motion_head(self, in_channels, hidden_size, num_decoder_layers):

        motion_reg_head =  common_layers.build_mlps(

            c_in=in_channels,

            mlp_channels=[hidden_size, hidden_size, self.num_future_frames * 7], ret_before_act=True

        )

        motion_cls_head =  common_layers.build_mlps(

            c_in=in_channels,

            mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True

        )

        motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])

        motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])

        motion_vel_heads = None

        return motion_reg_heads, motion_cls_heads, motion_vel_heads

    def apply_dense_future_prediction(self, obj_feature, obj_mask, obj_pos):

            """

            执行致密未来预测（Dense Future Prediction）。

            这是一个辅助任务：让每个障碍物利用简单的 MLP 预测自己的未来，

            然后把这个“对未来的预判”编码回特征中。

            """

            # obj_feature: (Batch, N_obj, C) - 障碍物的历史特征

            # obj_mask:    (Batch, N_obj)    - 有效性掩码

            # obj_pos:     (Batch, N_obj, 3) - 障碍物当前时刻的位置 (x, y, z)

            num_center_objects, num_objects, _ = obj_feature.shape

            # ============================================================

            # 1. 提取有效数据 (Flatten & Masking)

            # ============================================================

            # 利用布尔索引，把 Batch 中所有有效的障碍物特征提取出来拍扁成一维列表

            # obj_pos_valid: (M, 2) -> M 是整个 Batch 中所有有效障碍物的总数，只取 x, y

            obj_pos_valid = obj_pos[obj_mask][..., 0:2]

            # obj_feature_valid: (M, C_in)

            obj_feature_valid = obj_feature[obj_mask]

            # ============================================================

            # 2. 特征融合与简单预测 (MLP Prediction)

            # ============================================================

            # 把“当前位置”编码成特征向量 (M, 2) -> (M, C_hidden)

            obj_pos_feature_valid = self.obj_pos_encoding_layer(obj_pos_valid)

           

            # 将“位置特征”和“历史特征”拼接

            # obj_fused_feature_valid: (M, C_hidden + C_in)

            obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

            # 过 MLP 头，直接回归出未来所有帧的参数

            # 输出维度通常是: num_future_frames * 7 (x, y, vx, vy, heading, etc.)

            pred_dense_trajs_valid = self.dense_future_head(obj_fused_feature_valid)

           

            # 重塑形状: (M, T_future, 7)

            pred_dense_trajs_valid = pred_dense_trajs_valid.view(pred_dense_trajs_valid.shape[0], self.num_future_frames, 7)

            # ============================================================

            # 3. 坐标还原 (Offset -> Absolute)

            # ============================================================

            # 预测出来的通常是相对于当前位置的偏移量 (dx, dy)，需要加上当前位置 (x, y) 得到绝对坐标

            # temp_center: (M, T_future, 2)

            temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]

           

            # 把算好的绝对坐标拼回去，替换原来的前两维

            pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

            # ============================================================

            # 4. 未来特征反哺 (Feature Update)

            # ============================================================

            # 从预测结果中挑出重要的信息：位置(0,1) 和 速度/朝向(最后两维 -2,-1)

            # 将这些几何信息展平: (M, T_future, 4) -> (M, T_future * 4)

            obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1, end_dim=2)

           

            # 用一个 MLP 把这些几何轨迹信息编码成高维特征 (M, C_hidden)

            obj_future_feature_valid = self.future_traj_mlps(obj_future_input_valid)

            # 核心步骤：把“预测出的未来特征”和“原来的历史特征”拼在一起

            obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)

           

            # 再次融合，得到包含未来意图的新特征

            # obj_feature_valid: (M, C_out)

            obj_feature_valid = self.traj_fusion_mlps(obj_full_trajs_feature)

            # ============================================================

            # 5. 数据还原回 Batch 结构 (Scatter)

            # ============================================================

            # 准备一个全 0 的容器，形状和输入一样

            ret_obj_feature = torch.zeros_like(obj_feature)

            # 把算好的有效特征填回它们原来的位置（无效位置保持为 0）

            ret_obj_feature[obj_mask] = obj_feature_valid

            # 同样把预测出的轨迹也填回去，用于计算辅助 Loss

            ret_pred_dense_future_trajs = obj_feature.new_zeros(num_center_objects, num_objects, self.num_future_frames, 7)

            ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid

           

            # 存入字典，供 Loss 计算模块使用

            self.forward_ret_dict['pred_dense_trajs'] = ret_pred_dense_future_trajs

            # 返回更新后的特征（包含未来信息）和预测轨迹

            return ret_obj_feature, ret_pred_dense_future_trajs

    def get_motion_query(self, center_objects_type):

        num_center_objects = len(center_objects_type)

        if self.use_place_holder:

            raise NotImplementedError

        else:

            intention_points = torch.stack([

                self.intention_points[center_objects_type[obj_idx]]

                for obj_idx in range(num_center_objects)], dim=0)

            intention_points = intention_points.permute(1, 0, 2)  # (num_query, num_center_objects, 2)

            # 把intention_points这个物理坐标点，通过正弦位置编码（Sine Positional Encoding）和 MLP 升维后得到的高维语义特征

            # 它不仅包含了位置信息，还转换成了 Decoder 能够理解和处理的“查询语言”

            intention_query = position_encoding_utils.gen_sineembed_for_position(intention_points, hidden_dim=self.d_model)

            # 通过 MLP 再过一层，让网络学会如何更好地表达这个位置

            intention_query = self.intention_query_mlps(intention_query.view(-1, self.d_model)).view(-1, num_center_objects, self.d_model)  # (num_query, num_center_objects, C)

        return intention_query, intention_points

    def apply_cross_attention(self, kv_feature, kv_mask, kv_pos, query_content, query_embed, attention_layer,
                                dynamic_query_center=None, layer_idx=0, use_local_attn=False, query_index_pair=None,
                                query_content_pre_mlp=None, query_embed_pre_mlp=None):
            """
            Args:
                kv_feature (B, N, C): 环境特征（Key/Value）。比如周围所有车辆、车道线的特征。
                kv_mask (B, N): 掩码。标记哪些是真实数据，哪些是Padding的空数据。
                kv_pos (B, N, 3): 环境物体的位置坐标 (x, y, z)。
                query_content (M, B, C): Query的内容特征。M=64(模式数)。随着层数变深，这里面存的是"语义信息"（比如：前方有障碍）。
                query_embed (M, B, C): Query的意图嵌入。代表这64个模式原本的"身份信息"（比如：我是左转模式）。
                dynamic_query_center (M, B, 2): 动态锚点中心。重要！这是当前层Query在地图上的绝对坐标 (x, y)。
                attention_layer: Transformer的注意力层对象。
                query_index_pair (B, M, K): 局部注意力用的索引。如果开启Local模式，每个Query只看最近的K个邻居。
            """

            # 1. 预处理 (Pre-MLP)
            # 有时候在进入Attention之前，需要先通过一个全连接层(MLP)调整一下特征维度或进行非线性变换
            if query_content_pre_mlp is not None:
                query_content = query_content_pre_mlp(query_content)
            if query_embed_pre_mlp is not None:
                query_embed = query_embed_pre_mlp(query_embed)

            # 获取维度信息：num_q=64(模式数), d_model=256(特征维度)
            num_q, batch_size, d_model = query_content.shape

            # 2. 生成 Query 的几何位置编码 (Geometric Embedding)
            # dynamic_query_center 是这 64 个锚点当前在地图上的 (x, y) 坐标
            # 这里用正弦位置编码 (Sinusoidal PE) 把坐标变成向量，让 Attention 知道"我在哪"
            # searching_query: (M, B, C)
            searching_query = position_encoding_utils.gen_sineembed_for_position(dynamic_query_center, hidden_dim=d_model)

            # 3. 生成 Key/Value 的几何位置编码
            # kv_pos 原本是 (B, N, 3)，这里 permute 成 (N, B, 3)，并只取前两维 (x, y)
            kv_pos = kv_pos.permute(1, 0, 2)[:, :, 0:2]
            # 同样把环境物体的位置变成向量，让 Attention 知道"物体在哪"
            kv_pos_embed = position_encoding_utils.gen_sineembed_for_position(kv_pos, hidden_dim=d_model)

            # -----------------------------------------------------------------
            # 分支 A: 全局注意力 (Global Attention) - 默认模式
            # 每一个 Query 都会看环境里的每一个物体 (All-to-All)
            # -----------------------------------------------------------------
            if not use_local_attn:
                query_feature = attention_layer(
                    tgt=query_content,                # Query的内容 (Q_content)
                    query_pos=query_embed,            # Query的意图身份 (Q_pos)
                    query_sine_embed=searching_query, # Query的几何位置 (Q_geometry) -> 用来计算和物体的相对距离
                    memory=kv_feature.permute(1, 0, 2), # Key/Value的内容 (K, V) -> 环境特征
                    memory_key_padding_mask=~kv_mask,   # 告诉它别看 Padding 的空数据
                    pos=kv_pos_embed,                   # Key/Value的位置 (K_pos)
                    is_first=(layer_idx == 0)           #如果是第一层，会有特殊的初始化处理
                )  # 输出形状: (M, B, C) -> 更新后的 Query 特征

            # -----------------------------------------------------------------
            # 分支 B: 局部注意力 (Local Attention) - 加速/优化模式
            # 每一个 Query 只看离自己最近的 K 个物体 (Sparse Attention)
            # -----------------------------------------------------------------
            else:
                batch_size, num_kv, _ = kv_feature.shape
                
                # 扁平化操作 (Flatten)
                # 为了高效计算，把 Batch 维度和 N 维度拼起来 -> (B*N, C)
                kv_feature_stack = kv_feature.flatten(start_dim=0, end_dim=1)
                kv_pos_embed_stack = kv_pos_embed.permute(1, 0, 2).contiguous().flatten(start_dim=0, end_dim=1)
                kv_mask_stack = kv_mask.view(-1)

                # 记录每个样本有多少个物体，方便后续恢复索引
                key_batch_cnt = num_kv * torch.ones(batch_size).int().to(kv_feature.device)

                # query_index_pair 里面存的是 KNN 搜索得到的索引：即每个 Query 应该关注哪几个物体
                query_index_pair = query_index_pair.view(batch_size * num_q, -1)
                
                # 生成 Batch 索引辅助向量
                index_pair_batch = torch.arange(batch_size).type_as(key_batch_cnt)[:, None].repeat(1, num_q).view(-1)  
                assert len(query_index_pair) == len(index_pair_batch)

                # 调用支持局部索引的 Attention 层
                query_feature = attention_layer(
                    tgt=query_content,
                    query_pos=query_embed,
                    query_sine_embed=searching_query,
                    memory=kv_feature_stack,       # 拍扁后的环境特征
                    memory_valid_mask=kv_mask_stack,
                    pos=kv_pos_embed_stack,        # 拍扁后的环境位置
                    is_first=(layer_idx == 0),
                    key_batch_cnt=key_batch_cnt,
                    index_pair=query_index_pair,   # 核心：只对这些索引对进行 Attention 计算！ local_attention的新增指出 只看关键障碍物
                    index_pair_batch=index_pair_batch
                )

                # 恢复形状: (B*M, C) -> (B, M, C) -> (M, B, C)
                query_feature = query_feature.view(batch_size, num_q, d_model).permute(1, 0, 2)

            return query_feature # 返回吸收了环境信息后的 Query 特征
    
    def apply_dynamic_map_collection(self, map_pos, map_mask, pred_waypoints, base_region_offset, num_query, num_waypoint_polylines=128, num_base_polylines=256, base_map_idxs=None):

        map_pos = map_pos.clone()

        map_pos[~map_mask] = 10000000.0

        num_polylines = map_pos.shape[1]

        if base_map_idxs is None:

            base_points = torch.tensor(base_region_offset).type_as(map_pos)

            base_dist = (map_pos[:, :, 0:2] - base_points[None, None, :]).norm(dim=-1)  # (num_center_objects, num_polylines)

            base_topk_dist, base_map_idxs = base_dist.topk(k=min(num_polylines, num_base_polylines), dim=-1, largest=False)  # (num_center_objects, topk)

            base_map_idxs[base_topk_dist > 10000000] = -1

            base_map_idxs = base_map_idxs[:, None, :].repeat(1, num_query, 1)  # (num_center_objects, num_query, num_base_polylines)

            if base_map_idxs.shape[-1] < num_base_polylines:

                base_map_idxs = F.pad(base_map_idxs, pad=(0, num_base_polylines - base_map_idxs.shape[-1]), mode='constant', value=-1)

        dynamic_dist = (pred_waypoints[:, :, None, :, 0:2] - map_pos[:, None, :, None, 0:2]).norm(dim=-1)  # (num_center_objects, num_query, num_polylines, num_timestamps)

        dynamic_dist = dynamic_dist.min(dim=-1)[0]  # (num_center_objects, num_query, num_polylines)

        dynamic_topk_dist, dynamic_map_idxs = dynamic_dist.topk(k=min(num_polylines, num_waypoint_polylines), dim=-1, largest=False)

        dynamic_map_idxs[dynamic_topk_dist > 10000000] = -1

        if dynamic_map_idxs.shape[-1] < num_waypoint_polylines:

            dynamic_map_idxs = F.pad(dynamic_map_idxs, pad=(0, num_waypoint_polylines - dynamic_map_idxs.shape[-1]), mode='constant', value=-1)

        collected_idxs = torch.cat((base_map_idxs, dynamic_map_idxs), dim=-1)  # (num_center_objects, num_query, num_collected_polylines)

        # remove duplicate indices

        sorted_idxs = collected_idxs.sort(dim=-1)[0]

        duplicate_mask_slice = (sorted_idxs[..., 1:] - sorted_idxs[..., :-1] != 0)  # (num_center_objects, num_query, num_collected_polylines - 1)

        duplicate_mask = torch.ones_like(collected_idxs).bool()

        duplicate_mask[..., 1:] = duplicate_mask_slice

        sorted_idxs[~duplicate_mask] = -1

        return sorted_idxs.int(), base_map_idxs

    def apply_transformer_decoder(self, center_objects_feature, center_objects_type, obj_feature, obj_mask, obj_pos, map_feature, map_mask, map_pos):

        """
        MTR Decoder 的核心前向传播。
        执行多层 Transformer Decoder Block，每层都包含：
        1. Cross-Attn (Agent-Agent)
        2. Dynamic Map Collection (基于当前预测位置动态搜索地图)
        3. Cross-Attn (Agent-Map)
        4. Feature Fusion & Prediction
        5. Trajectory Update (用于下一层的搜索引导)
        """
        # 1. 获取意图锚点 (Intention Query)
        # 这里的 intention_points 是通过 K-Means 聚类得到的静态锚点 (比如 64 个终点)
        # intention_query: (num_query, Batch, C) - 这里的 num_query 也就是 num_modes (比如 64)， C=256
        intention_query, intention_points = self.get_motion_query(center_objects_type)

        # 初始化 Query 的内容特征，第一层输入通常为 0

        query_content = torch.zeros_like(intention_query)

        # 保存锚点供 Loss 计算使用
        self.forward_ret_dict['intention_points'] = intention_points.permute(1, 0, 2)  # (Batch, num_query, 2)
        num_center_objects = query_content.shape[1] # Batch Size
        num_query = query_content.shape[0]          # 模式数量 (64)

        # 2. 扩展中心对象特征 (Ego Feature Expansion)
        # None就是前面加一个维度1,repeat再复制64份
        # center_objects_feature: (Batch, C) -> (1, Batch, C) -> (num_query, Batch, C)
        # 每一个预测模式(Mode)都需要一份主角的历史特征作为基础

        center_objects_feature = center_objects_feature[None, :, :].repeat(num_query, 1, 1)
        base_map_idxs = None
 
        # 3. 初始化轨迹路点 (Waypoints)

        # 在第 0 层，我们将静态的意图锚点作为初始的“预测轨迹”

        # 形状变换: (num_query, Batch, 2) -> (Batch, num_query, 2) -> (Batch, num_query, 1, 2)

        # 这里的 '1' 代表时间步，此时只有一个终点

        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]
        # 初始化动态查询中心，用于计算相对位置编码
        dynamic_query_center = intention_points
        pred_list = []
        # ============================================================
        # 4. 进入迭代解码循环 (Iterative Refinement Loop)
        # ============================================================

        for layer_idx in range(self.num_decoder_layers):
            # --- A. 交互：Agent-Agent Attention ---
            # Query (意图) 去关注环境中的其他动态障碍物 (obj_feature)
            # 这一步是为了理解“如果不撞车，我该怎么走”

            obj_query_feature = self.apply_cross_attention(
                # 来源是 Encoder 输出的场景中所有 Agent 的历史特征。
                # 形状: (Batch, Num_Agents, C) -> 比如 (4, 128, 256)
                kv_feature=obj_feature, 
                
                # 掩码：告诉 Attention 哪些 Agent 是真的，哪些是填充(Padding)的空数据，不要看空数据。
                # 形状: (Batch, Num_Agents)
                kv_mask=obj_mask,       
                
                # 场景中所有 Agent 的历史最后位置 (x, y)。
                # 用于计算相对位置编码，让 Query 知道这些 Agent 离自己有多远。
                # 形状: (Batch, Num_Agents, 3)
                kv_pos=obj_pos,         

                # ================= 查询者 (Target / Query) =================
                # Query 的"内容"特征。
                # 第 0 层时通常是全 0 (或者来自主角的特征)，后面几层则是上一层的输出。
                # 它是 Transformer 真正要更新和学习的向量。
                # 形状: (num_query, Batch, C) -> 比如 (64, 4, 256)
                # 随着层数加深，它越来越丰富，包含了“环境里有辆卡车在逼近”这种语义信息
                query_content=query_content, 
                
                # Query 的"位置"嵌入 (Intention Query)。
                # 代表这 64 个意图锚点最初的几何含义（比如代表左转、直行、右转的向量）
                # 身份/位置。它告诉模型“我是第 3 号模式，我负责向东北方向预测”
                # 通常作为 Position Embedding 加在 Content 上
                # 形状: (num_query, Batch, C)
                query_embed=intention_query, 

                # ================= 模型组件与辅助信息 =================
                # 当前使用的 Transformer Decoder 层对象 (里面包含了 Linear 层和 MultiheadAttention)。
                attention_layer=self.obj_decoder_layers[layer_idx], 

                # 动态锚点中心 (x, y)。
                # MTR 是层级细化的，每一层预测完后，锚点的位置会动 (Refine)。
                # 这里传入的是当前这一层锚点在地图上的绝对坐标，用于和 kv_pos 计算相对距离。
                # 形状: (num_query, Batch, 2)
                dynamic_query_center=dynamic_query_center, 

                # 当前是第几层 Decoder (比如 0, 1, 2...)
                layer_idx=layer_idx
            )

            # 输出 obj_query_feature: 
            # 经过这一层"吸星大法"后，这 64 个 Query 吸收了环境特征，变得更强了。
            # 形状: (num_query, Batch, C)


            # --- B. 动态地图收集 (Dynamic Map Collection) ---
            # 这是 MTR 的精髓：不看全图，只看“我当前预测路径”附近的地图
            # pred_waypoints: 上一层的预测结果 (或初始锚点)。
            # 返回的 collected_idxs 是筛选出的地图线段索引

            collected_idxs, base_map_idxs = self.apply_dynamic_map_collection(

                map_pos=map_pos, map_mask=map_mask,

                pred_waypoints=pred_waypoints, # <--- 关键：用上一轮的预测去搜地图

                base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,

                num_waypoint_polylines=self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES,

                num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,

                base_map_idxs=base_map_idxs,

                num_query=num_query

            )

            # --- C. 交互：Agent-Map Attention ---
            # Query 去关注刚刚筛选出来的“局部地图特征”
            # use_local_attn=True: 说明这是一个稀疏注意力，只计算 collected_idxs 指定的元素

            map_query_feature = self.apply_cross_attention(
                kv_feature=map_feature, kv_mask=map_mask, kv_pos=map_pos,
                query_content=query_content, query_embed=intention_query,
                attention_layer=self.map_decoder_layers[layer_idx],
                layer_idx=layer_idx,
                dynamic_query_center=dynamic_query_center,
                use_local_attn=True,
                query_index_pair=collected_idxs, # 传入筛选后的索引 local attention用于只关注附近障碍物
                query_content_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_pre_mlp=self.map_query_embed_mlps
            )

            # --- D. 特征融合 (Feature Fusion) ---
            # 将三者拼起来：[主角历史, 环境障碍物交互, 地图交互]
            query_feature = torch.cat([center_objects_feature, obj_query_feature, map_query_feature], dim=-1)

            # 使用 MLP 融合特征，更新 query_content
            # 更新后的 content 包含了当前层对环境的所有理解
 
            # self.query_feature_fusion_layers[layer_idx]()就是个mlp ->  nn.Linear
            # flatten 把第 0 维和第 1 维合并(相乘)！第 2 维保持原样 -> mlp一般就接受两维的输入
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1)

            # --- E. 运动预测 (Motion Prediction Head) ---
            # 准备数据: (num_query * Batch, C)
            # 初始：[64个模式][4个路口][256特征]
            # permute：变形成 [4个路口][64个模式][256特征] （但内存里还是乱的）
            # contiguous：把内存搬整齐，确保第 1 个路口的 64 个模式在物理上挨在一起。
            # view：把前两个维度捏扁，变成 [256个待预测的意图][256特征]。
            query_content_t = query_content.permute(1, 0, 2).contiguous().view(num_center_objects * num_query, -1)
            # 1. 预测概率分数 (Classification)
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)           

            # 2. 预测轨迹 (Regression)

            if self.motion_vel_heads is not None:
                # 如果速度头是分开的
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 5)
                pred_vel = self.motion_vel_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 2)
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)

            else:

                # 否则直接回归 7 个参数 (x, y, heading, v_x, v_y, ...)
                pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 7)

            # 将当前层的预测结果存入列表 (用于计算 Auxiliary Loss)

            pred_list.append([pred_scores, pred_trajs])

            # --- F. 迭代更新 (Update for Next Layer) ---
            # 这是关键的一步：用当前层预测出来的轨迹，去更新 pred_waypoints
            # 下一层循环做 Dynamic Map Collection 时，就会以这个更准的轨迹为中心去搜地图
            pred_waypoints = pred_trajs[:, :, :, 0:2]

            # 更新 query center (通常取轨迹终点)，用于下一层 Attention 的位置编码基
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].contiguous().permute(1, 0, 2)  # (num_query, Batch, 2)
        if self.use_place_holder:
            raise NotImplementedError
        assert len(pred_list) == self.num_decoder_layers
        return pred_list

    def get_decoder_loss(self, tb_pre_tag=''):

        center_gt_trajs = self.forward_ret_dict['center_gt_trajs'].cuda()

        center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask'].cuda()

        center_gt_final_valid_idx = self.forward_ret_dict['center_gt_final_valid_idx'].long()

        assert center_gt_trajs.shape[-1] == 4

        pred_list = self.forward_ret_dict['pred_list']

        intention_points = self.forward_ret_dict['intention_points']  # (num_center_objects, num_query, 2)

        num_center_objects = center_gt_trajs.shape[0]

        center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2]  # (num_center_objects, 2)

        if not self.use_place_holder:

            dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  # (num_center_objects, num_query)

            center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)

        else:

            raise NotImplementedError

        tb_dict = {}

        disp_dict = {}

        total_loss = 0

        for layer_idx in range(self.num_decoder_layers):

            if self.use_place_holder:

                raise NotImplementedError

            pred_scores, pred_trajs = pred_list[layer_idx]

            assert pred_trajs.shape[-1] == 7

            pred_trajs_gmm, pred_vel = pred_trajs[:, :, :, 0:5], pred_trajs[:, :, :, 5:7]

            loss_reg_gmm, center_gt_positive_idx = loss_utils.nll_loss_gmm_direct(

                pred_scores=pred_scores, pred_trajs=pred_trajs_gmm,

                gt_trajs=center_gt_trajs[:, :, 0:2], gt_valid_mask=center_gt_trajs_mask,

                pre_nearest_mode_idxs=center_gt_positive_idx,

                timestamp_loss_weight=None, use_square_gmm=False,

            )

            pred_vel = pred_vel[torch.arange(num_center_objects), center_gt_positive_idx]

            loss_reg_vel = F.l1_loss(pred_vel, center_gt_trajs[:, :, 2:4], reduction='none')

            loss_reg_vel = (loss_reg_vel * center_gt_trajs_mask[:, :, None]).sum(dim=-1).sum(dim=-1)

            loss_cls = F.cross_entropy(input=pred_scores, target=center_gt_positive_idx, reduction='none')

            # total loss

            weight_cls = self.model_cfg.LOSS_WEIGHTS.get('cls', 1.0)

            weight_reg = self.model_cfg.LOSS_WEIGHTS.get('reg', 1.0)

            weight_vel = self.model_cfg.LOSS_WEIGHTS.get('vel', 0.2)

            layer_loss = loss_reg_gmm * weight_reg + loss_reg_vel * weight_vel + loss_cls.sum(dim=-1) * weight_cls

            layer_loss = layer_loss.mean()

            total_loss += layer_loss

            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}'] = layer_loss.item()

            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_gmm'] = loss_reg_gmm.mean().item() * weight_reg

            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_reg_vel'] = loss_reg_vel.mean().item() * weight_vel

            tb_dict[f'{tb_pre_tag}loss_layer{layer_idx}_cls'] = loss_cls.mean().item() * weight_cls

            if layer_idx + 1 == self.num_decoder_layers:

                layer_tb_dict_ade = motion_utils.get_ade_of_each_category(

                    pred_trajs=pred_trajs_gmm[:, :, :, 0:2],

                    gt_trajs=center_gt_trajs[:, :, 0:2], gt_trajs_mask=center_gt_trajs_mask,

                    object_types=self.forward_ret_dict['center_objects_type'],

                    valid_type_list=self.object_type,

                    post_tag=f'_layer_{layer_idx}',

                    pre_tag=tb_pre_tag

                )

                tb_dict.update(layer_tb_dict_ade)

                disp_dict.update(layer_tb_dict_ade)

        total_loss = total_loss / self.num_decoder_layers

        return total_loss, tb_dict, disp_dict

    def get_dense_future_prediction_loss(self, tb_pre_tag='', tb_dict=None, disp_dict=None):

        obj_trajs_future_state = self.forward_ret_dict['obj_trajs_future_state'].cuda()

        obj_trajs_future_mask = self.forward_ret_dict['obj_trajs_future_mask'].cuda()

        pred_dense_trajs = self.forward_ret_dict['pred_dense_trajs']  # (num_center_objects, num_objects, num_future_frames, 7)

        assert pred_dense_trajs.shape[-1] == 7

        assert obj_trajs_future_state.shape[-1] == 4

        pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

        loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')

        loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

        num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape

        fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1, 1)  # (num_center_objects * num_objects, 1)

        temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)

        temp_gt_idx = torch.zeros(num_center_objects * num_objects).cuda().long()  # (num_center_objects * num_objects)

        temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects, num_timestamps, 2)

        temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)

        loss_reg_gmm, _ = loss_utils.nll_loss_gmm_direct(

            pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs, gt_valid_mask=temp_gt_trajs_mask,

            pre_nearest_mode_idxs=temp_gt_idx,

            timestamp_loss_weight=None, use_square_gmm=False,

        )

        loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

        loss_reg = loss_reg_vel + loss_reg_gmm

        obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

        loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), min=1.0)

        loss_reg = loss_reg.mean()

        if tb_dict is None:

            tb_dict = {}

        if disp_dict is None:

            disp_dict = {}

        tb_dict[f'{tb_pre_tag}loss_dense_prediction'] = loss_reg.item()

        return loss_reg, tb_dict, disp_dict

    def get_loss(self, tb_pre_tag=''):

        loss_decoder, tb_dict, disp_dict = self.get_decoder_loss(tb_pre_tag=tb_pre_tag)

        loss_dense_prediction, tb_dict, disp_dict = self.get_dense_future_prediction_loss(tb_pre_tag=tb_pre_tag, tb_dict=tb_dict, disp_dict=disp_dict)

        total_loss = loss_decoder + loss_dense_prediction

        tb_dict[f'{tb_pre_tag}loss'] = total_loss.item()

        disp_dict[f'{tb_pre_tag}loss'] = total_loss.item()

        return total_loss, tb_dict, disp_dict

    def generate_final_prediction(self, pred_list, batch_dict):

        pred_scores, pred_trajs = pred_list[-1]

        pred_scores = torch.softmax(pred_scores, dim=-1)  # (num_center_objects, num_query)

        num_center_objects, num_query, num_future_timestamps, num_feat = pred_trajs.shape

        if self.num_motion_modes != num_query:

            assert num_query > self.num_motion_modes

            pred_trajs_final, pred_scores_final, selected_idxs = motion_utils.batch_nms(

                pred_trajs=pred_trajs, pred_scores=pred_scores,

                dist_thresh=self.model_cfg.NMS_DIST_THRESH,

                num_ret_modes=self.num_motion_modes

            )

        else:

            pred_trajs_final = pred_trajs

            pred_scores_final = pred_scores

        return pred_scores_final, pred_trajs_final

    def forward(self, batch_dict):

        """

        MTR Decoder 的前向传播入口

       

        Args:

            batch_dict: 包含所有输入信息的字典

                - obj_feature: (num_center_objects, num_objects, C_in)  环境中的其他障碍物特征

                - obj_mask: (num_center_objects, num_objects)           障碍物掩码 (1有效, 0无效)

                - map_feature: (num_center_objects, num_polylines, C_in) 地图车道线特征

                - center_objects_feature: (num_center_objects, C_in)     我们要预测的目标(中心)对象的特征

        """

        input_dict = batch_dict['input_dict']

       

        # 1. 解包 Encoder 传过来的特征

        # num_center_objects: 当前 Batch 里一共要预测多少个目标 (相当于 Batch Size)

        # num_objects: 每个目标周围有多少个环境障碍物 (context agents)

        # num_polylines: 每个目标周围有多少条地图线段

        obj_feature, obj_mask, obj_pos = batch_dict['obj_feature'], batch_dict['obj_mask'], batch_dict['obj_pos']
        map_feature, map_mask, map_pos = batch_dict['map_feature'], batch_dict['map_mask'], batch_dict['map_pos']
        # center_objects_feature.size: (num_center_objects, C)
        center_objects_feature = batch_dict['center_objects_feature']
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        # 2. 输入投影 (Input Projection)

        # 目的：将 Encoder 输出的特征维度映射到 Decoder 的 hidden_dim (d_model)
        # 2.1 投影中心对象特征

        # center_objects_feature: (num_center_objects, C_in) -> (num_center_objects, d_model)

        center_objects_feature = self.in_proj_center_obj(center_objects_feature)

        # 2.2 投影环境障碍物特征 (只计算有效元素以节省算力)

        # obj_feature[obj_mask] -> 选出所有有效的障碍物，形状变成 (Total_Valid_Objs, C_in)

        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])

        # 创建全 0 容器并填回

        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])

        obj_feature[obj_mask] = obj_feature_valid

        # 2.3 投影地图特征 (同理，只计算有效元素)

        map_feature_valid = self.in_proj_map(map_feature[map_mask])

        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])

        map_feature[map_mask] = map_feature_valid

        # 3. 致密未来预测 (Dense Future Prediction) -- 这是一个辅助任务

        # 作用：不仅预测中心车，还顺便预测周围所有障碍物的未来轨迹。单纯MLP 不用Transform

        # 关键点：它会将预测出的“未来信息”编码后融合回 obj_feature 中。

        # 结果：现在的 obj_feature 不仅包含了过去，还包含了对未来的“预判”，增强了上下文信息。

        obj_feature, pred_dense_future_trajs = self.apply_dense_future_prediction(

            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos

        )

        # 4. 进入核心解码器循环 (Decoder Layers) -- 全文最重要部分

        # 这里执行了: Intention Query -> Cross Attention -> Dynamic Map Collection -> Trajectory Update

        # pred_list: 包含每一层 decoder 的预测结果 [[scores, trajs], [scores, trajs], ...]

        pred_list = self.apply_transformer_decoder(

            center_objects_feature=center_objects_feature,

            center_objects_type=input_dict['center_objects_type'],

            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,

            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos

        )

        # 将预测结果存入字典，供 Loss 计算使用

        self.forward_ret_dict['pred_list'] = pred_list

        # 5. 根据模式处理输出

        if not self.training:

            # === 推理模式 (Inference) ===

            # 取最后一层的预测结果，进行 NMS (非极大值抑制) 去重，选出最好的几条轨迹

            pred_scores, pred_trajs = self.generate_final_prediction(pred_list=pred_list, batch_dict=batch_dict)
 
            # 将最终结果写入 batch_dict 返回

            batch_dict['pred_scores'] = pred_scores  # (num_center_objects, num_modes)

            batch_dict['pred_trajs'] = pred_trajs    # (num_center_objects, num_modes, num_future, 7)

        else:

            # === 训练模式 (Training) ===

            # 将 Ground Truth (真值) 存入 forward_ret_dict，稍后在 get_loss 中计算损失    

            # 目标车的真值轨迹

            self.forward_ret_dict['center_gt_trajs'] = input_dict['center_gt_trajs']

            self.forward_ret_dict['center_gt_trajs_mask'] = input_dict['center_gt_trajs_mask']

            self.forward_ret_dict['center_gt_final_valid_idx'] = input_dict['center_gt_final_valid_idx']

           

            # 环境车的真值轨迹 (用于 Dense Future Prediction Loss)

            self.forward_ret_dict['obj_trajs_future_state'] = input_dict['obj_trajs_future_state']

            self.forward_ret_dict['obj_trajs_future_mask'] = input_dict['obj_trajs_future_mask']

            # 对象类型 (用于按类别统计 Metrics)

            self.forward_ret_dict['center_objects_type'] = input_dict['center_objects_type']

        return batch_dict