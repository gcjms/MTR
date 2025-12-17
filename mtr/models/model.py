# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508

# Published at NeurIPS 2022

# Written by Shaoshuai Shi

# All Rights Reserved

import numpy as np

import os

import torch

import torch.nn as nn

import torch.nn.functional as F

from .context_encoder import build_context_encoder

from .motion_decoder import build_motion_decoder

class MotionTransformer(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.model_cfg = config

        # 构建上下文编码器（论文中的 Context Encoder），将输入轨迹/地图编码成统一维度的特征

        self.context_encoder = build_context_encoder(self.model_cfg.CONTEXT_ENCODER)

        # 构建运动解码器（论文中的 Motion Decoder），其输入通道需与 encoder 的输出通道一致

        self.motion_decoder = build_motion_decoder(

            in_channels=self.context_encoder.num_out_channels,

            config=self.model_cfg.MOTION_DECODER

        )

    def forward(self, batch_dict):

        """

        前向流程：

        1) context_encoder：对 batch_dict['input_dict'] 中的 obj/map 多段线等进行编码，写回 batch_dict

        2) motion_decoder：读取 encoder 产出的特征，执行跨注意力/查询解码并计算 loss 或预测

        训练模式 (self.training=True)：返回 (loss, tb_dict, disp_dict)

        评估/推理模式：返回包含预测结果的 batch_dict

        Args:

            batch_dict (dict): 外部 DataLoader 组装的字典，包含 input_dict 以及中间/输出键位

        Returns:

            - training: (loss: Tensor, tb_dict: dict, disp_dict: dict)

            - eval/inference: batch_dict (包含预测/中间结果)

        """

        # 调用论文中的 Encoder（MTREncoder），完成特征编码

        batch_dict = self.context_encoder(batch_dict)

        # 调用论文中的 Decoder（MTRDecoder），完成解码与（训练时的）损失计算

        batch_dict = self.motion_decoder(batch_dict)

        if self.training:

            # 训练模式下由 decoder 汇总损失与可视化字典

            loss, tb_dict, disp_dict = self.get_loss()

            tb_dict.update({'loss': loss.item()})

            disp_dict.update({'loss': loss.item()})

            return loss, tb_dict, disp_dict

        return batch_dict

    def get_loss(self):

        # 直接从 motion_decoder 汇总损失与日志字典

        loss, tb_dict, disp_dict = self.motion_decoder.get_loss()

        return loss, tb_dict, disp_dict

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):

        if not os.path.isfile(filename):

            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))

        loc_type = torch.device('cpu') if to_cpu else None

        checkpoint = torch.load(filename, map_location=loc_type)

        epoch = checkpoint.get('epoch', -1)

        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:

            logger.info('==> Loading optimizer parameters from checkpoint %s to %s'

                        % (filename, 'CPU' if to_cpu else 'GPU'))

            optimizer.load_state_dict(checkpoint['optimizer_state'])

        if 'version' in checkpoint:

            print('==> Checkpoint trained from version: %s' % checkpoint['version'])

        # logger.info('==> Done')

        logger.info('==> Done (loaded %d/%d)' % (len(checkpoint['model_state']), len(checkpoint['model_state'])))

        return it, epoch

    def load_params_from_file(self, filename, logger, to_cpu=False):

        if not os.path.isfile(filename):

            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))

        loc_type = torch.device('cpu') if to_cpu else None

        checkpoint = torch.load(filename, map_location=loc_type)

        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)

        if version is not None:

            logger.info('==> Checkpoint trained from version: %s' % version)

        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')

        model_state = self.state_dict()

        model_state_disk_filter = {}

        for key, val in model_state_disk.items():

            if key in model_state and model_state_disk[key].shape == model_state[key].shape:

                model_state_disk_filter[key] = val

            else:

                if key not in model_state:

                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')

                else:

                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)

        logger.info(f'Missing keys: {missing_keys}')

        logger.info(f'The number of missing keys: {len(missing_keys)}')

        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')

        logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)

        it = checkpoint.get('it', 0.0)

        return it, epoch

