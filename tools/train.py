# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved

import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from tensorboardX import SummaryWriter

from mtr.datasets import build_dataloader
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.utils import common_utils
from mtr.models import model as model_utils

from train_utils.train_utils import train_model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--without_sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--not_eval_with_train', action='store_true', default=False, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')

    parser.add_argument('--add_worker_init_fn', action='store_true', default=False, help='')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def build_optimizer(model, opt_cfg):
    if opt_cfg.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(
            [each[1] for each in model.named_parameters()],
            lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
        )
    elif opt_cfg.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg.LR, weight_decay=opt_cfg.get('WEIGHT_DECAY', 0))
    else:
        assert False

    return optimizer

def build_scheduler(optimizer, dataloader, opt_cfg, total_epochs, total_iters_each_epoch, last_epoch):
    decay_steps = [x * total_iters_each_epoch for x in opt_cfg.get('DECAY_STEP_LIST', [5, 10, 15, 20])]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * opt_cfg.LR_DECAY
        return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)

    if opt_cfg.get('SCHEDULER', None) == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2 * len(dataloader),
            T_mult=1,
            eta_min=max(1e-2 * opt_cfg.LR, 1e-6),
            last_epoch=-1,
        )
    elif opt_cfg.get('SCHEDULER', None) == 'lambdaLR':
        scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
    elif opt_cfg.get('SCHEDULER', None) == 'linearLR':
        total_iters = total_iters_each_epoch * total_epochs
        scheduler = lr_sched.LinearLR(optimizer, start_factor=1.0, end_factor=opt_cfg.LR_CLIP / opt_cfg.LR, total_iters=total_iters, last_epoch=last_epoch)
    else:
        scheduler = None

    return scheduler

def main():

    # 训练主入口：

    # 1) 解析配置与命令行参数

    # 2) 初始化分布式/日志/输出目录

    # 3) 构建训练/验证数据加载器

    # 4) 构建模型、优化器、学习率调度器

    # 5) （可选）加载预训练/断点恢复

    # 6) 真正的训练循环在 train_utils/train_utils.py 中的 train_model(...) 被调用

    # 7) 训练结束后，调用测试脚本对各轮 checkpoint 做评估

    args, cfg = parse_config()

    if args.launcher == 'none':

        dist_train = False

        total_gpus = 1

        args.without_sync_bn = True

    else:

        # 分布式训练初始化（pytorch 或 slurm），设置 LOCAL_RANK / 进程组

        if args.local_rank is None:

            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))

        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(

            args.tcp_port, args.local_rank, backend='nccl'

        )

        dist_train = True

    # 处理 batch_size 和总训练轮次（可被命令行覆写）

    if args.batch_size is None:

        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    else:

        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'

        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:

        # 固定随机种子，保证可复现

        common_utils.set_random_seed(666)

    # 创建输出目录、ckpt 目录与日志文件

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag

    ckpt_dir = output_dir / 'ckpt'

    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file

    logger.info('**********************Start logging**********************')

    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'

    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:

        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))

    for key, val in vars(args).items():

        logger.info('{:16} {}'.format(key, val))

    log_config_to_file(cfg, logger=logger)

    if cfg.LOCAL_RANK == 0:

        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # 构建训练数据集与 DataLoader

    train_set, train_loader, train_sampler = build_dataloader(

        dataset_cfg=cfg.DATA_CONFIG,

        batch_size=args.batch_size,

        dist=dist_train, workers=args.workers,

        logger=logger,

        training=True,

        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,

        total_epochs=args.epochs,

        add_worker_init_fn=args.add_worker_init_fn,

    )

    # 构建模型并移动到 GPU；根据需要转换为 SyncBN

    model = model_utils.MotionTransformer(config=cfg.MODEL)

    if not args.without_sync_bn:

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    # 构建优化器

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible

    start_epoch = it = 0

    last_epoch = -1

    # （可选）仅加载预训练模型权重，不含优化器/迭代状态

    if args.pretrained_model is not None:

        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    # （可选）从 ckpt 恢复：同时恢复模型权重、优化器状态、当前迭代与开始 epoch

    if args.ckpt is not None:

        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer,

                                                           logger=logger)

        last_epoch = start_epoch + 1

    else:

        # 若未显式指定 ckpt，则尝试从 ckpt 目录中寻找最新的断点继续训练

        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))

        if len(ckpt_list) > 0:

            ckpt_list.sort(key=os.path.getmtime)

            while len(ckpt_list) > 0:

                basename = os.path.basename(ckpt_list[-1])

                if basename == 'best_model.pth':

                    ckpt_list = ckpt_list[:-1]

                    continue

                try:

                    it, start_epoch = model.load_params_with_optimizer(

                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger

                    )

                    last_epoch = start_epoch + 1

                    break

                except:

                    ckpt_list = ckpt_list[:-1]

    # 构建学习率调度器（可选）

    scheduler = build_scheduler(

        optimizer, train_loader, cfg.OPTIMIZATION, total_epochs=args.epochs,

        total_iters_each_epoch=len(train_loader), last_epoch=last_epoch

    )

    # 进入训练模式；在包裹 DDP 之前可冻结部分参数

    model.train()  # before wrap to DistributedDataParallel to support to fix some parameters

    if dist_train:

        # 分布式包裹模型

        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)

    logger.info(model)

    num_total_params = sum([x.numel() for x in model.parameters()])

    logger.info(f'Total number of parameters: {num_total_params}')

    # 构建验证/测试 DataLoader（训练期间可用于周期性评估）

    test_set, test_loader, sampler = build_dataloader(

        dataset_cfg=cfg.DATA_CONFIG,

        batch_size=args.batch_size,

        dist=dist_train, workers=args.workers, logger=logger, training=False

    )

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'

    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------start training---------------------------

    # 注意：真正的训练循环在下面这行被调用

    # train_utils/train_utils.py -> train_model(...)

    # 该函数内部执行每个 iteration 的 forward/backward/optimizer.step()，并处理日志、保存 ckpt、评估等

    # 返回

    logger.info('**********************Start training %s/%s(%s)**********************'

                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(

        model,

        optimizer,

        train_loader,

        optim_cfg=cfg.OPTIMIZATION,

        start_epoch=start_epoch,

        total_epochs=args.epochs,

        start_iter=it,

        rank=cfg.LOCAL_RANK,

        ckpt_save_dir=ckpt_dir,

        train_sampler=train_sampler,

        ckpt_save_interval=args.ckpt_save_interval,

        max_ckpt_save_num=args.max_ckpt_save_num,

        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,

        tb_log=tb_log,

        scheduler=scheduler,

        logger=logger,

        eval_output_dir=eval_output_dir,

        test_loader=test_loader if not args.not_eval_with_train else None,

        cfg=cfg, dist_train=dist_train, logger_iter_interval=args.logger_iter_interval,

        ckpt_save_time_interval=args.ckpt_save_time_interval

    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'

                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # 训练结束后的独立评估流程（遍历 ckpt 做统一评测）

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %

                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    eval_output_dir = output_dir / 'eval' / 'eval_with_train'

    eval_output_dir.mkdir(parents=True, exist_ok=True)

    args.start_epoch = max(args.epochs - 0, 0)  # Only evaluate the last 10 epochs

    cfg.DATA_CONFIG.SAMPLE_INTERVAL.val = 1

    test_set, test_loader, sampler = build_dataloader(

        dataset_cfg=cfg.DATA_CONFIG,

        batch_size=args.batch_size,

        dist=dist_train, workers=args.workers, logger=logger, training=False

    )

    from test import repeat_eval_ckpt, eval_single_ckpt

    # repeat_eval_ckpt 会遍历 ckpt 目录里满足条件的权重文件并逐一评估

    repeat_eval_ckpt(

        model.module if dist_train else model,

        test_loader, args, eval_output_dir, logger, ckpt_dir,

        dist_test=dist_train

    )

    logger.info('**********************End evaluation %s/%s(%s)**********************' %

                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':

    main()