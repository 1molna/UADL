import argparse
import copy
import os
import os.path as osp
import time
import warnings

from sklearn.preprocessing import label_binarize

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from consolidate_loop import *
from mmcv.parallel import collate
from functools import partial

import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir)
sys.path.append(parentdir2)


from mmdet import __version__
from mmdet.apis import set_random_seed_new, train_detector_new
from mmdet.datasets import build_dataset_new, build_dataloader_new
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
import torch.utils.data as data

import json
import random
import numpy as np
import torch
import sklearn.mixture as sm
import argparse
import scipy.stats as st
from performance_metrics import *
from utils import fit_gmms, gmm_uncertainty
import tqdm
import os
import sys
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from base_dirs import *

import os
os.environ['CUDA_VISIBLE_DEVICES']='0, 1, 2, 3'


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--nums', help='the total number of selected samples')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dType', default = 'FRCNN', help='FRCNN or retinanet')
    parser.add_argument('--dataset', default = 'coco', help='voc or coco or bdd or kitti or idd')
    parser.add_argument('--saveNm', default = None, help='the save name of the results')
    parser.add_argument('--iouThresh', default = 0.5, type = float, help='the cutoff iou used to estimate class centres')
    parser.add_argument('--scoreThresh', default = 0.5, type = float, help='the cutoff score used to estimate class centres')
    parser.add_argument('--saveResults', default = False, type = bool, help='save results')
    parser.add_argument('--weights_path', default = 'latest.pth', help='model weights path for loading')


    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    total_select_nums = args.nums
    weights_path = args.weights_path
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed_new(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model.load_state_dict(torch.load(weights_path),False)

    datasets = [build_dataset_new(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset_new(val_dataset))

    
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    if args.dataset == 'voc':
        num_classes = 10
        num = 16550
        num_init = 3000
        num_label = 1000
    elif args.dataset == 'coco':
        num_classes = 40
        num = 117271
        num_init = 5000
        num_label = 5000
    elif args.dataset == 'bdd':
        num_classes = 10
        num = 69405
        num_init = 10000
        num_label = 1000
    elif args.dataset == 'kitti':
        num_classes = 9
        num = 3741
        num_init = 1000
        num_label = 600
    elif args.dataset == 'idd':
        num_classes = 13
        num = 31569
        num_init = 7000
        num_label = 2000
    else:
        num_classes = 80
        print('implement')
        exit()

    indices = list(range(num))
    random.shuffle(indices)
    labeled_set = indices[:num_init]
    unlabeled_set = indices
    # iterNum = int(num) // 8
    # print(iterNum)
    # batch_i = []
    # for i in range(12):
    #     batch_i.append(int(iterNum))
    unsupervised_data_loader = data.DataLoader(datasets, batch_size=1,
                                               num_workers=2,
                                               sampler=SequentialSampler(unlabeled_set),
                                               collate_fn=detection_collate,
                                               pin_memory=True)
    #print(batch_iterator)
    batch_iterator = iter(unsupervised_data_loader)
    #print(batch_i)
    print(batch_iterator)
    labeled_set, unlabeled_set = active_consolidate(
            batch_iterator,
            labeled_set,
            unlabeled_set,
            model,
            num_classes,
            num_label,
            num
        )

    if int(total_select_nums) > int(num_label):
        ff = open("labeled_training_set_" + args.dataset + '_' + str(len(labeled_set)) + '_id_' + str(total_select_nums - num_label) + ".txt",'r')
        lines = ff.readlines()# read all content 
        for line in lines:
            line1 = line.strip('\n')
            #line = np.fromstring(line, dtype=np.int64)
            line2 = int(line1)  
            labeled_set.append(line2)

    f = open("labeled_training_set_" + args.dataset + '_' + str(len(labeled_set)) + '_id_' + str(total_select_nums) + ".txt", 'w')
    for i in range(len(labeled_set)):
        f.write(str(labeled_set[i]))
        f.write("\n")
    f.close()

    train_detector_new(
        labeled_set,
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
