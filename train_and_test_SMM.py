import argparse
import itertools
import logging
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
from scipy.io import savemat

from utils.logger import setlogger
from utils.utils_SMM import utils_SMM

print(torch.__version__)
warnings.filterwarnings('ignore')

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('--random_number', type=int, default=42)

    parser.add_argument('--paradigm_name', type=str, default='SMM', help='the name of the paradigm')

    parser.add_argument("--Fine_grained_testing", type=bool, default=False, )
    parser.add_argument("--Data_dependency", type=bool, default=False, )

    # model parameters
    parser.add_argument('--model_name', type=str, default='CNN_1d', help='the name of the model')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='the directory to save the model')
    parser.add_argument('--class_innum', type=int, default=256)
    parser.add_argument('--class_outnum', type=int, default=10)

    # CWRU dataset parameter
    parser.add_argument('--data_name', type=str, default='CWRU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default=r'E:\Project_py\Review\DTL\Data\CWRU',
                        help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('--sample_number', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')

    # SMM
    parser.add_argument('--distance_loss', type=str, default='MK-MMD')

    # optimization information
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--middle_epoch', type=int, default=50, help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=300, help='max number of epoch')

    args = parser.parse_args()

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    args = parse_args()

    setup_seed(args.random_number)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    domains = [0, 1, 2, 3]
    domain_pairs = list(itertools.permutations(domains, 2))

    sub_dir = args.paradigm_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')

    for source_domain, target_domain in domain_pairs:

        number = 5

        if args.Fine_grained_testing:
            # 混淆矩阵以及数据量测试使用
            source_domain = 3
            target_domain = 0
            number = 1

        print(f"\n=== 正在执行迁移任务: 源域 {source_domain} → 目标域 {target_domain} ===")
        args.transfer_task = [[source_domain], [target_domain]]

        sub_dir1 = str(source_domain) + '_' + str(target_domain)
        save_dir = os.path.join(args.checkpoint_dir, sub_dir, sub_dir1)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for num in range(number):
            print(f"\n--- 实验 {num + 1}/5 ---")

            setup_seed(args.random_number + num)

            # set the logger
            log_file = os.path.join(save_dir, f'train_exp_{num + 1}.log')
            if os.path.exists(log_file):
                os.remove(log_file)
            setlogger(log_file)

            # save the args
            for k, v in args.__dict__.items():
                logging.info("{}: {}".format(k, v))

            # source_train
            ACC_st = np.zeros(args.max_epoch, dtype=float)
            Loss_CLA_st = np.zeros(args.max_epoch, dtype=float)
            Loss_DTL_st = np.zeros(args.max_epoch, dtype=float)
            Loss_st = np.zeros(args.max_epoch, dtype=float)

            # source_val
            ACC_sv = np.zeros(args.max_epoch, dtype=float)
            Loss_sv = np.zeros(args.max_epoch, dtype=float)

            # source_train
            ACC_tt = np.zeros(args.max_epoch, dtype=float)
            Loss_tt = np.zeros(args.max_epoch, dtype=float)

            # target_val
            ACC_tv = np.zeros(args.max_epoch, dtype=float)
            Loss_tv = np.zeros(args.max_epoch, dtype=float)

            trainer = utils_SMM(args, save_dir)
            trainer.setup()

            # 训练模型并获取结果
            results = trainer.train()

            # 根据是否为细粒度测试决定是否包含 labels
            if args.Fine_grained_testing:
                metrics = results[:-1]  # 前8个返回值是指标
                labels = results[-1]  # 最后一个是 labels
            else:
                metrics = results
                labels = None

            # 定义指标名称
            metric_names = ['ACC_st', 'Loss_st', 'ACC_sv', 'Loss_sv', 'ACC_tt', 'Loss_tt', 'ACC_tv', 'Loss_tv']

            # 构建保存的数据字典
            data_dict = {name: value for name, value in zip(metric_names, metrics)}

            # 如果有 labels，添加到字典中
            if labels is not None:
                data_dict['Labels'] = labels

            # 保存到 .mat 文件
            file_name = f"domain_{source_domain}_{target_domain}_exp_{num + 1}.mat"
            multiple_arrays_dir = os.path.join(save_dir, file_name)
            savemat(multiple_arrays_dir, data_dict)

            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
