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

import models
from utils.logger import setlogger
from utils.utils_PTFT import utils_PTFT

print(torch.__version__)
warnings.filterwarnings('ignore')

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('--random_number', type=int, default=42)

    parser.add_argument('--paradigm_name', type=str, default='PTFT', help='the name of the paradigm')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--fine_num', type=int, default=1)

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

    # pt：optimization information
    parser.add_argument('--lr_pt', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--weight_decay_pt', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--gamma_pt', type=float, default=0.1,
                        help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps_pt', type=str, default='100, 150', help='the learning rate decay for step and stepLR')

    # ft：optimization information
    parser.add_argument('--lr_ft', type=float, default=1e-4, help='the initial learning rate')
    parser.add_argument('--momentum_ft', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay_ft', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--gamma_ft', type=float, default=0.1,
                        help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps_ft', type=str, default='30, 60', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--max_epoch_pt', type=int, default=200, help='max number of epoch')
    parser.add_argument('--max_epoch_ft', type=int, default=100, help='max number of epoch')

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
            ACC_st_pt = np.zeros(args.max_epoch_pt, dtype=float)
            Loss_st_pt = np.zeros(args.max_epoch_pt, dtype=float)
            ACC_st_ft = np.zeros(args.max_epoch_ft, dtype=float)
            Loss_st_ft = np.zeros(args.max_epoch_ft, dtype=float)

            # source_val
            ACC_sv_pt = np.zeros(args.max_epoch_pt, dtype=float)
            Loss_sv_pt = np.zeros(args.max_epoch_pt, dtype=float)
            ACC_sv_ft = np.zeros(args.max_epoch_ft, dtype=float)
            Loss_sv_ft = np.zeros(args.max_epoch_ft, dtype=float)

            # source_train
            ACC_tt_pt = np.zeros(args.max_epoch_pt, dtype=float)
            Loss_tt_pt = np.zeros(args.max_epoch_pt, dtype=float)
            ACC_tt_ft = np.zeros(args.max_epoch_ft, dtype=float)
            Loss_tt_ft = np.zeros(args.max_epoch_ft, dtype=float)

            # target_val
            ACC_tv_pt = np.zeros(args.max_epoch_pt, dtype=float)
            Loss_tv_pt = np.zeros(args.max_epoch_pt, dtype=float)
            ACC_tv_ft = np.zeros(args.max_epoch_ft, dtype=float)
            Loss_tv_ft = np.zeros(args.max_epoch_ft, dtype=float)

            trainer = utils_PTFT(args, save_dir)
            trainer.setup()

            if args.pretrained:
                Basis_model_class = getattr(models, args.model_name)
                Basis_model = Basis_model_class()
                model_pt_state_dic = torch.load('XXXXX.pkl')
                Basis_model.load_state_dict(model_pt_state_dic)

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    device_count = torch.cuda.device_count()
                    logging.info('using {} gpus'.format(device_count))
                    assert args.batch_size % device_count == 0, "batch size should be divided by device count"
                else:
                    warnings.warn("gpu is not available")
                    device = torch.device("cpu")
                    device_count = 1
                    logging.info('using {} cpu'.format(device_count))

                Basis_model.to(device)

                trainer.setup_ft(basis_model=Basis_model)
                ACC_tt_ft, Loss_tt_ft, ACC_st_ft, Loss_st_ft, ACC_sv_ft, Loss_sv_ft, ACC_tv_ft, Loss_tv_ft = trainer.ft()

                file_name = f"domain_{source_domain}_{target_domain}_exp_{num + 1}.mat"
                multiple_arrays_dir = os.path.join(save_dir, file_name)
                savemat(multiple_arrays_dir, {
                    'ACC_st': ACC_st_ft,
                    'ACC_sv': ACC_sv_ft,
                    'ACC_tt': ACC_tt_ft,
                    'ACC_tv': ACC_tv_ft,
                    'Loss_st': Loss_st_ft,
                    'Loss_sv': Loss_sv_ft,
                    'Loss_tt': Loss_tt_ft,
                    'Loss_tv': Loss_tv_ft,
                })

            else:
                trainer.setup_pt()
                Basis_model, ACC_st_pt, Loss_st_pt, ACC_sv_pt, Loss_sv_pt, ACC_tt_pt, Loss_tt_pt, ACC_tv_pt, Loss_tv_pt = trainer.pt(num)

                trainer.setup_ft(basis_model=Basis_model)

                # 根据是否为细粒度测试
                if args.Fine_grained_testing:
                    ACC_tt_ft, Loss_tt_ft, ACC_st_ft, Loss_st_ft, ACC_sv_ft, Loss_sv_ft, ACC_tv_ft, Loss_tv_ft, labels = trainer.ft()

                    file_name = f"domain_{source_domain}_{target_domain}_exp_{num + 1}.mat"
                    multiple_arrays_dir = os.path.join(save_dir, file_name)
                    savemat(multiple_arrays_dir, {
                        'ACC_st': np.concatenate((ACC_st_pt, ACC_st_ft)),
                        'ACC_sv': np.concatenate((ACC_sv_pt, ACC_sv_ft)),
                        'ACC_tt': np.concatenate((ACC_tt_pt, ACC_tt_ft)),
                        'ACC_tv': np.concatenate((ACC_tv_pt, ACC_tv_ft)),
                        'Loss_st': np.concatenate((Loss_st_pt, Loss_st_ft)),
                        'Loss_sv': np.concatenate((Loss_sv_pt, Loss_sv_ft)),
                        'Loss_tt': np.concatenate((Loss_tt_pt, Loss_tt_ft)),
                        'Loss_tv': np.concatenate((Loss_tv_pt, Loss_tv_ft)),
                        'Labels': labels,
                    })
                else:
                    ACC_tt_ft, Loss_tt_ft, ACC_st_ft, Loss_st_ft, ACC_sv_ft, Loss_sv_ft, ACC_tv_ft, Loss_tv_ft = trainer.ft()

                    file_name = f"domain_{source_domain}_{target_domain}_exp_{num + 1}.mat"
                    multiple_arrays_dir = os.path.join(save_dir, file_name)
                    savemat(multiple_arrays_dir, {
                        'ACC_st': np.concatenate((ACC_st_pt, ACC_st_ft)),
                        'ACC_sv': np.concatenate((ACC_sv_pt, ACC_sv_ft)),
                        'ACC_tt': np.concatenate((ACC_tt_pt, ACC_tt_ft)),
                        'ACC_tv': np.concatenate((ACC_tv_pt, ACC_tv_ft)),
                        'Loss_st': np.concatenate((Loss_st_pt, Loss_st_ft)),
                        'Loss_sv': np.concatenate((Loss_sv_pt, Loss_sv_ft)),
                        'Loss_tt': np.concatenate((Loss_tt_pt, Loss_tt_ft)),
                        'Loss_tv': np.concatenate((Loss_tv_pt, Loss_tv_ft)),
                    })

            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
