import logging
import os
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch import optim

import datasets
import models


def set_freeze_by_id(model, layer_num_last=0, freeze_all=False):
    """

    """

    for param in model.parameters():
        param.requires_grad = False

    if freeze_all:
        print("All layers of the model have been frozen.")
        return

    # 解冻最后几层
    children = list(model.children())
    if layer_num_last > len(children):
        layer_num_last = len(children)
        print(
            f"Warning: The specified number of layers {layer_num_last} exceeds the total number of layers in the model. All layers will be unfrozen.")

    for child in children[-layer_num_last:]:
        for param in child.parameters():
            param.requires_grad = True

    print(f"The last {layer_num_last} layers of the model have been unfrozen.")


def check_requires_grad(model):
    for name, param in model.named_parameters():
        print(f"Layer: {name}, Requires Grad: {param.requires_grad}")


class utils_PTFT(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):

        args = self.args

        # ---------------------------------------- Consider the gpu or cpu condition -----------------------------------
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # ---------------------------------------- Load the datasets ---------------------------------------------------
        Dataset = getattr(datasets, args.data_name)

        self.datasets = {}
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets[
            'target_val'] = Dataset(args.data_dir, args.transfer_task, args.normlizetype).data_split(
            sample_number=args.sample_number, Data_dependency=args.Data_dependency)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           # shuffle=(False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(False))
                            # drop_last=(True if args.last_batch and x.split('_')[1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_train', 'target_val']}

    def setup_pt(self):

        args = self.args

        # ---------------------------------------- Define the basic model ----------------------------------------------
        self.model_pt = getattr(models, args.model_name)(data_set=args.data_name)
        self.classifier_pt = nn.Linear(args.class_innum, args.class_outnum)
        self.model_all_pt = nn.Sequential(self.model_pt, self.classifier_pt)

        # -------------------------------- Parallel training of models on multiple GPUs --------------------------------
        if self.device_count > 1:
            self.model_pt = torch.nn.DataParallel(self.model_pt)
            self.classifier_pt = torch.nn.DataParallel(self.classifier_pt)

        # -------------------------------- Define the learning parameters ----------------------------------------------
        parameter_list_pt = [{"params": self.model_pt.parameters(), "lr": args.lr_pt},
                             {"params": self.classifier_pt.parameters(), "lr": args.lr_pt}]

        # -------------------------------- Define the optimizer --------------------------------------------------------
        self.optimizer_pt = optim.AdamW(params=parameter_list_pt, lr=args.lr_pt, weight_decay=args.weight_decay_pt,
                                        eps=1e-8,
                                        betas=(0.9, 0.999))

        # -------------------------------- Define the learning rate decay ----------------------------------------------
        steps = [int(step) for step in args.steps_pt.split(',')]
        self.lr_scheduler_pt = optim.lr_scheduler.MultiStepLR(self.optimizer_pt, steps, gamma=args.gamma_pt)

        # -------------------------------- Invert the model ------------------------------------------------------------
        self.model_pt.to(self.device)
        self.classifier_pt.to(self.device)

        # -------------------------------- Define the distance loss ----------------------------------------------------
        self.criterion_pt = nn.CrossEntropyLoss()

        self.start_epoch_pt = 0

    def pt(self, num):

        args = self.args

        # source_train
        ACC_st_pt = np.zeros(args.max_epoch_pt, dtype=float)
        Loss_st_pt = np.zeros(args.max_epoch_pt, dtype=float)

        # source_val
        ACC_sv_pt = np.zeros(args.max_epoch_pt, dtype=float)
        Loss_sv_pt = np.zeros(args.max_epoch_pt, dtype=float)

        # target_train
        ACC_tt_pt = np.zeros(args.max_epoch_pt, dtype=float)
        Loss_tt_pt = np.zeros(args.max_epoch_pt, dtype=float)

        # target_val
        ACC_tv_pt = np.zeros(args.max_epoch_pt, dtype=float)
        Loss_tv_pt = np.zeros(args.max_epoch_pt, dtype=float)

        for epoch in range(self.start_epoch_pt, args.max_epoch_pt):

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch_pt - 1) + '-' * 5)

            # -------------------------------- Update the learning rate ------------------------------------------------
            if self.lr_scheduler_pt is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler_pt.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr_pt))

            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_train', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_length = 0
                epoch_acc = 0
                epoch_loss = 0.0

                # -------------------------------- Set model to train mode or test mode --------------------------------
                if phase == 'source_train':
                    self.model_pt.train()
                    self.classifier_pt.train()
                else:
                    self.model_pt.eval()
                    self.classifier_pt.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # ------------------------------------forward --------------------------------------------------
                        features = self.model_pt(inputs)
                        outputs = self.classifier_pt(features)

                        # ------------------------------------loss -----------------------------------------------------
                        logits = outputs
                        loss = self.criterion_pt(logits, labels)

                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        epoch_acc += correct
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_length += labels.size(0)

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer_pt.zero_grad()
                            loss.backward()
                            self.optimizer_pt.step()

                # -------------------------------- Print the train and val information via each epoch ------------------
                epoch_acc = epoch_acc / epoch_length
                epoch_loss = epoch_loss / epoch_length
                if phase == 'source_train':
                    ACC_st_pt[epoch] = epoch_acc
                    Loss_st_pt[epoch] = epoch_loss
                if phase == 'source_val':
                    ACC_sv_pt[epoch] = epoch_acc
                    Loss_sv_pt[epoch] = epoch_loss
                if phase == 'target_train':
                    ACC_tt_pt[epoch] = epoch_acc
                    Loss_tt_pt[epoch] = epoch_loss
                if phase == 'target_val':
                    ACC_tv_pt[epoch] = epoch_acc
                    Loss_tv_pt[epoch] = epoch_loss

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

            if self.lr_scheduler_pt is not None:
                self.lr_scheduler_pt.step()

        # save the basis model
        model_pt_state_dic = self.model_pt.module.state_dict() if self.device_count > 1 else self.model_pt.state_dict()
        file_name = f"basis_model_{num + 1}.pkl"
        torch.save(model_pt_state_dic, os.path.join(self.save_dir, file_name))
        return self.model_pt, ACC_st_pt, Loss_st_pt, ACC_sv_pt, Loss_sv_pt, ACC_tt_pt, Loss_tt_pt, ACC_tv_pt, Loss_tv_pt

    def setup_ft(self, basis_model):
        """

        """
        args = self.args

        # ---------------------------------------- Define the basic model ----------------------------------------------
        if args.fine_num > 0:
            set_freeze_by_id(basis_model, layer_num_last=args.fine_num)
        else:
            set_freeze_by_id(basis_model, freeze_all=True)
        self.model_ft = basis_model
        self.classifier_ft = nn.Linear(args.class_innum, args.class_outnum)
        self.model_all_ft = nn.Sequential(self.model_ft, self.classifier_ft)

        # -------------------------------- Parallel training of models on multiple GPUs --------------------------------
        if self.device_count > 1:
            self.model_ft = torch.nn.DataParallel(self.model_ft)
            self.classifier_ft = torch.nn.DataParallel(self.classifier_ft)

        # -------------------------------- Define the learning parameters ----------------------------------------------
        parameter_list_ft = [{"params": self.model_ft.parameters(), "lr": args.lr_ft},
                             {"params": self.classifier_ft.parameters(), "lr": args.lr_ft}]

        # -------------------------------- Define the optimizer --------------------------------------------------------
        self.optimizer_ft = optim.SGD(parameter_list_ft, lr=args.lr_ft, momentum=args.momentum_ft,
                                      weight_decay=args.weight_decay_ft)

        # -------------------------------- Define the learning rate decay ----------------------------------------------
        steps = [int(step) for step in args.steps_ft.split(',')]
        self.lr_scheduler_ft = optim.lr_scheduler.MultiStepLR(self.optimizer_ft, steps, gamma=args.gamma_ft)

        # -------------------------------- Invert the model ------------------------------------------------------------
        self.model_ft.to(self.device)
        self.classifier_ft.to(self.device)

        # -------------------------------- Define the distance loss ----------------------------------------------------
        self.criterion_ft = nn.CrossEntropyLoss()

        self.start_epoch_ft = 0

    def ft(self):

        args = self.args

        best_acc = 0.0
        min_loss = 100.0

        # target_train
        ACC_tt_ft = np.zeros(args.max_epoch_ft, dtype=float)
        Loss_tt_ft = np.zeros(args.max_epoch_ft, dtype=float)

        # source_train
        ACC_st_ft = np.zeros(args.max_epoch_ft, dtype=float)
        Loss_st_ft = np.zeros(args.max_epoch_ft, dtype=float)

        # source_val
        ACC_sv_ft = np.zeros(args.max_epoch_ft, dtype=float)
        Loss_sv_ft = np.zeros(args.max_epoch_ft, dtype=float)

        # target_val
        ACC_tv_ft = np.zeros(args.max_epoch_ft, dtype=float)
        Loss_tv_ft = np.zeros(args.max_epoch_ft, dtype=float)

        if args.Fine_grained_testing:
            Actual_labels = None
            Predicted_labels = None

        for epoch in range(self.start_epoch_ft, args.max_epoch_ft):

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch_ft - 1) + '-' * 5)

            # -------------------------------- Update the learning rate ------------------------------------------------
            if self.lr_scheduler_ft is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler_ft.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr_ft))

            # -------------------------------- Check the freezing state of parameters ----------------------------------
            if epoch <= 2:
                print("===== Freeze Status =====")
                check_requires_grad(self.model_ft)
                print("========================")
                check_requires_grad(self.classifier_ft)
                print("========================")

            for phase in ['target_train', 'source_train', 'source_val', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_length = 0
                epoch_acc = 0
                epoch_loss = 0.0

                # -------------------------------- Set model to train mode or test mode --------------------------------
                if phase == 'target_train':
                    self.model_ft.train()
                    self.classifier_ft.train()
                else:
                    self.model_ft.eval()
                    self.classifier_ft.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'target_train'):
                        # ------------------------------------forward --------------------------------------------------
                        features = self.model_ft(inputs)
                        outputs = self.classifier_ft(features)

                        # ------------------------------------loss -----------------------------------------------------
                        logits = outputs
                        loss = self.criterion_ft(logits, labels)

                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        epoch_acc += correct
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_length += labels.size(0)

                        if args.Fine_grained_testing:
                            if epoch == args.max_epoch_ft - 1 and phase == 'target_val':
                                if Actual_labels is None:
                                    Actual_labels = labels
                                else:
                                    Actual_labels = torch.cat([Actual_labels, labels])

                                if Predicted_labels is None:
                                    Predicted_labels = pred
                                else:
                                    Predicted_labels = torch.cat([Predicted_labels, pred])

                        # Calculate the training information
                        if phase == 'target_train':
                            # backward
                            self.optimizer_ft.zero_grad()
                            loss.backward()
                            self.optimizer_ft.step()

                # -------------------------------- Print the train and val information via each epoch ------------------
                epoch_acc = epoch_acc / epoch_length
                epoch_loss = epoch_loss / epoch_length
                if phase == 'target_train':
                    ACC_tt_ft[epoch] = epoch_acc
                    Loss_tt_ft[epoch] = epoch_loss
                if phase == 'source_train':
                    ACC_st_ft[epoch] = epoch_acc
                    Loss_st_ft[epoch] = epoch_loss
                if phase == 'source_val':
                    ACC_sv_ft[epoch] = epoch_acc
                    Loss_sv_ft[epoch] = epoch_loss
                if phase == 'target_val':
                    ACC_tv_ft[epoch] = epoch_acc
                    Loss_tv_ft[epoch] = epoch_loss

                if args.Fine_grained_testing and epoch == args.max_epoch_ft - 1 and phase == 'target_val':
                    matrix = torch.cat([Actual_labels.unsqueeze(0), Predicted_labels.unsqueeze(0)], dim=0)
                    matrix_np = matrix.cpu().numpy()
                    data_dict = {'matrix': matrix_np}

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                # save the model
                # if phase == 'target_val':
                #     # save the checkpoint for other learning
                #     model_state_dic = self.model_all_ft.state_dict()
                #     # save the best model according to the val accuracy
                #     if (epoch_acc > best_acc and epoch_loss < min_loss) and (epoch > 50 - 1):
                #         best_acc = epoch_acc
                #         min_loss = epoch_loss
                #         logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                #         torch.save(model_state_dic,
                #                    os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

            if self.lr_scheduler_ft is not None:
                self.lr_scheduler_ft.step()

        if args.Fine_grained_testing:
            return ACC_tt_ft, Loss_tt_ft, ACC_st_ft, Loss_st_ft, ACC_sv_ft, Loss_sv_ft, ACC_tv_ft, Loss_tv_ft, data_dict
        else:
            return ACC_tt_ft, Loss_tt_ft, ACC_st_ft, Loss_st_ft, ACC_sv_ft, Loss_sv_ft, ACC_tv_ft, Loss_tv_ft
