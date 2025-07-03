import logging
import math
import os
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch import optim

import datasets
import models
from loss.DAN import DAN


class utils_SMM(object):

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

        # ---------------------------------------- Define the basic model ----------------------------------------------
        self.model = getattr(models, args.model_name)(data_set=args.data_name)
        self.classifier_layer = nn.Linear(args.class_innum, args.class_outnum)
        self.model_all = nn.Sequential(self.model, self.classifier_layer)

        # -------------------------------- Parallel training of models on multiple GPUs --------------------------------
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # -------------------------------- Define the learning parameters ----------------------------------------------
        parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                          {"params": self.classifier_layer.parameters(), "lr": args.lr}]

        # -------------------------------- Define the optimizer --------------------------------------------------------
        self.optimizer = optim.AdamW(params=parameter_list, lr=args.lr, weight_decay=args.weight_decay, eps=1e-8,
                                     betas=(0.9, 0.999))

        # -------------------------------- Define the learning rate decay ----------------------------------------------
        steps = [int(step) for step in args.steps.split(',')]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)

        # -------------------------------- Invert the model ------------------------------------------------------------
        self.model.to(self.device)
        self.classifier_layer.to(self.device)

        # -------------------------------- Define the loss -------------------------------------------------------------
        self.distance_loss = DAN
        self.criterion = nn.CrossEntropyLoss()

        # --------------------------------------------------------------------------------------------------------------
        self.start_epoch = 0

    def train(self):

        args = self.args

        best_acc = 0.0
        min_loss = 100.0

        # source_train
        ACC_st = np.zeros(args.max_epoch, dtype=float)
        Loss_st = np.zeros(args.max_epoch, dtype=float)
        Loss_CLA_st = np.zeros(args.max_epoch, dtype=float)
        Loss_DTL_st = np.zeros(args.max_epoch, dtype=float)

        # source_val
        ACC_sv = np.zeros(args.max_epoch, dtype=float)
        Loss_sv = np.zeros(args.max_epoch, dtype=float)

        # target_train
        ACC_tt = np.zeros(args.max_epoch, dtype=float)
        Loss_tt = np.zeros(args.max_epoch, dtype=float)

        # target_val
        ACC_tv = np.zeros(args.max_epoch, dtype=float)
        Loss_tv = np.zeros(args.max_epoch, dtype=float)

        if args.Fine_grained_testing:
            Actual_labels = None
            Predicted_labels = None

        for epoch in range(self.start_epoch, args.max_epoch):

            step = 0

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)

            # -------------------------------- Update the learning rate ------------------------------------------------
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            iter_target = iter(self.dataloaders['target_train'])
            len_target_loader = len(self.dataloaders['target_train'])

            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_train', 'target_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_length = 0
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_loss_cla = 0.0
                epoch_loss_dtl = 0.0

                # -------------------------------- Set model to train mode or test mode --------------------------------
                if phase == 'source_train':
                    self.model.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    if phase != 'source_train' or epoch < args.middle_epoch:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                    else:
                        source_inputs = inputs
                        target_inputs, _ = next(iter_target)
                        inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                    if args.Data_dependency and phase == 'source_train' and epoch >= args.middle_epoch:
                        if (step + 1) % len_target_loader == 0:
                            iter_target = iter(self.dataloaders['target_train'])

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # ------------------------------------forward --------------------------------------------------
                        features = self.model(inputs)
                        outputs = self.classifier_layer(features)

                        # ------------------------------------loss -----------------------------------------------------
                        if phase != 'source_train' or epoch < args.middle_epoch:
                            logits = outputs
                            loss = self.criterion(logits, labels)
                        else:

                            step = step + 1

                            logits = outputs.narrow(0, 0, labels.size(0))
                            classifier_loss = self.criterion(logits, labels)

                            # -------------------------------- Calculate the distance metric ---------------------------
                            distance_loss = self.distance_loss(features.narrow(0, 0, labels.size(0)),
                                                               features.narrow(0, labels.size(0),
                                                                               inputs.size(0) - labels.size(0)))

                            # -------------------------------- Calculate the trade off parameter lam -------------------
                            lam_distance = (1 - math.exp(
                                -10 * ((epoch - args.middle_epoch) / (args.max_epoch - args.middle_epoch)))) / (
                                                   1 + math.exp(-10 * ((epoch - args.middle_epoch) / (
                                                   args.max_epoch - args.middle_epoch))))

                            loss = classifier_loss + lam_distance * distance_loss

                            loss_cla = classifier_loss
                            loss_dtl = lam_distance * distance_loss

                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        epoch_acc += correct

                        if args.Fine_grained_testing:
                            if epoch == args.max_epoch - 1 and phase == 'target_val':
                                if Actual_labels is None:
                                    Actual_labels = labels
                                else:
                                    Actual_labels = torch.cat([Actual_labels, labels])

                                if Predicted_labels is None:
                                    Predicted_labels = pred
                                else:
                                    Predicted_labels = torch.cat([Predicted_labels, pred])

                        loss_temp = loss.item() * labels.size(0)
                        epoch_loss += loss_temp
                        epoch_length += labels.size(0)

                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            if epoch < args.middle_epoch:
                                loss_cla_temp = loss_temp
                                loss_dtl_temp = 0.0
                            else:
                                if torch.is_tensor(loss_cla):
                                    loss_cla_temp = loss_cla.item() * labels.size(0)
                                else:
                                    loss_cla_temp = loss_cla * labels.size(0)
                                if torch.is_tensor(loss_dtl):
                                    loss_dtl_temp = loss_dtl.item() * labels.size(0)
                                else:
                                    loss_dtl_temp = loss_dtl * labels.size(0)

                            epoch_loss_cla += loss_cla_temp
                            epoch_loss_dtl += loss_dtl_temp

                # -------------------------------- Print the train and val information via each epoch ------------------
                epoch_acc = epoch_acc / epoch_length
                epoch_loss = epoch_loss / epoch_length
                if phase == 'source_train':
                    epoch_loss_cla = epoch_loss_cla / epoch_length
                    epoch_loss_dtl = epoch_loss_dtl / epoch_length
                    ACC_st[epoch] = epoch_acc
                    Loss_st[epoch] = epoch_loss
                    Loss_CLA_st[epoch] = epoch_loss_cla
                    Loss_DTL_st[epoch] = epoch_loss_dtl
                if phase == 'source_val':
                    ACC_sv[epoch] = epoch_acc
                    Loss_sv[epoch] = epoch_loss
                if phase == 'target_train':
                    ACC_tt[epoch] = epoch_acc
                    Loss_tt[epoch] = epoch_loss
                if phase == 'target_val':
                    ACC_tv[epoch] = epoch_acc
                    Loss_tv[epoch] = epoch_loss

                if args.Fine_grained_testing and epoch == args.max_epoch - 1 and phase == 'target_val':
                    matrix = torch.cat([Actual_labels.unsqueeze(0), Predicted_labels.unsqueeze(0)], dim=0)
                    matrix_np = matrix.cpu().numpy()
                    data_dict = {'matrix': matrix_np}

                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                # save the model
                # if phase == 'target_val':
                #     # save the checkpoint for other learning
                #     model_state_dic = self.model_all.state_dict()
                #     # save the best model according to the val accuracy
                #     if (epoch_acc > best_acc and epoch_loss < min_loss) and (epoch > args.middle_epoch - 1):
                #         best_acc = epoch_acc
                #         min_loss = epoch_loss
                #         logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                #         torch.save(model_state_dic,
                #                    os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # save the basis model
        if args.Fine_grained_testing:
            return ACC_st, Loss_st, ACC_sv, Loss_sv, ACC_tt, Loss_tt, ACC_tv, Loss_tv, data_dict
        else:
            return ACC_st, Loss_st, ACC_sv, Loss_sv, ACC_tt, Loss_tt, ACC_tv, Loss_tv
