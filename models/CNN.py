#!/usr/bin/python
# -*- coding:utf-8 -*-

from torch import nn


class CNN(nn.Module):
    def __init__(self, data_set, in_channel=1):
        super(CNN, self).__init__()

        if data_set == 'CWRU':
            self.layer1 = nn.Sequential(
                nn.Conv1d(in_channel, 16, kernel_size=15),
                nn.BatchNorm1d(16),
                nn.ReLU(inplace=True))

            self.layer2 = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2), )

            self.layer3 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True))

            self.layer4 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(4))

            self.layer5 = nn.Sequential(
                nn.Linear(128 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)

        return x


# class CNN(nn.Module):
#     def __init__(self, data_set, in_channel=1):
#         super(CNN, self).__init__()
#
#         if data_set == 'CWRU':
#             self.layer1 = nn.Sequential(
#                 nn.Conv1d(in_channel, 16, kernel_size=15),
#                 nn.BatchNorm1d(16),
#                 nn.ReLU(inplace=True))
#
#             self.layer2 = nn.Sequential(
#                 nn.Conv1d(16, 32, kernel_size=3),
#                 nn.BatchNorm1d(32),
#                 nn.ReLU(inplace=True),
#                 )
#
#             self.layer3 = nn.Sequential(
#                 nn.MaxPool1d(kernel_size=2, stride=2), )
#
#             self.layer4 = nn.Sequential(
#                 nn.Conv1d(32, 64, kernel_size=3),
#                 nn.BatchNorm1d(64),
#                 nn.ReLU(inplace=True))
#
#             self.layer5 = nn.Sequential(
#                 nn.Conv1d(64, 128, kernel_size=3),
#                 nn.BatchNorm1d(128),
#                 nn.ReLU(inplace=True),
#                 )
#
#             self.layer6 = nn.Sequential(
#                 nn.AdaptiveMaxPool1d(4))
#
#             self.layer7 = nn.Sequential(
#                 nn.Linear(128 * 4, 256),
#                 nn.ReLU(inplace=True),
#                 )
#
#             self.layer8 = nn.Sequential(
#                 nn.Dropout())
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = x.view(x.size(0), -1)
#         x = self.layer7(x)
#         x = self.layer8(x)
#
#         return x
