from torch import nn
import numpy as np


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    # 计算梯度反转系数，控制对抗训练的强度
    return np.float64(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    # 自定义反向传播时的梯度操作
    def fun1(grad):
        # 在前向传播时，GRL 不改变输入值；在反向传播时，将梯度乘以 -coeff，实现梯度反转
        return -coeff * grad.clone()
    return fun1

class AdversarialNet(nn.Module):
    def __init__(self, in_feature, hidden_size, max_iter=10000.0):
        super(AdversarialNet, self).__init__()
        self.ad_layer1 = nn.Sequential(
            nn.Linear(in_feature, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.ad_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.ad_layer3 = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

        # parameters
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1

        # -------------------------------- Calculate the trade off parameter lam ---------------------------------------
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)

        x = x * 1.0
        x.register_hook(grl_hook(coeff))

        x = self.ad_layer1(x)
        x = self.ad_layer2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y
