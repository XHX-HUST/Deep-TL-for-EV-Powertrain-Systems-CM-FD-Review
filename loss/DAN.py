
import torch

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    # ---------------------------------------------- 01 ----------------------------------------------------------------
    # # kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数
    # # kernel_num: 取不同高斯核的数量
    #
    # n_samples = int(source.size()[0])+int(target.size()[0])
    #
    # # 合并源域和目标域数据
    # total = torch.cat([source, target], dim=0)  # 形状: (2*batch, sample_num)
    # # 计算所有样本的模长平方（逐元素平方后按行求和）
    # norms = torch.sum(total * total, dim=1, keepdim=True)  # 形状: (2*batch, 1)
    # # 计算所有样本对的内积矩阵
    # inner_products = torch.mm(total, total.t())  # 形状: (2*batch, 2*batch)
    # # 计算L2距离平方矩阵
    # L2_distance = norms + norms.t() - 2 * inner_products  # 形状: (2*batch, 2*batch)
    # L2_distance = torch.clamp(L2_distance, min=0.0)  # 防止浮点数误差产生负数
    #
    # # 计算基础带宽（自动估计或固定值）
    # if fix_sigma:
    #     bandwidth = fix_sigma
    # else:
    #     bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2-n_samples)
    #
    # # 计算带宽缩放因子,改进后的：[σ/4, σ/2, σ, 2σ, 4σ]（覆盖小、中、大尺度），不计算缩放因子的：[σ, 2σ, 4σ, 8σ, 16σ]（仅大尺度）
    # # 通过调整初始带宽，生成以原始带宽为中心的对称几何级数带宽列表，使多核高斯核能够覆盖从局部到全局的多尺度特征，增强域适应模型对分布差异的捕捉能力。
    # # 如果删除这行代码，带宽列表将仅从原始带宽开始单调递增，导致核的尺度范围偏斜，可能遗漏关键特征尺度，影响 MMD 的有效性。
    # scale_factor = kernel_mul ** (kernel_num // 2)
    # adjusted_bandwidth = bandwidth / scale_factor  # 调整后的初始带宽
    #
    # # 生成对称的带宽列表
    # bandwidth_list = [adjusted_bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    #
    # # 计算多核矩阵并求和（权重均为1/M）
    # kernel_val = [torch.exp(-L2_distance / (1 * bandwidth_temp)) for bandwidth_temp in bandwidth_list]
    # return sum(kernel_val)  # / len(kernel_val)

    # ----------------------------------------------------- 02 ---------------------------------------------------------
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # 源域内部的距离
    YY = kernels[batch_size:, batch_size:]  # 目标域内部的距离
    XY = kernels[:batch_size, batch_size:]  # 源域到目标域的距离
    YX = kernels[batch_size:, :batch_size]  # 目标域到源域的距离
    loss = torch.mean(XX + YY - XY - YX)
    return loss
