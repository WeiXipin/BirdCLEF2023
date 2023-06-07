import pandas as pd
import numpy as np
import sklearn


def upsample_data(df, thr=20, seed=42):
    """
    Introduction:
        对于数据集df中样本数量不足thr的类别进行上采样，以达到更平衡的类别分布
    Args:
        df: 数据集，Pandas.DataFrame类型
        thr: 用于判断类别是否需要上采样的样本数量阈值，int类型，默认值为20
        seed: 随机种子，int类型，默认值为42
    Returns:
        up_df: 进行上采样后的数据集，Pandas.DataFrame类型
    """
    # 获取类别分布
    class_dist = df["primary_label"].value_counts()

    # 获取样本数量少于阈值的类别
    down_classes = class_dist[class_dist < thr].index.tolist()

    # 创建一个空列表，用于存放上采样后的数据
    up_dfs = []

    # 对样本数量不足的类别进行上采样
    for c in down_classes:
        # 获取当前类别的数据
        class_df = df.query("primary_label==@c")
        # 计算需要增加的样本数量
        num_up = thr - class_df.shape[0]
        # 进行上采样
        class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
        # 将上采样后的数据添加到列表中
        up_dfs.append(class_df)

    # 将上采样后的数据与原始数据合并
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df



def padded_cmap(y_true, y_score, padding_factor=5):
    """
    Introduction:
        对预测得分y_score和真实标签y_true进行padding，然后计算宏平均AP值
    Args:
        y_true: 真实标签，numpy.ndarray类型
        y_score: 预测得分，numpy.ndarray类型
        padding_factor: padding的数量，int类型，默认值为5
    Returns:
        score: 宏平均AP值，float类型
    """
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(y_true.shape[1])])
    padded_y_true = np.concatenate([y_true, new_rows])
    padded_y_score = np.concatenate([y_score, new_rows])
    score = sklearn.metrics.average_precision_score(
        padded_y_true, padded_y_score, average="macro"
    )
    return score


import wandb
# 初始化wandb
def wandb_start(cfg):
    """
    Introduction:
        使用配置文件中的参数初始化wandb
    Args:
        cfg: 配置文件，dict类型
    Returns:
        None
    """
    wandb.init(
        project=cfg["general"]["project_name"],
        name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
        group=f"{cfg['general']['save_name']}_cv" if cfg["general"]["cv"] else "all",
        job_type=cfg["job_type"],
        mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
        config=cfg,
    )




import torch
import torch.nn as nn

class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    '''
    introduction: 初始化函数，定义一些成员变量，包括平滑误差、权重和损失函数的计算方式
    args: 
    - smooth_eps: 平滑误差，默认为0.0025
    - weight: BCE损失函数的权重，默认为None
    - reduction: BCE损失函数的计算方式，默认为"mean"
    '''
    def __init__(self, smooth_eps=0.0025, weight=None, reduction="mean"):
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        self.smooth_eps = smooth_eps
        self.weight = weight
        self.reduction = reduction
        # 使用nn模块的BCEWithLogitsLoss函数创建一个损失函数对象，传入的参数为权重和计算方式
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(weight=self.weight, reduction=self.reduction)

    def forward(self, input, target):
        # 将真实标签值转为浮点型，并且用clamp函数确保标签值在[smooth_eps, 1.0 - smooth_eps]的范围内，做标签平滑处理
        target_smooth = torch.clamp(target.float(), self.smooth_eps, 1.0 - self.smooth_eps)
        # 进一步进行标签平滑处理，将平滑误差除以标签的数量，然后加到平滑后的标签值上
        target_smooth = target_smooth + (self.smooth_eps / target.size(1))
        # 使用BCEWithLogitsLoss计算平滑后的标签值和预测标签值的损失
        return self.bce_with_logits_loss(input, target_smooth)
