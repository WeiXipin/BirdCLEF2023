import argparse
import ast
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import yaml
from pytorch_lightning import seed_everything
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from pathlib import Path

from classifier import bird2023classifier
from utils import upsample_data, padded_cmap



def one_fold(skf, cfg, train_X, train_y, fold_n):
    """
    Introduction: 执行一次交叉验证的训练过程。
    Args:
        skf: 一个预设的sklearn.model_selection.StratifiedKFold实例。
        cfg: 配置字典，包含模型训练的各种参数。
        train_X: 训练数据特征。
        train_y: 训练数据标签。
        fold_n: 当前是第几折的交叉验证。
    Returns:
        valid_preds: 验证集的预测值。
        padded_cmap_score: 验证集的评分。
    """
    # 输出当前执行的交叉验证折数
    print(f"[fold_{fold_n}] start")
    # 设置随机种子以保证实验结果的可复现性
    seed_everything(cfg["general"]["seed"], workers=True)
    
    # 确保所有的类都存在于训练折中
    # 根据cv字段进行数据的划分
    train_y_minor = train_y[~train_X["cv"]].reset_index(drop=True)
    train_X_minor = train_X[~train_X["cv"]].reset_index(drop=True)
    train_y = train_y[train_X["cv"]]
    train_X = train_X[train_X["cv"]]
    train_y = train_y.reset_index(drop=True)
    train_X = train_X.reset_index(drop=True)
    
    # 使用StratifiedKFold进行分层采样，以保证每个类别的分布比例相同
    train_indices, valid_indices = list(skf.split(train_X, train_X["primary_label"]))[fold_n]
    train_X_cv, train_y_cv = (
        train_X.iloc[train_indices].reset_index(drop=True),
        train_y.iloc[train_indices].reset_index(drop=True),
    )
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )

    # 合并两部分训练数据
    train_X_cv = pd.concat([train_X_cv, train_X_minor], axis=0).reset_index(drop=True)
    train_y_cv = pd.concat([train_y_cv, train_y_minor], axis=0).reset_index(drop=True)
    
    # 打印类别数和训练数据条数
    print(train_X_cv["primary_label"].unique().__len__())
    print(len(train_X_cv))
    
    # 进行过采样以解决类别不平衡问题
    # 因为评价指标：sklearn.metrics.average_precision_score 选择的是macro,而不是micro
    # macro不在乎样本的差异,就算是小样本对总体也有非常大的影响
    
    # 合并训练数据和标签
    train_cv = pd.concat([train_X_cv, train_y_cv], axis=1).reset_index(drop=True)
    # 执行过采样
    train_cv = upsample_data(train_cv, thr=cfg["oversampling"], seed=cfg["general"]["seed"])
    # 分离训练数据和标签
    train_X_cv, train_y_cv = train_cv.iloc[:, :-264], train_cv.iloc[:, -264:]
    # 打印过采样后的训练数据条数
    print(len(train_X_cv))
        
    # 加入噪音音频数据
    datadir = Path(f"{cfg['general']['input_path']}/aicrowd2020_noise_30sec/noise_30sec/")
    all_audios = list(datadir.glob("*.ogg"))
    aicrowd2020 = ["aicrowd2020_noise_30sec/noise_30sec/" + ogg_name.name for ogg_name in all_audios]
    datadir = Path(f"{cfg['general']['input_path']}/ff1010bird_nocall/nocall/")
    all_audios = list(datadir.glob("*.ogg"))
    ff1010bird_nocall = ["ff1010bird_nocall/nocall/" + ogg_name.name for ogg_name in all_audios]
    datadir = Path(f"{cfg['general']['input_path']}/train_soundscapes/nocall/")
    all_audios = list(datadir.glob("*.ogg"))
    train_soundscapes = ["train_soundscapes/nocall/" + ogg_name.name for ogg_name in all_audios]
    
    # 构造噪音音频的特征和标签数据
    df_X = pd.DataFrame(columns=train_X_cv.columns)
    df_X["filename"] = aicrowd2020 + ff1010bird_nocall + train_soundscapes
    df_y = pd.DataFrame(columns=train_y_cv.columns, data=np.zeros((len(df_X), 264)))
    
    # 加入噪音音频数据
    train_X_cv = pd.concat([train_X_cv, df_X], axis=0).reset_index(drop=True)
    train_y_cv = pd.concat([train_y_cv, df_y], axis=0).reset_index(drop=True)
    print(len(train_X_cv))
    


    model = bird2023classifier.Bird2023Classifier(
        cfg, train_X=train_X_cv, train_y=train_y_cv, valid_X=valid_X_cv, valid_y=valid_y_cv
    )

    model.train(weight_path=None)
    valid_preds = model.predict(valid_X_cv)
    del model
    torch.cuda.empty_cache()




    print(f"[fold_{fold_n}]")
    # 是否存在二级标签
    if cfg["audio"]["second_label"]:
        valid_y_cv[valid_y_cv < 1.0] = 0.0
    # 计算padded_cmap_score评分
    padded_cmap_score = padded_cmap(valid_y_cv.values, valid_preds)
    # 记录评分
    print({"padded_cmap_score": padded_cmap_score})

    # 清空CUDA缓存
    torch.cuda.empty_cache()

    return valid_preds, padded_cmap_score




def main():
    with open("./train_effv2b0.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 如果参数中有指定的fold，则设置cfg的对应字段，并打印出来
    cfg["job_type"] = "train"

    # 设置随机种子
    seed_everything(cfg["general"]["seed"], workers=True)

    # 读取csv文件
    train = pd.read_csv(f"{cfg['general']['input_path']}/birdclef-2023/train_metadata.csv")
    # 组合生成音频文件的路径
    train["filename"] = "birdclef-2023/train_audio/" + train["filename"]
    # 将secondary_labels字段的字符串转换为列表
    train["secondary_labels"] = train["secondary_labels"].apply(ast.literal_eval)
    # 读取提交样本
    sample_submission = pd.read_csv(f"{cfg['general']['input_path']}/birdclef-2023/sample_submission.csv")
    # 提取目标（鸟类）的名字
    birds = sample_submission.columns[1:]
    # 对主要标签进行one-hot编码，并添加到原数据集中
    train = pd.concat([train, pd.get_dummies(train["primary_label"])], axis=1)
    # 修正列的顺序
    new_columns = list(train.columns.difference(birds)) + list(birds)
    train = train.reindex(columns=new_columns)

    # 搜索少数类
    counts = train.primary_label.value_counts()
    # 获取样本数小于5的类
    cond = train.primary_label.isin(counts[counts<5].index.tolist())
    # 获取包含训练索引的列
    no_cv = train[cond]["primary_label"].duplicated()
    no_cv_idx = train[cond][~no_cv].index
    # 在训练集中插入一个新列，用于标记是否参与交叉验证
    train.insert(0, "cv", True)
    # 对于样本数少于5的类，不参与交叉验证，设置cv列为False
    train.loc[no_cv_idx, "cv"] = False

    # 划分X和y
    train_X = train.iloc[:, :-264]
    # 对secondary_labels进行编码
    for idx, each_secondary_labels in enumerate(train["secondary_labels"]):
        for secondary_label in each_secondary_labels:
            for bird in birds:
                if secondary_label == bird:
                    train.loc[idx, bird] = 0.5  # 编码值设为0.5

    train_y = train.iloc[:, -264:]
    # 设置模型的类别数
    cfg["model"]["num_classes"] = len(train_y.columns)


    # 初始化StratifiedKFold交叉验证工具
    skf = StratifiedKFold(
        n_splits=cfg["general"]["n_splits"],
        shuffle=True,
        random_state=cfg["general"]["seed"],
    )
    valid_cmap_list = []

    # 对每一个fold进行训练和验证
    for fold_n in tqdm(cfg["general"]["fold"]):
        cfg["fold_n"] = fold_n
        _, valid_cmap = one_fold(skf, cfg, train_X, train_y, fold_n)
        valid_cmap_list.append(valid_cmap)  # 将每个fold的验证结果收集起来

    # 计算所有fold验证结果的平均值
    valid_cmap_mean = np.mean(valid_cmap_list, axis=0)
    print(f"cv mean pfbeta:{valid_cmap_mean}")  # 打印平均结果







if __name__ == "__main__":
    main()