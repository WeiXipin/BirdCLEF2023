import argparse
import ast
import json
import os

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import yaml
from pytorch_lightning import seed_everything
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from classifier import bird2023classifier




def main():
    
    # 使用yaml库打开并安全加载配置文件
    with open("./pretrain_effv2b0.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["job_type"] = "pretrain"
    seed_everything(cfg["general"]["seed"], workers=True)

    # 读取csv文件
    # 使用pandas库的read_csv函数读取csv文件
    bird2021 = pd.read_csv(f"{cfg['general']['pretrain_input_path']}/birdclef-2021/train_metadata.csv")
    bird2022 = pd.read_csv(f"{cfg['general']['pretrain_input_path']}/birdclef-2022/train_metadata.csv")
    bird2023 = pd.read_csv(f"{cfg['general']['input_path']}/birdclef-2023/train_metadata.csv")

    # 在bird2021和bird2022数据框中添加"birdclef"字段，并设置其值为相应的年份
    bird2021["birdclef"] = "2021"
    bird2022["birdclef"] = "2022"

    # 在bird2021, bird2022, bird2023数据框中构造"filepath"字段，根据其它字段的值来设定
    bird2021["filepath"] = "birdclef-2021/train_short_audio/" + bird2021["primary_label"] + "/" + bird2021["filename"]
    bird2022["filepath"] = "birdclef-2022/train_audio/" + bird2022["filename"]
    bird2023["filepath"] = "birdclef-2023/train_audio/" + bird2023["filename"]

    # 在bird2023数据框中设定"filename"字段，由"filepath"字段分割出文件名部分
    bird2023["filename"] = bird2023["filepath"].map(
        lambda x: x.split("/")[-1].split(".")[0]
    )

    # 将bird2021和bird2022数据框沿行(axis=0)方向合并，重置索引
    train = pd.concat(
        [bird2021, bird2022], axis=0
    ).reset_index(drop=True)

    # 对"filepath"字段进行处理，移除文件后缀部分，对"filename"字段进行处理，仅保留文件名部分
    train["filepath"] = train["filepath"].map(lambda x: x.split(".")[0])
    train["filename"] = train["filepath"].map(lambda x: x.split("/")[-1])

    # 删除重复的行，并设置相应的列名
    nodup_idx = train[["filename", "primary_label", "author"]].drop_duplicates().index
    train = train.loc[nodup_idx].reset_index(drop=True)

    # 删除train中出现在bird2023的"filename"字段中的行
    train = train[~train["filename"].isin(bird2023["filename"])].reset_index(drop=True)

    # 选择训练集需要的字段
    train = train[["filepath", "filename", "primary_label", "secondary_labels", "birdclef"]]
    #最主要鸟类，次要标签。预测主要鸟类就可以 
    # 对"secondary_labels"字段进行解析，使其从字符串形式转为列表形式
    train["secondary_labels"] = train["secondary_labels"].apply(ast.literal_eval)

    # 对"primary_label"字段进行one-hot编码，并将结果拼接到train数据框后面
    train = pd.concat([train, pd.get_dummies(train["primary_label"])], axis=1)
    cfg["model"]["num_classes"] = pd.get_dummies(train["primary_label"]).shape[1]
    birds = train.iloc[:, -cfg["model"]["num_classes"]:].columns

    # 打印编码后的"primary_label"字段的形状
    print(pd.get_dummies(train["primary_label"]).shape)

    # 查找数量较少的类别
    # 先拿到鸟类数量
    counts = train.primary_label.value_counts()
    # 设置条件，选择样本数少于`thr`的类别
    # 当数量少于5的时候，就作为no_cv，让它不在交叉验证范围里面，让它一定在训练集，而不在交叉验证集合里
    cond = train.primary_label.isin(counts[counts<5].index.tolist())
    # 检索包含的训练索引 
    no_cv = train[cond]["primary_label"].duplicated()
    no_cv_idx = train[cond][~no_cv].index
    # 添加一个新列以选择交叉验证的样本
    # 设置cv列，把不进行交叉验证的设置为false
    train.insert(0, "cv", True)
    # 对于样本数小于thr的类别，将cv设置为False
    train.loc[no_cv_idx, "cv"] = False

    # 划分特征X和目标y
    train_X = train.iloc[:, :-cfg["model"]["num_classes"]]

    # 对secondary label进行编码
    for idx, each_secondary_labels in enumerate(train["secondary_labels"]):
        for secondary_label in each_secondary_labels:
            for bird in birds:
                if secondary_label == bird:
                    train.loc[idx, bird] = 0.5

    # 获取目标变量
    train_y = train.iloc[:, -cfg["model"]["num_classes"]:]






    # 训练所有数据
    cfg["fold_n"] = "all"
    print("[all_train] start")
    seed_everything(cfg["general"]["seed"], workers=True)

    # 训练
    model = bird2023classifier.Bird2023Classifier(cfg, train_X, train_y, valid_X=None, valid_y=None)   
    model.train(weight_path=None)
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()