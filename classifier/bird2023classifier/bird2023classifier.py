import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from tqdm import tqdm

from . import bird2023model, dataset


class Bird2023Classifier: 
    def __init__(self, cfg, train_X, train_y, valid_X=None, valid_y=None):
        """
        introduction: 初始化函数，用于初始化数据集，数据加载器，回调函数，模型和训练器。
        args:
            cfg: 配置参数字典。
            train_X: 训练数据的输入特征。
            train_y: 训练数据的标签。
            valid_X: 验证数据的输入特征，默认为None。
            valid_y: 验证数据的标签，默认为None。
        """
        
        # 创建训练数据集
        train_dataset = dataset.Bird2023Dataset(
            cfg, train_X, train_y, train=True
        )

        # 创建训练数据加载器
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            **cfg["train_loader"],
        )

        # 创建验证数据集和数据加载器
        if valid_X is None:
            self.valid_dataloader = None
        else:
            valid_dataset = dataset.Bird2023Dataset(
                cfg, valid_X, valid_y, train=False
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset,
                **cfg["valid_loader"],
            )

        # 创建pytorch-lightning的回调函数列表
        # 学习率的设计
        callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="step")]

        # 定义早停策略
        if cfg["model"]["early_stopping_patience"] is not None:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    "valid_avg_loss",
                    patience=cfg["model"]["early_stopping_patience"],
                )
            )

        # 定义模型保存方法
        if cfg["model"]["model_save"]:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}",
                    filename=f"last_epoch_fold{cfg['fold_n']}"
                    if cfg["general"]["cv"]
                    else "last_epoch",
                    save_weights_only=cfg["model"]["save_weights_only"],
                )
            )

        # 创建模型和训练器
        self.model = bird2023model.Bird2023Model(cfg)
        self.trainer = Trainer(
            callbacks=callbacks,
            **cfg["pl_params"],
        )

        self.cfg = cfg  # 保存配置参数

    def train(self, weight_path=None):  
        """
        introduction: 训练模型的函数。
        args:
            weight_path: 模型权重路径，默认为None。
        """
        
        # 执行模型训练
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.valid_dataloader,
            ckpt_path=weight_path,
        )

    def predict(self, test_X, weight_path=None):  
        """
        introduction: 进行预测的函数。
        args:
            test_X: 测试数据的输入特征。
            weight_path: 模型权重路径，默认为None。
        returns: 
            preds: 预测结果。
        """
        
        # 进行预测
        preds = []
        test_dataset = dataset.Bird2023Dataset(self.cfg, test_X, train=False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model, dataloaders=test_dataloader, ckpt_path=weight_path
        )
        preds = torch.cat(preds, axis=0)  # 合并预测结果
        preds = preds.cpu().detach().numpy()  # 将预测结果转换为numpy数组
        return preds

    def load_weight(self, weight_path):  
        """
        introduction: 加载模型权重的函数。
        args:
            weight_path: 模型权重路径。
        """
        
        # 加载模型权重
        self.model = self.model.load_from_checkpoint(
            checkpoint_path=weight_path,
            cfg=self.cfg,
        )
        print(f"loaded model ({weight_path})")  # 输出模型权重加载信息









class Bird2023ClassifierInference:
    # 创建一个类来进行鸟类声音的分类推断

    def __init__(self, cfg, weight_path=None):
        # 初始化函数，用于初始化模型配置和权重路径
        # Args: 
        # cfg: 配置信息，字典形式
        # weight_path: 预训练权重的路径，字符串形式
        self.weight_path = weight_path  # 设置模型的权重路径
        self.cfg = cfg  # 设置模型的配置信息

        self.model = bird2023model.Bird2023Model(self.cfg)
        
        self.trainer = Trainer(**self.cfg["pl_params"])  # 初始化训练器，传入参数为配置信息中的"pl_params"字段

    def load_weight(self, weight_path):
        # 加载模型权重
        # Args: 
        # weight_path: 预训练权重的路径，字符串形式
        self.model = self.model.load_from_checkpoint(
            checkpoint_path=weight_path,  # 权重路径
            cfg=self.cfg,  # 模型的配置信息
        )
        print(f"loaded model ({weight_path})")  # 输出加载成功的信息

    def predict(self, test_X):
        # 预测函数
        # Args: 
        # test_X: 测试数据，numpy数组或者其他能被模型接收的数据格式
        # Returns:
        # preds: 模型预测的结果，numpy数组格式
        test_dataset = dataset.Bird2023Dataset(self.cfg, test_X, train=False)  # 使用配置信息和测试数据创建一个测试数据集
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,  # 加载创建好的测试数据集
            **self.cfg["test_loader"],  # 传入配置信息中的"test_loader"字段作为参数
        )
        preds = self.trainer.predict(
            self.model,  # 使用模型
            dataloaders=test_dataloader,  # 用于预测的数据加载器
            ckpt_path=self.weight_path  # 预训练模型的权重路径
        )
        preds = torch.cat(preds, axis=0)  # 将所有预测结果在0轴上进行拼接
        preds = preds.cpu().detach().numpy()  # 将预测结果从GPU移至CPU，并从计算图中移除，然后转化为numpy数组

        return preds  # 返回预测结果

    def test_predict(self, ogg_name_list, sample_submission):
        # 进行测试预测的函数，生成提交的csv文件
        # Args: 
        # ogg_name_list: .ogg音频文件名称的列表，用于生成提交文件的row_id
        # sample_submission: 提交的示例文件，用于生成提交的csv文件的列名
        # Returns:
        # preds: 模型预测的结果，numpy数组格式
        sub_df = pd.DataFrame(columns=sample_submission.columns)  # 创建一个空的dataframe，列名为sample_submission的列名
        test_dataset = dataset.Bird2023TestDataset(self.cfg, ogg_name_list)  # 使用配置信息和.ogg文件名列表创建一个测试数据集

        self.load_weight(self.weight_path)  # 加载模型权重
        self.model.to("cpu")  # 将模型移至CPU
        self.model.eval()  # 设置模型为评估模式

        for i, data in enumerate(test_dataset):  # 对于测试数据集中的每一条数据
            preds = []  # 创建一个空列表用于存放预测结果
            ogg_name = ogg_name_list[i][:-4]  # 提取.ogg文件名，作为row_id的一部分
            for start in tqdm(
                range(0, len(data), self.cfg["test_loader"]["batch_size"])
            ):  # 对于数据的每一个批次
                with torch.no_grad():  # 关闭自动求导
                    pred = self.model(
                        data[start : start + self.cfg["test_loader"]["batch_size"]]
                    )  # 对该批次数据进行预测
                    pred = torch.sigmoid(pred).to("cpu")  # 对预测结果进行sigmoid激活，并移至CPU
                preds.append(pred)  # 将预测结果添加到preds列表中
            preds = torch.cat(preds)  # 将所有预测结果在0轴上进行拼接
            preds = preds.cpu().detach().numpy()  # 将预测结果从GPU移至CPU，并从计算图中移除，然后转化为numpy数组
            row_ids = [f"{ogg_name}_{(i+1)*5}" for i in range(len(preds))]  # 根据.ogg文件名和预测结果的数量生成row_id
            df = pd.DataFrame(columns=sample_submission.columns)  # 创建一个空的dataframe，列名为sample_submission的列名
            df["row_id"] = row_ids  # 设置dataframe的"row_id"列为生成的row_id列表
            df[df.columns[1:]] = preds  # 将预测结果设置为其他列的值
            sub_df = pd.concat([sub_df, df]).reset_index(drop=True)  # 将新生成的dataframe拼接到sub_df，并重设index

        return preds  # 返回预测结果

