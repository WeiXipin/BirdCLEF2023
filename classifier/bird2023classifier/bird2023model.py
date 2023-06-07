# 引入所需库
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn.metrics
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append("../..")
from utils import padded_cmap, LabelSmoothingBCEWithLogitsLoss 


def init_bn(bn):
    # 初始化批量归一化层(BatchNorm)的权重和偏置
    # 输入：bn - 需要被初始化的批量归一化层

    bn.bias.data.fill_(0.0)  # 将偏置初始化为0
    bn.weight.data.fill_(1.0)  # 将权重初始化为1

# 定义函数 init_layer，用于初始化网络层
def init_layer(layer):
    # 使用 Xavier 均匀分布初始化权重
    nn.init.xavier_uniform_(layer.weight)

    # 如果该层有偏置项
    if hasattr(layer, "bias"):
        # 如果偏置项不为空
        if layer.bias is not None:
            # 将偏置项设为 0
            layer.bias.data.fill_(0.0)


class AttBlockV2(nn.Module):
    """
    创建一个继承了nn.Module的类，具有注意力机制的模块。

    Args:
      in_features: int, 输入的特征数量
      out_features: int, 输出的特征数量
      activation: str, 激活函数的类型，默认为线性激活函数

    Returns:
      输出张量x, 注意力权重 norm_att, 和激活后的结果 cla
    """

    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()  # 调用父类nn.Module的构造函数

        self.activation = activation  # 设置激活函数的类型

        # 定义一个一维卷积层，用于实现注意力机制。输入特征数为in_features，输出特征数为out_features，卷积核大小为1。
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        
        # 定义另一个一维卷积层，用于进行分类。输入特征数为in_features，输出特征数为out_features，卷积核大小为1。
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.init_weights()  # 初始化权重

    def init_weights(self):
        """
        初始化权重的方法。
        """
        init_layer(self.att)  # 初始化att层的权重
        init_layer(self.cla)  # 初始化cla层的权重

    def forward(self, x):
        """
        前向传播函数

        Args:
          x: 输入的张量, 形状为 (n_samples, n_in, n_time)

        Returns:
          x: 输出的张量
          norm_att: 注意力权重
          cla: 经过非线性变换后的cla
        """
        # 将att层的输出通过tanh函数和softmax函数进行变换，得到注意力权重norm_att
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        
        # 将cla层的输出通过非线性变换函数进行变换，得到cla
        cla = self.nonlinear_transform(self.cla(x))
        
        # norm_att和cla进行element-wise乘法，然后在第2维上求和，得到输出x
        x = torch.sum(norm_att * cla, dim=2)

        return x, norm_att, cla

    def nonlinear_transform(self, x):
        """
        非线性变换函数

        Args:
          x: 输入的张量

        Returns:
          x: 经过非线性变换后的张量
        """
        # 根据self.activation的值选择不同的激活函数进行变换
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)  # sigmoid函数进行非线性变换
        


class EncoderModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()  # 调用父类的初始化函数
        self.cfg = cfg  # 将配置字典保存为类的属性
        
        # 使用timm库创建基础模型，可以自动下载并加载预训练权重
        base_model = timm.create_model(
            model_name=cfg["model"]["model_name"],
            pretrained=cfg["model"]["pretrained"],
            in_chans=cfg["model"]["in_chans"],
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
        )

        self.bn0 = nn.BatchNorm2d(cfg["mel_specgram"]["n_mels"])

        # 根据基础模型的结构，获取不同的层并保存为encoder（编码器）
        if hasattr(base_model, "fc"):
            layers = list(base_model.children())[:-2]
            self.encoder = nn.Sequential(*layers)
            in_features = base_model.fc.in_features
        elif "eca_nfnet" in self.cfg["model"]["model_name"]:
            layers = list(base_model.children())[:-1]
            self.encoder = nn.Sequential(*layers)
            in_features = base_model.head.fc.in_features
        else:
            layers = list(base_model.children())[:-2]
            self.encoder = nn.Sequential(*layers)
            in_features = base_model.classifier.in_features


        self.in_features = in_features
        self.fc1 = nn.Linear(self.in_features, self.in_features, bias=True)
        self.att_block = AttBlockV2(
            self.in_features, cfg["model"]["num_classes"], activation="sigmoid"
        )
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)  # 初始化第一个批量归一化层
        init_layer(self.fc1)  # 初始化第一个全连接层

    def set_head(self, out_features):
        # 使用AttBlockV2（注意力模块），设置输入特征数为self.in_features，输出特征数为out_features，激活函数为sigmoid
        self.att_block = AttBlockV2(
            self.in_features, out_features, activation="sigmoid"
        )




class Bird2023Model(pl.LightningModule):
    # 类的构造函数
    def __init__(self, cfg):
        super().__init__()  # 调用父类构造函数
        self.cfg = cfg  # 保存配置信息
        
        # 根据配置信息判断是否有预训练模型路径，有则加载预训练模型
        if cfg["model"]["pretrained_path"]:
            # 保存类别数量
            tmp_num_classes = cfg["model"]["num_classes"]
            # 设置预训练模型类别数
            cfg["model"]["num_classes"] = cfg["model"]["pretrained_classes"]
            # 从指定路径加载预训练模型
            sed_model = EncoderModel(cfg)
            sed_model = sed_model.load_from_checkpoint(
                checkpoint_path=cfg["model"]["pretrained_path"],
                cfg=cfg
            )
            sed_model.set_head(tmp_num_classes)

            cfg["model"]["num_classes"] = tmp_num_classes

            
            # 使用timm库创建基础模型，该库中包含了大量预训练模型
            base_model = timm.create_model(
                model_name=cfg["model"]["model_name"],  # 模型名称
                pretrained=cfg["model"]["pretrained"],  # 是否使用预训练模型
                in_chans=cfg["model"]["in_chans"],  # 输入通道数
                num_classes=cfg["model"]["num_classes"],  # 类别数
                drop_rate=cfg["model"]["drop_rate"],  # dropout比率
                drop_path_rate=cfg["model"]["drop_path_rate"],  # 路径丢弃率

            )



            layers = list(sed_model.encoder) + list(base_model.children())[-2:]

            self.model = nn.Sequential(*layers)
            del sed_model, base_model
            torch.cuda.empty_cache()
            # pretrained_dict = torch.load(cfg["model"]["pretrained_path"], map_location=torch.device('cpu'))
            # pretrained_dict = pretrained_dict["state_dict"]
            # pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            # model_dict = self.model.state_dict()
            # model_dict.update(pretrained_dict)
            # self.model.load_state_dict(model_dict)

        else:
            # 没有预训练模型，直接使用timm创建模型
            self.model = timm.create_model(
                model_name=cfg["model"]["model_name"],  # 模型名称
                pretrained=cfg["model"]["pretrained"],  # 是否使用预训练模型
                in_chans=cfg["model"]["in_chans"],  # 输入通道数
                num_classes=cfg["model"]["num_classes"],  # 类别数
                drop_rate=cfg["model"]["drop_rate"],  # dropout比率
                drop_path_rate=cfg["model"]["drop_path_rate"],  # 路径丢弃率
            )
            # 如果配置中avg_and_max为True则进行平均和最大值池化，这部分功能还未实现

        # 根据配置文件选择损失函数
        if cfg["model"]["criterion"] == "bce_smooth":
            self.criterion = LabelSmoothingBCEWithLogitsLoss()  # 使用平滑交叉熵损失，让标签loss更靠近0.5,不是更自信的（标签不那么干净的时候）
        else:
            self.criterion = nn.__dict__[cfg["model"]["criterion"]]()  # 使用指定名称的损失函数

        # 如果开启梯度检查点，则设置模型使用梯度检查点
        if cfg["model"]["grad_checkpointing"]:
            print("grad_checkpointing true")
            self.model.set_grad_checkpointing(enable=True)

    # 前向传播函数
    def forward(self, X):
        outputs = self.model(X)
        return outputs

    # 随机生成边界框函数，用于裁剪图像进行数据增强
    def rand_bbox(self, size, lam):
        # 获取宽度和高度
        W = size[2]
        H = size[3]
        # 计算裁剪比例
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # 在图像中随机选择一个点作为裁剪中心
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # 计算裁剪区域的边界坐标
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 返回裁剪区域的边界坐标
        return bbx1, bby1, bbx2, bby2

    # 进行cutmix数据增强的函数
    def cutmix_data(self, x, y, alpha=1.0):
        # 随机生成一个索引序列
        indices = torch.randperm(x.size(0))
        # 根据索引序列获取打乱的数据和标签
        shuffled_data = x[indices]
        shuffled_target = y[indices]

        # 随机生成混合参数
        lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
        # 随机生成一个裁剪区域
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        # 对数据进行裁剪并混合
        new_data = x.clone()
        new_data[:, :, bby1:bby2, bbx1:bbx2] = x[indices, :, bby1:bby2, bbx1:bbx2]
        # 调整混合参数以匹配像素比例
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        # 返回混合后的数据，原始标签，打乱的标签和混合参数
        return new_data, y, shuffled_target, lam

    # 进行mixup数据增强的函数
    def mixup_data(self, x, y, alpha=1.0, return_index=False):
        # 随机生成混合参数
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        # 获取批量大小
        batch_size = x.size()[0]
        # 随机生成一个索引序列
        index = torch.randperm(batch_size)
        # 根据索引序列和混合参数对数据进行混合
        mixed_x = lam * x + (1 - lam) * x[index, :]
        # 获取混合后的标签
        y_a, y_b = y, y[index]

        # 返回混合后的数据和标签，以及混合参数，如果return_index为True，还返回索引序列
        if return_index:
            return mixed_x, y_a, y_b, lam, index
        else:
            return mixed_x, y_a, y_b, lam


    def mix_criterion(self, pred, y_a, y_b, lam, criterion="default"):
        """
        Introduction: 定义混合损失函数，用于计算两种数据增强方式的损失
        Args: 
            pred: 模型预测的输出
            y_a, y_b: mixup或cutmix混合的两种标签
            lam: mixup或cutmix中的混合比例
            criterion: 损失函数，默认为self.criterion
        Returns: 计算后的混合损失
        """
        # 如果用户没有传入特定的损失函数，就使用默认的损失函数
        if criterion == "default":
            criterion = self.criterion
        # 计算混合损失
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def training_step(self, batch, batch_idx):
        """
        Introduction: 定义训练步骤，根据配置文件中的设置进行不同的训练方式
        Args: 
            batch: 输入的一批数据，包含了输入X和标签y
            batch_idx: 当前批次的索引
        Returns: 计算后的损失
        """
        # 如果配置中指定进行数据增强，并且随机概率大于0.5
        if self.cfg["model"]["aug_mix"] and torch.rand(1) < 0.5:
            X, y = batch
            # 随机选择使用mixup或者cutmix的方式进行数据增强
            if torch.rand(1) >= 0.5:
                mixed_X, y_a, y_b, lam = self.mixup_data(X, y, alpha=self.cfg["model"]["mixup_alpha"])
            else:
                mixed_X, y_a, y_b, lam = self.cutmix_data(X, y, alpha=self.cfg["model"]["cutmix_alpha"])
            pred_y = self.forward(mixed_X)
            # 计算混合损失
            loss = self.mix_criterion(pred_y, y_a, y_b, lam)
        else:
            X, y = batch
            pred_y = self.forward(X)
            # 计算损失
            loss = self.criterion(pred_y, y)
        # 记录训练损失
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def training_epoch_end(self, outputs):
        """
        Introduction: 定义训练周期结束时的操作
        Args: 
            outputs: 训练过程中的所有输出
        Returns: None
        """
        # 从输出中取出所有的损失
        loss_list = [x["loss"] for x in outputs]
        # 计算平均损失
        avg_loss = torch.stack(loss_list).mean()
        # 记录平均训练损失
        self.log("train_avg_loss", avg_loss, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred_y = self.forward(X)
        # 计算损失
        loss = self.criterion(pred_y, y)
        # 对预测结果进行sigmoid变换，并处理NaN值
        pred_y = torch.sigmoid(pred_y)
        pred_y = torch.nan_to_num(pred_y)
        return {"valid_loss": loss, "preds": pred_y, "targets": y}


    def validation_epoch_end(self, outputs):
        """
        Introduction: 定义验证周期结束时的操作
        Args: 
            outputs: 验证过程中的所有输出
        Returns: 平均损失
        """
        # 从输出中取出所有的验证损失
        loss_list = [x["valid_loss"] for x in outputs]
        # 从输出中取出所有的预测结果和真实目标，并进行处理
        preds = torch.cat([x["preds"] for x in outputs], dim=0).cpu().detach().numpy()
        targets = (
            torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach().numpy()
        )
        # 如果配置中指定了二级标签，将目标值小于1的设为0
        if self.cfg["audio"]["second_label"]:
            targets[targets < 1.0] = 0.0
        # 计算平均损失和padded cmap分数
        avg_loss = torch.stack(loss_list).mean()
        padded_cmap_score = padded_cmap(targets, preds)
        # 记录平均验证损失和padded cmap分数
        self.log("valid_avg_loss", avg_loss, prog_bar=True)
        self.log("valid_padded_cmap_score", padded_cmap_score, prog_bar=True)
        return avg_loss


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X, y = batch
        pred_y = self.forward(X)
        # 对预测结果进行sigmoid变换，并处理NaN值
        pred_y = torch.sigmoid(pred_y)
        pred_y = torch.nan_to_num(pred_y)
        return pred_y


    def configure_optimizers(self):
        # 根据配置文件创建优化器
        optimizer = optim.__dict__[self.cfg["model"]["optimizer"]["name"]](
            self.parameters(), **self.cfg["model"]["optimizer"]["params"]
        )
        # 如果没有指定学习率调整器，只返回优化器
        if self.cfg["model"]["scheduler"] is None:
            return [optimizer]
        else:
            # 根据配置文件创建学习率调整器
            if self.cfg["model"]["scheduler"]["name"] == "OneCycleLR":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    total_steps=self.trainer.estimated_stepping_batches,
                    **self.cfg["model"]["scheduler"]["params"],
                )
                scheduler = {"scheduler": scheduler, "interval": "step"}
            else:
                scheduler = optim.lr_scheduler.__dict__[
                    self.cfg["model"]["scheduler"]["name"]
                ](optimizer, **self.cfg["model"]["scheduler"]["params"])
            # 返回优化器和学习率调整器
            return [optimizer], [scheduler]
