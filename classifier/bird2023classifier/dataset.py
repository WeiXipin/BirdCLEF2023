import audiomentations
import numpy as np
import torch
import torchaudio
import torchvision
import colorednoise as cn



class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply  # 初始化always_apply属性
        self.p = p  # 初始化p属性

    def __call__(self, y: np.ndarray, sr):
        """
        Args:
            y (np.ndarray): 输入的音频数据
            sr (int): 音频的采样率
        Returns:
            np.ndarray: 转换后的音频数据
        """
        if self.always_apply:
            return self.apply(y, sr=sr)  # 如果always_apply为True，总是应用转换
        else:
            if np.random.rand() < self.p:  # 根据概率p决定是否应用转换
                return self.apply(y, sr=sr)
            else:
                return y  # 不应用转换，返回原音频数据

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError  # 抛出未实现错误


class PinkNoise(AudioTransform):
    # 根据信噪比，音频能量的等级来调整随机波动的一个值，即信噪比小波动小，信噪比大的时候波动大
    # 原因：增加模型的泛化性
    """生成粉红噪声的音频转换类"""
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)  # 初始化父类的属性

        self.min_snr = min_snr  # 初始化最小信噪比，默认为5
        self.max_snr = max_snr  # 初始化最大信噪比，默认为20

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)  # 随机生成一个在最小和最大信噪比之间的信噪比
        a_signal = np.sqrt(y ** 2).max()  # 计算音频信号的最大幅值
        a_noise = a_signal / (10 ** (snr / 20))  # 根据信噪比计算噪声的幅值

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))  # 生成粉红噪声
        a_pink = np.sqrt(pink_noise ** 2).max()  # 计算粉红噪声的最大幅值
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)  # 将粉红噪声添加到音频信号中，并保持数据类型一致
        return augmented  # 返回添加噪声后的音频数据


class Bird2023Dataset(torch.utils.data.Dataset):
    """这是一个用于处理音频数据的数据集类，继承自PyTorch的数据集基类。"""

    def __init__(self, cfg, X, y=None, train=True):
        self.cfg = cfg  # 保存配置字典
        self.train = train  # 保存训练状态标识符
        self.df = X  # 保存输入数据
        # 根据是否在训练状态来确定音频的长度
        self.audio_length = cfg["audio"]["sample_rate"] * self.cfg["audio"]["train_duration"] if train else cfg["audio"]["sample_rate"] * self.cfg["audio"]["valid_duration"] 
        
        # 如果没有提供标签数据，就创建一个形状为(len(X), num_columns_X)的零张量
        if y is None:
            self.y = torch.zeros(
                (len(self.df), len(self.df.columns)), dtype=torch.float32
            )
        # 否则，将标签数据转化为float32类型的张量
        else:
            self.y = torch.tensor(y.values, dtype=torch.float32)

        # 定义音频的背景噪音增强方法
        # 数据增强的python库：audiomentations
        self.augmentation_backgroundnoise = audiomentations.OneOf(
                    [
                        # 增加来自ff1010bird_nocall/nocall路径的背景噪音，信噪比在0到3db之间
                        audiomentations.AddBackgroundNoise(
                            sounds_path=f"{cfg['general']['input_path']}/birdclef2021-background-noise/ff1010bird_nocall/nocall",
                            min_snr_in_db=0,
                            max_snr_in_db=3,
                            p=0.5,
                        ),
                        # 增加来自train_soundscapes/nocall路径的背景噪音，信噪比在0到3db之间
                        audiomentations.AddBackgroundNoise(
                            sounds_path=f"{cfg['general']['input_path']}/birdclef2021-background-noise/train_soundscapes/nocall",
                            min_snr_in_db=0,
                            max_snr_in_db=3,
                            p=0.25,
                        ),
                        # 增加来自aicrowd2020_noise_30sec/noise_30sec路径的背景噪音，信噪比在0到3db之间
                        audiomentations.AddBackgroundNoise(
                            sounds_path=f"{cfg['general']['input_path']}/birdclef2021-background-noise/aicrowd2020_noise_30sec/noise_30sec",
                            min_snr_in_db=0,
                            max_snr_in_db=3,
                            p=0.25,
                        ),
                    ],
                    p=0.5,
                )
        # 定义音频的高斯噪音增强方法
        self.augmentation_gaussiannoise = audiomentations.OneOf(
                    [
                        audiomentations.AddGaussianSNR(p=0.5),  # 添加高斯信噪比
                        audiomentations.AddGaussianNoise(p=0.5)  # 添加高斯噪声
                    ],
                    p=0.5
                )
        # 定义音频的增益增强方法
        self.augmentation_gain = audiomentations.OneOf(
                    [
                        audiomentations.Gain(p=0.5),  # 增加固定增益
                        audiomentations.GainTransition(p=0.5),  # 增加变化的增益
                    ],
                    p=0.5
                )
        # 定义音频的粉红噪声增强方法
        self.augmentation_pinknoise = PinkNoise(p=0.5)

        # 定义音频的归一化变换
        self.normalize_waveform = audiomentations.Normalize(p=1.0)

        # 定义梅尔频谱图的时间拉伸增强方法
        self.aug_timestretch = torchaudio.transforms.TimeStretch()

        # 定义梅尔频谱图的归一化变换
        self.normalize_melspecgram = torchvision.transforms.Normalize(
            mean=self.cfg["model"]["mean"], std=self.cfg["model"]["std"]
        )

    def __len__(self):
        return len(self.df)

    def min_max_0_1(self, x):
        """最大最小归一化"""
        return (x - x.min()) / (x.max() - x.min())
    
    def z_normalize(self, x):
        """标准差归一化"""
        mean = torch.mean(x)
        std = torch.std(x) + 1e-6
        x_normalized = (x - mean) / std
        return x_normalized

    def repeat_waveform(self, audio, target_len):
        """
        Introduction: 如果音频的长度小于目标长度，就重复音频直到长度不小于目标长度。
        
        Args: 
            audio: 需要重复的音频张量
            target_len: 目标长度
            
        Returns: 重复后的音频张量
        """
        # 获取音频的长度
        audio_len = audio.shape[1]
        # 计算需要重复的次数
        repeat_num = (target_len // audio_len) + 1
        # 重复音频
        audio = audio.repeat(1, repeat_num)
        return audio



    def crop_or_pad_waveform_random(self, audio, target_len):
        """
        Introduction: 对音频波形进行随机的裁剪或填充，以使其长度符合目标长度

        Args:
            audio (Tensor): 输入的音频波形
            target_len (int): 目标长度

        Returns:
            audio (Tensor): 裁剪或填充后的音频波形
        """
        # 获取音频波形的长度
        audio_len = audio.shape[1]
        # 如果输入音频的长度小于目标长度，则在音频的随机位置上进行填充
        if audio_len < target_len:
            # 计算输入音频与目标长度之间的差值
            diff_len = target_len - audio_len
            # 随机选择一个位置进行填充
            pad_left = torch.randint(0, diff_len, size=(1,))
            pad_right = diff_len - pad_left
            # 对音频数据进行填充
            audio = torch.nn.functional.pad(
                audio, (pad_left, pad_right), mode="constant", value=0
            )
        # 如果输入音频的长度大于目标长度，则在音频的随机位置上进行裁剪
        elif audio_len > target_len:
            # 随机选择一个位置进行裁剪
            idx = torch.randint(0, audio_len - target_len, size=(1,))
            # 对音频数据进行裁剪
            audio = audio[:, idx : (idx + target_len)]
        # 返回裁剪或填充后的音频数据
        return audio

    def crop_or_pad_waveform_constant(self, audio, target_len):
        """
        Introduction: 对音频波形进行固定的裁剪或填充，以使其长度符合目标长度

        Args:
            audio (Tensor): 输入的音频波形
            target_len (int): 目标长度

        Returns:
            audio (Tensor): 裁剪或填充后的音频波形
        """
        # 获取音频波形的长度
        audio_len = audio.shape[1]
        # 如果输入音频的长度小于目标长度，则对音频的左侧进行填充
        if audio_len < target_len:
            # 计算输入音频与目标长度之间的差值
            diff_len = target_len - audio_len
            # 对音频数据进行填充
            audio = torch.nn.functional.pad(
                audio, (0, diff_len), mode="constant", value=0
            )
        # 如果输入音频的长度大于目标长度，则对音频的开始部分进行裁剪，使其长度等于目标长度
        elif audio_len > target_len:
            # 对音频数据进行裁剪
            audio = audio[:, :target_len]
        # 返回裁剪或填充后的音频数据
        return audio

    def __getitem__(self, index):
        """
        Introduction: 根据索引获取数据集中的元素，包括音频波形和对应的标签

        Args:
            index (int): 数据集中的索引

        Returns:
            mel_specgram (Tensor): 音频的Mel频谱图
            y (int or float): 音频文件的标签
        """
        # 根据索引获取音频文件的标签
        y = self.y[index]
        # 根据文件路径加载音频波形和其采样率
        if self.cfg["job_type"] == "pretrain":
            # 获取给定索引处的音频文件名
            file_path = self.df.loc[index, "filepath"]
            waveform, sample_rate = torchaudio.load(
                self.cfg["general"]["pretrain_input_path"] + "/" + file_path + ".ogg"
            )
        else:
            # 获取给定索引处的音频文件名
            file_path = self.df.loc[index, "filename"]
            waveform, sample_rate = torchaudio.load(
                self.cfg["general"]["input_path"] + "/" + file_path
            )

        # 如果采样率不等于目标采样率，对波形进行重采样
        if sample_rate != self.cfg["audio"]["sample_rate"]:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.cfg["audio"]["sample_rate"]
            )

        # 重复波形
        waveform = self.repeat_waveform(waveform, self.audio_length)
        # 对波形进行裁剪或填充，以达到固定的持续时间
        if self.train and torch.rand(1) >= 0.5:
            waveform = self.crop_or_pad_waveform_random(waveform, self.audio_length)
        else:
            waveform = self.crop_or_pad_waveform_constant(waveform, self.audio_length)
        
        # 如果波形是立体声（有两个通道），则将其转换为单声道
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # 将波形转换为numpy数组
        waveform = waveform.numpy()
        waveform = np.squeeze(waveform)
        if self.train:
            # 对波形进行增强
            waveform = self.augmentation_backgroundnoise(
                samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
            )
            waveform = self.augmentation_gaussiannoise(
                samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
            )
            # 对波形进行增益处理
            #waveform = self.augmentation_gain(samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"])
        # 对波形进行归一化
        waveform = self.normalize_waveform(
            samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
        )
        # 将波形转换为张量
        waveform = waveform[np.newaxis, :]
        waveform = torch.from_numpy(waveform)

        # 计算波形的Mel频谱图
        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg["audio"]["sample_rate"],
            n_fft=self.cfg["mel_specgram"]["n_fft"],
            win_length=self.cfg["mel_specgram"]["win_length"],
            hop_length=self.cfg["mel_specgram"]["hop_length"],
            f_min=self.cfg["mel_specgram"]["f_min"],
            f_max=self.cfg["mel_specgram"]["f_max"],
            n_mels=self.cfg["mel_specgram"]["n_mels"],
        )(waveform)
        # 将Mel频谱图转换为分贝尺度
        mel_specgram = torchaudio.transforms.AmplitudeToDB(
            top_db=self.cfg["mel_specgram"]["top_db"]
        )(mel_specgram)

        # 对Mel频谱图进行增强
        if self.train and torch.rand(1) >= 0.5:
            mel_specgram = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=mel_specgram.shape[1] // 5
            )(mel_specgram)
            mel_specgram = torchaudio.transforms.TimeMasking(
                time_mask_param=mel_specgram.shape[2] // 5
            )(mel_specgram)

        # 对Mel频谱图进行最小-最大归一化
        mel_specgram = self.min_max_0_1(mel_specgram)

        # 将Mel频谱图扩展到3个通道
        mel_specgram = mel_specgram.repeat(self.cfg["model"]["in_chans"], 1, 1)

        # 对Mel频谱图进行z归一化，使得值有零均值和单位方差
        mel_specgram = self.normalize_melspecgram(mel_specgram)
        
        # 将Mel频谱图中的NaN值替换为0
        mel_specgram = torch.nan_to_num(mel_specgram)

        return mel_specgram, y



class Bird2023TestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, ogg_name_list):
        """
        构造函数，用于初始化数据集。

        Args:
            cfg (dict): 配置信息的字典，包括音频和模型相关的配置。
            ogg_name_list (list): 音频文件名称列表。

        Returns:
            None
        """
        self.cfg = cfg  # 存储配置信息的字典
        self.ogg_name_list = ogg_name_list  # 存储音频文件名称列表
        self.audio_length = cfg["audio"]["sample_rate"] * self.cfg["audio"]["test_duration"]  # 计算每段音频的长度（采样率*时长）
        self.step = cfg["audio"]["sample_rate"] * 5  # 步长，用于分割音频

        # 定义波形的正则化变换
        self.normalize_waveform = audiomentations.Normalize(p=1.0)

        # 定义mel spectrogram的正则化变换
        self.normalize_melspecgram = torchvision.transforms.Normalize(
            mean=self.cfg["model"]["mean"], std=self.cfg["model"]["std"]
        )

    def __len__(self):
        return len(self.ogg_name_list)  # 返回音频文件的数量

    def min_max_0_1(self, x):
        """最大最小归一化"""

        return (x - x.min()) / (x.max() - x.min())  # 执行最小-最大规范化

    def audio_to_mel_specgram(self, waveform):
        """
        将音频波形转换为mel spectrogram。

        Args:
            waveform (torch.Tensor): 输入的音频波形。

        Returns:
            mel_specgram (torch.Tensor): 转换后的mel spectrogram。
        """

        # 将张量转换为numpy数组
        waveform = waveform.numpy()
        waveform = np.squeeze(waveform)

        # 对波形进行归一化处理
        waveform = self.normalize_waveform(
            samples=waveform, sample_rate=self.cfg["audio"]["sample_rate"]
        )

        # 将numpy数组转回张量
        waveform = waveform[np.newaxis, :]
        waveform = torch.from_numpy(waveform)

        # 计算波形的mel spectrogram
        mel_specgram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.cfg["audio"]["sample_rate"],
            n_fft=self.cfg["mel_specgram"]["n_fft"],
            win_length=self.cfg["mel_specgram"]["win_length"],
            hop_length=self.cfg["mel_specgram"]["hop_length"],
            f_min=self.cfg["mel_specgram"]["f_min"],
            f_max=self.cfg["mel_specgram"]["f_max"],
            n_mels=self.cfg["mel_specgram"]["n_mels"],
        )(waveform)

        # 将mel spectrogram转换为分贝尺度
        mel_specgram = torchaudio.transforms.AmplitudeToDB(
            top_db=self.cfg["mel_specgram"]["top_db"]
        )(mel_specgram)

        # 将mel spectrogram的值范围缩放到0和1之间
        mel_specgram = self.min_max_0_1(mel_specgram)

        # 如果需要的通道数大于1，那么就复制mel spectrogram来获得更多的通道
        if self.cfg["model"]["in_chans"] > 1:
            mel_specgram = mel_specgram.repeat(self.cfg["model"]["in_chans"], 1, 1)

        # 对mel spectrogram执行Z标准化，使其具有零均值和单位方差
        mel_specgram = self.normalize_melspecgram(mel_specgram)
        mel_specgram = torch.nan_to_num(mel_specgram)

        return mel_specgram

    def __getitem__(self, index):
        """
        Returns:
            mel_specgrams (torch.Tensor): 提取的mel spectrogram。
        """

        # 获取给定索引处的音频文件名
        file_path = self.ogg_name_list[index]

        # 从文件路径加载音频波形和采样率
        waveform, sample_rate = torchaudio.load(
            self.cfg["general"]["input_path"] + "/test_soundscapes/" + file_path
        )

        # 如果采样率与目标采样率不一致，则对波形进行重新采样
        if sample_rate != self.cfg["audio"]["sample_rate"]:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.cfg["audio"]["sample_rate"]
            )(waveform)

        waveforms = []
        row_id = []

        # 将音频波形分割为多个片段
        for i in range(self.audio_length, waveform.shape[1] + self.step, self.step):
            start = max(0, i - self.audio_length)
            end = start + self.audio_length
            waveforms.append(waveform[:, start:end])
            row_id.append(f"{file_path[:-4]}_{end//self.cfg['audio']['sample_rate']}")

        # 如果最后一个片段的长度小于音频长度，那么就删除它
        if waveforms[-1].shape[1] < self.audio_length:
            waveforms = waveforms[:-1]
            row_id = row_id[:-1]

        # 计算所有片段的mel spectrogram
        mel_specgrams = [self.audio_to_mel_specgram(waveform) for waveform in waveforms]
        mel_specgrams = torch.stack(mel_specgrams)

        return mel_specgrams