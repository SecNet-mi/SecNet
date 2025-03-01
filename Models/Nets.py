import os
import sys
import torch.nn as nn
import torch
sys.path.append(os.path.dirname(__file__))
from utils import *
from lightning.pytorch import seed_everything

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)
# @ Author: Wei Liang
# @ Email: liangwei.gd@foxmail.com

'''
If these codes help you, we'd appreciate it if you could cite our paper:

Liang, W., Allison, B. Z., Xu, R., He, X., Wang, X., Cichocki, A., & Jin, J. (2025).  
SecNet: A second order neural network for MI-EEG.  
*Information Processing & Management*, 62(3), 104012.

'''
class BaseModel(nn.Module):
    def __init__(self, configs, *args,**kwargs):
        """
        初始化模型
        :param configs: 配置字典，包含模型所需的参数
        :param args: 其他参数
        :param kwargs: 其他关键字参数

        Liang, W., Allison, B. Z., Xu, R., He, X., Wang, X., Cichocki, A., & Jin, J. (2025).  SecNet: A second order neural network for MI-EEG.  *Information Processing & Management*, 62(3), 104012.
        """
        class_num = configs.get('class_num', 4)
        channelNum = configs.get('channelNum', 22)
        width = configs.get('width', 300)
        drop_att = configs.get('drop_att', 0.2)
        p = configs.get('p', 3)
        if not isinstance(class_num, int) or class_num <= 0:
            raise ValueError("class_num must be a positive integer")
        if not isinstance(channelNum, int) or channelNum <= 0:
            raise ValueError("channelNum must be a positive integer")
        if not isinstance(width, int) or width <= 0:
            raise ValueError("input_width must be a positive integer")

        super(BaseModel, self).__init__()
        self.input_width = width
        self.input_channels = 150
        self.channelNum = channelNum

        try:
            # 创建卷积块
            self.block1 = self.create_block(15)
            self.block2 = self.create_block(95)
            self.block3 = self.create_block(55)
            self.fusion=Concat()
            # 创建其他层
            in_size=self.input_channels * 3 if isinstance(self.fusion,Concat) else self.input_channels
            self.Sconv3 = nn.Sequential(PointwiseConv2d(in_size, 100))
            self.attention_module=R_attention(100,p,drop_att)
            self.log_layer1 = LogmLayer()
            self.vec = Vec(100)
            self.FC = nn.Sequential(nn.Linear(5050, class_num))

            # 初始化参数
            self.apply(self.initParms)
        except Exception as e:
            print(f"An error occurred: {e}")


    def create_block(self, kernel_size):
        """
        创建卷积块
        :param kernel_size: 卷积核大小
        :return: 卷积块
        """
        return nn.Sequential(
            Conv2dWithConstraint(1, self.input_width, kernel_size=(self.channelNum, 1), padding=0, bias=False,
                                 groups=1),
            PointwiseConv2d(self.input_width, self.input_channels),
            LayerNormalization(1000),

            nn.Conv2d(self.input_channels, self.input_channels, kernel_size=(1, kernel_size), padding='same',
                      bias=False, groups=self.input_channels),
            LayerNormalization(1000),
        )
    def initParms(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, feature):
        if len(feature.shape)==3:
            feature=feature.unsqueeze(1)

        h1=self.block1(feature)
        h2=self.block2(feature)
        h3=self.block3(feature)
        h=self.fusion([h1,h2,h3])

        h=self.Sconv3(h)
        # add attention
        h = self.attention_module(h)
        feature=self.log_layer1(h)
        h = self.FC(self.vec(feature))
        return h,feature.flatten(1)


class EEGNet(nn.Module):
    def __init__(self,configs,*args,**kwargs):
        super(EEGNet, self).__init__()
        self.F1 = configs.get('F1', 8)
        self.D = configs.get('D', 2)
        self.C1 = configs.get('C1', 64)
        self.F2 = self.D * self.F1
        self.nChan = configs.get('channelNum', 22)
        class_num = configs.get('class_num', 4)

        self.block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding = 'same', bias =False),
                # time_wind(inchannel=8, input_time_dim=1000, win_len=250, stride_len=250),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1), padding = 0, bias = False, max_norm = 1,groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1,4), stride = 4),
                nn.Dropout(p = 0.5))
        self.block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 16),
                                     padding = (0, 16//2) , bias = False,
                                     groups=self.F1* self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1,1),
                          stride =1, bias = False, padding = 0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1,8), stride = 8),
                nn.Dropout(p = 0.5)
                )
        self.fc=nn.Linear(496,class_num)

    def forward(self, input_tensor):
        if len(input_tensor.shape)==3:
            input_tensor=input_tensor.unsqueeze(1)
        out=self.block1(input_tensor)
        out=self.block2(out)
        feature=out.flatten(1)
        out=self.fc(feature)
        return out,feature

class bciiv2a(BaseModel):
    def __init__(self, configs, *args,**kwargs):
        configs['class_num']=4
        configs['channelNum']=22
        super().__init__(configs, *args, **kwargs)

class openbmi(BaseModel):
    def __init__(self, configs, *args,**kwargs):
        configs['class_num']=2
        configs['channelNum']=20
        super().__init__(configs, *args, **kwargs)