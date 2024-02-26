import math
import os

import torch
import torch.nn as nn
from attention.ca import CALayer
import torch.utils.model_zoo as model_zoo

BatchNorm2d = nn.BatchNorm2d

#-----------------------------------------#
# 对输入进来的特征层进行卷积标准化+激活函数
#-----------------------------------------#
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


#-------------------------#
# MobileNetV2的倒残差网络
#-------------------------#
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        #--------------------------#
        # 不对输入的特征层进行升维
        #--------------------------#
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #----------------------------------------------#
                # 进行3x3的逐层卷积, 进行跨特征点的特征提取(dw)
                #----------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #---------------------------------#
                # 利用1x1卷积进行通道数的调整(pw)
                #---------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

        #--------------------------#
        # 对输入的特征层进行升维
        #--------------------------#
        else:
            self.conv = nn.Sequential(
                #---------------------------------#
                # 利用1x1卷积进行通道数的上升(pw)
                #---------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                #----------------------------------------------#
                # 进行3x3的逐层卷积, 进行跨特征点的特征提取(dw)
                #----------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # CA
                CALayer(hidden_dim),
                #---------------------------------------------#
                # 利用1x1卷积进行通道数的下降(具备更低的运算量)
                #---------------------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm2d(oup),
            )

    def forward(self, x):
        #--------------------#
        # 使用残差边
        #--------------------#
        if self.use_res_connect:
            #------------------------------#
            # 将输入和卷积特征的结果进行相加
            #------------------------------#
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            #-------------------------------------------------#
            # t --> 是否要进行1×1卷积通道数上升(expand_ratio)
            # c --> 输出通道数的大小(output_channel)
            # n --> 进行Inverted residual block的次数
            # s --> 输入的特征层是否要进行高和宽的压缩, 代表步长
            #-------------------------------------------------#
            [1, 16, 1, 1],  # 256, 256, 32 --> 256, 256, 16 不进行卷积通道数的上升
            [6, 24, 2, 2],  # 256, 256, 16 --> 128, 128, 24   2
            [6, 32, 3, 2],  # 128, 128, 24 --> 64, 64, 32     4
            [6, 64, 4, 2],  # 64, 64, 32 --> 32, 32, 64       7
            [6, 96, 3, 1],  # 32, 32, 64 --> 32, 32, 96
            [6, 160, 3, 2], # 32, 32, 96 --> 16, 16, 160     14
            [6, 320, 1, 1], # 16, 16, 160 --> 16, 16, 320
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        #------------------------------#
        # 对特征层进行高和宽的压缩
        # 1024×2048×3-->512×1024×3
        #------------------------------#
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        #-------------------------#
        # 利用1×1卷积调整通道数
        #-------------------------#
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            # 官方的nn.Dropout(0.1)
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)

def mobilenetv2(pretrained=False, **kwargs):
    model = MobileNetV2(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'), strict=False)
    return model

if __name__ == "__main__":
    model = mobilenetv2()
    for i, layer in enumerate(model.features):
        print(i, layer)
