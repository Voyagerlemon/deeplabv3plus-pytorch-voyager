'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-14 09:32:42
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-08-22 11:38:53
FilePath: \deeplabv3_plus-voyager\nets\mobilenetv3.py
Description: 构建MobileNetV3网络
'''

import torch
import torch.nn as nn
import torchvision

BatchNorm2d = nn.BatchNorm2d

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        #-------------------------------------------------------#
        # `nn.ReLU6()`默认是False, ReLU6(x) = min(max(0, x), 6)
        #-------------------------------------------------------#
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        #--------------------#
        # `h-swish`激活函数
        #--------------------#
        return x * self.relu6(x + 3) / 6
    
#--------------------------------#
# 带有批归一化和激活函数的卷积层
#--------------------------------#
def ConvBNActivation(in_channels, out_channels, kernel_size, stride, activate):
    return nn.Sequential(
        #-----------------------------------------------------------------------------------#
        # 将`groups`设置为`groups=in_channels`-->如果`in_channels=out_channels`则进行深度卷积
        #-----------------------------------------------------------------------------------#
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding = (kernel_size-1)//2, groups=in_channels),
        BatchNorm2d(out_channels),
        #---------------------------#
        # `activate`-->选择激活函数
        #---------------------------#
        nn.ReLU6(inplace=True) if activate == "relu" else HardSwish()
    )

#----------------------------------#
# 带有批归一化和激活函数的1×1卷积层
#----------------------------------#
def Conv1x1BNActivation(in_channels, out_channels, activate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True) if activate == "relu" else HardSwish()
    )

#---------------------------#
# 带有批归一化的1×1卷积层
#---------------------------#
def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
        BatchNorm2d(out_channels)
    )

#-----------------------------------#
# 定义`squeeze-and-excite(SE)`模块
#-----------------------------------#
class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, se_kernel_size, divide=4):
        super(SqueezeAndExcite, self).__init__()
        #-------------------#
        # 计算中间特征通道数
        #-------------------#
        mid_channels = in_channels // divide
        self.pool = nn.AvgPool2d(kernel_size=se_kernel_size, stride=1)
        self.SEblock = nn.Sequential(
            #--------------------------------------------------------------#
            # 对特征图进行线性变换-->将输入的特征图通道数降低道中间特征通道数
            #--------------------------------------------------------------#
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            HardSwish(inplace=True)
        )
    #---------------#
    # 前向传播函数
    #---------------#
    def forward(self, x):
        b, c, h, w = x.size()
        #--------------------------------#
        # 对卷积出的结果进行平均池化操作
        #--------------------------------#
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x
    
#-------------------------------#
# 定义`MobileNet3 + SE`模块
#-------------------------------#
class SEInvertedBottleneck():
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, activate, use_se, se_kernel_size=1):
        super(SEInvertedBottleneck, self).__init__()
        self.stride = stride
        #-----------------#
        # 是否使用SE模块
        #-----------------#
        self.use_se = use_se
        self.conv = Conv1x1BNActivation(in_channels, mid_channels, activate)
        self.depth_conv = ConvBNActivation(mid_channels, mid_channels, kernel_size, stride, activate)
        
        if self.use_se:
            self.SEblock = SqueezeAndExcite(mid_channels, mid_channels, se_kernel_size)
        
        self.point_conv = Conv1x1BNActivation(mid_channels, out_channels, activate)

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)
    
    def forward(self, x):
        out = self.depth_conv(self.conv(x))
        if self.use_se:
            out = self.SEblock(out)
        out = self.point_conv(out)
        out = (out + self.shortcut(x)) if self.stride ==1 else out
        return out

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, type="large"):
        super(MobileNetV3, self).__init__()
        self.type = type

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(16),
            HardSwish(inplace=True),
        )

        if type == "large":
            self.large_bottleneck = nn.Sequential(
                SEInvertedBottleneck(in_channels=16,  mid_channels=16,  out_channels=16,  kernel_size=3, stride=1, activate='relu',   use_se=False                  ),
                SEInvertedBottleneck(in_channels=16,  mid_channels=64,  out_channels=24,  kernel_size=3, stride=2, activate='relu',   use_se=False                  ),
                SEInvertedBottleneck(in_channels=24,  mid_channels=72,  out_channels=24,  kernel_size=3, stride=1, activate='relu',   use_se=False                  ),
                SEInvertedBottleneck(in_channels=24,  mid_channels=72,  out_channels=40,  kernel_size=5, stride=2, activate='relu',   use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40,  mid_channels=120, out_channels=40,  kernel_size=5, stride=1, activate='relu',   use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40,  mid_channels=120, out_channels=40,  kernel_size=5, stride=1, activate='relu',   use_se=True, se_kernel_size=28),
                SEInvertedBottleneck(in_channels=40,  mid_channels=240, out_channels=80,  kernel_size=3, stride=1, activate='hswish', use_se=False                  ),
                SEInvertedBottleneck(in_channels=80,  mid_channels=200, out_channels=80,  kernel_size=3, stride=1, activate='hswish', use_se=False                  ),
                SEInvertedBottleneck(in_channels=80,  mid_channels=184, out_channels=80,  kernel_size=3, stride=2, activate='hswish', use_se=False                  ),
                SEInvertedBottleneck(in_channels=80,  mid_channels=184, out_channels=80,  kernel_size=3, stride=1, activate='hswish', use_se=False                  ),
                SEInvertedBottleneck(in_channels=80,  mid_channels=480, out_channels=112, kernel_size=3, stride=1, activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=112, kernel_size=3, stride=1, activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=112, mid_channels=672, out_channels=160, kernel_size=5, stride=2, activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=160, mid_channels=960, out_channels=160, kernel_size=5, stride=1, activate='hswish', use_se=True, se_kernel_size=7),
            )

            self.large_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1),
                BatchNorm2d(960),
                HardSwish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=960, out_channels=1280, kernel_size=1, stride=1),
                HardSwish(inplace=True),
            )
        else:
            self.small_bottleneck = nn.Sequential(
                SEInvertedBottleneck(in_channels=16, mid_channels=16, out_channels=16, kernel_size=3, stride=2,activate='relu', use_se=True, se_kernel_size=56),
                SEInvertedBottleneck(in_channels=16, mid_channels=72, out_channels=24, kernel_size=3, stride=2,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=88, out_channels=24, kernel_size=3, stride=1,activate='relu', use_se=False),
                SEInvertedBottleneck(in_channels=24, mid_channels=96, out_channels=40, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=240, out_channels=40, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=40, mid_channels=120, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=48, mid_channels=144, out_channels=48, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=14),
                SEInvertedBottleneck(in_channels=48, mid_channels=288, out_channels=96, kernel_size=5, stride=2,activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
                SEInvertedBottleneck(in_channels=96, mid_channels=576, out_channels=96, kernel_size=5, stride=1,activate='hswish', use_se=True, se_kernel_size=7),
            )
            self.small_last_stage = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=576, kernel_size=1, stride=1),
                BatchNorm2d(576),
                HardSwish(inplace=True),
                nn.AvgPool2d(kernel_size=7, stride=1),
                nn.Conv2d(in_channels=576, out_channels=1280, kernel_size=1, stride=1),
                HardSwish(inplace=True),
            )

        self.classifier = nn.Linear(in_features=1280,out_features=num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        if self.type == "large":
            x = self.large_bottleneck(x)
            x = self.large_last_stage(x)
        else:
            x = self.small_bottleneck(x)
            x = self.small_last_stage(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

def mobilenetv3():
    model = MobileNetV3(num_classes=1000, type="large")
    return model

if __name__ == '__main__':
    model = mobilenetv3()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)