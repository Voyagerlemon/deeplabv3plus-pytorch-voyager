'''
Author: xuhy 1727317079@qq.com
Date: 2023-08-16 16:22:26
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-08-16 16:43:49
FilePath: \deeplabv3plus-pytorch-voyager\nets\SEResNet.py
Description: SEResNet
'''

import torch
import torchvision
from torch import nn
from torchsummary import summary
from attention.senet import SELayer
from torch.nn import functional as F

#-----------------------------#
# 实现SEResNet18与SEResNet34
#-----------------------------#
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        #---------------------#
        # 使用SENet注意力机制
        #---------------------#
        self.se = SELayer(out_channel, 16)
        self.right = shortcut
 
    def forward(self, x):
        out = self.left(x)
        out= self.se(out)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)
    
class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.model_name = 'resnet34'
        
        #------------------#
        # 前几层: 图像转换
        #------------------#
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
 
        self.layer1 = self._make_layer(64, 64, blocks[0])
        self.layer2 = self._make_layer(64, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, blocks[3], stride=2)
 
        self.fc = nn.Linear(512, num_classes)
 
    def _make_layer(self, in_channel, out_channel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
 
        layers = []
        layers.append(ResidualBlock(in_channel, out_channel, stride, shortcut))
 
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channel, out_channel))
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.pre(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
 
def Se_ResNet18():
    return ResNet([2, 2, 2, 2])
 
def Se_ResNet34():
    return ResNet([3, 4, 6, 3])

#-----------------------------------------------------------------------------------------------------------------------------#

#------------------------------------------#
# 实现SEResNet50、SEResNet101与SEResNet152
#------------------------------------------#
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, down_sampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.down_sampling = down_sampling
 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )
        #---------------------#
        # 使用SENet注意力机制
        #---------------------#
        self.se = SELayer(places * self.expansion, 16)
        if self.down_sampling:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out = self.se(out)
        if self.down_sampling:
            residual = self.down_sample(x)
 
        out += residual
        out = self.relu(out)
        return out


class ResNetPro(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion
 
        self.conv1 = Conv1(in_planes=3, places=64)
 
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)
 
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, down_sampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * self.expansion, places))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def SEResNet50():
    return ResNetPro([3, 4, 6, 3])
 
def SEResNet101():
    return ResNetPro([3, 4, 23, 3])
 
def SEResNet152():
    return ResNetPro([3, 8, 36, 3])

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SEResNet101()
    model.to(device)
    summary(model, (3, 224, 224))