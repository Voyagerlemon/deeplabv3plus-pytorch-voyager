'''
Author: xuhy 1727317079@qq.com
Date: 2023-08-15 19:54:51
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-10-22 21:38:44
FilePath: \deeplabv3plus-pytorch-voyager\/attention\senet.py
Description: SENet
'''
#---------------------------------------------------------------------------------------#
# SENet是2017年提出重点是获得输入进来的特征层, 关注每一个通道的权值
# 具体实现方式:
#     1. 对输入进来的特征层进行全局平均池化
#     2. 进行两次全连接, 第一次全连接神经元个数较少, 第二次全连接神经元个数和输入特征层相同
#     3. 完成两次全连接后, 再取一次sigmoid将值固定到0-1之间, 获得输入特征层每一个通道的权值
#     4. 获得这个权值后, 将此权值乘上原输入特征层
#---------------------------------------------------------------------------------------#

import torch
from torch import nn

class SELayer(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(SELayer, self).__init__()
        #-------------------------------#
        # 自适应全局平均池化的高宽设置为1
        #-------------------------------#
        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        # 定义两次全连接
        self.fc         = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
            nn.Sigmoid()
        )

    # 前向传播
    def forward(self, x):
        b, c, h, w = x.size()
        # view()用来重塑tensor的shape
        # b, c, h, w -->b, c, 1, 1
        avg = self.avg_pool2d(x).view([b, c])

        # b, c --> b, c // ratio -->b, c -->b, c, 1, 1
        fc  = self.fc(avg).view([b, c, 1, 1])

        return x * fc
    
if __name__ == "__main__":
    # 测试代码
    model = SELayer(512)
    print(model)
    inputs = torch.ones([2, 512, 26, 26])
    outputs = model(inputs)