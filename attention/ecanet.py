'''
Author: xuhy 1727317079@qq.com
Date: 2023-08-16 19:07:55
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-08-16 19:19:27
FilePath: \deeplabv3plus-pytorch-voyager\attention\ecanet.py
Description: ECANet
'''
import torch
import math
from torch import nn

class ECALayer(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECALayer, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #------------------#
        # 看作时间序列模型
        #------------------#
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    
if __name__ == "__main__":
    # 测试代码
    model = ECALayer(512)
    print(model)
    inputs = torch.ones([2, 512, 26, 26])
    outputs = model(inputs)