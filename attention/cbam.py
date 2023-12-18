'''
Author: xuhy 1727317079@qq.com
Date: 2023-08-16 16:55:10
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-08-16 19:04:54
FilePath: \deeplabv3plus-pytorch-voyager\attention\cbam.py
Description: CBAM-->通道注意力机制和空间注意力机制的结合
'''
import torch
from torch import nn

#-----------------#
# 通道注意力机制
#-----------------#
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        
        #-----------------------#
        # 输出特征层的高和宽是1
        #-----------------------#
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)

        #--------------------------------------#
        # 共享全连接层, 利用1×1卷积代替全连接层
        #--------------------------------------#
        self.fc1        = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1      = nn.ReLU(inplace=True)
        self.fc2        = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

#-----------------#
# 空间注意力机制
#-----------------#
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        #------------------------#
        # 卷积核的大小必须是3或7
        #------------------------#
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        #--------------------------------------------#
        # CBAM的空间注意力机制的输入通道数是2, 输出是1
        #--------------------------------------------#
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAMLayer(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAMLayer, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio = ratio)
        self.spatial_attention = SpatialAttention(kernel_size = kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
    
if __name__ == "__main__":
    # 测试代码
    model = CBAMLayer(512)
    print(model)
    inputs = torch.ones([2, 512, 26, 26])
    outputs = model(inputs)