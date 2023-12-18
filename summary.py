'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-02 10:28:10
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-08-17 16:34:47
FilePath: \deeplabv3_plus-voyager\summary.py
Description: 网络结构
'''
#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    input_shape     = [1024, 2048]
    num_classes     = 19
    backbone        = 'mobilenet'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = DeepLab(num_classes = num_classes, backbone = backbone, downsample_factor = 16, pretrained = False, att = 1).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
