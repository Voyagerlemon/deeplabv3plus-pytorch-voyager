'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-14 09:32:42
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-07-09 17:25:28
FilePath: \deeplabv3_plus-voyager\nets\mobilenetv3.py
Description: 构建MobileNetV3网络
'''

import torch.nn as nn
import math
import torch

__all__ = ['build_mobilenetv3_small', 'build_mobilenetv3_large']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    #--------------------------------------#
    # 确保调整后的值不会比原始值下降超过10%
    #--------------------------------------#
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Sigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu6(x + 3) / 6


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.sigmoid = Sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SEInvertedBottleneck(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEInvertedBottleneck, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            # nn.ReLU(inplace=True),
            nn.ReLU6(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            # Sigmoid()
            HardSwish(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.SEblock(y).view(b, c, 1, 1)
        return x * y

#--------------------------------#
# 带有批归一化和激活函数的卷积层
#--------------------------------#
def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        HardSwish()
    )

#----------------------------------#
# 带有批归一化和激活函数的1×1卷积层
#----------------------------------#
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        HardSwish()
    )

#-------------------------#
# MobileNetV3的倒残差网络
#-------------------------#
class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                #----------------------------------------------#
                # 进行3x3的逐层卷积, 进行跨特征点的特征提取(dw)
                #----------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU6(inplace=True),

                #---------------#
                # 使用SE模块
                #---------------# 
                SEInvertedBottleneck(hidden_dim) if use_se else nn.Identity(),

                #---------------------------------------#
                # 利用1x1卷积进行通道数的调整(pw-linear)
                #---------------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                #---------------------------------#
                # 利用1x1卷积进行通道数的调整(pw)
                #---------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True),

                #----------------------------------------------#
                # 进行3x3的逐层卷积, 进行跨特征点的特征提取(dw)
                #----------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),

                #---------------#
                # 使用SE模块
                #---------------# 
                SEInvertedBottleneck(hidden_dim) if use_se else nn.Identity(),
                HardSwish() if use_hs else nn.ReLU(inplace=True),

                #----------------------------------------#
                # 利用1x1卷积进行通道数的调整(pw-linear)
                #----------------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(MobileNetV3, self).__init__()

        #------------------#
        # 设置倒残差网络块
        #------------------#
        self.cfgs = cfgs
        assert mode in ['large', 'small']

        #--------------#
        # 构建第一层
        #--------------#
        input_channel = _make_divisible(16 * width_mult, 8)

        #------------------------------#
        # 对特征层进行高和宽的压缩
        # 1024×2048×3-->512×1024×3
        #------------------------------#
        self.features = [conv_3x3_bn(3, input_channel, 2)]

        #------------------#
        # 构建倒残差网络块
        #------------------#
        block = InvertedResidual

        exp_size = None
        output_channel = None

        for i in range(4):
            for k, t, c, use_se, use_hs, s in self.cfgs[i]:
                output_channel = _make_divisible(c * width_mult, 8)
                exp_size = _make_divisible(input_channel * t, 8)
                self.features.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
                input_channel = output_channel

        #------------------#
        # 构建最后几层网络
        #------------------#
        self.features = nn.Sequential(*self.features)
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
            mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            HardSwish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        #--------------------------------------#
        # downsample: [2  4  8  16  32]
        #--------------------------------------# 
        x = self.features(x) 
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv3_large(**kwargs):
    cfgs = [
        # k, t, c, SE, HS, s
        [[3, 1, 16, 0, 0, 1],
         [3, 4, 24, 0, 0, 2],
         [3, 3, 24, 0, 0, 1]],
        [[5, 3, 40, 1, 0, 2],
         [5, 3, 40, 1, 0, 1],
         [5, 3, 40, 1, 0, 1]],
        [[3, 6, 80, 0, 1, 1],
         [3, 2.5, 80, 0, 1, 1],
         [3, 2.3, 80, 0, 1, 1],
         [3, 2.3, 80, 0, 1, 1],
         [3, 6, 112, 1, 1, 1],
         [3, 6, 112, 1, 1, 1]],
        [[5, 6, 160, 1, 1, 1],
         [5, 6, 160, 1, 1, 1],
         [5, 6, 160, 1, 1, 1]]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    cfgs = [
        # k, t, c, SE, HS, s
        [[3, 1, 16, 1, 0, 2]],
        [[3, 4.5, 24, 0, 0, 2],
         [3, 3.67, 24, 0, 0, 1]],
        [[5, 4, 40, 1, 1, 1],
         [5, 6, 40, 1, 1, 1],
         [5, 6, 40, 1, 1, 1],
         [5, 3, 48, 1, 1, 1],
         [5, 3, 48, 1, 1, 1]],
        [[5, 6, 96, 1, 1, 1],
         [5, 6, 96, 1, 1, 1],
         [5, 6, 96, 1, 1, 1]],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)


def load_and_convert(net, state_dict):
    net_dict = net.state_dict().copy()
    net_list = list(net_dict.keys())
    trained_list = list(state_dict.keys())
    assert len(net_list) == len(trained_list), 'Learning parameters do not match, check net and trained state_dict'
    for i in range(len(net_list)):
        net_dict[net_list[i]] = state_dict[trained_list[i]]
    net.load_state_dict(net_dict)


def build_mobilenetv3_large(pretrained=True, width_mult=1.):
    net = mobilenetv3_large(width_mult=width_mult)
    if pretrained:
        eps = 1e-5
        if abs(1.0 - width_mult) < eps:
            weights = '../model_pretrain/mobilenetv3-large-1cd25616.pth'
            state_dict = torch.load(weights)
        elif abs(0.75 - width_mult) < eps:
            weights = '../model_pretrain/mobilenetv3-large-0.75-9632d2a8.pth'
            state_dict = torch.load(weights)
        else:
            raise RuntimeError("Not support width_mult: {}".format(width_mult))
        load_and_convert(net, state_dict)
    return net


def build_mobilenetv3_small(pretrained=True, width_mult=1.):
    net = mobilenetv3_small(width_mult=width_mult)
    if pretrained:
        eps = 1e-5
        if abs(1.0 - width_mult) < eps:
            weights = '../model_pretrain/mobilenetv3-small-55df8e1f.pth'
            state_dict = torch.load(weights)
        elif abs(0.75 - width_mult) < eps:
            weights = '../model_pretrain/mobilenetv3-small-0.75-86c972c3.pth'
            state_dict = torch.load(weights)
        else:
            raise RuntimeError("Not support width_mult: {}".format(width_mult))
        load_and_convert(net, state_dict)
    return net


if __name__ == '__main__':

    def params(net):
        return sum(param.numel() for param in net.parameters())

    net = build_mobilenetv3_large(pretrained=False, width_mult=1.)
    input = torch.randn((1, 3, 224, 224))
    out = net(input)
    print('Out shape ', out.size())