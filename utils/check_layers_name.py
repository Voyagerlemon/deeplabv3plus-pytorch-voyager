import torch
from nets.deeplabv3_plus import DeepLab

input_shape     = [1024, 2048]
num_classes     = 19
backbone        = 'mobilenet'
    
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model   = DeepLab(num_classes = num_classes, backbone = backbone, downsample_factor = 16, pretrained = False, att = 1).to(device)
for name in model.named_modules():
    print(name)