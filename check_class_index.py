import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from torchcam.utils import overlay_mask
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from torchvision.io.image import read_image
from nets.deeplabv3_plus import DeepLab


input_shape = [1024, 2048]
model_path  = "model_data/deeplabv3p_mobilenetv2_adam89.pth"
model       = DeepLab(num_classes=19, backbone="mobilenet", downsample_factor=8, pretrained=False)

device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model       = model.eval()

# 注册 hook 函数，用于提取特定层的特征图
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((1024, 2048)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取图像
image_path = "img/berlin_000002_000019_leftImg8bit.png"
image      = Image.open(image_path)

input_tensor = preprocess(image)
input_batch  = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)

num_classes = output.size(1)
# 打印类别数量
print("Number of classes:", num_classes)

# 显示模型输出的每个类别的得分
print("Model output scores:")
print(output.squeeze())

labels = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
                   "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                   "bicycle"]

# 打印每个类别对应的索引和名称
for i in range(num_classes):
    print("Index:", i, "Label:", labels[i])