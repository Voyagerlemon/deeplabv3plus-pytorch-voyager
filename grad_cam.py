import torch
from torchvision import transforms
from nets.deeplabv3_plus import DeepLab
from torchcam.utils import overlay_mask
from torchvision.io.image import read_image
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import normalize, resize, to_pil_image


input_shape = [256, 1024]
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
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取图像
image_path = "img/test/2233_118.7973379_32.01332392_panorama.png"
image      = read_image(image_path)
input_tensor = normalize(resize(image, (256, 1024)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


model.cls_conv.register_forward_hook(get_activation("cls_conv"))
with torch.no_grad():
    output = model(input_tensor.unsqueeze(0))

num_classes = output.size(1)
# 指定类别索引
class_index = 11  # 这里假设取第11个类别

# 确保类别索引不超出范围
if class_index >= num_classes:
    print("Error: Class index out of bounds.")
    exit()

# 获取目标层的特征图
# featur_map = activation["cls_conv"].squeeze()

with SmoothGradCAMpp(model, target_layer='cls_conv') as cam_extractor:
    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(class_index, out)

result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0][0, :, :].cpu().squeeze(0), mode='F'), alpha=0.5)
result.save("img_out/2233_118.7973379_32.01332392_panorama.png")