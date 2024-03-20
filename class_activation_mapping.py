import cv2
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image

input_shape = [256, 1024]
model_path  = "model_data/deeplabv3p_mobilenetv2_adam89.pth"
model       = DeepLab(num_classes=19, backbone="mobilenet", downsample_factor=8, pretrained=False)

device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model       = model.eval()

def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens,size))
    if titles == False:
        titles="0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def tensor2img(tensor,heatmap=False, shape=(224,224)):
    np_arr=tensor.detach().numpy()[0]
    #对数据进行归一化
    if np_arr.max()>1 or np_arr.min()<0:
        np_arr=np_arr-np_arr.min()
        np_arr=np_arr/np_arr.max()
    np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0]==1:
        np_arr=np.concatenate([np_arr,np_arr,np_arr],axis=0)
    np_arr=np_arr.transpose((1,2,0))
    if heatmap:
        np_arr = cv2.resize(np_arr, shape)
        np_arr = cv2.applyColorMap(np_arr, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    return np_arr/255
 
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())
    print("backward_hook:",grad_in[0].shape,grad_out[0].shape)
 
def farward_hook(module, input, output):
    fmap_block.append(output)
    print("farward_hook:",input[0].shape,output.shape)


# 注册hook
fh=model.cat_conv.register_forward_hook(farward_hook)
bh=model.cat_conv.register_backward_hook(backward_hook)
 
#定义存储特征和梯度的数组
fmap_block = list()
grad_block = list()

# summary (model=model, input_size=(32, 3, 300, 300), col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

image_path   = "img/2233_118.7973379_32.01332392_panorama.png"
image        = Image.open(image_path)
image        = cvtColor(image)
orininal_h  = np.array(image).shape[0]
orininal_w  = np.array(image).shape[1]

image_data, nw, nh  = resize_image(image, (input_shape[1], input_shape[0]))
image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)


images = torch.from_numpy(image_data)
# images = images.to(device)
pred     = model(images)
pr = pred.argmax(axis=-1)
print("pred type:", pr)
#构造label，并进行反向传播
clas=72926
trues=torch.ones((1,), dtype=torch.int64)*clas
target = torch.randint(1, 11, (1, 256, 1024))
ce_loss=nn.CrossEntropyLoss()
loss=ce_loss(pred, target)
loss.backward()

# 卸载hook
fh.remove()
bh.remove()

#取出相应的特征和梯度
layer1_grad=grad_block[-1] #layer1_grad.shape [1, 19, 256, 512]
layer1_fmap=fmap_block[-1]
 
#将梯度与fmap相乘
cam=layer1_grad[0,0].mul(layer1_fmap[0,0])

for i in range(1,layer1_grad.shape[1]):
    cam+=layer1_grad[0,i].mul(layer1_fmap[0,i])
layer1_grad=layer1_grad.sum(1,keepdim=True) #layer1_grad.shape [1, 1, 256, 512]
layer1_fmap=layer1_fmap.sum(1,keepdim=True) #为了统一在tensor2img函数中调用
cam=cam.reshape((1,1,*cam.shape))
 
#进行可视化
img_np=tensor2img(images)
#layer1_fmap=torchvision.transforms.functional.resize(layer1_fmap,[224, 224])
layer1_grad_np=tensor2img(layer1_grad,heatmap=True,shape=(1024,256))
layer1_fmap_np=tensor2img(layer1_fmap,heatmap=True,shape=(1024,256))
cam_np=tensor2img(cam,heatmap=True,shape=(1024,256))
print("颜色越深（红），表示该区域的值越大")
myimshows([img_np,cam_np,cam_np*0.4+img_np*0.6],['image','cam','cam + image'])  
