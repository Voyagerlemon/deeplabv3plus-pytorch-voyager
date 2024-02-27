'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-02 10:28:10
LastEditors: Voyagerlemon xuhaiyangw@163.com
LastEditTime: 2024-02-26 20:13:32
FilePath: \deeplabv3_plus-voyager\voc_annotation.py
Description: 划分voc数据集
'''
import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   想要增加测试集修改trainval_percent 
#   修改train_percent用于改变验证集的比例 9:1
#   当前该库将测试集当作验证集使用，不单独划分测试集
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC数据集所在的文件夹
#   默认指向根目录下的VOC数据集
#-------------------------------------------------------#
VOC_path      = 'datasets'
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in Cityscapes.")
    annotation_path     = os.path.join(VOC_path, 'Annotations')
    saveBasePath        = os.path.join(VOC_path, "ImageSets", "Segmentation")
    mkdir(saveBasePath)
    temp_seg = os.listdir(annotation_path)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size: ", tv)
    print("train size: ", tr)
    f_trainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    f_test       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    f_train      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    f_val        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            f_trainval.write(name)  
            if i in train:  
                f_train.write(name)  
            else:  
                f_val.write(name)  
        else:  
            f_test.write(name)  
    
    f_trainval.close()  
    f_train.close()  
    f_val.close()  
    f_test.close()
    print("Generate txt in Cityscapes done.")

    print("Check datasets format, this may take a while.")
    classes_nums        = np.zeros([256], np.int32)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(annotation_path, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s, 请查看具体路径下文件是否存在以及后缀是否为png"%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s, 不属于灰度图或者八位彩图, 请仔细检查数据集格式"%(name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图, 标签的每个像素点的值就是这个像素点所属的种类"%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255, 数据格式有误")
        print("二分类问题需要将标签修改为背景的像素点值为0, 目标的像素点值为1")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点, 数据格式有误, 请仔细检查数据集格式")

    print("JPEGImages中的图片应当为.jpg文件, Annotations中的图片应当为.png文件")
    print("如果格式有误, 参考: \n https://github.com/bubbliiiing/segmentation-format-fix")