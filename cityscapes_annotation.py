'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-09 10:30:09
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-06-10 19:14:36
FilePath: \deeplabv3_plus-voyager\cityscapes_annotation.py
Description: 划分Cityscapes数据集-->注意标签路径
'''

import os
import random

#-----------------------------#
# datasets
#     -Annotations
#         -train
#         -val
#         -test
#-----------------------------#

#-------------------------------------------------------#
#   指向Cityscapes数据集所在的文件夹
#   默认指向根目录下的Cityscapes数据集
#-------------------------------------------------------#
cityscapes_path      = 'datasets'
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in Cityscapes.")
    #------------------#
    # 图像标签的路径
    #------------------#
    annotation_path     = os.path.join(cityscapes_path, 'Annotations')
    saveBasePath        = os.path.join(cityscapes_path, "ImageSets", "Segmentation")
    mkdir(annotation_path)
    mkdir(saveBasePath)

    #---------------------------------------------------------------------------------------------#
    cities_train = os.listdir(os.path.join(annotation_path, "train"))
    temp_train_seg = []

    for city in cities_train:
       temp_train_seg += os.listdir(os.path.join(annotation_path, "train/" + city))

    #---------------------#
    # 训练集的颜色分割图
    #---------------------#
    total_train_seg = []
    for seg in temp_train_seg:
        if seg.endswith("color.png"):
            total_train_seg.append(seg)
    
    tr = len(total_train_seg)
    print("train size: ", tr)

    f_train      = open(os.path.join(saveBasePath, "train.txt"), 'w')  
    for i in total_train_seg:
        #---------------#
        # 去掉png后缀
        #---------------#
        name = i[:-16] + '\n'
        f_train.write(name)
    f_train.close()

    #---------------------------------------------------------------------------------------------#
    cities_val = os.listdir(os.path.join(annotation_path, "val"))
    temp_val_seg = []
    for city in cities_val:
        temp_val_seg += os.listdir(os.path.join(annotation_path, "val/" + city))
    total_val_seg = []
    for seg in temp_val_seg:
        if seg.endswith("color.png"):
            total_val_seg.append(seg)
    
    tv = len(total_val_seg)
    print("val size: ", tv)

    f_val        = open(os.path.join(saveBasePath, "val.txt"), 'w')  
    for i in total_val_seg:
        name = i[:-16] + '\n'
        f_val.write(name)
    f_val.close()
   
    #---------------------------------------------------------------------------------------------#
    cities_test = os.listdir(os.path.join(annotation_path, 'test'))
    temp_test_seg = []
    for city in cities_test:
        temp_test_seg += os.listdir(os.path.join(annotation_path, 'test/' + city))
    total_test_seg = [] 
    for seg in temp_test_seg:
        if seg.endswith("color.png"):
            total_test_seg.append(seg)

    ts = len(total_test_seg)
    print("test size: ", ts)
    f_test = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    for i in total_test_seg:
        name = i[:-16] + '\n'
        f_test.write(name)
    f_test.close()
    print("Generate txt in Cityscapes done.")