'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-13 20:40:02
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-06-13 21:49:35
FilePath: \deeplabv3_plus-voyager\cropping_cityscapes.py
Description: Cityscapes转换为512*1024
'''

import os
import argparse
import cv2
from tqdm import tqdm
from PIL import Image

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="datasets", help="path to Dataset")
    return parser
    
def concat_files(parent_folder):
    #--------------------#
    # 裁剪图像大小和位置
    #--------------------#
    print("开始resize")
    #--------------------------------#
    # 遍历父文件夹下的所有子文件夹
    #--------------------------------#
    for category in tqdm(os.listdir(parent_folder)):
      #------------------------#
      # 不获取txt标签数据集
      #------------------------#
      if category == "ImageSets": continue
      elif category == "Annotations":
          #----------------------#
          # 遍历原图像和标签图像
          #----------------------#
          files = os.listdir(os.path.join(parent_folder, category))
          #---------------------#
          # 移动文件到目标文件夹
          #---------------------#
          new_files_path = os.path.join("datasets5121024", category)
          mkdir(new_files_path)

          for file in tqdm(files):
             file_prefix = os.path.basename(file)[:-16]
             image = cv2.imread(os.path.join(parent_folder, category, file), cv2.IMREAD_UNCHANGED)
             resized_image = cv2.resize(image, (1024, 512))
             cv2.imwrite(f"{new_files_path}/{file_prefix}_leftImg8bit.png", resized_image)

      elif category == "JPEGImages":
        files_original = os.listdir(os.path.join(parent_folder, category))
        new_original_path = os.path.join("datasets5121024", category)
        mkdir(new_original_path)

        for file_original in tqdm(files_original):
            
            file_original_prefix = os.path.basename(file_original)[:-16]
            image_original = Image.open(os.path.join(parent_folder, category, file_original))
            cropped_original_image = image_original.resize((1024, 512))
            cropped_original_image.save(f"{new_original_path}/{file_original_prefix}_leftImg8bit.jpg") 
      else: print("参数有误, 请重新输入")
    print(f"-----原始图像和标签图像已经从1024*2048resize为512*1024, 共有{len(files_original)}张-----")
    

def main():
    #---------------------------------------------------------------------------------------------#
    opts = get_argparse().parse_args()
    if opts.data_root == 'datasets':
        concat_files(opts.data_root)
    #---------------------------------------------------------------------------------------------#
    
if __name__ == "__main__":
    main()

