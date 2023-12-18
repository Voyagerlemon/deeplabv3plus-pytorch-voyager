'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-12 09:49:54
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-08-18 10:15:57
FilePath: \deeplabv3_plus-voyager\utils\cropping_image.py
Description: 裁剪Cityscapes数据集-->512*1024
'''
import os
import argparse
from PIL import Image
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../datasets", help="path to Dataset")
    return parser
    
def concat_files(parent_folder):
    #--------------------#
    # 裁剪图像大小和位置
    #--------------------#
    crop_boxes = [
    # 以图像的左下角为锚点
    (0, 0, 1024, 512),     # 左下角子图像
    (0, 512, 1024, 1024),  # 左上角子图像
    (1024, 0, 2048, 512),   # 右下角子图像
    (1024, 512, 2048, 1024) # 右上角子图像
]
    print("开始裁剪")
    #--------------------------------#
    # 遍历父文件夹下的所有子文件夹
    #--------------------------------#
    for category in os.listdir(parent_folder):
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
          new_files_path = os.path.join("../datasets5121024", category)
          progress_bar = tqdm(total=files, desc=f"{category} {len(files)}", postfix=dict, mininterval= 0.3)
          mkdir(new_files_path)
          for file in files:
             file_prefix = os.path.basename(file)[:-16]
             image = Image.open(os.path.join(parent_folder, category, file))
             for i, crop_box in enumerate(crop_boxes):
                 cropped_image = image.crop(crop_box)
                 cropped_image.save(f"{new_files_path}/{file_prefix}_{i+1}_leftImg8bit.png")
                 progress_bar.update(1)
          progress_bar.close()

      elif category == "JPEGImages":
        files_original = os.listdir(os.path.join(parent_folder, category))
        new_original_path = os.path.join("../datasets5121024", category)
        mkdir(new_original_path)

        progress_original_bar = tqdm(total=files, desc=f"{category} {len(files_original)}", postfix=dict, mininterval= 0.3)

        for file_original in files_original:
            
            file_original_prefix = os.path.basename(file_original)[:-16]
            image_original = Image.open(os.path.join(parent_folder, category, file_original))
            for j, crop_box in enumerate(crop_boxes):
                cropped_image = image_original.crop(crop_box)
                cropped_image.save(f"{new_original_path}/{file_original_prefix}_{j+1}_leftImg8bit.jpg")
                progress_original_bar.update(1)
        progress_original_bar.close()      
        
    print(f"-----原始图像和标签图像已经从1024*2048裁剪为四张512*1024, 共有{len(files_original)*4}张-----")
    

def main():
    #---------------------------------------------------------------------------------------------#
    opts = get_argparse().parse_args()
    if opts.data_root == 'datasets':
        concat_files(opts.data_root)
    #---------------------------------------------------------------------------------------------#
    
if __name__ == "__main__":
    main()