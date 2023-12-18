'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-03 20:49:13
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-07-22 16:23:10
FilePath: \deeplabv3_plus-voyager\cityscapes_move_gtFine.py
Description: 筛选标签数据
'''
import os
import shutil
import argparse
from glob import glob
from PIL import Image

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="cityscapes/gtFine", help="path to Dataset")
    return parser

def png2jpg(png_path, jpg_path, quality=95):
    try: 
        png_image = Image.open(png_path)
        png_image = png_image.convert("RGB")
        png_image.save(jpg_path, format="JPEG", quality=quality)
    except Exception as e:
        print(f"转换失败: {e}")
        
def concat_files(parent_folder):
    #--------------------------------#
    # 遍历父文件夹下的所有子文件夹
    #--------------------------------#
    for category in os.listdir(parent_folder):
      #------------------------#
      # 不获取test数据集
      #------------------------#
      if category == 'test': continue
      for city in os.listdir(os.path.join(parent_folder, category)):
          #----------------------#
          # 遍历不同城市里的文件
          #----------------------#
          files = glob(os.path.join(parent_folder, category, city) + "/*gtFine_labelTrainIds.png")
          #---------------------#
          # 复制文件到目标文件夹
          #---------------------#
          new_files_path = "datasets/Annotations"
          mkdir(new_files_path)
          for file in files:
             shutil.copy2(file, new_files_path) 
    print("-----标签png图片筛选完成-----")
    
    #---------------------#
    # 更改标签的文件名
    #---------------------#
    annotation_files = os.listdir(new_files_path)
    for file in annotation_files:
        base_filename = os.path.basename(file)[:-25]
        new_file_name = base_filename + "_leftImg8bit.png"
        old_path = os.path.join(new_files_path, file)
        new_path = os.path.join(new_files_path, new_file_name)

        os.rename(old_path, new_path)
        print(f"rename: {file} -> {new_file_name}")
    print(f"total files: {len(annotation_files)}") 
def main():
    #---------------------------------------------------------------------------------------------#
    opts = get_argparse().parse_args()
    if opts.data_root == "cityscapes/gtFine":
        concat_files(opts.data_root)

    #---------------------------------------------------------------------------------------------#
    elif opts.data_root == "cityscapes":
        parent_folder = opts.data_root + "/gtFine"
        for category in os.listdir(parent_folder):
            if category == "test": continue
            for city in os.listdir(os.path.join(parent_folder, category)):
                files = glob(os.path.join(parent_folder, category, city) + "/*.json")
                new_files_path = "datasets/JSONfiles"
                mkdir(new_files_path)
                for file in files:
                    shutil.copy2(file, new_files_path)
        print("-----标签json文件筛选完成-----")
    
    #---------------------------------------------------------------------------------------------#
    elif opts.data_root == "cityscapes/gtCoarse":
        print("-----开始筛选gtCoarse图片-----")
        #--------------------------------#
        # 遍历父文件夹下的所有子文件夹
        #--------------------------------#
        for category in os.listdir(opts.data_root):
            for city in os.listdir(os.path.join(opts.data_root, category)):
                files = glob(os.path.join(opts.data_root, category, city) + "/*gtCoarse_labelTrainIds.png")
                new_files_path = "datasetsCoarse/Annotations"
                mkdir(new_files_path)
                progress = 0
                print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
                for file in files:
                    shutil.copy2(file, new_files_path)
                    progress += 1 
                    print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        print("-----标签png图片筛选完成-----")

        annotation_files = os.listdir(new_files_path)
        for file in annotation_files:
            base_file_name = os.path.basename(file)[:-27]
            new_file_name = base_file_name + "_leftImg8bit.png"
            old_path = os.path.join(new_files_path, file)
            new_path = os.path.join(new_files_path, new_file_name)

            os.rename(old_path, new_path)
        print(f"total files: {len(annotation_files)}")
        print("Done!")

        print("-----开始筛选leftImg8bitCoarse图片-----")
        leftImg8bit_coarse = "cityscapes/leftImg8bitCoarse"
        progress = 0
        print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        for leftImg in os.listdir(leftImg8bit_coarse):
            for sub_leftImg in os.listdir(os.path.join(leftImg8bit_coarse, leftImg)):
                for train_extra in os.listdir(os.path.join(leftImg8bit_coarse, leftImg, sub_leftImg)):
                    files = glob(os.path.join(leftImg8bit_coarse, leftImg, sub_leftImg, train_extra))
                    new_leftImg_path = "datasetsCoarse/JPEGImages"
                    mkdir(new_leftImg_path)
                    
                    for file in files:
                        file_name = os.path.basename(file)
                        jpeg_path = os.path.join(new_leftImg_path, file_name)
                        png2jpg(file, jpeg_path)
        progress += 1 
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        print("Done!")
        
    #---------------------------------------------------------------------------------------------#
    elif opts.data_root == "datasets":
        cities_train = os.listdir(os.path.join(opts.data_root, "JPEGImages/train"))
        cities_val = os.listdir(os.path.join(opts.data_root, "JPEGImages/val"))
        temp_train_seg = []
        temp_val_seg = []
        saveBasePath        = os.path.join(opts.data_root, "ImageSets", "Segmentation")
        for city_train in cities_train:
            temp_train_seg += os.listdir(os.path.join(opts.data_root, "JPEGImages/train/" + city_train))
        for city_val in cities_val:
            temp_val_seg += os.listdir(os.path.join(opts.data_root, "JPEGImages/val/" + city_val))
       
        #---------------------#
        # 训练集的颜色分割图
        #---------------------#
        total_train_seg = []
        total_val_seg =[]
        for seg_train in temp_train_seg:
            if seg_train.endswith(".png"):
                total_train_seg.append(seg_train)
        for seg_val in temp_val_seg:
            if seg_val.endswith(".png"):
                total_val_seg.append(seg_val)

        tr = len(total_train_seg)
        tv = len(total_val_seg)
        print("train size: ", tr)
        print("val size: ", tv)

        f_train      = open(os.path.join(saveBasePath, "train.txt"), 'w')  
        f_val        = open(os.path.join(saveBasePath, "val.txt"), 'w')
        for i in total_train_seg:
            #---------------#
            # 去掉png后缀
            #---------------#
            name_train = i[:-4] + '\n'
            f_train.write(name_train)
        
        for j in total_val_seg:
            name_val = j[:-4] + '\n'
            f_val.write(name_val)
        f_train.close()
        f_val.close()
    #---------------------------------------------------------------------------------------------#
    else: print("-----参数输入有误-----")

if __name__ == "__main__":
    main()