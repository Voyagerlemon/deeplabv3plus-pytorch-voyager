'''
Author: xuhy 1727317079@qq.com
Date: 2022-05-05 10:34:09
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-06-03 10:52:40
FilePath: \utils\data_process.py
Description: cityscapes数据集处理
'''

import numpy as np
import os
from glob import glob
import cv2


def genarate_dataset(data_dir, convert_dict, target_size, save_dir=None, flags=["train", "val"]):
    for flag in flags:
        save_num = 0
        # 获取图片和标签的路径
        images_paths = glob(data_dir + "leftImg8bit/" +
                            flag + "/*/*__leftImg8bit.png")
        images_paths = sorted(images_paths)
        gts_paths = glob(data_dir + "gtFine/" + flag +
                         "/*/*gtFine_labelIds.png")
        gts_paths = sorted(gts_paths)
        print(len(gts_paths))

        # 遍历每一张照片
        for image_path, gt_path in zip(images_paths, gts_paths):
            # 图片与标签要一一对应
            image_name = os.path.split(image_path)[-1].split('_')[0:3]
            gt_name = os.path.split(gt_path)[-1].split('_')[0:3]
            assert image_name == gt_name

            # 读取图片和标签，并转换标签为0-19类
            image = cv2.imread(image_path)
            gt = cv2.imread(gt_path, 0)
            binary_gt = np.zeros_like(gt)
            # 循环遍历字典的key，并累加value值
            for key in convert_dict.keys():
                index = np.where(gt == key)
                binary_gt[index] = convert_dict[key]

            # 尺寸
            target_height, target_width = target_size

            # resize，参数输入是：宽*高
            resize_image = cv2.resize(
                image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            resize_gt = cv2.resize(
                binary_gt, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

            # 保存路径
            image_save_path = save_dir + flag + \
                "/images/" + str(save_num) + "_resize.png"
            gt_save_path = save_dir + flag + \
                "/gts/" + str(save_num) + "_resize.png"
            # 保存
            cv2.imwrite(image_save_path, resize_image)
            cv2.imwrite(gt_save_path, resize_gt)

            # 每保存一次图片和标签，计数加一
            save_num += 1

            h_arr = [0, 256, 512]
            w_arr = [0, 512, 1024]

            # 遍历长宽起始坐标列表，将原始图片随机裁剪为512
            for h in h_arr:
                for w in w_arr:
                    # 裁剪
                    crop_image = image[h: h + target_height,
                                       w: w + target_width, :]
                    crop_gt = binary_gt[h: h +
                                        target_height, w: w + target_width]

                    # 保存路径
                    image_save_path = save_dir + flag + \
                        "/images_crop/" + str(save_num) + "_crop.png"
                    gt_save_path = save_dir + flag + \
                        "/gts_crop/" + str(save_num) + "_crop.png"

                    # 保存
                    cv2.imwrite(image_save_path, crop_image)
                    cv2.imwrite(gt_save_path, crop_gt)
                    # 每保存一次图片和标签，计数加一
                    save_num += 1


# 将trainId里面的255定为第0类，原本的0-18类向顺序后加1
pixLables = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0, 10: 0, 11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6,
             18: 0, 19: 7, 20: 8, 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16, 29: 0, 30: 0, 31: 17, 32: 18, 33: 19, -1: 0}

genarate_dataset(data_dir='./cityspaces', convert_dict=pixLables, target_size=(512, 1024), save_dir="./cityspacesDataset")
