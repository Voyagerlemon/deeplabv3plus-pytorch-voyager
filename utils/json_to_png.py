'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-05 12:56:36
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-08-16 09:59:24
FilePath: \deeplabv3_plus-voyager\json_to_png.py
Description: 将json标签转换为png图片
'''
import base64
import json
import os

import numpy as np
from labelme import utils

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    png_path   = "../datasets/SegmentationClass"
    mkdir(png_path)
    classes     = ["_background_","road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", 
                   "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", 
                   "bicycle", "license plate"]
    
    count = os.listdir("../datasets/JSONfiles") 
    for i in range(0, len(count)):
        prefix_json = count[0][:-21]
        path = os.path.join("../datasets/JSONfiles", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
            
            JPEGImages_path = "../datasets/JPEGImages"
            jpeg_image_path = prefix_json + "_leftImg8bit.jpg"

            imagePath = os.path.join(JPEGImages_path, jpeg_image_path)
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')

            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for object in data['objects']:
                label_name = object['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))
            
            lbl = utils.shapes_to_label(img.shape, data['objects'], label_name_to_value)
            
                

            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all*(np.array(lbl) == index_json)

            utils.lblsave(os.path.join(png_path, count[i].split(".")[0]+'.png'), new)
            print('Saved '+ count[i].split(".")[0] + '.png')
