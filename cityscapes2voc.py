'''
Author: xuhy 1727317079@qq.com
Date: 2023-06-03 11:06:06
LastEditors: xuhy 1727317079@qq.com
LastEditTime: 2023-06-03 21:52:36
FilePath: \deeplabv3_plus-voyager\cityscapes2voc.py
Description: Cityscapes to voc
'''

from pascal_voc_writer import Writer
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import glob
import time
from shutil import copy

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def polygon_to_bbox(polygon):
    x_coordinates, y_coordinates = zip(*polygon)
    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]

#--------------------------------------------#
# read a json file and convert to voc format
#--------------------------------------------#
def read_json(file):
    #--------------------------------------------#
    # if no relevant objects found in the image
    # don't save the xml for the image
    #--------------------------------------------#
    relevant_file = False

    data = []
    with open(file, 'r') as f:
        file_data = json.load(f)

        for object in file_data['objects']:
            label, polygon = object['label'], object['polygon']
            #-------------------------------------#
            # process only if label found in voc
            #-------------------------------------#
            if label in classes_keys:
                polygon = np.array([x for x in polygon])
                bbox = polygon_to_bbox(polygon)
                data.append([classes[label]] + bbox)
        #----------------------------------------------------------#
        # if relevant objects found in image, set the flag to True
        #----------------------------------------------------------#
        if data:
            relevant_file = True

    return data, relevant_file


#---------------------------#
# function to save xml file
#---------------------------#
def save_xml(img_path, img_shape, data, save_path):
    writer = Writer(img_path,img_shape[0], img_shape[1])
    for element in data:
        writer.addObject(element[0],element[1],element[2],element[3],element[4])
    writer.save(save_path)


if __name__ == "__main__":

    #-----------------#
    # arguments
    #-----------------#
    cityscapes_dir = 'cityscapes'
    save_path = os.path.join(cityscapes_dir, "cityscapes_voc_format")

    cityscapes_dir_gt = os.path.join(cityscapes_dir, 'gtFine')


    #----------------------------#
    # Valid classes dictionary
    #----------------------------#
    classes = {'road':'road', 'sidewalk':'sidewalk', 'building':'building', 'wall':'wall', 'fence':'fence',
               'pole':'pole', 'traffic light':'traffic light', 'traffic sign':'traffic sign', 'vegetation':'vegetation',
               'terrain':'terrain', 'sky':'sky', 'person':'person', 'rider':'rider', 'car':'car', 'truck':'truck',
               'bus':'bus',  'train':'train', 'motorcycle':'motorcycle', 'bicycle':'bicycle'}
    classes_keys = list(classes.keys())

    #------------------------------------------#
    # reading json files from each subdirectory
    #------------------------------------------#
    valid_files = []
    trainval_files = []
    test_files = []

    # make Annotations target directory if already doesn't exist
    ann_dir = os.path.join(save_path, 'VOC2007', 'Annotations')
    make_dir(ann_dir)

    count=0

    start = time.time()
    for category in os.listdir(cityscapes_dir_gt):
        #------------------------#
        # no GT for test data
        #------------------------#
        if category == 'test': continue

        for city in os.listdir(os.path.join(cityscapes_dir_gt, category)):
            #---------------#
            # read files
            #---------------#
            files = glob.glob(os.path.join(cityscapes_dir, 'gtFine', category, city) + '/*.json')
            #------------------------#
            # process json files
            #------------------------#
            for file in files:
                data, relevant_file = read_json(file)

                if relevant_file:
                    count += 1
                    base_filename = os.path.basename(file)[:-21]
                    xml_filepath = os.path.join(ann_dir, base_filename + '_leftImg8bit.xml')
                    img_name = base_filename + '_leftImg8bit.png'
                    img_path = os.path.join(cityscapes_dir, 'leftImg8bit', category, city,
                                            base_filename + '_leftImg8bit.png')
                    img_shape = plt.imread(img_path).shape
                    valid_files.append([img_path, img_name])
                    #--------------------------------------------------------#
                    # make list of train, val and test files for voc format
                    # lists will be stored in txt files
                    #--------------------------------------------------------#
                    trainval_files.append(img_name[:-4]) if category == 'train' else test_files.append(img_name[:-4])

                    # save xml file
                    save_xml(img_path, img_shape, data, xml_filepath)

    end = time.time() - start
    print('Total Time taken: ', end)
    print('file nums=',count)

    # ----------------------------
    # copy files into target path
    # ----------------------------
    images_savepath = os.path.join(save_path, 'VOC2007', 'JPEGImages')
    make_dir(images_savepath)

    start = time.time()
    for file in valid_files:
        copy(file[0], os.path.join(images_savepath, file[1]))

    # ---------------------------------------------
    # create text files of trainval and test files
    # ---------------------------------------------
    print("len trainval=",len(trainval_files))
    print("len test=", len(test_files))

    textfiles_savepath = os.path.join(save_path, 'VOC2007', 'ImageSets', 'Main')
    make_dir(textfiles_savepath)

    traival_files_wr = [x + '\n' for x in trainval_files]
    test_files_wr = [x + '\n' for x in test_files]

    with open(os.path.join(textfiles_savepath, 'trainval.txt'), 'w') as f:
        f.writelines(traival_files_wr)

    with open(os.path.join(textfiles_savepath, 'test.txt'), 'w') as f:
        f.writelines(test_files_wr)
