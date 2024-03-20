import os
from utils.dataloader import DeeplabDataset


input_shape     = [1024, 2048]
num_classes     = 19
cityscapes_path = 'datasets'

with open(os.path.join(cityscapes_path, "ImageSets/Segmentation/test.txt"), "r") as f:
    train_lines = f.readlines()
train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, cityscapes_path)
print(train_dataset[0])
print(train_dataset[1])
print(train_dataset[2])
print(train_dataset[3])
print(train_dataset[4])
print(train_dataset[5])
print(train_dataset[6])
print(train_dataset[7])
print(train_dataset[8])
print(train_dataset[9])
