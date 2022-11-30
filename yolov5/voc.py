# -*- coding: utf-8 -*-

import os
from sklearn.model_selection import train_test_split
    
path = 'data/Annotations'
files = os.listdir(path)


#以下製作voc.data裡需要的train和valid (擇一)
train, val = train_test_split(files, test_size = 0.2, random_state = 42)

path2 = 'data/ImageSets'

f = open(path2 + '/trainval.txt', 'w')
for fname in files:
    f.write(fname[:-4] + '\n')  #VOC格式
    #f.write(path + '/' + fname + '\n')  #yolo格式
    # fname = fname.replace('.xml', '')
    # f.write(fname + '\n')
f.close()

f = open(path2 + '/train.txt', 'w')
for fname in train:
    f.write(fname[:-4] + '\n')  #VOC格式
    #f.write(path + '/' + fname + '\n')  #yolo格式
    # fname = fname.replace('.xml', '')
    # f.write(fname + '\n')
f.close()

f = open(path2 + '/val.txt', 'w')
for fname in val:
    f.write(fname[:-4] + '\n')  #VOC格式
#     fname = fname.replace('.xml', '')
#     f.write(fname + '\n')  #yolo格式
f.close()


print("Trainval:", len(files))
print("Train:", len(train))
print("val:", len(val))