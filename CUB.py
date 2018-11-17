# -*- coding: utf-8 -*-
'''
CUB-200-2011 Dataset.
'''
from __future__ import print_function

import numpy as np
import warnings

from PIL import Image
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
import scipy.io as sio
import os
import time

def load_data(data_folder, target_size=(224, 224), bounding_box=True):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    #data_folder = '/home/hankai/data/CUB_200_2011'
    images_file = data_folder+'/images.txt'
    label_file = data_folder+'/image_class_labels.txt'
    attributes_file = data_folder+'/attributes/image_attribute_labels.txt'
    class_attributes_file = data_folder+'/attributes/class_attribute_labels_continuous.txt'
    split_file = data_folder+'/train_test_split.txt'
    bb_file = data_folder+'/bounding_boxes.txt'
    attribute_name_file = data_folder+'/attributes.txt'
    processed_attribute_file = data_folder+'/processed_attributes.txt'
    # train test split
    split_rf = open(split_file,'r')
    train_test_list = []
    train_idx = []
    test_idx = []
    i=0
    for line in split_rf.readlines():
        strs = line.strip().split(' ')
        train_test_list.append(strs[1])
        if(strs[1]=='1'):
            train_idx.append(i)
        else:
            test_idx.append(i)
        i+=1
    split_rf.close()
    # bb
    bb_rf = open(bb_file,'r')
    bb_list = []
    for line in bb_rf.readlines():
        strs = line.strip().split(' ')
        bb_list.append((float(strs[1]),float(strs[2]),float(strs[1])+float(strs[3])
            ,float(strs[2])+float(strs[4])))
    bb_rf.close()
    # images
    i = 0
    images_rf = open(images_file,'r')
    for line in images_rf.readlines():
        strs = line.strip().split(' ')
        img = image.load_img(data_folder+'/images/'+strs[1])
        if(bounding_box):
            img = img.crop(bb_list[int(strs[0])-1])
        img = img.resize(target_size)
        x = image.img_to_array(img)
        if(train_test_list[int(strs[0])-1]=='1'):
            X_train.append(x)
        else:
            X_test.append(x)
        i += 1
        if(i%1000==0):
            print(i,' images load.')
    images_rf.close()
    # label
    label_rf = open(label_file,'r')
    for line in label_rf.readlines():
        strs = line.strip().split(' ')
        if(train_test_list[int(strs[0])-1]=='1'):
            y_train.append(int(strs[1])-1)
        else:
            y_test.append(int(strs[1])-1)
    label_rf.close()   
    # attributes
    A_all = np.genfromtxt(processed_attribute_file, dtype=int, delimiter=' ')
    A_train = A_all[train_idx]
    A_test = A_all[test_idx]
    # class attributes
    C_A = np.zeros((200,312))
    class_attr_rf = open(class_attributes_file,'r')
    i = 0
    for line in class_attr_rf.readlines():
        strs = line.strip().split(' ')
        for j in range(len(strs)):
            C_A[i][j] = 0 if strs[j]=='0.0' else float(1.0/float(strs[j]))
        i+=1
    class_attr_rf.close()

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)
    # theano or tensorflow
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, target_size[0], target_size[1])
        X_test = X_test.reshape(X_test.shape[0], 3, target_size[0], target_size[1])
    else:
        X_train = X_train.reshape(X_train.shape[0], target_size[0], target_size[1], 3)
        X_test = X_test.reshape(X_test.shape[0], target_size[0], target_size[1], 3)  
    return (X_train,y_train), (X_test,y_test), (A_train,A_test,C_A)


if __name__ == '__main__':
    (X_train,y_train), (X_test,y_test) = load_data()



