''' A3M for fine-grained recognition
'''

from __future__ import print_function
import sys
sys.path.append("..")
sys.setrecursionlimit(10000)
import numpy as np
np.random.seed(2208)  # for reproducibility

import time
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, RepeatVector, Permute, merge
from keras.layers import BatchNormalization, Lambda, Bidirectional, GRU
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D
from keras.layers import GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
#from keras.utils.visualize_util import plot
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import scipy.misc
from sklearn import preprocessing
import CUB

# args
net = sys.argv[1]
data_folder = sys.argv[2]

# model config
flag_test = False
batch_size = 10
nb_epoch = 10
dropout = 0.5
final_dim = 512 if net=='VGG16' else 2048
emb_dim = 512
shared_layer_name = 'block5_pool' if net=='VGG16' else 'activation_49'
model_weight_path = './model/weights_resnet50_86.1.h5'
lambdas = [0.2,0.5,1.0]
attr_equal = False
region_equal = False

# dataset config
dataset = 'CUB'
nb_classes = 200
nb_attributes = [10, 16, 16, 16, 5, 16, 7, 16, 12, 16, 16, 15, 4, 16, 16, 16, 16, 6, 6, 15, 5, 5, 5, 16, 16, 16, 16, 5]
img_rows, img_cols = 448, 448
L = 14*14
lr_list = [0.001,0.003,0.001,0.001,0.001,0.001,0.001,0.0001]

def init_classification(input_fea_map, dim_channel, nb_class, name=None):
    # conv
    fea_map = Convolution1D(dim_channel, 1, border_mode='same')(share_fea_map)
    fea_map = BatchNormalization(axis=2)(fea_map)
    fea_map = Activation('relu')(fea_map)
    # pool
    pool = GlobalAveragePooling1D(name=name+'_avg_pool')(fea_map)
    pool = BatchNormalization()(pool)
    pool = Activation('relu')(pool)
    # classification
    prob = Dropout(dropout)(pool)
    prob = Dense(nb_class)(pool)
    prob = Activation(activation='softmax',name=name)(prob)
    return prob,pool,fea_map

# model define
alphas = [lambdas[1]*1.0/len(nb_attributes)]*len(nb_attributes)
loss_dict = {}
weight_dict = {}
# input and output
inputs = Input(shape=(3, img_rows, img_cols))
out_list = []

# shared CNN
model_raw = eval(net)(input_tensor=inputs, include_top=False, weights='imagenet')
share_fea_map = model_raw.get_layer(shared_layer_name).output
share_fea_map = Reshape((final_dim, L), name='reshape_layer')(share_fea_map)        
share_fea_map = Permute((2, 1))(share_fea_map) 

# loss-1: identity classification
id_prob,id_pool,id_fea_map = init_classification(share_fea_map, emb_dim, nb_classes, name='p0')
out_list.append(id_prob)
loss_dict['p0'] = 'categorical_crossentropy'
weight_dict['p0'] = lambdas[0]

# loss-2: attribute classification
attr_fea_list = []
for i in range(len(nb_attributes)):
    name ='attr'+str(i)
    attr_prob,attr_pool,_ = init_classification(share_fea_map, emb_dim, nb_attributes[i], name)
    out_list.append(attr_prob)
    attr_fea_list.append(attr_pool)
    loss_dict[name] = 'categorical_crossentropy'
    weight_dict[name] = alphas[i]

# attention generation
region_score_map_list = []
attr_score_list = []
for i in range(len(nb_attributes)):
    attn1 = merge([id_fea_map,attr_fea_list[i]], mode='dot', dot_axes=(2,1)) 
    fea_score = merge([id_pool,attr_fea_list[i]], mode='dot', dot_axes=(1,1))
    region_score_map_list.append(attn1)
    attr_score_list.append(fea_score)

# regional feature fusion
region_score_map = merge(region_score_map_list, mode='ave', name='attn')
region_score_map = BatchNormalization()(region_score_map)
region_score_map = Activation('sigmoid', name='region_attention')(region_score_map)
region_fea = merge([id_fea_map,region_score_map], mode='dot', dot_axes=(1,1))
region_fea = Lambda(lambda x: x*(1.0/L))(region_fea)
region_fea = BatchNormalization()(region_fea)

# attribute feature fusion
attr_scores = merge(attr_score_list, mode='concat')
attr_scores = BatchNormalization()(attr_scores)
attr_scores = Activation('sigmoid')(attr_scores)
attr_fea = merge(attr_fea_list, mode='concat')
attr_fea = Reshape((emb_dim, len(nb_attributes)))(attr_fea) 
equal_attr_fea = GlobalAveragePooling1D()(attr_fea)
attr_fea = merge([attr_fea,attr_scores], mode='dot', dot_axes=(2,1))
attr_fea = Lambda(lambda x: x*(1.0/len(nb_attributes)))(attr_fea)
attr_fea = BatchNormalization()(attr_fea)

# loss-3: final classification
if(attr_equal):
    attr_fea = equal_attr_fea
if(region_equal):
    region_fea = id_pool
final_fea = merge([attr_fea,region_fea], mode='concat')
final_fea = Activation('relu', name='final_fea')(final_fea)
final_fea = Dropout(dropout)(final_fea)
final_prob = Dense(nb_classes)(final_fea)
final_prob = Activation(activation='softmax',name='p')(final_prob)
out_list.append(final_prob)
loss_dict['p'] = 'categorical_crossentropy'
weight_dict['p'] = lambdas[2]

model = Model(inputs, out_list)
if(flag_test):
    model.load_weights(model_weight_path)

model.summary()
#plot(model, show_shapes=True, to_file='./fig/'+net+'_attention.png')

# the data, shuffled and split between train and test sets
(X_train, y_train),(X_test, y_test),(A_train,A_test,C_A)=eval(dataset).load_data(
    data_folder, target_size=(img_rows, img_cols), bounding_box=True)

print(X_train[100][1][50:60,100:110])
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

# concat Y A
yA_train = np.concatenate((np.expand_dims(y_train,1), A_train), axis=1)
yA_test = np.concatenate((np.expand_dims(y_test,1), A_test), axis=1)
print('yA_train shape:', yA_train.shape)
print('yA_test shape:', yA_test.shape)

# train/test
for lr in lr_list:
    # test
    if(flag_test):
        label_test_list = []
        label_test_list.append(np_utils.to_categorical(y_test, nb_classes))
        for i in range(len(nb_attributes)):
            label_test_list.append(np_utils.to_categorical(A_test[:,i], nb_attributes[i]))
        label_test_list.append(np_utils.to_categorical(y_test, nb_classes))
        scores = model.evaluate(X_test, label_test_list, verbose=0)
        print('\nval-loss: ',scores[:1+len(loss_dict)], '\nval-acc: ', scores[1+len(loss_dict):])
        break
    # train
    if(not flag_test):
        if(lr==0.011):
            for layer in model.layers:
                if(layer.name=='reshape_layer'):
                    break
                layer.trainable=False
        else:
            for layer in model.layers:
                layer.trainable=True
        opt = SGD(lr=lr, decay=5e-4, momentum=0.9, nesterov=True)
        model.compile(loss=loss_dict,
                      loss_weights=weight_dict,
                      optimizer=opt, metrics=['accuracy'])
        # data augment this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            zoom_range=[0.75,1.33],
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # train for nb_epoch epoches
        for e in range(nb_epoch):
            time1 = time.time()
            print('Epoch %d/%d' % (e+1,nb_epoch))
            batches = 1
            ave_loss = np.zeros(1+2*len(loss_dict))
            for X_batch, yA_batch in datagen.flow(X_train, yA_train, batch_size=batch_size):
                y_batch = yA_batch[:,:1]
                attr_batch = yA_batch[:,1:]
                label_batch_list = []
                label_batch_list.append(np_utils.to_categorical(y_batch, nb_classes))
                for i in range(len(nb_attributes)):
                    label_batch_list.append(np_utils.to_categorical(attr_batch[:,i], nb_attributes[i]))
                label_batch_list.append(np_utils.to_categorical(y_batch, nb_classes))
                loss = model.train_on_batch(X_batch, label_batch_list)
                # print
                ave_loss = ave_loss*(batches-1)/batches + np.array(loss)/batches
                show_idx = [0,len(loss_dict)+1,len(loss_dict)+2,2*len(loss_dict)]
                sys.stdout.write('\rtrain-loss: %.4f, train-acc: %.4f %.4f %.4f'
                    % tuple(ave_loss[show_idx].tolist()))
                sys.stdout.flush()
                batches += 1
                if batches > len(X_train)/batch_size:
                    sys.stdout.write("\r  \r\n")
                    break
            # test
            label_test_list = []
            label_test_list.append(np_utils.to_categorical(y_test, nb_classes))
            for i in range(len(nb_attributes)):
                label_test_list.append(np_utils.to_categorical(A_test[:,i], nb_attributes[i]))
            label_test_list.append(np_utils.to_categorical(y_test, nb_classes))
            scores = model.evaluate(X_test, label_test_list, verbose=0)
            print('\nval-loss: ',scores[:1+len(loss_dict)], '\nval-acc: ', scores[1+len(loss_dict):])
            print('Main acc: %f' %(scores[-1]))
        # save model
        model.save_weights('./model/weights_'+net+str(lr)+'.h5')
        print('train stage:',lr,' sgd done!')

