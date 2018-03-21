# -*- coding: utf-8 -*-
"""
Created on Thu Aug  16 13:19:28 2017

@author: fly
"""

from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,Dropout,Deconvolution2D,BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
#from keras.utils.vis_utils import model_to_dot, plot_model
from PIL import Image
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
#from models import model_from_json
#from data import load_train_data, load_test_data
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 256
img_cols = 256

smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((3, img_rows, img_cols))#3*512*512
    conv1_1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)#64*512*512
    conv1_2 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1_1)#64*512*512
 #   B1=BatchNormalization()(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)#32*59*59
    

    conv2_1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)#64*57*57
    conv2_2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2_1)#64*55*55
   # B2=BatchNormalization()(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)#64*54*54
    

    conv3_1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)#128*53*53
    conv3_2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3_1)#128*51*51
   # B3=BatchNormalization()(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)#128*50*50
    

    conv4_1 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4_2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4_1)#128*46*46
   # B4=BatchNormalization()(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)#256*45*45

    conv5_1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5_1)#512*41*41
    #B5=BatchNormalization()(conv5_2)
    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5_2), conv4_2], mode='concat', concat_axis=1)
    conv6_1 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6_2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6_1)
   # B6=BatchNormalization()(conv6_2)


    up7 = merge([UpSampling2D(size=(2, 2))(conv6_2), conv3_2], mode='concat', concat_axis=1)
    conv7_1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7_1)
   # B7=BatchNormalization()(conv7)


    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2_2], mode='concat', concat_axis=1)
    conv8_1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8_1)
   # B8=BatchNormalization()(conv8)


    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1_2], mode='concat', concat_axis=1)
    conv9_1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9_1)

    output = Convolution2D(1, 1, 1, activation='sigmoid',name='out_put0')(conv9)



###summary every map
    model = Model(input=inputs, output=output)
    json_string = model.to_json()
    fh = open("model_cons.pb", "w")
    fh.write(json_string)
    fh.close()
   
   # visualize model
   # plot_model(model, to_file='U_Net.png',show_shapes=True)
    sgd = SGD(lr=0.0001,decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=[dice_coef])#'binary_crossentropy'loss_weights=[0.2,0.2,0.2,0.2,0.2,1.0]

    return model


def preprocess(imgs):
    imgs=imgs.reshape(imgs.shape[0],1,imgs.shape[-2],imgs.shape[-1])
    return imgs

def preprocess3(imgs):
    imgs=imgs.reshape(imgs.shape[0],3,imgs.shape[-2],imgs.shape[-1])
    return imgs

def preprocess1(imgs):
    imgs=imgs.reshape(imgs.shape[0],1,imgs.shape[-2],imgs.shape[-1])
    return imgs


def train_and_predict1():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    test_x = np.load('imgs_test.npy')
    test_y = np.load('imgs_mask_test.npy')
    test_x=(np.array(test_x))
    test_x = test_x.astype('float32')


    test_y=(np.array(test_y))/255.0
    test_x = preprocess3(test_x)
    test_y = preprocess1(test_y)
    test_y=test_y.astype('float32')

    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')

    imgs_train=(np.array(imgs_train))
    imgs_train = imgs_train.astype('float32')

    imgs_mask_train=(np.array(imgs_mask_train))/255.0
    imgs_mask_train=imgs_mask_train.astype('float32')

    imgs_train = preprocess3(imgs_train)
    imgs_mask_train = preprocess1(imgs_mask_train)


    print('trainsamples',imgs_train.shape)
    print('testsamples',test_x.shape)


    print('test_y',test_y.shape)
    print('trainsamples',imgs_mask_train.shape)

    model = get_unet()
#    model.load_weights('AugUnet1.hdf5')
# model_checkpoint = ModelCheckpoint('test.hdf5', monitor='val_loss', save_best_only=True)
    model_checkpoint = ModelCheckpoint('test_model_1.hdf5', monitor='val_loss', save_best_only=True)   
#early_stopping = EarlyStopping(monitor='val_loss', patience=1)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    hist= model.fit(imgs_train,imgs_mask_train, batch_size=8, nb_epoch=2000, verbose=2, shuffle=True,validation_data=[test_x,test_y],callbacks=[model_checkpoint])#[imgs_mask_train,imgs_mask_train,imgs_mask_train,imgs_mask_train,imgs_mask_train,imgs_mask_train]
    print(hist.history)


if __name__ == '__main__':
    train_and_predict1()
#    model = get_unet()
#    model.load_weights('AugUnet1.hdf5')
#    model.save('new_Unet.hdf5')
