# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 12:47:12 2017

@author: fly
"""

from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional
from PIL import Image
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import os
#import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#from post_deal import post_deal_mine
#from unet import get_unet
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code 
smooth = 1
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection+ smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

img_rows = 256
img_cols = 256

    
def preprocess3(imgs):
    imgs=imgs.reshape(imgs.shape[0],3,imgs.shape[2],imgs.shape[3])
    return imgs

def preprocess1(imgs):
    imgs=imgs.reshape(imgs.shape[0],1,imgs.shape[-2],imgs.shape[-1])
    return imgs    
    
def get_segemation3():
    test_x = np.load('./imgs_train.npy')
    test_y = np.load('./imgs_mask_test.npy')
    test_x=((np.array(test_x))).astype('float32')
    test_y=(np.array(test_y))/255.0
    test_x = preprocess3(test_x)
    test_y = preprocess1(test_y)
    test_y=test_y.astype('float32')

    im =test_x
    print im

    gt=test_y
    scores=[]
    print(im.shape[0])
    print(im.shape)
    print(gt.shape)
#   model = get_unet()
#    model.load_weights('FCN.hdf5')
    model = load_model('test.hdf5',custom_objects={"dice_coef_loss":dice_coef_loss,"dice_coef":dice_coef})
  #  model = Model(input=model.input, output=model.layers[-2].output)
    IM=[]
    GT=[]
    RE=[]
    DF=[]
    for i in range(im.shape[0]):#li3:
        x=im[i]
        y=gt[i]  
        x=x.reshape(1,3,img_rows,img_cols)
        y=y.reshape(1,1,img_rows,img_cols)
        scores1 = model.evaluate(x,y,verbose=2)#[y,y,y,y,y,y]
        print(scores1[-1])
        scores.append(scores1[-1])
        features = model.predict(x)
        dna = features[-1].squeeze()
#        features28.append(dna)
#        sum_dna=np.zeros((1,288,288))
#        for i in range(dna.shape[0]):
#            sum_dna+=dna[i]
#        #print(sum_dna.shape)
#        features1.append(sum_dna)
            
     #   dna = post_deal_mine(dna)
#       IM.append(x.reshape(3,img_rows,img_cols))
#       GT.append(y.reshape(img_rows,img_cols))
        RE.append(dna.reshape(img_rows,img_cols))
#        DF.append(y.reshape(img_rows,img_cols)-dna.reshape(img_rows,img_cols).reshape(img_rows,img_cols))

#            dna1name='IM/im'+str(i)+'.png'
#            plt.imsave(dna1name,x.reshape(288,288),cmap='binary_r')
#            dna1name='IM/re'+str(i)+'.png'
#            plt.imsave(dna1name,dna,cmap='binary')
#            dna1name='IM/gt'+str(i)+'.png'
#            plt.imsave(dna1name,y.reshape(288,288),cmap='binary')
#            dna1nam='IM/DF'+str(i)+'_'+str(scores1[-1])+'.png'
#            plt.imsave(dna1name,y.reshape(288,288)-dna,cmap='binary')

    print('mean:',np.mean(scores))
    print('median:',np.median(scores))
    print('max:',np.max(scores))
    print('min:',np.min(scores))
    path='./'
#    np.save(path+'_result.npy',scores)
    np.save(path+'IM.npy',IM)
#   np.save(path+'GT.npy',GT)
    np.save(path+'RE.npy',RE)
#    np.save(path+'_DF.npy',DF)


    
   # np.save('fcn_result.npy',scores)

def add():
    x1=np.load('../1D/test_x.npy')
    x2=np.load('AugUnet_test_RE.npy')
    print(x1.shape)
    print(x2.shape)
    xx=np.zeros((x1.shape[0],x1.shape[1],x1.shape[2]))
    for i in range(x1.shape[0]):
    #    print((x1[i]+x2[i]).shape)
        xx[i,:,:]=x1[i]+x2[i]
    print(xx.shape)
    np.save('Ntest_x.npy',xx)
def get_add_segemation():
    test_x = np.load('../1D/test_x.npy')
    test_y = np.load('../1D/test_y.npy')
    test_x=((np.array(test_x))/255.0).astype('float32')
    test_y=(np.array(test_y))
    test_x = preprocess1(test_x)
    test_y = preprocess1(test_y)
    test_y=test_y.astype('float32')

    im =test_x
    gt=test_y
    scores=[]
    print(im.shape[0])
    print(im.shape)
    print(gt.shape)
#    model = FCN()
#    model.load_weights('FCN.hdf5')
    model = load_model('AugUnet.hdf5',custom_objects={"dice_coef_loss":dice_coef_loss,"dice_coef":dice_coef})
    IM=[]
    GT=[]
    RE=[]
    DF=[]

    for i in range(im.shape[0]):#li3:
        x=im[i]
        y=gt[i]  
        x=x.reshape(1,1,img_rows,img_cols)
        y=y.reshape(1,1,img_rows,img_cols)
        scores1 = model.evaluate(x,y,verbose=2)#[y,y,y,y,y,y]
        print(scores1[-1])
        scores.append(scores1[-1])
#        if scores1[-1]<0.80 or scores1[-1]==1.0:
        features = model.predict(x)
        dna = features[-1].squeeze()
     #   dna = post_deal_mine(dna)
    #    IM.append(x.reshape(img_rows,img_cols))
   #     GT.append(y.reshape(img_rows,img_cols))
        RE.append(dna.reshape(img_rows,img_cols))
   #     DF.append(y.reshape(img_rows,img_cols)-dna.reshape(img_rows,img_cols).reshape(img_rows,img_cols))

#            dna1name='IM/im'+str(i)+'.png'
#            plt.imsave(dna1name,x.reshape(288,288),cmap='binary_r')
#            dna1name='IM/re'+str(i)+'.png'
#            plt.imsave(dna1name,dna,cmap='binary')
#            dna1name='IM/gt'+str(i)+'.png'
#            plt.imsave(dna1name,y.reshape(288,288),cmap='binary')
#            dna1nam='IM/DF'+str(i)+'_'+str(scores1[-1])+'.png'
#            plt.imsave(dna1name,y.reshape(288,288)-dna,cmap='binary')

    print('mean:',np.mean(scores))
    print('median:',np.median(scores))
    print('max:',np.max(scores))
    print('min:',np.min(scores))
    print(test_x.shape)
    NewTrainx=preprocess1(test_x)+preprocess1(np.array(RE))
    print(NewTrainx.shape)
#    path='Unet_train'
#    #np.save(path+'_result.npy',scores)
#   # np.save(path+'_IM.npy',IM)
#  #  np.save(path+'_GT.npy',GT)
    np.save('../1D/'+'NewTestx.npy',NewTrainx)    
def get_features32():
    test_x = np.load('../1D/train_x.npy')
    test_y = np.load('../1D/train_y.npy')
    test_x=((np.array(test_x))/255.0).astype('float32')
    test_y=(np.array(test_y))
    test_x = preprocess1(test_x)
    test_y = preprocess1(test_y)
    test_y=test_y.astype('float32')

    im =test_x
    gt=test_y
    scores=[]
    print(im.shape[0])
    print(im.shape)
    print(gt.shape)
    model = get_unet()
    model = load_model('Unet.hdf5',custom_objects={"dice_coef_loss":dice_coef_loss,"dice_coef":dice_coef})
    model1 = Model(input=model.input, output=model.layers[-2].output)
    IM=[]
    GT=[]
    RE=[]
    DF=[]
    features28=[]
    for i in range(im.shape[0]):#li3:
        x=im[i]
        y=gt[i]  
        x=x.reshape(1,1,img_rows,img_cols)
        y=y.reshape(1,1,img_rows,img_cols)
        scores1 = model.evaluate(x,y,verbose=2)#[y,y,y,y,y,y]
        print(scores1[-1])
        scores.append(scores1[-1])
        features = model1.predict(x)
        dna = features[-1].squeeze()
        features28.append(dna)
#        sum_dna=np.zeros((1,288,288))
#        for i in range(dna.shape[0]):
#            sum_dna+=dna[i]
#        #print(sum_dna.shape)
#        features1.append(sum_dna)
            
     #   dna = post_deal_mine(dna)
#        IM.append(x.reshape(img_rows,img_cols))
#        GT.append(y.reshape(img_rows,img_cols))
#        RE.append(dna.reshape(img_rows,img_cols))
#        DF.append(y.reshape(img_rows,img_cols)-dna.reshape(img_rows,img_cols).reshape(img_rows,img_cols))

#            dna1name='IM/im'+str(i)+'.png'
#            plt.imsave(dna1name,x.reshape(288,288),cmap='binary_r')
#            dna1name='IM/re'+str(i)+'.png'
#            plt.imsave(dna1name,dna,cmap='binary')
#            dna1name='IM/gt'+str(i)+'.png'
#            plt.imsave(dna1name,y.reshape(288,288),cmap='binary')
#            dna1nam='IM/DF'+str(i)+'_'+str(scores1[-1])+'.png'
#            plt.imsave(dna1name,y.reshape(288,288)-dna,cmap='binary')

    print('mean:',np.mean(scores))
    print('median:',np.median(scores))
    print('max:',np.max(scores))
    print('min:',np.min(scores))
    path='./'
    print(np.array(features28).shape)
    np.save(path+'trainx_features32.npy',np.array(features28))
#    np.save(path+'_IM.npy',IM)
#    np.save(path+'_GT.npy',GT)
#    np.save(path+'_RE.npy',RE)
#    np.save(path+'_DF.npy',DF)
    
def get_features1():
    test_x = np.load('../1D/train_x.npy')
    test_y = np.load('../1D/train_y.npy')
    test_x=((np.array(test_x))/255.0).astype('float32')
    test_y=(np.array(test_y))
    test_x = preprocess1(test_x)
    test_y = preprocess1(test_y)
    test_y=test_y.astype('float32')

    im =test_x
    gt=test_y
    scores=[]
    print(im.shape[0])
    print(im.shape)
    print(gt.shape)
    model = get_unet()
    model = load_model('Unet.hdf5',custom_objects={"dice_coef_loss":dice_coef_loss,"dice_coef":dice_coef})
   # model1 = Model(input=model.input, output=model.layers[-2].output)
    IM=[]
    GT=[]
    RE=[]
    DF=[]
    features28=[]
    for i in range(im.shape[0]):#li3:
        x=im[i]
        y=gt[i]  
        x=x.reshape(1,1,img_rows,img_cols)
        y=y.reshape(1,1,img_rows,img_cols)
        scores1 = model.evaluate(x,y,verbose=2)#[y,y,y,y,y,y]
        print(scores1[-1])
        scores.append(scores1[-1])
        features = model.predict(x)
        dna = features[-1].squeeze()
        features28.append(dna)
#        sum_dna=np.zeros((1,288,288))
#        for i in range(dna.shape[0]):
#            sum_dna+=dna[i]
#        #print(sum_dna.shape)
#        features1.append(sum_dna)
            
     #   dna = post_deal_mine(dna)
#        IM.append(x.reshape(img_rows,img_cols))
#        GT.append(y.reshape(img_rows,img_cols))
#        RE.append(dna.reshape(img_rows,img_cols))
#        DF.append(y.reshape(img_rows,img_cols)-dna.reshape(img_rows,img_cols).reshape(img_rows,img_cols))

#            dna1name='IM/im'+str(i)+'.png'
#            plt.imsave(dna1name,x.reshape(288,288),cmap='binary_r')
#            dna1name='IM/re'+str(i)+'.png'
#            plt.imsave(dna1name,dna,cmap='binary')
#            dna1name='IM/gt'+str(i)+'.png'
#            plt.imsave(dna1name,y.reshape(288,288),cmap='binary')
#            dna1nam='IM/DF'+str(i)+'_'+str(scores1[-1])+'.png'
#            plt.imsave(dna1name,y.reshape(288,288)-dna,cmap='binary')

    print('mean:',np.mean(scores))
    print('median:',np.median(scores))
    print('max:',np.max(scores))
    print('min:',np.min(scores))
    path='./'
    print(np.array(features28).shape)
    np.save(path+'trainx_features1.npy',np.array(features28))
#    np.save(path+'_IM.npy',IM)
#    np.save(path+'_GT.npy',GT)
#    np.save(path+'_RE.npy',RE)
#    np.save(path+'_DF.npy',DF)    
if __name__ == '__main__':
#    get_features()
#    get_features32()
    get_segemation3()
#    path='unet_result/422/'
#    #np.save(path+'_result.npy',scores)
#    scores=np.load(path+'Unet_result.npy')
#    print(len(scores))
#    print('mean:',np.mean(scores))
#    print('median:',np.median(scores))
#    print('max:',np.max(scores))
#    print('min:',np.min(scores))
#    x1=x[1]/np.max(x[1])
#    print x[1]
    
#    get_segemation3()
  #   get_add_segemation()
#    xx=np.load('ULSTM4_result.npy')
#    print(np.median(xx))
