# -*- coding: utf-8 -*-
from __future__ import print_function
import Image
import os
import numpy as np
import cv2
import pandas as pd
import sys
import os
import os.path
import string
import scipy.io
import pdb
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import skimage
import skimage.measure

data_path = './'
save_path = './npy_data/'
image_rows = 256
image_cols = 256

def create_train_data():
    train_data_path = os.path.join(data_path, 'val')
    images = sorted(os.listdir(train_data_path))
    total = len(images)
    print (images)
    
    imgs = np.ndarray((total, image_rows, image_cols,3),dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        # if 'mask' in image_name:
        #     continue
        img = cv2.imread(os.path.join(train_data_path, image_name))
        img = np.array([img])

        imgs[i] = img

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    print(imgs.shape)
    imgs = imgs.transpose((0,3,1,2))
    print (imgs.shape)

    np.save('imgs_test.npy', imgs)
    print('Saving to .npy files done.')


def create_mask_data():
    train_mask_path = os.path.join(data_path, 'val_seg')
    images = sorted(os.listdir(train_mask_path))
    total = len(images)
    imgs = np.ndarray((total,image_rows, image_cols,1), dtype=np.float32)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    # img = cv2.imread(os.path.join(train_mask_path, images[0]),cv2.IMREAD_GRAYSCALE)

    # for row in range(0, 256):
    #     for col in range(0,256):
    #         if (img[row, col] > 130):
    #             img[row,col] = 255
    #         elif(img[row, col] < 120):
    #             img[row, col] = 0
    #         print(img[row, col])
    #
    # img = np.array([img][0])
    # print(img.shape)
    # imgs[0, :, :, 0] = img[:, :]
    # cv2.imshow("imgs1", imgs[0])
    # cv2.waitKey(0)

    for image_name in images:
        img = cv2.imread(os.path.join(train_mask_path, image_name),cv2.IMREAD_GRAYSCALE)
        img = np.array([img][0])

        for row in range(0, 256):
            for col in range(0, 256):
                if (img[row, col] > 130):
                    img[row, col] = 255
                elif (img[row, col] < 120):
                    img[row, col] = 0
        imgs[i, :, :, 0] = img[:, :]
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    imgs = imgs.transpose((0, 3, 1, 2))
    np.save('imgs_mask_test.npy', imgs)
    print('Saving to .npy files done.')


def load_img_from_npy():

    imgs1=np.load("IM.npy")
    imgs2=np.load("RE.npy")
    imgs3=np.load("GT.npy")

    # 显示图片
    r = Image.fromarray(imgs1[11][0]).convert('L')
    g = Image.fromarray(imgs1[11][1]).convert('L')
    b = Image.fromarray(imgs1[11][2]).convert('L')
    image = Image.merge("RGB", (b,g,r))

    # plt.imshow(image)
    # plt.figure('0')
    # plt.imshow(image)
    # plt.show()
    # cv2.imshow("imgs1",imgs1[21].reshape((256,256,3)))

    threshold = 0.5
    for num in range(imgs2.shape[0]):
        for i in range(imgs2.shape[1]):
            for j in range(imgs2.shape[2]):

                if(imgs2[num][i][j] < threshold):
                    imgs2[num][i][j] = 0.0
                else:
                    imgs2[num][i][j] = 1.0

        imgs2[num] *= 255.0
        print(imgs2.shape)
        cv2.imwrite("./preImgs_threshold_0.8/"+str(num)+".png",imgs2[num])
        cv2.imwrite("./preImgs_threshold_0.9/"+str(num)+".png",imgs2[num])
        # np.save("./preImgs/"+str(num)+".png",imgs2[num])

    # cv2.imshow("imgs2",imgs2[11])
    # cv2.imshow("imgs3",imgs3[11])
    # cv2.waitKey(0)
    # cv2.destroyWindow("imgs1")
    # cv2.destroyWindow("imgs2")
    # cv2.destroyWindow("imgs3")

    # np.save("./test.png",imgs2)
    # img = img*255.0
    # print (img[255][190])
    # cv2.imwrite("test1.png",img)

if __name__ == '__main__':
    create_train_data()
    create_mask_data()
# load_img_from_npy()
    


