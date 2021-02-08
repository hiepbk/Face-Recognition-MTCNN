import tensorflow as tf
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
import numpy as np
import random
import os
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from imutils import paths
from matplotlib import pyplot as plt
import cv2
import time
import pickle
fold_elip_txt = ['2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-01-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-02-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-03-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-04-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-05-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-06-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-07-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-08-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-09-ellipseList.txt',
                 '2.FDDB/FDDB-folds/FDDB-folds/FDDB-fold-10-ellipseList.txt',
                 ]

def get_training_data(fold_elip_txt):
    #'''Returns two lists: a list of the names of the image files and a list of the bounding boxes'''
    # Open file
    path = []
    toa_do_elip = np.empty([0, 5])
    for i in range(0,10):
        print('Đọc dữ liệu từ:\n',fold_elip_txt[i])
        f = open(fold_elip_txt[i],'r')
        file = f.readlines()
        count = 0
        for i in file:
            # Grab the files with only 1 face
            if i == '1\n':
                filename = file[count - 1]
                bb = file[count + 1]

                # Get rid of the /n
                filename = filename[:-1] +".jpg"
                # Convert into array of integer coordinates
                num = bb.split(' ')
                num = num[:5]
                num = [float(i) for i in num]

                # Save into array
                path.append(filename)
                toa_do_elip = np.vstack([toa_do_elip,num])
            count = count + 1
    print("Mảng path đọc từ file txt:\n", path) #Tra ve duong dan den thu muc anh vi du 2002/08/11/big/img_591
    print("Tọa độ elip đọc từ file txt\n", toa_do_elip) #Tra ve toa do tung elip major_axis_radius ,minor_axis_radius,angle,center_x, center_y
    return path, toa_do_elip

def ellipse_to_rectangle(toa_do_elip):
    toa_do_rectangle= np.zeros([toa_do_elip.shape[0],8],dtype=int) #ma tran chua co hang nao va 8 cot
    toa_do_rectangle[:, [0]] = np.subtract(toa_do_elip[:, [3]], toa_do_elip[:, [1]]) #x điểm 1
    toa_do_rectangle[:, [1]] = np.subtract(toa_do_elip[:, [4]], toa_do_elip[:, [0]]) #y điểm 1
    toa_do_rectangle[:, [2]] = np.add(toa_do_elip[:, [3]], toa_do_elip[:, [1]]) #x điểm 2
    toa_do_rectangle[:, [3]] = np.add(toa_do_elip[:, [4]], toa_do_elip[:, [0]]) #y điểm 2
    toa_do_rectangle[:, [4]] = toa_do_elip[:, [3]] #x trung tâm
    toa_do_rectangle[:, [5]] = toa_do_elip[:, [4]] #y trung tâm
    toa_do_rectangle[:, [6]] = np.subtract(toa_do_rectangle[:, [2]],toa_do_rectangle[:, [0]])
    toa_do_rectangle[:, [7]] = np.subtract(toa_do_rectangle[:, [3]],toa_do_rectangle[:, [1]])
    print("Tọa độ chuyển sang rectangle:\n",toa_do_rectangle)
    return toa_do_rectangle
IMG_SIZE = 100
training_data = []
def creat_training_data(path, toa_do_rectangle,IMG_SIZE):
    for i in range(len(path)):
        img = cv2.imread("./2.FDDB/originalPics/" + path[i], cv2.IMREAD_COLOR)
        #crop = cv2.getRectSubPix(img,(toa_do_rectangle[0,2]-toa_do_rectangle[0,0],toa_do_rectangle[0,3]-toa_do_rectangle[0,1]),(toa_do_rectangle[4],toa_do_rectangle[5]))
        crop = cv2.getRectSubPix(img, (toa_do_rectangle[i,6], toa_do_rectangle[i,7]), ( toa_do_rectangle[i,4], toa_do_rectangle[i,5]))
        img_resize = cv2.resize(crop,(IMG_SIZE ,IMG_SIZE))
        cv2.imwrite(f'./crop/img{i}.jpg',img_resize)
        cv2.imshow('test',img_resize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
path, toa_do_elip = get_training_data(fold_elip_txt)
toa_do_rectangle = ellipse_to_rectangle(toa_do_elip)
creat_training_data(path, toa_do_rectangle,IMG_SIZE)




