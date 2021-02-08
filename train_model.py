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
from keras.utils import np_utils
import time
import pickle

def get_crop_image():
    training_data = []
    for i in range(1700):
        img = cv2.imread(f"./crop/img{i}.jpg",cv2.IMREAD_COLOR)
        training_data.append(img)
        # X = np.array().reshape(-1, xl.IMG_SIZE, xl.IMG_SIZE, 1)
        # X = X/255.0
        # y = np.ones([X.shape[0],1],int)
        # print(X.shape)
    return training_data

training_data = get_crop_image()

X = np.array(training_data).reshape(-1, 100, 100,3)
print(X)
Y = np.ones([1700,1])
# data normalization
num_classes = 10
X = X/255.
# 4. One hot encoding label (Y)
#from keras.utils import to_categorical
#y = to_categorical(y)
# Load model VGG 16 của ImageNet dataset, include_top=False để bỏ phần Fully connected lay
baseModel = VGG16(weights='imagenet', include_top=False, \
                  input_tensor=Input(shape=(100, 100, 3)))
# Buil layer
fcHead = baseModel.output
# Flatten
fcHead = Flatten()(fcHead)
# Add FC
fcHead = Dense(512, activation='relu')(fcHead)
fcHead = Dropout(0.5)(fcHead)
# Output layer with softmax activation
fcHead = Dense(5, activation='softmax')(fcHead)
# modle
model = model = Model(inputs=baseModel.input, outputs=fcHead)
model.compile(loss = "sparse_categorical_crossentropy",
                    optimizer = SGD(lr=1e-5, momentum=0.9),
                    metrics=["accuracy"])
history = model.fit(X, Y, batch_size=32, epochs=500, validation_split=0.1)

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)

model.save_weights("model2.h5")
print("Saved model to disk")

model.save('your_model.model')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()