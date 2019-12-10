# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 02:26:14 2019

@author: Asl
"""
import gc
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from glob import glob
import random
import warnings

def read_image_path(DIR, SET):
    image_path = []
    for path in glob(DIR + '\\' + SET + '.*.jpg'):
        image_path.append(path)  
    image_path = np.array(image_path)
    return image_path


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)


def prep_data(images_path):
    count = len(images_path)
    data = np.ndarray((count, HEIGHT, WIDTH, CHANNELS))

    for i, image_file in enumerate(images_path):
        image = read_image(image_file)/255
        data[i] = image
        if i % 100 == 0: 
            print('已处理图片 {} / {}'.format(i, count))
        if i == len(images_path) - 1: 
            print('已处理图片 {} / {}'.format(i + 1, count))
    return data

def model_to_train(x_train_, y_train_):
    warnings.filterwarnings('ignore')
    model = Sequential()
    
    model.add(Conv2D(filters=3, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(224, 224, 3)))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=3, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=6, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=6, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(1, activation="sigmoid"))
    
    

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-04, decay=0.0)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    model.fit(x_train_, y_train_, epochs = EPOCHS, batch_size=BATCH_SIZE)
    
    return model

def valid_score(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_pred_classes = (y_pred > 0.5)
    y_pred_classes.astype(int)
    score = (y_pred_classes == y_val).astype(int).sum()/len(y_val)
    return score


if __name__ == "__main__":
    TRAIN_DIR = 'D:\\code\\py3\\kaggle_DOGvsCAT\\train1' #train*: *=1 2000, *=2 4000, *=3 6000

    HEIGHT = 224  # 输入的图片高度
    WIDTH = 224  # 输入的图片宽度
    CHANNELS = 3  # 输入的图片信道

    BATCH_SIZE = 16  # 每次传入模型的图片数量

    print('-' * 15 + '开始读取图片到内存' + '-' * 15)
    dog_path = read_image_path(TRAIN_DIR, 'dog')
    cat_path = read_image_path(TRAIN_DIR, 'cat')
    train_path = np.concatenate((dog_path, cat_path), axis=0)
    random.shuffle(train_path)
    del cat_path, dog_path
    
    y_train = np.zeros((len(train_path), 1))
    for i in range(len(y_train)):
        if train_path[i][35:38] == 'dog':
            y_train[i] = 1
        else:
            y_train[i] = 0
    del i
    
    x_train = prep_data(train_path)
    
    print('-' * 15 + '处理完毕' + '-' * 15)
    print("x_train shape: {}".format(x_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    
    print('-' * 15 + '划分训练集跟验证集' + '-' * 15)
    x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    print("x_train_ shape: {}".format(x_train_.shape))
    print("x_val shape: {}".format(x_val.shape))
    del x_train, y_train
    
    print('-' * 15 + '开始训练模型' + '-' * 15)
    EPOCHS = 6    # 训练集迭代的轮数
    print('-' * 15 + 'EPOCHS:' + str(EPOCHS) + '-' * 15)
    model = model_to_train(x_train_, y_train_)
    val_accuracy = valid_score(model, x_val, y_val)
    print('val_accuracy: ' + str(val_accuracy))
    