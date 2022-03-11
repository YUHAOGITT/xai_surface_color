# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 05:39:45 2020

@author: Yuhao
"""

import numpy as np
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import math
import scipy
import scipy.io
from PIL import Image
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model
from tensorflow import set_random_seed
from keras.callbacks import ModelCheckpoint
import pydot
import pydotplus
import graphviz
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import random
import pickle


set_random_seed(1)
np.random.seed(1)

batch_size = 32
num_classes = 2
epochs = 130

# Input image dimensions
img_rows, img_cols = 51, 100

if K.image_data_format() == 'channels_last':
#      
#    X_orig = scipy.io.loadmat('X_half_sq_10x.mat')['half_sq']
#    
#    X_orig = X_orig.reshape((1,img_rows,img_cols,600))
#    X = X_orig[0,:,:,0].reshape((1,img_rows,img_cols)) 
#    for i in range(1,600):
#        X_i = X_orig[0,:,:,i].reshape((1,img_rows,img_cols))
#        X = np.vstack((X, X_i))  
#    X = X.reshape((600,img_rows,img_cols,1))
#    
##    Y_orig = scipy.io.loadmat('Y_osq300.mat')['Y']
##    Y = keras.utils.to_categorical(Y_orig, num_classes)
#    
#    np.save('x_sq600_clean_10x_py.npy', X)
#    np.save('y_osq300_py.npy',Y)
    
    X = np.load('x_sqOur_600_py.npy')
    Y = np.load('y_300_py.npy')
    X_test = np.zeros((60,img_rows,img_cols))
    X_train = np.zeros((540,img_rows,img_cols))
    Y_test = np.zeros((60,2))
    Y_train = np.zeros((540,2))
    
#    a = random.sample(range(0,300),30)
    a = [170, 32, 60, 114, 161, 69, 6, 35, 178, 223, 117, 16, 184, 263, 242, 5, 11, 273, 65, 253, 130, 298, 299, 284, 102, 83, 224, 245, 45, 183]
    b = [429, 518, 538, 336, 466, 458, 494, 344, 477, 543, 378, 331, 492, 412, 474, 578, 443, 432, 421, 581, 473, 580, 410, 574, 362, 521, 463, 500, 508, 583]
    
    k = 0
    j = 0
    for i in range(0,600):
        if i in a or i in b:
            X_test[k,:,:] = X[i,:,:,0]
            Y_test[k,:] = Y[i,:]
            k = k+1
        else:
            X_train[j,:,:] = X[i,:,:,0]
            Y_train[j,:] = Y[i,:]
            j = j+1
    X_test = X_test.reshape((60,img_rows,img_cols,1))
    X_train = X_train.reshape((540,img_rows,img_cols,1))
    
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(3, 3),strides =(1,1),padding = 'same',
                     activation='relu',
                     input_shape=(img_rows,img_cols,1)))
    model.add(MaxPooling2D(pool_size=(3, 4)))
    model.add(Conv2D(8, (2, 2), strides =(1,1),padding = 'valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    #model.add(Dropout(0.05))
    model.add(Flatten())
    model.add(Dense(num_classes,activation = 'softmax'))
    
    adam = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer = adam, loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint('bestmodel_sq600_clean_10x_our.h5', monitor='accuracy', verbose=1, 
                    save_best_only=True, save_weights_only=False,
                    mode='auto', period=1)
    
    history = model.fit(X_train, Y_train,
          batch_size = batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=[checkpoint],
          validation_data=(X_test, Y_test))
    
#    score = model.evaluate(X_test, Y_test, verbose=0)
#    print('Test loss:', score[0])
#    print('Test accuracy:', score[1])
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1,len(acc)+1)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Testing acc')
    plt.title('Training and testing accuracy')
    #plt.title('Training accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Testing loss')
    plt.title('Training and testing loss')
    #plt.title('Training loss')
    plt.legend()
    plt.show()
    
    
 