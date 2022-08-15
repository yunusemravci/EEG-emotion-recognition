#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:30:22 2022

@author: yunusemreavci
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pywt
import pickle as pickle
import pandas as pd
#from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.signal import welch
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import tensorflow as tf
import keras.backend as K
#from keras.layers import Dense,LSTM, Bidirectional
##from keras.models import Sequential,Model
#from keras.layers.convolutional import Conv1D
#from keras.layers.convolutional import MaxPooling1D
#from keras.layers import Flatten, Activation
#from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
#import keras
#import timeit
#import warnings
#from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.utils import to_categorical



def read_eeg_signal_from_file(filename):
  x = pickle._Unpickler(open(filename,"rb"))
  x.encoding = "latin1"
  p = x.load()
  return p


def discrete_wavelet_transform(sub,mother_wavelet,pair_id,pair_first_chNum,pair_second_chNum):
  
  for subfile in sub:
    detail_coeffs = []
    channel_select_pair = []
    with open("data/s"+subfile + ".dat","rb") as file:
      subject = pickle.load(file, encoding='latin1')
      #subject = np.load(file,allow_pickle=True)
      data = subject["data"]
      channel1 = data[:,pair_first_chNum]
      channel2 = data[:,pair_second_chNum]
      channel_select_pair = np.array([*channel1, *channel2])
      channel_select_pair = channel_select_pair.reshape(40,16128)
      #print("concatenation")
      #print(channel_select_pair.shape)
      
      c_detail = pywt.wavedec(channel_select_pair,mother_wavelet,level=3)
      #detail_coeffs.append(c_detail[1])
      detail_coeffs = np.array(c_detail[1])
      #print("details")
      #print(detail_coeffs.shape)
      #detail_coeffs = np.reshape(detail_coeffs, (2,40,1014))
      #detail_coeffs = np.reshape(detail_coeffs,(40,2028))
      #print("-save each subjects channel pair into diff file-")
      np.save("wavelet_details_s"+ subfile +  "_"+ mother_wavelet +str(pair_id) + '_channelpair', detail_coeffs, allow_pickle=True, fix_imports=True)
      
  return detail_coeffs

def train_test_split(pair_id,mother_wavelet):

  data_training = []
  label_training = []
  data_testing = []
  label_testing = []

  for subjects in subjectList:
      with open("wavelet_details_s"+ subjects +  "_"+ mother_wavelet+ str(pair_id) + '_channelpair' + '.npy', 'rb') as f1, open("/content/drive/My Drive/Colab Notebooks/Biyomedikal/data/s"+ subjects + ".dat","rb") as f2:
          sub = np.load(f1,allow_pickle=True)
          labels = pickle.load(f2, encoding='latin1')
          labels = labels["labels"]
          #print(sub.shape[0])
          for i in range (0,sub.shape[0]):
              if i % 4 == 0 or i % 4 == 0:
                  data_testing.append(sub[i])
                  label_testing.append(labels[i])
            
              else:
                  data_training.append(sub[i])
                  label_training.append(labels[i])
  #np.save('data_training_' + str(pair_id), np.array(data_training), allow_pickle=True, fix_imports=True)
  #np.save('label_training', np.array(label_training), allow_pickle=True, fix_imports=True)
  print("training dataset with label:", np.array(data_training).shape, np.array(label_training).shape)
  return data_training,data_testing,label_training,label_testing
#------------------------#
labels = []
data = []
print("aliveli ")
#channel pairs
cp1 = [3,4]
cp2 = [8,26]
cp3 = [4,21]
cp4 = [5,6]
cp5 = [22,23]
cp6 = [7,8]
cp7 = [25,26]
band = [4,8,12,16,25,45] #5 bands
window_size = 256 #Averaging band power of 2 sec
step_size = 16 #Each 0.125 sec update once
sample_rate = 70 #Sampling rate of 128 Hz
subjectList = ['01','02','03','04','05','06','07','08','09','10','11','12',
              '13','14','15','16','17','18','19','20','21','22','23','24',
             '25','26','27','28','29','30','31','32']
#List of subjects



for n in subjectList:
  filename = "data/s" + n + '.dat'
  all_data = read_eeg_signal_from_file(filename)
  
  labels.append(all_data["labels"])
  data.append(all_data["data"])

#---------------------------------#
labels = np.array(labels)
labels = labels.flatten()
labels = labels.reshape(1280,4)
data = np.array(data)
data = data.flatten()
data = data.reshape(1280,40,8064)

pair1 = []
pair1.append(data[:,3])
pair1.append(data[:,4])
pair1 = np.reshape(pair1, (len(data), 2, len(data[0,0])))

pair1c = pair1.reshape(1280,16128)
print(pair1c.shape)


df_label_ratings = pd.DataFrame({"Valence": labels[:,0], "Arousal": labels[:,1]})
print(df_label_ratings.describe())

#---------------------------------#
sym4details1 = discrete_wavelet_transform(subjectList,"sym4",1,3,4)
sym4details2 = discrete_wavelet_transform(subjectList,"sym4",2,8,26)
sym4details3 = discrete_wavelet_transform(subjectList,"sym4",3,4,21)
sym4details4 = discrete_wavelet_transform(subjectList,"sym4",4,5,6)
sym4details5 = discrete_wavelet_transform(subjectList,"sym4",5,23,22)
sym4details6 = discrete_wavelet_transform(subjectList,"sym4",6,8,7)
sym4details7 = discrete_wavelet_transform(subjectList,"sym4",7,25,26)

sym4details1 = discrete_wavelet_transform(subjectList,"db4",1,3,4)
sym4details2 = discrete_wavelet_transform(subjectList,"db4",2,8,26)
sym4details3 = discrete_wavelet_transform(subjectList,"db4",3,4,21)
sym4details4 = discrete_wavelet_transform(subjectList,"db4",4,5,6)
sym4details5 = discrete_wavelet_transform(subjectList,"db4",5,23,22)
sym4details6 = discrete_wavelet_transform(subjectList,"db4",6,8,7)
sym4details7 = discrete_wavelet_transform(subjectList,"db4",7,25,26)

#---------------------------------#
splitted_datas = train_test_split(1,"sym4")

#---------------------------------#
temp = np.array(splitted_datas[3])
L = np.ravel(temp[:,[3]])

y_test = to_categorical(L)

temp2 = np.array(splitted_datas[2])
Z = np.ravel(temp2[: , [2]])
y_train = to_categorical(Z)
print(y_train.shape)


scaler = StandardScaler()
x_train = scaler.fit_transform(splitted_datas[0])
x_test = scaler.fit_transform(splitted_datas[1])


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

batch_size = 256
num_classes = 10
epochs = 200
input_shape1 = (x_train.shape[1],1)


model = Sequential()

model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape1))
model.add(Dropout(0.6))

model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.6))

model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.6))

model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=32))
model.add(Dropout(0.4))

model.add(Dense(units=16))
model.add(Activation('relu'))

model.add(Dense(units=num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
model.summary()

m=model.fit(x_train, y_train,epochs=200,batch_size=256,verbose=1,validation_data=(x_test,y_test))