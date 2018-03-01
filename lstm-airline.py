#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy 
import matplotlib.pyplot as plt 
from pandas import read_csv 
import math 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error 
#%matplotlib inline


# load the dataset 
dataframe = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3) 
dataset = dataframe.values # 将整型变为float 
dataset = dataset.astype('float32') 
plt.plot(dataset) 
plt.show()





