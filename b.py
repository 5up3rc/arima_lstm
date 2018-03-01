#!/usr/bin/python
#-*- coding: utf-8 -*-
'''
Created on 2018年2月24日
@author: 孙辽东
'''

'''
导入依赖的包
'''

import sys
#print(sys.version) #3.6.4 (v3.6.4:d48eceb, Dec 19 2017, 06:54:40) [MSC v.1900 64 bit (AMD64)]

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
#%matplotlib inline  


'''
生成数据
'''
random.seed(111)
rng = pd.date_range(start = '2000', periods = 209, freq = 'M')
ts = pd.Series(np.random.uniform(-10,10,size=len(rng)),rng).cumsum()
ts.plot(c='b', title = 'test data')
plt.show()
print(ts.head(10))


'''

'''

TS = np.array(ts)
num_periods = 20
f_horizon = 1
x_data = TS[:(len(TS)-(len(TS)%num_periods))]
x_batches = x_data.reshape(-1,20,1)

y_data = TS[1:(len(TS)-(len(TS)%num_periods))+f_horizon]
y_batches = y_data.reshape(-1,20,1)


print(len(x_batches))
print(x_batches.shape)
print(x_batches[0:2])

print(y_batches[0:1])
print(y_batches.shape)

























