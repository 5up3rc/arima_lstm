#!/usr/bin/python
#-*- coding: utf-8 -*-
'''
Created on 2018年2月24日
@author: 孙辽东
'''

'''
导入依赖的包
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline  
import tensorflow as tf
from sklearn.metrics import mean_absolute_error,mean_squared_error  
from sklearn.preprocessing import MinMaxScaler  

import warnings   
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的警告

##################################################################################################
'''
导入数据
'''
#f = open('usabled_days.xls', 'r', encoding='UTF-8') 
f = open('usabled_days.xls', 'rb') 
#mem_usabled	cpu_usabled	stor_usabled	record_time
df = pd.read_excel(f , names = ['mem_usabled' ,'cpu_usabled' ,'stor_usabled','record_time'])     #读入集群的历史性能数据
#df = pd.read_excel(f)
#print(df.head())
#print(df)
#data = df.loc[:,['cpu_usabled','record_time']].values  #读取需要计算的数据
#data = df.loc[:,['cpu_usabled']].values  #读取需要计算的数据
#data = np.array(df['cpu_usabled'])   #获取cpu的性能数据
data = df.iloc[:,0:3].values   #取第1-3列,（mem_usabled	cpu_usabled	stor_usabled）,需要和input_size保持一致  
#data = data[::-1]      #反转，使数据按照日期先后顺序排列
#print(data)
##################################################################################################




##################################################################################################
'''
定义常量并初始化权重：
'''
#定义常量  
time_step       = 20                                    #时间步
rnn_unit        = 10                                    #hidden layer units
batch_size      = 60                                    #每一批次训练多少个样例
input_size      = 3                                     #输入层维度
output_size     = 1                                     #输出层维度
lr              = 0.0006                                #学习率

data_len = len(data)
data_test_len = 20
data_train_len = data_len - data_test_len
data_size = 3                                           #原始数据列数
tf.reset_default_graph()  
#输入层、输出层权重、偏置  
weights={  
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),  
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))  
         }  
biases={  
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),  
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))  
        }  
##################################################################################################




##################################################################################################
'''
分割数据集，将数据分为训练集和验证集（最后20天做验证，其他做训练）：
'''
def get_data(batch_size = 60,time_step = 20,train_begin = 0,train_end = data_train_len):  
    batch_index=[]  
          
    scaler_for_x = MinMaxScaler(feature_range=(0,1))  #按列做minmax缩放 
    #print(scaler_for_x)
    scaler_for_y = MinMaxScaler(feature_range=(0,1))  
    #print(data[:,-1])
    #fit_transform()先拟合数据，再标准化
    scaled_x_data = scaler_for_x.fit_transform(data)
    scaled_y_data = scaler_for_y.fit_transform(data)  


    #scaled_x_data = scaler_for_x.fit_transform(data[:,:-1])  
    #scaled_y_data = scaler_for_y.fit_transform(data[:,:-3])  
    #scaled_y_data=scaler_for_y.fit_transform(data[:,-1])  
      
    label_train = scaled_y_data[train_begin:train_end]  
    label_test = scaled_y_data[train_end:]  
    normalized_train_data = scaled_x_data[train_begin:train_end]  
    normalized_test_data = scaled_x_data[train_end:]  
      
    train_x,train_y=[],[]   #训练集x和y初定义  
    for i in range(len(normalized_train_data)-time_step):  
        if i % batch_size==0:  
            batch_index.append(i)  
        x=normalized_train_data[i:i+time_step,:data_size]  
        y=label_train[i:i+time_step,np.newaxis]  
        train_x.append(x.tolist())  
        train_y.append(y.tolist())  
    batch_index.append((len(normalized_train_data)-time_step))  
      
    size = (len(normalized_test_data)+time_step-1)//time_step  #有size个sample   
    print(size)
    test_x,test_y=[],[]    
    for i in range(size-1):  
        x = normalized_test_data[i*time_step:(i+1)*time_step,:data_size]  
        y = label_test[i*time_step:(i+1)*time_step]  
        test_x.append(x.tolist())  
        test_y.extend(y)  
    test_x.append((normalized_test_data[(i+1)*time_step:,:data_size]).tolist())  
    test_y.extend((label_test[(i+1)*time_step:]).tolist())      

    return batch_index,train_x,train_y,test_x,test_y,scaler_for_y 
##################################################################################################







##################################################################################################
'''
#——————————————————定义神经网络变量——————————————————  
'''
def lstm(X):    
    batch_size = tf.shape(X)[0]  
    time_step = tf.shape(X)[1]  
    #print(time_step)
    w_in = weights['in']  
    b_in = biases['in']    
    input = tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入  
    input_rnn = tf.matmul(input,w_in)+b_in  
    input_rnn = tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入  
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)  
    #cell=tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_unit)  
    init_state = cell.zero_state(batch_size,dtype=tf.float32)  
    output_rnn,final_states = tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果  
    output = tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入  
    #print(output_rnn.shape)
    #print(output)
    w_out = weights['out']  
    b_out = biases['out']  
    pred = tf.matmul(output,w_out) + b_out  
    return pred,final_states  
##################################################################################################







##################################################################################################
'''
#——————————————————训练模型——————————————————  
'''
def train_lstm(batch_size = 80,time_step = 15,train_begin = 0,train_end = data_train_len):  
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])  
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])  
    batch_index,train_x,train_y,test_x,test_y,scaler_for_y = get_data(batch_size,time_step,train_begin,train_end)  
    print(batch_index)
    #print(train_y)
    #print(train_x)
    pred,_= lstm(X) 
    #print(pred)
    #损失函数  
    #print(tf.reshape(Y, [-1]))
    loss = tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    #print(loss)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)    
    #print(train_op)
    #print(train_y[batch_index[0]:batch_index[1]])
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())  
        #重复训练500次  
        iter_time = 500  
        for i in range(iter_time):  
            for step in range(len(batch_index)-1):  
                #print(step)
                #print(train_y[batch_index[step]:batch_index[step+1]])
                _,loss_ = sess.run(
                                    [train_op,loss],
                                    feed_dict = {
                                                    X:train_x[batch_index[step]:batch_index[step+1]],
                                                    Y:train_y[batch_index[step]:batch_index[step+1]]
                                                }
                                   )  
            if i % 100 == 0:      
                print('iter:',i,'loss:',loss_)  
        ####predict####  
        test_predict=[]  
        for step in range(len(test_x)):  
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})     
            predict=prob.reshape((-1))  
            test_predict.extend(predict)  
              
        test_predict = scaler_for_y.inverse_transform(test_predict)  
        test_y = scaler_for_y.inverse_transform(test_y)  
        rmse=np.sqrt(mean_squared_error(test_predict,test_y))  
        mae = mean_absolute_error(y_pred=test_predict,y_true=test_y)  
        print ('mae:',mae,'   rmse:',rmse)  
    return test_predict  

##################################################################################################



##################################################################################################
'''
调用train_lstm()函数，完成模型训练与预测的过程，并统计验证误差（mae和rmse）:
'''
test_predict = train_lstm(batch_size = 40,time_step = 15,train_begin = 0,train_end = data_train_len)  

plt.figure(figsize=(24,8))  
plt.plot(data[:, -1])  
plt.plot([None for _ in range(data_train_len)] + [x for x in test_predict])  
plt.show() 
##################################################################################################



##################################################################################################