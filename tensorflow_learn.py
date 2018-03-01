#!/usr/bin/python
#-*- coding: utf-8 -*-
# 在下面所有代码中，都去掉了这一行，默认已经导入
#import tensorflow as tf
#a = tf.zeros(shape=[1,2])
#a = tf.zeros((1,2))

'''
	在训练开始前，所有的数据都是抽象的概念，也就是说，此时a只是表示这应该是个1*5的零矩阵，
	而没有实际赋值，也没有分配空间，所以如果此时print,就会出现如下情况:
'''

#print(a)