#!/usr/bin/python
#-*- coding: utf-8 -*-

#倒入用到的python库和类
import numpy as np 
import pandas as pd 
import re 
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima_model import ARIMA 
start_date = '2017-08-15';
end_date = '2017-09-11';
forecast_end_date = '2017-10-01';
#读取数据
train_df = pd.read_csv('2017-08-15_2017-09-11.csv').fillna(0) # 用０来补充缺失值
#print(train_df)

#转换数据类型，释放内存空间
# 数据为浮点数类型的整数，消耗内存较大，转换为整型数据释放一些内存600M减少到300M
for col in train＿df.columns[1:]:
    train_df[col] = pd.to_numeric(train_df[col],downcast='integer')
#print(train_df)
#正则表达提取网页的语种信息
# 正则表达匹配的对象是一个不可迭代的对象，可以通过group()转换为一个字符串对象
def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group()[:2]
    return 'na'

#新增语言列
train_df['lang'] = train_df['Page'].map(get_language)
#用一个字典对象来保存，不同语言的网页的流量数据，key为语言标记，value为对应的dataFarame对象
lang_sets = {}
lang_sets['en'] = train_df[train_df.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = train_df[train_df.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train_df[train_df.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train_df[train_df.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train_df[train_df.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train_df[train_df.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train_df[train_df.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train_df[train_df.lang=='es'].iloc[:,0:-1]

#计算每种语言wiki页面的日平均流量
sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
#print(sums)
# 字典转换为DataFrame对象绘图方便，Nan表示在网页的地址中没有明确表示文字格式
traffic_sum = pd.DataFrame(sums) 

#更新列名
traffic_sum.columns=['German','English','Spanish','French','Japanese','Nan','Russian','Chinese']

new_index = pd.date_range(start = start_date, end = end_date, freq='D');
traffic_sum.index = pd.Index(new_index)
#每种语言wiki页面的日平均浏览量 
traffic_sum.plot(figsize=(12,6))

plt.show()

print("##########################################原始数据#####################################################")
print(traffic_sum)
print("##########################################原始数据#####################################################")
#下面利用上面的数据绘制不用wiki页面浏览数据的自相关和部分自相关图，以估计用于训练ARIMA模型的超参数。
'''
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

for key in sums:
    fig = plt.figure(1,figsize=[12,4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    data = np.array(sums[key])
    #print(data)
    #print(len(data))
    autocorr = acf(data,nlags = 27)
    #print(autocorr)
    #print(len(autocorr))
    pac = pacf(data,nlags = 27)
    #print(pac)
    #print(len(pac))
	

    x = [x for x in range(len(pac))]
    #print(x)
    ax1.plot(x[1:],autocorr[1:])
    ax1.grid(True)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')

    ax2.plot(x[1:],pac[1:])
    ax2.grid(True)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    print(key)
    plt.show()

'''

'''
观察上面的图表
英语，俄语，法语，Nan语种页面的浏览量的自相关系数都有较快的收敛，序列比较平稳，所以不需要在进行处理．其他的时间序列有明显的周期性趋势，并没有较好的收敛，需要对序列进行处理，这里通过一阶差分来是序列趋于平稳．
对于日语，汉语，德语和西班牙语他们的流量数据下的自相关系数没７天左右会出现一个高峰，阶数Ｐ取７，其他的取３／４都ok,特别在意的话可以查找更多关于ARIMA模型定阶的资料．
平稳序列的部分相关系数都有较快的收敛，所以q=0,非平稳序列的收敛情况不一，为了方便计算，统一q=1.总之定阶的问题，我也比较头疼，还在摸索．
'''

#下面对不同语言的序列用ARIMA模型进行预测未来的流量
params = {'en': [4,1,0], 'ja': [7,1,1], 'de': [7,1,1], 'na': [4,1,0], 
          'fr': [4,1,0], 'zh': [7,1,1], 'ru': [4,1,0], 'es': [7,1,1]}

for key in sums:
    print(key)
    data = np.array(sums[key])
    result = None
    arima = ARIMA(data,params[key])
    result = arima.fit(disp=False)
    print(result)
    print(arima)
    print(result.params)
    pred = result.predict(1,typ='levels')
    #pred = result.predict(end = forecast_end_date, dynamic = False,typ = 'levels') #预测
    x = pd.date_range(start_date,forecast_end_date)


    print(key)
    plt.figure(figsize=(10,5))
    #plt.plot(x[:2],data[2:] ,label='Data')
    plt.plot(data[2:] ,label='Data')
	
    #plt.plot(x,pred,label='ARIMA Model')
    plt.plot(pred,label='ARIMA Model')
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.legend()
    plt.show()