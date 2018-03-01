#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.api import qqplot
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator

#==================================1.基础数据处理=====================================================================================
#先将数据读入data
#读取数据，指定日期列为指标，Pandas自动将“日期”列识别为Datetime格式，filePath为数据文件路径
#data = pd.read_excel('d:/Documents/arima/mem_usabled.xls', index_col = u'record_time') #这种方式横坐标不显示时间，日期自动识别失败
#data = pd.read_excel('d:/Documents/arima/mem_usabled.xls').head(100)
#可以使用fillna(0)补充缺失数据
data = pd.read_excel('mem_usabled_days_new.xls')
#设置预测时间
forecast_start_date = '2018-01-03';
forecast_end_date = '2018-03-01';
forecast_typ = 'linear';
#设置图形大小
fig_size = (10,5)
#
lagnum = 40
#设置需要预留的数据
forecast_size = 20
#预留数据
data_obligate = data.tail(forecast_size)
data_size = len(data)
#读取前几行数据-预留后续数据做校对
data = data.head(data_size - forecast_size)
record_time_array = data['record_time'];
start_date = record_time_array[0];
end_date = record_time_array[data_size - forecast_size - 1];


#设置索引
#data.set_index("record_time", inplace=True)
#data.index = pd.DatetimeIndex(data.index)
new_index = pd.date_range(start = start_date, end = end_date, freq='D');
data.index = pd.Index(new_index)
#print(data.index)

#print(data1.info())
#分析数据
data_analysis = data['mem_usabled'];


#================绘制时序图,原始数据图形展现===================
#data.plot() 
data.plot(figsize = fig_size ) #设置图片显示大小

# 设置X轴的坐标刻度线显示间隔
#plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#以天为单位分隔，可以改成自己想要的方式，譬如以月分隔，以年分隔
#plt.gca().xaxis.set_major_locator(MonthLocator())
plt.show() # 画出时序图
#plt.gcf().autofmt_xdate()


'''
#===================================2. 构建模型=================================================================================

#1.获取被观测系统时间序列数据；
#2.对数据绘图，观测是否为平稳时间序列；对于非平稳时间序列要先进行d阶差分运算，化为平稳时间序列；
#3.经过第二步处理，已经得到平稳时间序列。要对平稳时间序列分别求得其自相关系数ACF 和偏自相关系数PACF ，
   通过对自相关图和偏自相关图的分析，得到最佳的阶层 p 和阶数 q
#4.由以上得到的d、q、p，得到ARIMA模型。然后开始对得到的模型进行模型检验。


#2.1 我们首先要做的是对观测值进行平稳性检验，如果不平稳需要进行差分运算直到差分后的数据平稳
#2.2 在数据平稳后进行白噪声检验，如果没有通过白噪声检验，就进行模型识别，识别模型属于AR,MA和ARMA中的哪一种模型，
#    并通过贝叶斯信息准则对模型进行定阶，确定ARIMA模型的p、q参数 。
#2.3 在模型识别后需要进行模型检验，检测模型残差序列是否为白噪声序列，
#    如果没有通过检验，需要对其进行重新识别，对已经通过检验的模型采用极大似然估计方法进行模型参数估计。
#2.4 应用模型进行预测，将实际值与预测值进行误差分析。如果误差比较小，表明模型拟合效果较好，则模型可以结束，反之需要重新估计参数。
'''


#2.1.1 平稳性检测，为了确定原始数据序列中没有随机趋势或确定趋势，需要对数据进行平稳性检验，否则会产生伪回归现象
#      ADF检验可以得到单位根检验统计量对应的p值，若此值显著大于0.05，则该序列非平稳
#      偏差之后图形展现

#定义d阶差分
diff = 0
adf =  (1,1);
diff_data = data_analysis
while adf[1] >= 0.05:#adf[1]为p值，p值小于0.05认为是平稳的
    print(u'循环执行次数%s'%(diff))
    if diff == 0:
        adf = ADF(data_analysis)
    else:
        diff_data = data_analysis.diff(diff).dropna()
        adf = ADF(diff_data)
    print(u'原始序列经过%s阶差分后，adf值为%s'%(diff,adf))
    diff = diff + 1
print(u'原始序列经过%s阶差分后归于平稳，p值为%.10f'%(diff-1,adf[1]))
diff_data.plot()
plt.show()

#输出结果分析------数据平稳
#(-5.126199820007796, 1.241445473324911e-05, 54, 56471, {'1%': -3.430465804524, '5%': -2.8615911833505536, '10%': -2.5667972431822057}, 954817.6633813755)



'''
#2.1.2 白噪声检验：为了验证平稳序列中有用信息是否已被提取完，需要对序列进行白噪声检验，
#      如果序列是白噪声的说明序列中的有用信息已经被提取完毕了，剩下的全是随机扰动，无法进行预测和使用
#      检验时间序列是否为白噪声序列，一般如果统计量的P值小于0.05时，则可以拒绝原假设，认为该序列为非白噪声序列
#      若差分后的序列不平稳，修改差分阶数重新差分，或者继续对序列进行差分运算，运算后继续进行平稳性和白噪声检验，直至差分后的序列平稳为止。

[[lb],[p]] = acorr_ljungbox(data_analysis,lags=1)#滞后阶数
diff_1 = 0;
while p < 0.05: #非白噪声，需要递归
    diff_1 = diff_1 + 1;
    [[lb],[p]] = acorr_ljungbox(data_analysis.diff(diff_1).dropna(),lags=1)
print(u'原始序列经过%s阶差分后为白噪声序列，p值为%.10f'%(diff_1,p))
'''


'''
#2.2.1 模型识别
#      采用极大似然比方法进行模型的参数估计，估计各个参数值，然后针对各个不同模型，
#      采用BIC信息准则（贝叶斯信息准则）对模型进行定阶，确定p,q参数，从而选择最优模型。

#      确定最佳p、d、q值,d来源于之前的阶数diff字段
#      p和q的值可以通过对自相关系数图和偏自相关系数图人为识别来确定，
#      确定方法是根据自相关系数图和偏自相关系数图的拖尾和截尾的性质来确定的
#      ，同样我们也可以通过BIC矩阵中的最小BIC信息量的位置来决定.

'''


#######################################相关性####################################################
# p=偏自相关截尾的lag。如果偏自相关截尾后，自相关仍然拖尾，考虑增加p
# q=自相关截尾的lag。如果自相关截尾后，偏自相关仍然拖尾，考虑增加q
# 以上仅是理论上的说法，具体最合适的p,q，还是要以AIC、BIC或预测结果残差最小来循环确定
# ARIMA模型是通过优化AIC, HQIC, BIC等得出的, 一般来说AIC较小的模型拟合度较好,但拟合度较好不能说明预测能力好

# AR（p）模型：自相关系数拖尾，偏自相关系数p阶截尾。
# MA（q）模型：自相关系数q阶截尾，偏自相关系数拖尾。
# ARMA(ｐ，ｑ)模型：自相关系数拖尾，偏自相关系数拖尾。

# 平稳的序列的自相关图和偏相关图不是拖尾就是截尾。
# 截尾就是在某阶之后，系数都为 0 
# 拖尾，拖尾就是有一个衰减的趋势，但是不都为 0 。 

# 如果自相关是拖尾，偏相关截尾，则用 AR 算法 
# 如果自相关截尾，偏相关拖尾，则用 MA 算法 
# 如果自相关和偏相关都是拖尾，则用 ARMA 算法， ARIMA 是 ARMA 算法的扩展版，用法类似 。 

#其中lags 表示滞后的阶数，以下分别得到acf 图和pacf 图
fig = plt.figure(figsize = fig_size )
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_analysis,lags = lagnum,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_analysis,lags = lagnum,ax=ax2)
plt.show() # 在Scala IDE要输入这个命令才能显示图片

#######################################相关性####################################################



#######################################BIC矩阵###################################################
#定阶
# 一阶差分后平稳，所以d=1。p,q参数使用循环暴力调参确定最佳值
d = diff - 1
if d > 0:
    forecast_typ = 'levels'
pmax = int(len(data_analysis)/10)#一般阶数不超过length/10
pmax = 6

qmax = int(len(data_analysis)/10)#一般阶数不超过length/10
qmax = 6

bic_matrix=[]#bic矩阵
pdq_result = pd.DataFrame()

for p in np.arange(0, pmax + 1):
    tmp=[]
    for q in np.arange(0, qmax + 1):
        try:#存在报错，用try来跳过报错
            model = ARIMA(data_analysis,order=(p, d, q))  # ARIMA的参数: order = p, d, q
            results_ARIMA = model.fit()
            pdq_result = pdq_result.append(pd.DataFrame(
                {'p': [p], 'd': [d],'q': [q], 'AIC': [results_ARIMA.aic], 'BIC': [results_ARIMA.bic],
                 'HQIC':[results_ARIMA.hqic]}))
            tmp.append(results_ARIMA.bic)
            #print(results_ARIMA.summary())  #summary2()也可以使用
        except:
            print('NG')
            tmp.append(None)
    bic_matrix.append(tmp)
print("============pdq_result==================")
print(pdq_result)
print("============pdq_result==================")
print("============bic_matrix==================")
bic_matrix=pd.DataFrame(bic_matrix)
print(bic_matrix)
print("============bic_matrix==================")
p,q=bic_matrix.stack().idxmin()#从中可以找出最小值
print (u'BIC最小的p值和q值为：%s,%s'%(p,q))
#######################################BIC矩阵###################################################




#2.3.1 模型检测--ARIMA(0,1,0)
#      在指数平滑模型下，观察ARIMA模型的残差是否是平均值为0且方差为常数的正态分布（服从零均值、方差不变的正态分布），
#      同时也要观察连续残差是否（自）相关。
#print("##############data_analysis##################") 
#print("data_analysis=",data_analysis)
#print("##############data_analysis##################")

#print("#############data_analysis_pred###################")  
#建立ARIMA(0,1,0)模型  
arima = ARIMA(data_analysis, (p, d, q)).fit() #建立并训练模型  


#2.3.2 检验下残差序列
resid = arima.resid
fig = plt.figure(figsize = fig_size )
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags = lagnum, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags = lagnum, ax=ax2)
plt.show()

#2.3.3 做D-W检验 德宾-沃森（Durbin-Watson）检验。
#      德宾-沃森检验,简称D-W检验，是目前检验自相关性最常用的方法，但它只使用于检验一阶自相关性。
#      因为自相关系数ρ的值介于-1和1之间，所以 0≤DW≤４。并且DW＝O＝＞ρ＝１　　 即存在正自相关性 
#      DW＝４＜＝＞ρ＝－１　即存在负自相关性 
#      DW＝２＜＝＞ρ＝０　　即不存在（一阶）自相关性 
#      因此，当DW值显著的接近于O或４时，则存在自相关性，
#      而接近于２时，则不存在（一阶）自相关性。
#      这样只要知道ＤＷ统计量的概率分布，在给定的显著水平下，根据临界值的位置就可以对原假设Ｈ０进行检验。
print("==================D-W===========================")
print(sm.stats.durbin_watson(resid.values))
print("==================D-W===========================")

#2.3.4 观察是否符合正态分布
#      这里使用QQ图，它用于直观验证一组数据是否来自某个分布，或者验证某两组数据是否来自同一（族）分布。
#      在教学和软件中常用的是检验数据是否来自于正态分布。

fig = plt.figure(figsize = fig_size )
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
plt.show()

#2.3.5 Ljung-Box检验
#      Ljung-Box test是对randomness的检验,或者说是对时间序列是否存在滞后相关的一种统计检验。
#      对于滞后相关的检验，我们常常采用的方法还包括计算ACF和PCAF并观察其图像，但是无论是ACF还是PACF都仅仅考虑是否存在某一特定滞后阶数的相关。
#      LB检验则是基于一系列滞后阶数，判断序列总体的相关性或者说随机性是否存在。 
#      时间序列中一个最基本的模型就是高斯白噪声序列。
#      而对于ARIMA模型，其残差被假定为高斯白噪声序列，所以当我们用ARIMA模型去拟合数据时，
#      拟合后我们要对残差的估计序列进行LB检验，判断其是否是高斯白噪声，如果不是，那么就说明ARIMA模型也许并不是一个适合样本的模型。

#      检验的结果就是看最后一列前十二行的检验概率（一般观察滞后1~12阶），
#      如果检验概率小于给定的显著性水平，比如0.05、0.10等就拒绝原假设，其原假设是相关系数为零。
#      就结果来看，如果取显著性水平为0.05，那么相关系数与零没有显著差异，即为白噪声序列。

print("===========================Ljung-Box检验========================================")
r_Ljung_Box,q_Ljung_Box,p_Ljung_Box = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r_Ljung_Box[1:], q_Ljung_Box, p_Ljung_Box]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
print("===========================Ljung-Box检验========================================")


#6.数据预测
###################################使用forecast预测数据#########################################################################
print("##########使用forecast预测数据###################") 
# forecast返回值为有3个元素的元组(tuple)，每个元素都是一个array，
# 说明：forecast : array， stderr : array，conf_int : array2D
predict_dta = arima.forecast(forecast_size) # 连续预测N个值
print(predict_dta)
print("##########使用forecast预测数据###################") 
###################################使用forecast预测数据#########################################################################

###################################使用plot_predict预测数据#####################################################################
print("##########使用plot_predict预测数据###################") 

if d == 0:
    predict_dta2 = arima.predict(start = forecast_start_date, end = forecast_end_date,dynamic = False)
else:
    predict_dta2 = arima.predict(start = forecast_start_date, end = forecast_end_date,dynamic = False,typ = forecast_typ)
print(predict_dta2)

xdata_pred2,ax = plt.subplots(figsize = fig_size )
ax = data_analysis.ix[1:].plot(ax=ax)
xdata_pred2 = arima.plot_predict(start = forecast_start_date,end = forecast_end_date,dynamic = False, ax = ax, plot_insample = False)
plt.show()
#print(xdata_pred2)
print("##########使用plot_predict预测数据###################") 
###################################使用plot_predict预测数据######################################################################


###################################使用predict预测数据######################################################################
#dynamic=False参数确保我们产生一步前进的预测，这意味着每个点的预测都将使用到此为止的完整历史生成
#语法参考：http://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMAResults.predict.html#statsmodels.tsa.arima_model.ARIMAResults.predict
print("##########使用predict预测数据###################")
if d == 0:
    xdata_pred = arima.predict(end = forecast_end_date, dynamic = False) #预测
else:
    xdata_pred = arima.predict(end = forecast_end_date, dynamic = False,typ = forecast_typ) #预测
print(xdata_pred)

plt.figure(figsize = fig_size )
plt.plot(data_analysis ,label='Old Data')
plt.plot(xdata_pred,label='Forecast Data')
plt.xlabel('Days')
plt.ylabel('Views')
plt.legend()
plt.show()
print("##########使用predict预测数据###################")
###################################使用predict预测数据######################################################################
#7.数据校对

#模型确定后，检验其残差序列是否为白噪声，如果不是白噪声说明残差中还存在有用信息，需要修改模型或者进一步提取
print("==========================pred_error======================================================")
pred_error = (xdata_pred - data_analysis).dropna()#计算残差(观察值与拟合值之间的差)
print(pred_error)
print("==========================pred_error======================================================")
#白噪声检验
lb,p1 = acorr_ljungbox(pred_error,lags = lagnum)
print(lb)
print(p1)
h = ( p1 < 0.05 ).sum()#p1值小于0.05，认为是非白噪声
print(u'模型arima(%s,%s,%s)，h值为%.10f'%(p,d,q,h))
if h > 0:
    print (u'模型不符合白噪声检验')
else:
    print (u'模型符合白噪声检验')
#模型评价
#采用三个衡量模型预测精度的统计指标：平均绝对值、均方根误差、均方根误差和平均绝对误差，这三个指标从不同侧面反映了算法的预测精度。

#获取实际值

print("======================实际值========================")
print(data_obligate)
print("======================实际值========================")

print("======================预计值========================")
print(predict_dta)
print("======================预计值========================")

abs_= (predict_dta[0] - data_obligate['mem_usabled']).abs()
mae_=abs_.mean()#平均绝对误差
rmse_=((abs_**2).mean())**0.5#均方根误差
mape_=(abs_ / data_obligate['mem_usabled']).mean()#平均绝对百分误差
print (u'平均绝对误差:%0.4f,\n均方根误差为:%0.4f,\n平均绝对百分误差:%0.6f'%(mae_,rmse_,mape_))
