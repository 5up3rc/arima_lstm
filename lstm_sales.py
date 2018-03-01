#!/usr/bin/python
#-*- coding: utf-8 -*-

#https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/?spm=a2c4e.11153940.blogcont174270.16.52b9a3b47UDGdq
'''
1. 从 CSV 文件加载数据集。
2. 转换数据集以使其适合 LSTM 模型, 包括:
	2.1 将数据转换为有监督的学习问题。
	2.2 将数据转换为静止的。
	2.3 转换数据, 使其具有-1 到1的比例。
3. 将有状态 LSTM 网络模型拟合到培训数据。
4. 对测试数据的静态 LSTM 模型进行评估。
5. 报告预测的执行情况。


有关示例的一些注意事项:
	1. 缩放和反转缩放行为已移动到函数缩放 ()和invert_scale ()以实现简洁。
	2. 测试数据是使用器对训练数据进行调整, 以确保测试数据的最小/最大值不影响模型。
	3. 为了方便起见, 对数据转换的顺序进行了调整, 以使数据平稳, 然后进行有监督的学习问题, 然后进行缩放。
	4. 为了方便起见, 在将整个数据集拆分为训练和测试集之前, 对其进行了差分。我们可以很容易地收集观察, 在前进的验证和差异, 因为我们去。我决定对它的可读性。
'''
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import numpy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #忽略烦人的警告

# load and plot dataset
# load dataset

# date-time parsing function for loading the dataset
def parser(x):
    print(x)
    return datetime.strptime('190' + str(x), '%Y-%m')
'''
LSTM 数据准备
1.将时间序列转换为有监督的学习问题
2.转换时间序列数据, 使其静止不动。
3.将观测值转换为特定的刻度。
'''

'''
1.1 变换时间序列到监督学习
# Keras 中的 LSTM 模型假定您的数据分为输入 (X) 和输出 (y) 组件。
# 对于一个时间序列的问题, 我们可以通过使用最后一步 (t-1) 的观测作为输入, 在当前时间步长 (t) 作为输出的观察来实现这一点。
# frame a sequence as a supervised learning problem
# 采用了原始时间序列数据的 NumPy 数组, 并将滞后或移位的系列数作为输入来创建和使用。
'''
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df
'''
#2.1将时间序列转换为静止
# 消除趋势的标准方法是通过区分数据。这是从前一个时间步 (t-1) 的观察减去当前观测 (t)。
# 这消除了趋势, 我们留下了一个不同的系列, 或对观察的变化从一个时间步到下一个。
# 请注意, 原始数据集中的第一个观察是从倒置的差异数据中删除的。此外, 最后一组数据与预期值匹配。
'''

#下面是一个函数, 称为差异 () , 用于计算差分系列。请注意, 将跳过该系列中的第一个观察, 因为没有用于计算差分值的预先观察。
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
	
#我们还需要反转这一过程, 以使差分系列的预测回到原来的规模。
#下面的函数(称为 inverse_difference ())将反转此操作。

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]
'''
3.1将时间序列转换为缩放
与其他神经网络一样, LSTMs 希望数据在网络使用的激活函数的范围内。
LSTMs 的默认激活函数是双曲正切 (tanh), 它输出介于-1 和1之间的值。这是时间序列数据的首选范围。
为了使实验公平, 必须在训练数据集上计算比例系数 (min 和 max) 值, 并应用于对测试数据集和任何预测进行缩放。
这是为了避免用来自测试数据集的知识来污染实验, 这可能会给模型一个小的边缘。
'''
# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled
# 颠倒预测上的比例, 将值返回到原始刻度, 以便可以解释结果, 并计算出可比较的错误评分。
# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]
'''
LSTM 模型开发
长短期记忆网络 (LSTM) 是一种递归神经网络 (RNN)。
这种类型的网络的好处是它可以学习和记住长序列, 不依赖于预先指定的窗口滞后观察作为输入。
在 Keras 中, 这被称为 "有状态", 并且在定义 LSTM 层时涉及将 "有状态" 参数设置为 "True"。
默认情况下, Keras 中的 LSTM 层在一个批处理中维护数据之间的状态。
一批数据是来自培训数据集的固定大小的行数, 用于定义在更新网络权重之前要处理的模式数量。
默认情况下, 批处理之间的 LSTM 层中的状态被清除, 因此我们必须使 LSTM 有状态。
这使我们可以通过调用reset_states ()函数来对何时清除 LSTM 层的状态进行细粒度的控制。
LSTM 层期望输入与维度的矩阵: [示例, 时间步骤, 功能]。
	示例: 这些是来自域 (通常是数据行) 的独立观察。
	时间步骤: 这些是给定变量的单独时间步骤, 用于给定的观察。
	功能: 这些是观察时观察到的单独措施。
'''

#来训练和返回一个 lstm 模型。作为论据, 它采取训练数据集以被监督的学习的格式, 批量大小, 许多世纪和许多神经元。
#batch_size 必须设置为1。这是因为它必须是训练和测试数据集大小的一个因素。
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    #由于训练数据集定义为 X 输入和 y 输出, 因此必须将其重新转换为示例/TimeSteps/特征格式, 
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	'''
		输入数据的形状必须在 LSTM 层中指定, 方法是使用 "batch_input_shape" 参数作为一个元组, 指定要读取的每个批处理所需的观察次数、时间步长数和特征数。
		批次大小通常比样本总数小得多。它与时代的数量一起定义了网络学习数据的速度 (权重的更新频率)。
		定义 LSTM 层的最后一个导入参数是神经元的数量, 也称为内存单元或块的数量。这是一个相当简单的问题, 1 和5之间的数字应该足够。
	'''
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

'''
LSTM 预测:选定可用于训练的lstm模型来完成数据的预测

我们可以决定在所有的训练数据中, 
	一次调整模型然后从测试数据中一次预测每一个新的时间步长 (我们称之为固定方法), 
	或者我们可以重新适应模型, 
	或者在测试数据的每一次步骤中更新模型, 
作为新的观察测试数据可用 (我们称之为动态方法)。

若要进行预测, 可以对模型调用forecast()函数。这需要一个3D 的 NumPy 数组输入作为参数。
在这种情况下, 它将是一个数组的一个值, 观察在上一次的时间步长。

在训练期间, 内部状态在每个世纪以后被重置。在预测时, 我们不会希望重置预测之间的内部状态。实际上, 我们希望模型在测试数据集中的每个时间步长时建立状态。
这就引出了一个问题, 即在预测测试数据集之前, 什么是网络的良好初始状态。
我们将对培训数据集中的所有示例进行预测, 从而对该状态进行播种。理论上, 应建立内部状态, 准备下一阶段的预测。
我们现在有所有的部分, 以适应 LSTM 网络模型的洗发水销售数据集, 并评估其性能。
'''

#forecast()函数返回一个预测数组, 一个用于提供每个输入行。因为我们提供了一个单一的输入, 输出将是一个 2D NumPy 阵列的一个值。
#我们可以在下面列出的名为forecast()的函数中捕获此行为。
#给定一个适合模型, 在拟合模型 (例如 1) 和测试数据中的一行时使用的批处理大小, 该函数将从测试行中分离出输入数据, 对其进行整形, 并将预测作为单个浮点值返回。
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]
def forecast(model, batch_size, row):
	X = row[0:-1]
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

############################################################################################################################################################
series = read_csv('sales-of-shampoo-over-a-three-ye.csv', header=0, parse_dates = [0], index_col = 0, squeeze = True, date_parser = parser)
#series = read_csv('sales-of-shampoo-over-a-three-ye.csv', header = 0, parse_dates = [0], index_col = 0, squeeze = True)
# summarize first few rows
print(series)
# line plot
series.plot()
pyplot.show()

'''
#We will split the Shampoo Sales dataset into two parts: a training and a test set.

# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]
print("========================train=============================")
print(train)
print("========================train=============================")

print("========================test=============================")
print(test)
print("========================test=============================")

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])

print("=========================predictions==============================")
print(predictions)
print("=========================predictions==============================")
print("=========================history==============================")
print(history)
print("=========================history==============================")
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(test)
pyplot.plot(predictions)
pyplot.show()
'''

#supervised = timeseries_to_supervised(X, 1)
#print("====================timeseries_to_supervised===========================")
#print(supervised)
#print("====================timeseries_to_supervised===========================")

# transform data to be stationary
print("==================difference============================")
raw_values = series.values
diff_values = difference(raw_values, 1)
#differenced = difference(series, 1)
#print(differenced)
print(diff_values)
print("==================difference============================")

# transform data to be supervised learning
print("====================supervised_values===========================")
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values
print(supervised_values)
print("====================supervised_values===========================")

# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

'''
# fit the model
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 1, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()
'''

# repeat experiment
repeats = 30
error_scores = list()
for r in range(repeats):
	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
	# forecast the entire training dataset to build up state for forecasting
	train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
	lstm_model.predict(train_reshaped, batch_size=1)
	# walk-forward validation on the test data
	predictions = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions.append(yhat)
	# report performance
	rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
	print('%d) Test RMSE: %.3f' % (r+1, rmse))
	error_scores.append(rmse)

# summarize results
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
pyplot.show()


'''
print("==================inverted============================")
# invert transform
inverted = list()
for i in range(len(differenced)):
	value = inverse_difference(series, differenced[i], len(series)-i)
	inverted.append(value)
inverted = Series(inverted)
print(inverted)
print("==================inverted============================")
'''

'''
# transform scale
X_1 = X.reshape(len(X), 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(X_1)
scaled_X = scaler.transform(X_1)
scaled_series = Series(scaled_X[:, 0])
print("=============scaled_series=====================")
print(scaled_series.head())
print("=============scaled_series=====================")
# invert transform
inverted_X = scaler.inverse_transform(scaled_X)
inverted_series = Series(inverted_X[:, 0])
print("============inverted_series======================")
print(inverted_series.head())
print("============inverted_series======================")
'''

############################################################################################################################################################