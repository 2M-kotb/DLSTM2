# from numpy.random import seed
# seed(1)
import numpy as np
import tensorflow as tf
import random as rn
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras import regularizers
from math import sqrt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate

def parser(x):
	return datetime.strptime(x, '%Y%m')

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

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

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], X.shape[1],1 )
	model = Sequential()
	model.add(GRU(neurons[0], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True,return_sequences=True))
	model.add(Dropout(0.3))
	model.add(GRU(neurons[1], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True,return_sequences=True))
	model.add(Dropout(0.3))
	model.add(GRU(neurons[2], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dropout(0.3))
	# model.add(LSTM(neurons[3], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	# model.add(Dropout(0.3))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		print('epoch:',i+1)
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
		model.reset_states()
	return model
	


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, len(X), 1)
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataset = np.insert(dataset,[0]*look_back,0)    
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataY= np.array(dataY)        
	dataY = np.reshape(dataY,(dataY.shape[0],1))
	dataset = np.concatenate((dataX,dataY),axis=1)  
	return dataset


# compute RMSPE
def RMSPE(x,y):
	result=0
	for i in range(len(x)):
		result += ((x[i]-y[i])/x[i])**2
	result /= len(x)
	result = sqrt(result)
	result *= 100
	return result

# compute MAPE
def MAPE(x,y):
	result=0
	for i in range(len(x)):
		result += abs((x[i]-y[i])/x[i])
	result /= len(x)
	result *= 100
	return result


def experiment(series,look_back,neurons,n_epoch):

	raw_values = series.values
	# transform data to be stationary
	diff = difference(raw_values, 1)
	

	# create dataset x,y
	dataset = diff.values
	dataset = create_dataset(dataset,look_back)


	# split into train and test sets
	train_size = int(dataset.shape[0] * 0.8)
	test_size = dataset.shape[0] - train_size
	train, test = dataset[0:train_size], dataset[train_size:]


	# transform the scale of the data
	scaler, train_scaled, test_scaled = scale(train, test)



	# fit the model
	lstm_model = fit_lstm(train_scaled, 1, n_epoch, neurons)
	# forecast the entire training dataset to build up state for forecasting
	print('Forecasting Training Data')   
	predictions_train = list()
	for i in range(len(train_scaled)):
		# make one-step forecast
		X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(raw_values)-i)
		# store forecast
		predictions_train.append(yhat)
		expected = raw_values[ i+1 ] 
		print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

	# report performance
	rmse_train = sqrt(mean_squared_error(raw_values[1:len(train_scaled)+1], predictions_train))
	print('Train RMSE: %.4f' % rmse_train)
	#report performance using RMSPE
	rmspe_train = RMSPE(raw_values[1:len(train_scaled)+1],predictions_train)
	print('Train RMSPE: %.4f' % rmspe_train)
	MAE_train = mean_absolute_error(raw_values[1:len(train_scaled)+1], predictions_train)
	print('Train MAE: %.5f' % MAE_train)
	MAPE_train = MAPE(raw_values[1:len(train_scaled)+1], predictions_train)
	print('Train MAPE: %.5f' % MAPE_train)

	# forecast the test data
	print('Forecasting Testing Data')
	predictions_test = list()
	for i in range(len(test_scaled)):
		# make one-step forecast
		X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
		# store forecast
		predictions_test.append(yhat)
		expected = raw_values[len(train) + i + 1]
		print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

	# report performance using RMSE
	rmse_test = sqrt(mean_squared_error(raw_values[-len(test_scaled):], predictions_test))
	print('Test RMSE: %.4f' % rmse_test)
	#report performance using RMSPE
	rmspe_test = RMSPE(raw_values[-len(test_scaled):],predictions_test)
	print('Test RMSPE: %.4f' % rmspe_test)
	MAE_test = mean_absolute_error(raw_values[-len(test_scaled):], predictions_test)
	print('Test MAE: %.5f' % MAE_test)
	MAPE_test = MAPE(raw_values[-len(test_scaled):], predictions_test)
	print('Test MAPE: %.5f' % MAPE_test)

	predictions = np.concatenate((predictions_train,predictions_test),axis=0)

	# line plot of observed vs predicted
	fig, ax = plt.subplots(1)
	ax.plot(raw_values, label='original', color='blue')
	ax.plot(predictions, label='predictions', color='red')
	ax.axvline(x=len(train_scaled)+1,color='k', linestyle='--')
	ax.legend(loc='upper right')
	ax.set_xlabel('Time',fontsize = 16)
	ax.set_ylabel('oil production '+ r'$(10^4 m^3)$',fontsize = 16)
	plt.show()

	 

def run():

	#load dataset
	series = read_csv('oil_production.csv', header=0,parse_dates=[0],index_col=0, squeeze=True,date_parser=parser)
	look_back= 2
	neurons=[ 3 , 1 , 4 ]
	n_epoch=1152
	experiment(series,look_back,neurons,n_epoch)

run()

