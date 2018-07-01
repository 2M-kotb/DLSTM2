
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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from deap import base, creator, tools, algorithms
from keras import regularizers
from math import sqrt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from numpy import concatenate
import random


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
	model.add(LSTM(neurons[0], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True,return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(neurons[1], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True,return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(neurons[2], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dropout(0.3))
	# model.add(LSTM(neurons[3], batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	# model.add(Dropout(0.3))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		#print('Epoch:',i)
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model
	

# Update LSTM model
def update_model(model, train, batch_size, updates):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], X.shape[1],1 )
	for i in range(updates):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()


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


def train_evaluate(ga_individual_solution): 

	n_epoch = ga_individual_solution[0]
	neuron1 = ga_individual_solution[1]
	neuron2 = ga_individual_solution[2]
	neuron3 = ga_individual_solution[3]
	look_back = ga_individual_solution[4]
	updates = ga_individual_solution[5]
	

	print('\nnum of epochs = ',n_epoch,"num_neurons = [",neuron1,",",neuron2,",",neuron3,"]","window size = ",look_back,"updates = ",updates)

	neurons = []
	neurons.append(neuron1)
	neurons.append(neuron2)
	neurons.append(neuron3)


	#load dataset
	series = read_csv('oil_production.csv', header=0,parse_dates=[0],index_col=0, squeeze=True,date_parser=parser)
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

	# divide train into train and valid
	train2_size = int(train_scaled.shape[0] * 0.8)
	train_scaled2, valid = train_scaled[0:train2_size], train_scaled[train2_size:]

	


	# fit the model
	lstm_model = fit_lstm(train_scaled2, 1, n_epoch, neurons)
	

	# forecast the valid data
	print('Forecasting valid Data')
	valid_copy = np.copy(train_scaled2)
	predictions_valid = list()
	for i in range(len(valid)):
		# update model
		if i > 0:
			update_model(lstm_model, valid_copy, 1, updates)
		# make one-step forecast
		X, y = valid[i, 0:-1], valid[i, -1]
		yhat = forecast_lstm(lstm_model, 1, X)
		# invert scaling
		yhat = invert_scale(scaler, X, yhat)
		# invert differencing
		yhat = inverse_difference(raw_values, yhat, len(test_scaled)+len(valid)+1-i)
		# store forecast
		predictions_valid.append(yhat)
		# add to training set
		valid_copy = concatenate((valid_copy, valid[i,:].reshape(1, -1)))
		expected = raw_values[len(train_scaled2) + i + 1]
		#print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

	# report performance using RMSE
	rmse_valid = sqrt(mean_squared_error(raw_values[len(train_scaled2)+1:len(valid)+len(train_scaled2)+1], predictions_valid))
	print('Test RMSE: %.3f' % rmse_valid)
	
	return rmse_valid,
	
	
	 


def genetic():

	population_size = 5    # num of solutions in the population
	num_generations = 10 # num of time we generate new population
	
	# create a minimizing fitness function, cause we want to minimize RMSE
	creator.create('FitnessMax', base.Fitness, weights = (-1.0,))
	# create a list to encode solution in it (binary list)
	creator.create('Individual', list , fitness = creator.FitnessMax)

	# create an object of Toolbox class
	toolbox = base.Toolbox()

	toolbox.register("attr_int2", random.randint, 100,2000)
	toolbox.register("attr_int3",random.randint,1,5)
	toolbox.register("attr_int4",random.randint,1,5)
	toolbox.register("attr_int5",random.randint,1,5)
	toolbox.register("attr_int6",random.randint,1,6)
	toolbox.register("attr_int7",random.randint,1,4)
	toolbox.register("individual", tools.initCycle, creator.Individual,
             (toolbox.attr_int2, toolbox.attr_int3,toolbox.attr_int4,toolbox.attr_int5,toolbox.attr_int6,toolbox.attr_int7),
              n=1)

	toolbox.register('population', tools.initRepeat, list , toolbox.individual)

	toolbox.register('mate', tools.cxTwoPoint)
	toolbox.register('mutate', tools.mutUniformInt,low = [100,1,1,1,1,1],up = [2000,5,5,5,6,4],indpb = 0.6)
	toolbox.register('select', tools.selRoulette)
	toolbox.register('evaluate', train_evaluate)

	# create population by calling population function
	population = toolbox.population(n = population_size)

	hof = tools.HallOfFame(3)

	# start GA 
	r = algorithms.eaSimple(population, toolbox, cxpb = 0.4, mutpb = 0.1, 
	ngen = num_generations, halloffame = hof , verbose = False)


	# Print top N solutions 
	best_individuals = tools.selBest(hof,k = 3)
	best_epochs = None
	best_neuron1 = None
	best_neuron2 = None
	best_neuron3 = None
	best_window = None
	best_updates = None
	
	print("\nBest solution is:")
	for bi in best_individuals:
		best_epochs = bi[0]
		best_neuron1 = bi[1]
		best_neuron2 = bi[2]
		best_neuron3 = bi[3]
		best_window = bi[4]
		best_updates = bi[5]
		print("epochs = ",best_epochs,",neurons = [",best_neuron1,",",best_neuron2,",",best_neuron3,"]", ",window size = ",best_window,",updates = ",best_updates)

genetic()

