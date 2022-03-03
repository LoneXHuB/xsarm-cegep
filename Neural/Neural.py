import keras
import pandas as pd
import numpy as np
from sklearn.feature_selection import r_regression
from sklearn.linear_model import LinearRegression
from joblib import dump, load

def get_model(n_inputs, n_outputs):
	model = keras.Sequential()
	model.add(keras.layers.Dense(3, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(keras.layers.Dense(n_outputs))
	model.compile(loss='mae', optimizer='adam')
	model.summary()
	return model

def get_data():
	data = pd.read_csv("SensorPuppet.csv")
	print("Aquired data...")
	print("Shape: "+ str(data.shape))
	print(data)
	X_train = data[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']]
	Y_train = data[['joint0', 'joint1', 'joint2', 'joint3', 'joint4']]
	print("Xtrain Shape:")
	print(X_train.shape)
	return X_train,Y_train


X_train, Y_train = get_data()
n_inputs = X_train.shape[1]
n_outputs = Y_train.shape[1]

regression = LinearRegression()
regression.fit(X_train,Y_train)

row = X_train.iloc[1]
newX = np.asarray([row])
print(newX)
pred = regression.predict(newX)
print('Predicted: %s' % pred[0])
print('Actual: %s' % Y_train.iloc[1])
error = np.zeros(5)
	
for indx,x in enumerate(pred[0]):
	error[indx] = x - Y_train.iloc[1][indx]
print('Error: %s' % error)

filename = 'python_demos/regression_puppet.sav'
dump(regression, filename)

regression = load(filename)
result = regression.score(X_train, Y_train)
print(result) 

"""
#fit on dense NNy
model = get_model(n_inputs, n_outputs)
model.fit(X_train, Y_train, verbose=0, epochs= 1000, validation_split=0.1)
#model = keras.models.load_model('Puppet')

row = X_train.iloc[1]
newX = np.asarray([row])
print(newX)
pred = model.predict(newX)
print('Predicted: %s' % pred[0])
print('Actual: %s' % Y_train.iloc[1])
error = np.zeros(5)
	
for indx,x in enumerate(pred[0]):
	error[indx] = x - Y_train.iloc[1][indx]
print('Error: %s' % error)
model.save('python_demos/Puppet')
"""