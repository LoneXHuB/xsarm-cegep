from audioop import ratecv
from hashlib import new
import keras
import pandas as pd
import numpy as np
from sklearn.feature_selection import r_regression
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from joblib import dump, load
import pickle

CSV_HEADER = ['accel_x', 'accel_y', 'accel_z']
CLASS_LENGTH = 50

def get_model(n_inputs, n_outputs):
	model = keras.Sequential()
	model.add(keras.layers.Dense(174, input_dim=n_inputs, activation='relu'))
	model.add(keras.layers.Dense(174, activation='relu'))
	model.add(keras.layers.Dense(n_outputs, activation='softmax'))
	model.compile(optimizer="adam",
              loss=keras.losses.CategoricalCrossentropy(from_logits=False), # default from_logits=False
              metrics=[keras.metrics.CategoricalAccuracy()])
	model.summary()
	return model

def get_data():
	global CSV_HEADER
	h = np.asarray(CSV_HEADER).astype('U10')
	final_header = np.resize(h, CLASS_LENGTH * len(CSV_HEADER) + 6)
	final_header = np.append(final_header, 'move')

	batch_count = 0
	for indx,x in enumerate(final_header):
		if indx % 6 == 0 :
			batch_count += 1
		final_header[indx] = x + str(batch_count)

	CSV_HEADER = final_header

	data = pd.read_csv("ClassSensorPuppet.csv")
	print("Aquired data...")
	data.describe()
	X_train = data
	class_label = data[['move52']]
	
	X_train = preprocessing.normalize(data[CSV_HEADER[:-1]])
	Y_train = np.zeros((len(class_label),5))
	
	for indx, x in enumerate(Y_train):
		i = class_label.iloc[indx]
		Y_train[indx , i] = int(1)
	
	print("Xtrain Shape:")
	print(X_train)
	print(" max is : " + str(X_train.max()))
	X_train = X_train / X_train.max()
	print("Normalized Xtrain Shape:")
	print(X_train)
	print("Ytrain Shape:")
	print(Y_train)
	print("Shape: "+ str(Y_train.shape))
	
	return X_train,Y_train

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
data = pd.read_csv("ClassSensorPuppet.csv")

test = data.iloc[100,:-1]
pred = loaded_model.predict([test])
print(pred[0])

"""
X_train, Y_train = get_data()
n_inputs = X_train.shape[1]
n_outputs = Y_train.shape[1]


model = get_model(n_inputs, n_outputs)
model.fit(X_train, Y_train, verbose=1, epochs=200, validation_split=0.1)


row = X_train[1]
newX = np.asarray([row])
print(newX)
pred = model.predict(newX)
print('Predicted: %s' % pred[0])
print('Actual: %s' % Y_train[100])
error = np.zeros(5)
	
for indx,x in enumerate(pred[0]):
	error[indx] = x - Y_train[100,indx]
print('Error: %s' % error)
"""
"""
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