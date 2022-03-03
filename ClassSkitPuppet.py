# Import libraries and classes required for this example:
from scipy.fftpack import cs_diff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
import pickle
import pandas as pd 
import numpy as np

# Import dataset:
url = "ClassSensorPuppet.csv"

# Assign column names to dataset:
CSV_HEADER = ['accel_x', 'accel_y', 'accel_z']
CLASS_LENGTH = 10
h = np.asarray(CSV_HEADER).astype('U10')
final_header = np.resize(h, CLASS_LENGTH * len(CSV_HEADER) + len(CSV_HEADER))
final_header = np.append(final_header, 'move')

batch_count = 0
for indx,x in enumerate(final_header):
    if indx % len(CSV_HEADER) == 0 :
        batch_count += 1
    final_header[indx] = x + str(batch_count)

CSV_HEADER = final_header

# Convert dataset to a pandas dataframe:
dataset = pd.read_csv(url, names=CSV_HEADER) 

# Use head() function to return the first 5 rows: 
dataset.head() 
# Assign values to the X and y variables:
X = dataset.iloc[1:, :-1].values
y = dataset.iloc[1:, -1].values 

print(X)
print(y)
# Split dataset into random train and test subsets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 

# Standardize features by removing mean and scaling to unit variance:
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test) 

# Use the KNN classifier to fit data:
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train) 

# Predict y data with classifier: 
y_predict = classifier.predict(X_test)

filename = 'finalized_model.sav'
pickle.dump(classifier, open(filename, 'wb'))
# Print results: 
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict)) 
