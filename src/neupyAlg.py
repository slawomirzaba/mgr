import numpy as np
import pandas as pd
from neupy.algorithms import GradientDescent
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
x = df.iloc[0:150, [0,1,2,3]].values
y = df.iloc[0:150, 4].values
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

encoder = preprocessing.LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(x, dummy_y, test_size=0.2, random_state=42)

bpnet = GradientDescent((4, 3, 3), verbose=False, step=0.1)
bpnet.train(X_train, y_train, epochs=100)
result = bpnet.predict(X_test)
print('results', result)
print('\ncorrect', y_test)
