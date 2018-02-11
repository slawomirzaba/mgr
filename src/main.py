from DeltaRule import DeltaRule
from BackPropagation import BackPropagation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils

epochs = 100
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
x = df.iloc[0:150, [0,1,2,3]].values
y = df.iloc[0:150, 4].values
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
hot_encoded_y = np_utils.to_categorical(encoded_Y)
X_train, X_test, y_train, y_test = train_test_split(x, hot_encoded_y, test_size=0.2, random_state=42)

delta = DeltaRule(epochs = epochs)
delta.train(X_train, y_train)

results = delta.predict(X_test)

print(results)
print(y_test)
