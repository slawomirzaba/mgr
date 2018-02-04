from DeltaRule import DeltaRule
from BackPropagation import BackPropagation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
x = df.iloc[0:150, [0,1,2,3]].values
y = df.iloc[0:150, 4].values
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = y/2
x = x/np.amax(x, axis=0) # maximum of X array
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# # setosa and versicolor
# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', 0, 1)
# # sepal length and petal length
# X = df.iloc[0:100, [0,2]].values
# X_std = np.copy(X)
# X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
# X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

epochs = 100
print(epochs)
# delta = DeltaRule(epochs = epochs)
# back = BackPropagation(epochs = epochs)
#
# delta.train(X_train, y_train)
# back.train(X_train, y_train)

# print ('deltaError', np.mean(np.square(y_test - delta.predict(X_test))))
array = [3, 4, 5, 10, 20, 50, 100, 200, 500]
for value in array:
    print(value)
    back = BackPropagation(epochs=epochs, hidden_size=value)
    back.train(X_train, y_train)
    print('backError', np.mean(np.square(y_test - back.predict(X_test))))




# plt.plot(range(1, len( delta.mean_squared_errors)+1), delta.mean_squared_errors, marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('Sum-squared-error for delta')
# plt.show()
# #
# plt.plot(range(1, len( back.mean_squared_errors)+1), back.mean_squared_errors, marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('Sum-squared-error for back')
# plt.show()
