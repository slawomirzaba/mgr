import numpy
from DataProvider import DataProvider
from NetworkWrapper import NetworkWrapper
import constants

csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_of_features = [0,1,2,3]
column_of_class = 4
data_provider = DataProvider(csv_url, column_of_features, column_of_class)
x_train, y_train, x_test, y_test = data_provider.get_whole_data()

# ---------------------------------------------------------------------------------------------------

network_architecture = (x_train.shape[1], 3, y_train.shape[1])
GradientDescentNetwork = NetworkWrapper('GradientDescent', network_architecture, step=0.1)
GradientDescentNetwork.train(x_train, y_train, 10000)
results = GradientDescentNetwork.predict(x_test)

print('results', numpy.argmax(results, axis=1))
print('correct', numpy.argmax(y_test, axis=1))

errors = numpy.argmax(y_test, axis=1)-numpy.argmax(results, axis=1)
errors_number = numpy.count_nonzero(errors)

print (errors_number)
