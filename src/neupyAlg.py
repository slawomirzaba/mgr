import numpy
import constants
from DataProvider import DataProvider
from NetworkWrapper import NetworkWrapper

data_inf = constants.DATABASE_INFORMATIONS['breast_cancer']
csv_url = data_inf['url']
features_column = data_inf['features_column']
class_column = data_inf['class_column']
data_provider = DataProvider(csv_url, features_column, class_column, test_size=0.2)
x_train, y_train, x_test, y_test = data_provider.get_whole_data()

# ---------------------------------------------------------------------------------------------------

network_architecture = (x_train.shape[1], 3, y_train.shape[1])
neuralNetwork = NetworkWrapper('QuasiNewton', network_architecture, step=0.1)
neuralNetwork.train(x_train, y_train, 100)
results = neuralNetwork.predict(x_test)

# print('results', numpy.argmax(results, axis=1))
# print('correct', numpy.argmax(y_test, axis=1))

# errors = numpy.argmax(y_test, axis=1)-numpy.argmax(results, axis=1)
# errors_number = numpy.count_nonzero(errors)

errors_number = neuralNetwork.get_number_of_errors(results, y_test)
corrects_number = neuralNetwork.get_number_of_corrects(results, y_test)
print (corrects_number, errors_number)

# neuralNetwork.plot_errors()
# neuralNetwork.plot_structure()