import numpy
import constants
from DataProvider import DataProvider
from NetworkWrapper import NetworkWrapper
from ParametersExecutorProvider import ParametersExecutorProvider

# program parameters
program_parameters = ParametersExecutorProvider().get_parameters()
algorithm_name = program_parameters['algorithm_name']
test_size = program_parameters['test_size']
learning_rate = program_parameters['learning_rate']
epochs_number = program_parameters['epochs_number']
hidden_units_number = program_parameters['hidden_units_number']
hidden_layers_number = program_parameters['hidden_layers_number']

print(constants.START_MESSAGE.format(algorithm_name, test_size, learning_rate, epochs_number,
                                     hidden_units_number, hidden_layers_number))

# database informations
data_inf = constants.DATABASE_INFORMATIONS['breast_cancer']
csv_url = data_inf['url']
features_column = data_inf['features_column']
class_column = data_inf['class_column']

# preparing data
data_provider = DataProvider(
    csv_url, features_column, class_column, test_size=test_size)
x_train, y_train, x_test, y_test = data_provider.get_whole_data()

# ---------------------------------------------------------------------------------------------------

# creating architecture, train and test
input_layer = [x_train.shape[1]]
hidden_layers = [hidden_units_number for _ in range(hidden_layers_number)]
output_layer = [y_train.shape[1]]
network_architecture = tuple(
    sum([input_layer, hidden_layers, output_layer], []))

neuralNetwork = NetworkWrapper(
    algorithm_name, network_architecture, step=learning_rate)
neuralNetwork.train(x_train, y_train, epochs_number)
results = neuralNetwork.predict(x_test)

# results
errors_number = neuralNetwork.get_number_of_errors(results, y_test)
corrects_number = neuralNetwork.get_number_of_corrects(results, y_test)
print("correctly classified:", corrects_number)
print("wrongly classified:", errors_number)

neuralNetwork.plot_structure()
neuralNetwork.plot_errors()
