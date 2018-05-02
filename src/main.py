import numpy
import constants
import time
from DataProvider import DataProvider
from NetworkWrapper import NetworkWrapper
from ParametersExecutorProvider import ParametersExecutorProvider
from neupy import algorithms

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

time_start = time.process_time()
neuralNetwork = NetworkWrapper(
    algorithm_name, 
    network_architecture, 
    step=learning_rate, 
    shuffle_data=False,
    mu_update_factor=2,
    mu=0.1)
neuralNetwork.train(x_train, y_train, epochs_number)
elapsed_time = round((time.process_time() - time_start) * 1000, 2)
results = neuralNetwork.predict(x_test)

# -------------------------------------------------------------------------------------------
# results

errors_number = neuralNetwork.get_number_of_errors(results, y_test)
corrects_number = neuralNetwork.get_number_of_corrects(results, y_test)
percent_correct = round(corrects_number * 100 / (corrects_number + errors_number), 2)
confusion_matrix = neuralNetwork.get_confusion_matrix(results, y_test)
TP, FP = confusion_matrix[0][0], confusion_matrix[0][1]
FN, TN = confusion_matrix[1][0], confusion_matrix[1][1]
sensitivity = round(TP / (TP + FN), 2)
specifity = round(TN / (TN + FP), 2)
precision = round(TP / (TP + FP), 2)

print("correctly classified:", corrects_number)
print("wrongly classified:", errors_number)
print("percent correct:", percent_correct)
print("confusion matrix:\n", confusion_matrix)
print("sensitivity:", sensitivity)
print("specifity:", specifity)
print("precision:", precision)
print("learning time:", elapsed_time)

# neuralNetwork.plot_structure()
neuralNetwork.plot_errors()
