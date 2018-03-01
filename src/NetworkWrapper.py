from neupy import algorithms, plots
import copy
import constants
import importlib
import numpy

class NetworkWrapper:
    def __init__(self, algorithm_name, networks_architecture, **kwargs):
        if algorithm_name not in constants.ALGORITHMS.keys():
            error_message = 'Alghoritm {0} is not supported!'.format(algorithm_name)
            raise Exception(error_message)

        required_parameters = constants.ALGORITHMS[algorithm_name]['required_parameters']
        correct_parameters = self.__get_initialise_correct_parameters(required_parameters, kwargs)
        algorithm = getattr(algorithms, algorithm_name)
        self.network = algorithm(networks_architecture, verbose=False, **correct_parameters)

    def train(self, x_train, y_train, epochs=100):
        self.__network_checker()

        self.network.train(x_train, y_train, epochs=epochs)

    def predict(self, x_test):
        self.__network_checker()

        return self.network.predict(x_test)
    
    def get_number_of_errors(self, predictions, y_test):
        errors_array = numpy.argmax(y_test, axis=1) - numpy.argmax(predictions, axis=1)
        return numpy.count_nonzero(errors_array)
    
    def get_number_of_corrects(self, predictions, y_test):
        errors_number = self.get_number_of_errors(predictions, y_test)
        return len(y_test) - errors_number

    def plot_errors(self):
        self.__network_checker()

        plots.error_plot(self.network)

    def plot_structure(self):
        self.__network_checker()

        plots.layer_structure(self.network)

    def __get_initialise_correct_parameters(self, required_parameters, passed_parameters):
        correct_parameters = copy.deepcopy(passed_parameters)
        for parameter_name, _ in passed_parameters.items():
            if parameter_name not in required_parameters:
                del correct_parameters[parameter_name]

        return correct_parameters

    def __network_checker(self):
        if not self.network:
            raise Exception('first network must be initialised!')
