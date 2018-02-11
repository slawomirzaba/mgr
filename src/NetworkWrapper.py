from neupy import algorithms
import copy
import constants
import importlib

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
        if not self.network:
            raise Exception('first network must be initialised!')

        self.network.train(x_train, y_train, epochs=epochs)

    def predict(self, x_test):
        if not self.network:
            raise Exception('first network must be initialised!')

        return self.network.predict(x_test)

    def __get_initialise_correct_parameters(self, required_parameters, passed_parameters):
        correct_parameters = copy.deepcopy(passed_parameters)
        for parameter_name, value in passed_parameters.items():
            if parameter_name not in required_parameters:
                del correct_parameters[parameter_name]

        return correct_parameters
