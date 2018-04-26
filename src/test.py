import numpy
import constants
from DataProvider import DataProvider
from NetworkWrapper import NetworkWrapper
from neupy import algorithms
import os
import time

class TestExecutor:
    def __init__(self):
        self.header = 'Lp.;iteracja;Nazwa algorytmu;Liczba epok;Wspolczynnik uczenia;Liczba warstw ukrytych;Czas uczenia [ms];Poprawnie sklasyfikowane; Niepoprawnie sklasyfikowane; Poprawnie sklasyfikowane [%]'
        self.path_to_file = constants.EXPERIMENTS_DIR + 'testy.csv'
        self.lp = 0
        # database informations
        self.data_inf = constants.DATABASE_INFORMATIONS['breast_cancer']
        self.csv_url = self.data_inf['url']
        self.features_column = self.data_inf['features_column']
        self.class_column = self.data_inf['class_column']
        # program parameters const
        self.test_size = 0.3
        self.hidden_units_number = 3

    def __save_result_to_file(self, text):
        with open(self.path_to_file, "a+") as file:
            file.write(text)

    def __execute_algorithm(self, algorithm_name, x_train, y_train, x_test, y_test, hidden_layers_number, learning_rate, epochs_number):
        input_layer = [x_train.shape[1]]
        hidden_layers = [
            self.hidden_units_number for _ in range(hidden_layers_number)]
        output_layer = [y_train.shape[1]]
        network_architecture = tuple(
            sum([input_layer, hidden_layers, output_layer], []))

        neuralNetwork = NetworkWrapper(
            algorithm_name, network_architecture, step=learning_rate, shuffle_data=False)
        time_start = time.process_time()
        neuralNetwork.train(x_train, y_train, epochs_number)
        elapsed_time = round((time.process_time() - time_start) * 1000, 2)
        results = neuralNetwork.predict(x_test)

        errors_number = neuralNetwork.get_number_of_errors(results, y_test)
        corrects_number = neuralNetwork.get_number_of_corrects(results, y_test)
        percent_correct = round(
            corrects_number * 100 / (corrects_number + errors_number), 2)
        self.lp += 1
        new_result = '\n{0};{1};{2};{3};{4};{5};{6};{7};{8};{9}'.format(
            self.lp, self.iteration, algorithm_name, epochs_number, learning_rate, hidden_layers_number, elapsed_time, corrects_number, errors_number, percent_correct)

        self.__save_result_to_file(new_result)
        print(new_result)

    def execute_test(self):
        self.__save_result_to_file(self.header)

        for self.iteration in range(1, 5):
            data_provider = DataProvider(
                self.csv_url, self.features_column, self.class_column, self.test_size, 42)
            x_train, y_train, x_test, y_test = data_provider.get_whole_data()

            for algorithm_data in constants.ALGORITHMS_DATA_TO_TEST:
                for epochs_number in algorithm_data['epochs_numbers']:
                    for hidden_layers_number in algorithm_data['hidden_layers_numbers']:
                        if not 'learning_rates' in algorithm_data:
                            self.__execute_algorithm(
                                algorithm_data['name'], x_train, y_train, x_test, y_test, hidden_layers_number, '-', epochs_number)
                        else:
                            for learning_rate in algorithm_data['learning_rates']:
                                self.__execute_algorithm(
                                    algorithm_data['name'], x_train, y_train, x_test, y_test, hidden_layers_number, learning_rate, epochs_number)


testExecutor = TestExecutor()
testExecutor.execute_test()
