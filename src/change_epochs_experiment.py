import os
import numpy
import constants
from DataProvider import DataProvider
from NetworkWrapper import NetworkWrapper


class ChangeEpochsExperiment:
    def __init__(self, data_base_name='breast_cancer', step=0.1, test_size=0.4, iterations_per_size=3):
        # database informations
        data_inf = constants.DATABASE_INFORMATIONS[data_base_name]
        self.csv_url = data_inf['url']
        self.features_column = data_inf['features_column']
        self.class_column = data_inf['class_column']

        # neural network informations
        self.step = 0.1
        self.test_size = test_size

        # test informations
        self.file_name = 'change_epochs_experiment.csv'
        self.iterations_range = range(1, iterations_per_size + 1)

    def execute_test(self):
        path_to_file = constants.EXPERIMENTS_DIR + self.file_name
        epochs_range = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 12000, 15000, 20000, 50000, 100000]
        csv_results = self.__prepare_csv_header()

        for epochs in epochs_range:
            csv_results += '\n{0}'.format(round(epochs, 2))
            percent_corrects_array = []
            for _ in self.iterations_range:
                correct_number, errors_number, correct_percent = self.__get_result_of_change_test_size_experiment(epochs)
                csv_results += ',{0},{1},{2}'.format(correct_number, errors_number, correct_percent)
                percent_corrects_array.append(correct_percent)
            average = round(numpy.average(percent_corrects_array), 2)
            csv_results += ',{0}'.format(average)

        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        with open(path_to_file, "w") as file:
            file.write(csv_results)

    def __prepare_csv_header(self):
        csv_header = ''

        for iteration in self.iterations_range:
            csv_header += ',{0} iteration,,'.format(iteration)

        csv_header += ',\nepochs'

        for iteration in self.iterations_range:
            csv_header += ',correct,non correct,percent correct'

        csv_header += ',percent_average'
        return csv_header

    def __get_result_of_change_test_size_experiment(self, epochs):
        data_provider = DataProvider(self.csv_url, self.features_column, self.class_column, test_size=self.test_size)
        x_train, y_train, x_test, y_test = data_provider.get_whole_data()
        network_architecture = (x_train.shape[1], 3, y_train.shape[1])
        neuralNetwork = NetworkWrapper('GradientDescent', network_architecture, step=self.step)
        neuralNetwork.train(x_train, y_train, epochs)
        predictions = neuralNetwork.predict(x_test)
        errors_number = neuralNetwork.get_number_of_errors(predictions, y_test)
        correct_number = neuralNetwork.get_number_of_corrects(predictions, y_test)
        correct_percent = correct_number * 100 / len(y_test)
        return correct_number, errors_number, round(correct_percent, 2)


first_test = ChangeEpochsExperiment()
first_test.execute_test()
