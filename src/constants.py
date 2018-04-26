import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiment_results/')
ALGORITHMS = {
    'GradientDescent': {
        'required_parameters': ['step', 'shuffle_data']
    },
    'ConjugateGradient': {
        'required_parameters': ['step', 'shuffle_data']
    },
    'LevenbergMarquardt': {
        'required_parameters': ['shuffle_data', 'mu_update_factor', 'mu']
    },
    'QuasiNewton': {
        'required_parameters': ['shuffle_data']
    }
}

DATABASE_INFORMATIONS = {
    'iris': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        'features_column': list(range(4)),
        'class_column': 4
    },
    'breast_cancer': {
        'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        # 'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
        'features_column': list(range(1, 10)),
        # 'features_column': list(range(2,32)),
        'class_column': 10
        # 'class_column': 1
    }
}

START_MESSAGE = """
program executed with params:
    algorithm_name: {0},
    test_size: {1},
    learning_rate: {2},
    epochs_number: {3},
    hidden_units_number: {4}
    hidden_layers_number: {5}
"""

HELP_INSTRUCTIONS = """ 
Program to test supervised learning in neural networks. User can do tests based on database "breast-cancer-wisconsin", 
which is taken from ics.uci repository. User can manipulate paramateres, which are described below.

options:

    -h | --help - display help

    -a | --alghorithm_name <alghorithm_name> - learning algorithm (possible: GradientDescent, ConjugateGradient, LevenbergMarquardt), default: LevenbergMarquardt

    -s | --test_size <test_size> - the percentage size of the test set (range from 0.1 to 0.9), default: 0.4

    -l | --learning_rate <learning_rate> - learning rate (range from 0.01 to 0.99), default: 0.1

    -e | --epochs_number <epochs_number> - epochs number, must be integer greater than 0, default: 100
    
    -u | --hidden_units_number <hidden_units_number> - hidden units number in neural network, must be integer greater than 0, default: 3
    
    -i | --hidden_layers_number <hidden_layers_number> - hidden layers number in neural network, must be integer greater than 0, default: 1

Author:
    Written by Slawomir Zaba"""


ALGORITHMS_DATA_TO_TEST = [{
    'name': 'QuasiNewton',
    'epochs_numbers': [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    'hidden_layers_numbers': [1, 3, 5]
}, {
    'name': 'GradientDescent',
    'learning_rates': [0.01, 0.2, 0.4, 0.6, 0.9],
    'epochs_numbers': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    'hidden_layers_numbers': [1, 3, 5]
}, {
    'name': 'ConjugateGradient',
    'learning_rates': [0.01, 0.2, 0.4, 0.6, 0.9],
    'epochs_numbers': [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    'hidden_layers_numbers': [1, 3, 5]
}]
