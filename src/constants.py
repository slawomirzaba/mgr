import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiment_results/')
ALGORITHMS = {
    'GradientDescent': {
        'required_parameters': ['step']
    },
    'ConjugateGradient': {
        'required_parameters': ['step']
    },
    'LevenbergMarquardt': {
        'required_parameters': []
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
        'features_column': list(range(1,10)),
        'class_column': 10
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
main - program to test supervised learning in neural networks

options:
    -h | --help - display help

    -a | --alghorihm_name <alghorihm_name> - learning algorithm (possible: GradientDescent, ConjugateGradient, LevenbergMarquardt), default: LevenbergMarquardt

    -s | --test_size <test_size> - the percentage size of the test set (range from 0.1 to 0.9), default: 0.4

    -l | --learning_rate <learning_rate> - learning rate (range from 0.01 to 0.99), default: 0.1

    -e | --epochs_number <epochs_number> - epochs number, must be integer greater than 0, default: 100
    
    -u | --hidden_units_number <hidden_units_number> - hidden units number in neural network, must be integer greater than 0, default: 3
    
    -i | --hidden_layers_number <hidden_layers_number> - hidden layers number in neural network, must be integer greater than 0, default: 1

Author:
    Written by Slawomir Zaba"""
