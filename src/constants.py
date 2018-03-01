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
    'QuasiNewton': {
        'required_parameters': []
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
