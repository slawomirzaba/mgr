import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ALGORITHMS = {
    'GradientDescent': {
        'required_parameters': ['step']
    },
    'QuasiNewton': {
        'required_parameters': []
    }
}
