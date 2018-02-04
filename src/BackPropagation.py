import numpy as np

class BackPropagation:
    def __init__(self, epochs=100, learning_rate=0.01, hidden_size=3):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.first_layer_output = None
        self.input_weights = None
        self.output_weights = None,
        self.mean_squared_errors = []

    def predict(self, x):
        self.first_layer_output = self.sigmoid(np.dot(x, self.input_weights))
        return self.sigmoid(np.dot(self.first_layer_output, self.output_weights))

    def backward(self, x, y, output):
        output_error = reshape_to_2d(y) - reshape_to_2d(output)
        delta_for_output_weights = output_error * self.sigmoidPrime(reshape_to_2d(output))

        hidden_layer_error = delta_for_output_weights.dot(reshape_to_2d(self.output_weights).T)
        delta_for_input_weights = hidden_layer_error * self.sigmoidPrime(self.first_layer_output)

        self.input_weights += self.learning_rate * x.T.dot(delta_for_input_weights)
        self.output_weights += reshape_to_1d(self.learning_rate * self.first_layer_output.T.dot(delta_for_output_weights))

    def train (self, x, y):
        self.input_weights = np.random.randn(x.shape[1], self.hidden_size)
        self.output_weights = np.random.randn(self.hidden_size)
        self.mean_squared_errors = []

        for i in range(self.epochs):
            output = self.predict(x)
            mean_squared_error = np.mean(np.square(y - output))
            self.mean_squared_errors.append(mean_squared_error)
            self.backward(x, y, output)

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        return self.sigmoid(s) * (1 - self.sigmoid(s))

def reshape_to_2d(vector):
    if len(vector.shape) > 1: return vector

    return vector.reshape(vector.shape[0], 1)

def reshape_to_1d(vector):
    return vector.reshape(vector.shape[0])
