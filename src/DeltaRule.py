import numpy as np
#delta rule:
# δ = z - y / errors = y - output
# w' = w + learnig_rate * δ * x
# chodzi o minimalizacje: suma(δ^2)/2

class DeltaRule:
    def __init__(self, epochs=20, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self, x, y):
        self.weights = np.random.randn(x.shape[1])
        self.mean_squared_errors = []

        for i in range(self.epochs):
            output = self.net_input(x)
            errors = y - output
            self.weights += self.learning_rate * x.T.dot(errors)
            mean_squared_error = np.mean(np.square(errors))
            self.mean_squared_errors.append(mean_squared_error)
        return self

    def predict(self, x):
        return 1 / (1 + np.exp(-1 * self.net_input(x)))

    def net_input(self, x):
        return np.dot(x, self.weights)
