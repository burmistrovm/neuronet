import numpy as np
from Layer import Layer


class Network(object):
    # network has any layers and manages fiting and predicting

    def __init__(self, input_dim, eta = 0.1):
        self._input_dim = input_dim
        self._layers = np.array([])
        self._learning_rate = eta # must be dynamical

    def add_layer(self, units, activation = 'logistic'):
        self._layers = np.array([])
        if self._layers.size == 0:
            new_layer = Layer(input_dim = self._input_dim, units = units, activation = activation)
        else:
            new_layer_dim = self._layers[self._layers.size - 1].input_dim
            new_layer = Layer(input_dim = new_layer_dim, units = units, activation = activation)

        self._layers = np.append(self._layers, new_layer)

    def delete_last_layer(self):
        self._layers = np.delete(self._layers, self._layers.size - 1, 0)
        
    def predict(self, X):
        for layer in self._layers:
            X = layer.predict(X)
        return X