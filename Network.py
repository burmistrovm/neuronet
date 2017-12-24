import numpy as np
from Layer import Layer


class Network(object):
    # network has any layers and manages fiting and predicting

    def __init__(self, input_dim, eta = 0.1):
        self._input_dim = input_dim
        self._output_dim = 0
        self._num_layers = 0
        self._layers = np.array([])
        self._learning_rate = eta # must be dynamical

    def add_layer(self, units, activation = 'logistic'):
        if self._layers.size == 0:
            new_layer = Layer(input_dim = self._input_dim, units = units, activation = activation)
        else:
            new_layer_dim = self._layers[self._layers.size - 1].units
            new_layer = Layer(input_dim = new_layer_dim, units = units, activation = activation)
        self._num_layers += 1
        self._output_dim = units
        self._layers = np.append(self._layers, new_layer)

    def delete_last_layer(self):
        self._num_layers -= 1
        self._layers = np.delete(self._layers, self._layers.size - 1, 0)
        
    def predict(self, X):
        for layer in self._layers:
            X = layer.predict(X)
        return X

    def fit(self, X, target, n_epoch = 1):
        if target.shape[1] != self._output_dim:
            raise ValueError("Size of outer layer must be equal to size of target vector")

        obj_num = X.shape[0]
        #obj_size = X.shape[1]
        for i in range(0, obj_num ):
            current_object = X[i]
            inputs = []
            outputs = []
            for layer in self._layers: # Forward propagation
                current_object = layer.predict(current_object)
                outputs.append(current_object)

            errors = target[i] - outputs[-1]
            j = len(self._layers) - 1
            for layer in self._layers[::-1]: # Backprop
                if (j == len(self._layers) - 1):
                    errors = layer.correct_weights(errors, outputs[j], outputs[j - 1])
                    print(errors)
                    j -= 1