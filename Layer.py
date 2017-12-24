import numpy as np
#Class for fully connected layer
class Layer:
    #_weigt is 2 dimensional array of weight
    #_weigt[i][j] is weight between j input and i neuron of layer

    def __init__(self, input_dim, units, eta = 0.01):
        self.input_dim = input_dim
        self.units = units
        self._weights = np.random.uniform(-1, 1, [units, input_dim + 1]) # bias is _weights[i][0]
        self._eta = eta #eta must be computed in learning loop

    def correct_weights(self, errors, last_predict, last_input):
        dsigma = last_predict * (1 - last_predict)
        dw =  errors * dsigma
        last_layer_errors = dw.dot(self._weights)
        last_input = np.insert(last_input, 0, 1)
        for i in range(len(dw)):
            self._weights[i] = self._weights[i] + last_input * dw[i] * self._eta
        return last_layer_errors
        
    def predict(self, X):
        if len(X.shape) == 1:
            X = np.insert(X, 0, 1) # first element will be multuplied on bias
        else:
            X = np.insert(X, 0, 1, axis = 1)
        z = self._weights.dot(X.T).T # rows of weights matrix are neurons, cols of X - features
        prediction = 1 / (1 + np.exp(-z))
        return prediction