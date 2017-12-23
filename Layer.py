import numpy as np
#Class for fully connected layer
class Layer:
    #_weigt is 2 dimensional array of weight
    #_weigt[i][j] is weight between j input and i neuron of layer

    def __init__(self, input_dim, units, activation='logistic', eta = 0.1):
        self._weights = np.random.uniform(-1, 1, [2, 3]) # bias is _weights[i][0]
        self._eta = eta #eta must be computed in learning loop

    # def fit(self, X, target, n_epoch = 1):
    #     obj_num = X.shape[0]
    #     obj_size = X.shape[1]
    #     X = np.insert(X, 0, 1, axis = 1)
    #     for i in range(0, obj_num): # must be n_epoch iterations
    #         z = self._weights[i].dot(X[i])
    #         prediction = 1 / (1 + np.exp(-z))
    #         self._weights = self._weights + self._eta * (target[i] - prediction) * X[i]

    def predict(self, X, target):
        obj_num = X.shape[0]
        obj_size = X.shape[1]
        
        X = np.insert(X, 0, 1, axis = 1)
        z = self._weights[i].dot(X[i])
        prediction = 1 / (1 + np.exp(-z))
        return prediction