import numpy as np

STUDENTY_NUMBER = 4645251

class Perceptron:
    def __init__(self, input_dim, n_classes, weight_mult=1):
        self.W = weight_mult * np.random.randn(input_dim, n_classes)  # the weights matrix

    def forward(self, X):  # X needs to have shape (batch size, input_dim + 1) where X[i,input_dim + 1] = 1 always
        Y_pred = X @ self.W
        return Y_pred