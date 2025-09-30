import numpy as np


def one_hot_encode(y, n_classes):
    row_idx = np.arange(len(y))
    one_hot_y = np.zeros((len(y), n_classes))
    one_hot_y[row_idx, y] = 1
    return one_hot_y

def softmax(Z):
    return (np.exp(Z).T / np.sum(np.exp(Z), axis=1)).T

def accuracy(Y_pred, y_true):
    class_pred = np.argmax(Y_pred, axis=1)
    accuracy = np.mean(class_pred==y_true)
    return accuracy

def process_learning_curve(matrix):
    avg = np.mean(matrix, axis=0)
    max = np.max(matrix, axis=0)
    min = np.min(matrix, axis=0)
    return avg, min, max
