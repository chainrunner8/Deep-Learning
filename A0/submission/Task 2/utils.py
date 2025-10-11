import numpy as np
import argparse


def one_hot_encode(y, n_classes):
    row_idx = np.arange(len(y))
    one_hot_y = np.zeros((len(y), n_classes))
    one_hot_y[row_idx, y] = 1
    return one_hot_y

def softmax(Z):
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)  # this shift is to prevent the exp from overflowing
    return (np.exp(Z_shift).T / np.sum(np.exp(Z_shift), axis=1)).T

def minmax(Z):
    return ((Z.T - np.min(Z, axis=1)) / (np.max(Z, axis=1) - np.min(Z, axis=1))).T

def accuracy(Y_pred, y_true):
    class_pred = np.argmax(Y_pred, axis=1)
    accuracy = np.mean(class_pred==y_true)
    return accuracy

def process_learning_curve(matrix):
    avg = np.mean(matrix, axis=0)
    max = np.max(matrix, axis=0)
    min = np.min(matrix, axis=0)
    return avg, min, max

def parse_args():
    parser = argparse.ArgumentParser(description='Train perceptron on MNIST')
    parser.add_argument('--experiment', type=int, default=1, help='1 to run the 100 epoch - 20 run experiment one time; 2 for the learning rate sensitivity analyis.')
    parser.add_argument('--lr', type=float, default=0.0001, help='if experiment=1')
    parser.add_argument('--weight_mult', type=float, default=0.01, help='if experiment=1')
    return parser.parse_args()