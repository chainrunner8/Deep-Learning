import numpy as np
from perceptron import Perceptron
from utils import *


n_epochs      = 100
n_classes     = 10
learning_rate = 0.001
n_trials      = 20
test_freq     = 5

X_train = np.loadtxt('train_in.csv', delimiter=',')
y_train = np.loadtxt('train_out.csv', delimiter=',', dtype=int)
X_test  = np.loadtxt('test_in.csv', delimiter=',')
y_test  = np.loadtxt('test_out.csv', delimiter=',', dtype=int)
Y_train = one_hot_encode(y_train, n_classes)

n_train = len(y_train)
n_test  = len(y_test)
X_train = np.hstack([X_train, np.ones((n_train,1))])
X_test  = np.hstack([X_test, np.ones((n_test, 1))])

perceptron            = Perceptron(input_dim=X_train.shape[1], n_classes=n_classes)
train_accuracy_matrix = np.zeros((n_trials, n_epochs))
test_accuracy_matrix  = np.zeros((n_trials, n_epochs//test_freq))
loss_matrix           = np.zeros((n_trials, n_epochs))


# training:
for trial in range(n_trials):
    for epoch in range(n_epochs):
        # weight update:
        Y_pred = perceptron.forward(X_train)
        Y_pred_softmax = softmax(Y_pred)
        # X_train is (batch size, input_dim + 1), weight matrix is (input_dim + 1, n_classes)
        # and Y_pred & Y_train are (batch size, n_classes), so we need to transpose X:
        grad_W = 2 * X_train.T @ (Y_pred_softmax - Y_train)
        perceptron.W -= learning_rate * grad_W
        loss = -np.sum(Y_train * np.log(Y_pred_softmax))  # the cross-entropy loss
        loss_matrix[trial, epoch] = loss
        # accuracy:
        train_accuracy_matrix[trial, epoch] = accuracy(Y_pred=Y_pred, y_true=y_train)
        print(f'Epoch {epoch} complete.')

        if (epoch+1)%test_freq==0:
            Y_pred = perceptron.forward(X_test)
            test_accuracy = accuracy(Y_pred=Y_pred, y_true=y_test)
            test_accuracy_matrix[trial, epoch//test_freq - 1] = test_accuracy
            print(test_accuracy_matrix)
            print('test')

train_acc_avg, train_acc_min, train_acc_max = process_learning_curve(train_accuracy_matrix)
train_loss_avg, train_loss_min, train_loss_max = process_learning_curve(loss_matrix)
test_acc_avg, test_acc_min, test_acc_max = process_learning_curve(test_accuracy_matrix)

accuracy_curves = [
    {
        'mean': {'x': np.arange(n_epochs) + 1, 'y': train_acc_avg}
        , 'min': train_acc_min
        , 'max': train_acc_max
        , 'label': 'training accuracy'
        , 'c': 'blue'
        }
    , {
        'mean': {'x': np.arange(test_freq, n_epochs + test_freq, test_freq), 'y': test_acc_avg}
        , 'min': test_acc_min
        , 'max': test_acc_max
        , 'label': 'test accuracy'
        , 'c': 'cyan'
        }
]
loss_curve = {
    'mean': {'x': np.arange(n_epochs) + 1, 'y': train_loss_avg}
    , 'min': train_loss_min
    , 'max': train_loss_max
    , 'label': 'training loss'
    , 'c': 'orange'
    }

# plotting:
plt.figure(figsize=(12,8))
for curve in accuracy_curves:
    plt.plot(curve['mean']['x'], curve['mean']['y'], c=curve['c'], marker='o', linewidth=2, ms=2, label=curve['label'])
    plt.fill_between(curve['mean']['x'], curve['min'], curve['max'], alpha=0.3, label=curve['label'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Training & test accuracy over {n_epochs} epochs')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(loss_curve['mean']['x'], loss_curve['mean']['y'], c=loss_curve['c'], marker='o', linewidth=2, ms=2)
plt.fill_between(loss_curve['mean']['x'], loss_curve['min'], loss_curve['max'], alpha=0.3)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title(f'Training loss over {n_epochs} epochs')
plt.grid(True)
plt.show()

