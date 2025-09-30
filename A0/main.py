import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from utils import *


N_EPOCHS  = 100
N_CLASSES = 10
N_TRIALS  = 20
TEST_FREQ = 5

X_train = np.loadtxt('train_in.csv', delimiter=',')
y_train = np.loadtxt('train_out.csv', delimiter=',', dtype=int)
X_test  = np.loadtxt('test_in.csv', delimiter=',')
y_test  = np.loadtxt('test_out.csv', delimiter=',', dtype=int)
Y_train = one_hot_encode(y_train, N_CLASSES)

n_train = len(y_train)
n_test  = len(y_test)
X_train = np.hstack([X_train, np.ones((n_train,1))])
X_test  = np.hstack([X_test, np.ones((n_test, 1))])


# training:
def train_perceptron(learning_rate, weight_init_mult):
    train_accuracy_matrix = np.zeros((N_TRIALS, N_EPOCHS))
    test_accuracy_matrix  = np.zeros((N_TRIALS, N_EPOCHS//TEST_FREQ))
    loss_matrix           = np.zeros((N_TRIALS, N_EPOCHS))
    for trial in range(N_TRIALS):
        np.random.seed(trial)
        perceptron = Perceptron(input_dim=X_train.shape[1], n_classes=N_CLASSES, weight_mult=weight_init_mult)
        for epoch in range(N_EPOCHS):
            # weight update:
            Y_pred = perceptron.forward(X_train)
            Y_pred_softmax = softmax(Y_pred)
            # X_train is (batch size, input_dim + 1), weight matrix is (input_dim + 1, n_classes)
            # and Y_pred & Y_train are (batch size, n_classes), so we need to transpose X:
            grad_W = 2 * X_train.T @ (Y_pred_softmax - Y_train)
            perceptron.W -= learning_rate * grad_W
            loss = np.mean((Y_pred_softmax - Y_train)**2)  # the cross-entropy loss
            loss_matrix[trial, epoch] = loss
            # accuracy:
            train_accuracy_matrix[trial, epoch] = accuracy(Y_pred=Y_pred, y_true=y_train)
            print(f'Epoch {epoch} complete.')

            if (epoch+1)%TEST_FREQ==0:
                Y_pred = perceptron.forward(X_test)
                test_accuracy = accuracy(Y_pred=Y_pred, y_true=y_test)
                test_accuracy_matrix[trial, (epoch+1)//TEST_FREQ - 1] = test_accuracy
    
    return train_accuracy_matrix, test_accuracy_matrix, loss_matrix


def first_experiment(lr, weight_init_mult):
    train_accuracy_matrix, test_accuracy_matrix, loss_matrix = train_perceptron(lr, weight_init_mult)

    train_acc_avg, train_acc_min, train_acc_max = process_learning_curve(train_accuracy_matrix)
    train_loss_avg, train_loss_min, train_loss_max = process_learning_curve(loss_matrix)
    test_acc_avg, test_acc_min, test_acc_max = process_learning_curve(test_accuracy_matrix)

    accuracy_curves = [
        {
            'mean': {'x': np.arange(N_EPOCHS) + 1, 'y': train_acc_avg}
            , 'min': train_acc_min
            , 'max': train_acc_max
            , 'label': 'training accuracy'
            , 'c': 'blue'
            }
        , {
            'mean': {'x': np.arange(TEST_FREQ, N_EPOCHS + TEST_FREQ, TEST_FREQ), 'y': test_acc_avg}
            , 'min': test_acc_min
            , 'max': test_acc_max
            , 'label': 'test accuracy'
            , 'c': 'cyan'
            }
    ]
    loss_curve = {
        'mean': {'x': np.arange(N_EPOCHS) + 1, 'y': train_loss_avg}
        , 'min': train_loss_min
        , 'max': train_loss_max
        , 'label': 'training loss'
        , 'c': 'orange'
        }

    print('training acc mean: ', np.mean(train_accuracy_matrix[:,-1])
          , '\n', 'std: ', np.std(train_accuracy_matrix[:,-1]))
    print('test acc mean: ', np.mean(test_accuracy_matrix[:,-1])
          , '\n', 'std: ', np.std(test_accuracy_matrix[:,-1]))

    # plotting:
    plt.figure(figsize=(12,8))
    for curve in accuracy_curves:
        plt.plot(curve['mean']['x'], curve['mean']['y'], c=curve['c'], marker='o', linewidth=2, ms=2, label=curve['label'])
        plt.fill_between(curve['mean']['x'], curve['min'], curve['max'], alpha=0.3, label='min-max ' + curve['label'].split('accuracy')[0])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training & test accuracy over {N_EPOCHS} epochs (learning rate = {lr})')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

    plt.figure(figsize=(12,8))
    plt.plot(loss_curve['mean']['x'], loss_curve['mean']['y'], c=loss_curve['c'], marker='o', linewidth=2, ms=2, label='training loss')
    plt.fill_between(loss_curve['mean']['x'], loss_curve['min'], loss_curve['max'], alpha=0.3, label='min-max')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Training loss')
    plt.title(f'Training loss over {N_EPOCHS} epochs (learning rate = {lr})')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show()


def lr_sensitivity_analysis(learning_rates):
    train_acc_mean, train_acc_std, test_acc_mean, test_acc_std = [], [], [], []
    for lr in learning_rates:
        train_acc_matrix, test_acc_matrix, _ = train_perceptron(learning_rate=lr)
        train_acc_mean.append(np.mean(train_acc_matrix[:, -1]))
        train_acc_std.append(np.std(train_acc_matrix[:, -1]))
        test_acc_mean.append(np.mean(test_acc_matrix[:, -1]))
        test_acc_std.append(np.std(test_acc_matrix[:, -1]))
    # plotting:
    x = np.arange(len(learning_rates))
    plt.figure(figsize=(12,8))
    plt.errorbar(x, train_acc_mean, yerr=train_acc_std, fmt='o', ms=3, color='blue', ecolor='blue', capsize=5, label='Train')
    plt.errorbar(x, test_acc_mean, yerr=test_acc_std, fmt='o', ms=3, color='cyan', ecolor='cyan', capsize=5, label='Test')
    for i in range(len(x)):
        plt.text(x[i], train_acc_mean[i]+0.01, f"{train_acc_mean[i]:.3f}", 
                ha='center', color='blue')
        plt.text(x[i], test_acc_mean[i]+0.01, f"{test_acc_mean[i]:.3f}", 
                ha='center', color='cyan')
    plt.xlabel('Learning rate')
    plt.ylabel('Accuracy')
    plt.title('Training & test accuracy for various learning rates (avg over 20 runs)')
    plt.xticks(x, learning_rates)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    if args.experiment == 1:
        first_experiment(lr=args.lr, weight_init_mult=args.weight_mult)
    elif args.experiment == 2:
        learning_rates = [0.0001, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.01]
        lr_sensitivity_analysis(learning_rates)
    