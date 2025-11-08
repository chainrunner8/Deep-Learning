# MAIN.PY

from cnn_class import OneHeadedCNN, TwoHeadedCNN
from utils import *
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


STUDENT_NUMBER = 4645251
N_EPOCHS = 50
SMOOTH_WINDOW = 5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# step 1: load data
images = np.load('A1_data_150/images.npy')
labels = np.load('A1_data_150/labels.npy')

N = len(labels)
n_train = int(0.8*N)
n_val = n_test = int(0.1*N)
batch_size_train = 128
log_freq = n_train // batch_size_train // 5

# shuffle data & train-val-test split:
index = TrainValTestIndex(n_train, n_val, n_test)
# index = TrainValTestIndex(load=True)


''' FUNCTIONS '''

def train_val_onehead_cnn(task, seed, loader_train, loader_val, nclasses, learning_rate, test=False, loader_test=None):
    torch.manual_seed(seed)
    if task == 'classification':
        cnn = OneHeadedCNN(out_dim=nclasses).to(DEVICE)
    elif task == 'regression':
        cnn = OneHeadedCNN(out_dim=1).to(DEVICE)
    optimiser = optim.Adam(cnn.parameters(), lr=learning_rate)
    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    errors_train, errors_val = np.zeros(N_EPOCHS), np.zeros(N_EPOCHS)

    for epoch in range(N_EPOCHS):
        # training:
        cnn.train()
        running_loss = 0
        total_error = 0
        for i, data in enumerate(loader_train, 0):
            inputs, targets = data[0].unsqueeze(1), data[1]
            # cf Torch docs ("Guide on good usage of non_blocking and pin_memory() in PyTorch")
            # : pinning tensors + asynchronous transfer to gpu (non_blocking=True)
            # is the fastest combo. We already pinned wen building the DataLoader
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimiser.zero_grad()
            logits = cnn(inputs)
            if task=='classification':
                loss = cross_entropy_loss(logits, targets)
            elif task=='regression':
                loss= mse_loss(logits.squeeze(1), targets)
            loss.backward()
            optimiser.step()
            if task=='classification':
                _, predictions = torch.max(logits, 1)
            elif task=='regression':
                predictions = logits.squeeze(1)
            error = circular_time_error(y_true=targets, y_pred=predictions, nclasses=nclasses)
            total_error += error.sum().item()
            running_loss += loss.item()
            if i%log_freq == log_freq - 1:
                print(f'[lr={learning_rate}, {epoch+1}, {i+1:5d}] running loss: {running_loss/log_freq:.4f}')
                running_loss = 0
        errors_train[epoch] = total_error / n_train

        # validation:
        cnn.eval()
        with torch.no_grad():
            total_error = 0
            for data in loader_val:
                inputs, targets = data[0].unsqueeze(1), data[1]
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                logits = cnn(inputs)
                if task=='classification':
                    _, predictions = torch.max(logits, 1)
                elif task=='regression':
                    predictions = logits.squeeze(1)
                error = circular_time_error(y_true=targets, y_pred=predictions, nclasses=nclasses)
                total_error += error.sum().item()
        errors_val[epoch] = total_error / n_val
        print(f'epoch {epoch+1} val error: {total_error / n_val:.3f}')

    if test:
        with torch.no_grad():
            total_error = 0
            for data in loader_test:
                inputs, targets = data[0].unsqueeze(1), data[1]
                inputs = inputs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                logits = cnn(inputs)
                if task=='classification':
                    _, predictions = torch.max(logits, 1)
                elif task=='regression':
                    predictions = logits.squeeze(1)
                error = circular_time_error(y_true=targets, y_pred=predictions, nclasses=nclasses)
                total_error += error.sum().item()
        error_test = total_error / n_test
        return errors_train, errors_val, error_test

    return errors_train, errors_val 

def class24_train_val(seeds, images, labels, index, batch_size_train, learning_rate):
    loader_train, loader_val, _ = make_data_loaders(
        'classification'
        , images
        , labels
        , index
        , batch_size_train
        , nclasses=24
    )
    train_matrix = np.zeros((len(seeds), N_EPOCHS))
    val_matrix = np.zeros((len(seeds), N_EPOCHS))
    for i, seed in enumerate(seeds):
        errors_train, errors_val = train_val_onehead_cnn('classification', seed, loader_train, loader_val, 24, learning_rate)
        train_matrix[i] = errors_train
        val_matrix[i] = errors_val
    avg_train_errors = smooth(np.mean(train_matrix, axis=0), SMOOTH_WINDOW)
    avg_val_errors = smooth(np.mean(val_matrix, axis=0), SMOOTH_WINDOW)
    os.makedirs('lr_sensitivity', exist_ok=True)
    np.save(f'lr_sensitivity/train_err_lr{learning_rate}.npy', avg_train_errors)
    np.save(f'lr_sensitivity/val_err_lr{learning_rate}.npy', avg_val_errors)
    steps = np.arange(1,N_EPOCHS+1)
    # plot:
    plt.figure(figsize=(5, 4))
    plt.plot(steps, avg_val_errors, color='#72C7C3', linewidth=2.5, label='Validation')
    plt.plot(steps, avg_train_errors, color='#CC7A4D', linewidth=2.5, label='Train')
    x_ticks = plt.xticks()[0]
    plt.xticks(x_ticks, [str(int(t)) if t  == x_ticks[-1] or t == x_ticks[0] else '' for t in x_ticks])
    plt.title(f'24 classes, lr={learning_rate}')
    plt.xlabel('Epoch')
    plt.ylabel('Error (30-min)')
    plt.grid(False)
    plt.box(True)
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'lr_sensitivity/lr{learning_rate}.png')

def learning_rate_search():
    print('Running learning rate search experiment...')
    learning_rates = [5e-5, 1e-4, 2.5e-4, 5e-4]
    seeds = np.arange(3) + STUDENT_NUMBER
    for lr in learning_rates:
        class24_train_val(
            seeds
            , images
            , labels
            , index
            , batch_size_train
            , learning_rate=lr
        )

def train_val_test(task, n_seeds, images, labels, index, batch_size_train, nclasses, learning_rate):
    seeds = np.arange(n_seeds) + STUDENT_NUMBER
    loader_train, loader_val, loader_test = make_data_loaders(
        task
        , images
        , labels
        , index
        , batch_size_train
        , nclasses
    )
    train_matrix = np.zeros((len(seeds), N_EPOCHS))
    val_matrix = np.zeros((len(seeds), N_EPOCHS))
    test_vec = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        errors_train, errors_val, error_test = train_val_onehead_cnn(task, seed, loader_train, loader_val, nclasses, learning_rate, test=True, loader_test=loader_test)
        train_matrix[i] = errors_train
        val_matrix[i] = errors_val
        test_vec[i] = error_test
    train_mean = np.mean(train_matrix, axis=0)
    train_sd = np.std(train_matrix, axis=0)
    min_train_error_idx = np.argmin(train_mean)
    min_train_error = dict(mean=train_mean[min_train_error_idx], sd=train_sd[min_train_error_idx])
    val_mean = np.mean(val_matrix, axis=0)
    val_sd = np.std(val_matrix, axis=0)
    min_val_error_idx = np.argmin(val_mean)
    min_val_error = dict(mean=val_mean[min_val_error_idx], sd=val_sd[min_val_error_idx])
    test_error = dict(mean=np.mean(test_vec), sd=np.std(test_vec))
    print(f'{task} results:')
    print(f'Minimum training error: {min_train_error['mean']:.2f}+-{min_train_error['sd']:.2f}')
    print(f'Minimum validation error: {min_val_error['mean']:.2f}+-{min_val_error['sd']:.2f}')
    print(f'Test error: {test_error['mean']:.2f}+-{test_error['sd']:.2f}')

def classification_24():
    print('Running 24-class experiment...')
    n_seeds = 3
    train_val_test(
        'classification'
        , n_seeds
        , images
        , labels
        , index
        , batch_size_train
        , nclasses=24
        , learning_rate=1e-4
    )

def classification_720():
    print('Running 720-class experiment...')
    n_seeds = 3
    train_val_test(
        'classification'
        , n_seeds
        , images
        , labels
        , index
        , batch_size_train
        , nclasses=720
        , learning_rate=1e-4
    )

def regression():
    print('Running regression experiment...')
    n_seeds = 3
    train_val_test(
        'regression'
        , n_seeds
        , images
        , labels
        , index
        , batch_size_train
        , nclasses=12  # looks weird here, but this is for the circular time error.
        , learning_rate=1e-4
    )


# step 5: multi-head experiment

def train_test_twohead_cnn(seed, loader_train, loader_test, learning_rate):
    torch.manual_seed(seed)
    cnn = TwoHeadedCNN(out_dim_hours=12, out_dim_minutes=1).to(DEVICE)
    optimiser = optim.Adam(cnn.parameters(), lr=learning_rate)
    cross_entropy_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss() 
    errors_train = np.zeros(N_EPOCHS)
    for epoch in range(N_EPOCHS):
        running_loss = 0
        total_error = 0
        for i, data in enumerate(loader_train, 0):
            inputs, targets = data[0].unsqueeze(1), data[1]
            # cf Torch docs (Guide on good usage of non_blocking and pin_memory() in PyTorch)
            # : pinning tensors + asynchronous transfer to gpu (non blocking)
            # is the fastest combo
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            target_hours = targets[:,0]
            target_minutes = targets[:,1].float()

            optimiser.zero_grad()
            logits_hours, logits_minutes = cnn(inputs)

            loss_hours = cross_entropy_loss(logits_hours, target_hours)
            loss_minutes = mse_loss(logits_minutes.squeeze(1), target_minutes/60)
            loss = 0.5*loss_hours + 0.5*loss_minutes
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            _, hours_pred = torch.max(logits_hours, 1)
            minutes_pred = logits_minutes.squeeze(1)
            y_true = target_hours*60 + target_minutes
            y_pred = hours_pred*60 + minutes_pred
            error = circular_time_error(y_true, y_pred, nclasses=720)
            total_error += error.sum().item()
            if i%log_freq == log_freq - 1:
                print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/log_freq:.3f}')
            errors_train[epoch] = total_error / n_train

    with torch.no_grad():
        total_error = 0
        for data in loader_test:
            inputs, targets = data[0].unsqueeze(1), data[1]
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            target_hours = targets[:,0]
            target_minutes = targets[:,1]
            logits_hours, logits_minutes = cnn(inputs)
            y_true = target_hours*60 + target_minutes
            y_pred = hours_pred*60 + minutes_pred
            error = circular_time_error(y_true, y_pred, nclasses=720)
            total_error += error.sum().item()
    error_test = total_error / n_test
    return errors_train, error_test


def two_headed(learning_rate=1e-4):
    n_seeds = 3
    seeds = np.arange(n_seeds) + STUDENT_NUMBER
    loader_train, _, loader_test = make_data_loaders(
        'two-heads'
        , images
        , labels
        , index
        , batch_size_train
    )
    train_matrix = np.zeros((len(seeds), N_EPOCHS))
    test_vec = np.zeros(len(seeds))
    for i, seed in enumerate(seeds):
        errors_train, error_test = train_test_twohead_cnn(seed, loader_train, loader_test, learning_rate)
        train_matrix[i] = errors_train
        test_vec[i] = error_test
    train_mean = np.mean(train_matrix, axis=0)
    train_sd = np.std(train_matrix, axis=0)
    min_train_error_idx = np.argmin(train_mean)
    min_train_error = dict(mean=train_mean[min_train_error_idx], sd=train_sd[min_train_error_idx])
    test_error = dict(mean=np.mean(test_vec), sd=np.std(test_vec))
    print(f'two-headed results:')
    print(f'Minimum training error: {min_train_error['mean']:.2f}+-{min_train_error['sd']:.2f}')
    print(f'Test error: {test_error['mean']:.2f}+-{test_error['sd']:.2f}')


if __name__=='__main__':
    args = parse_args()
    if args.experiment == 1:
        learning_rate_search()
    elif args.experiment == 2:
        classification_24()
    elif args.experiment == 3:
        classification_720()
    elif args.experiment == 4:
        regression()
    # two_headed()