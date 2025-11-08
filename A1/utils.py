import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import savgol_filter
import argparse


class TrainValTestIndex:
    def __init__(self, load=False, n_train=None, n_val=None, n_test=None):
        if load:
            self.train = np.load('index_splits/idx_train.npy')
            self.val = np.load('index_splits/idx_val.npy')
            self.test = np.load('index_splits/idx_test.npy')
        else:
            n = n_train + n_val + n_test
            index_vector = np.arange(n)
            np.random.seed(0)
            np.random.shuffle(index_vector)
            self.train = index_vector[:n_train]
            self.val = index_vector[n_train:n_train+n_val]
            self.test = index_vector[-n_test:]
            self._save_idx()
    
    def _save_idx(self):
        files = {
            'idx_train': self.train
            , 'idx_val': self.val
            , 'idx_test': self.test
        }
        os.makedirs('index_splits', exist_ok=True)
        for file, arr in files.items():
            np.save(f'index_splits/{file}.npy', arr)


def to_time_class(labels, nclasses):
    minutes = labels[:,0]*60 + labels[:,1]
    return minutes//(720//nclasses)

def to_time_float(labels):
    return labels[:,0] + labels[:,1]/60

def circular_time_error(y_true, y_pred, nclasses):
    diff = torch.abs(y_true - y_pred)
    return torch.min(diff, nclasses - diff)

def make_data_loaders(task, images, labels, index, batch_size_train, nclasses=None):
    '''
    Processes images, and labels according to the task specified.
    Builds 3 DataLoader instances for training, validation and test.
    Batch size for validation and test is 3 times the batch size of training (arbitrary choice).
    '''
    images_normalised = (images/255-0.5)/0.5  # source: Pytorch docs CIFAR-10 where they normalise pixel values from [0, 255] down to [-1, 1]
    images_tensor = torch.from_numpy(images_normalised).float()
    labels = process_labels(task, labels, nclasses)
    labels_tensor = torch.from_numpy(labels)
    if task == 'regression':
        labels_tensor = labels_tensor.float()

    loader_train = build_data_loader(
        images_tensor[index.train]
        , labels_tensor[index.train]
        , batch_size_train
    )
    loader_val = build_data_loader(
        images_tensor[index.val]
        , labels_tensor[index.val]
        , batch_size_train*3
        , shuffle=False
    )
    loader_test = build_data_loader(
        images_tensor[index.test]
        , labels_tensor[index.test]
        , batch_size_train*3
        , shuffle=False
    )
    return loader_train, loader_val, loader_test

def process_labels(task, labels, nclasses):
    if task == 'classification':
        if not nclasses or type(nclasses)!=int or nclasses<=0:
            raise RuntimeError('When running classification, provide a positive integer for nclasses.')
        labels = to_time_class(labels, nclasses)
    elif task == 'regression':
        labels = to_time_float(labels)
    elif task == 'two-heads':
        pass  # labels are already in correct format
    else:
        raise RuntimeError('task must be one of ("classification", "regression", "two-heads")')
    return labels

def build_data_loader(images_tensor, labels_tensor, batch_size, shuffle=True):
    dataset = TensorDataset(images_tensor, labels_tensor)
    loader = DataLoader(
        dataset
        , batch_size=batch_size
        , shuffle=shuffle
        , pin_memory=True  # recommended by torch docs wen doing gpu acc
        , num_workers=8
    )
    return loader

def smooth(y, window, poly=2):
    return savgol_filter(y, window, poly)

def parse_args():
    parser = argparse.ArgumentParser(description='CNN "Tell the time"')
    parser.add_argument(
        '--experiment'
        , type=int
        , default=1
        , help='1 to run the lr search; 2 for the 24-class task; 3 for the 720-class task; 4 for the regression task.'
    )
    return parser.parse_args()