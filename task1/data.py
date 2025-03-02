import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def read_data(filename, length=None):
    data = np.loadtxt(filename, delimiter=',')
    data = torch.as_tensor(data).float()
    if length is not None:
        data = data[:length]
    return data


def split_data(data: torch.Tensor, input_dim, training_ratio=0.8):
    shuffle = True
    training_len = round(data.shape[0] * training_ratio)
    if not shuffle:
        training_data = data[0:training_len]
        testing_data = data[training_len:]
    else:
        training_indices, testing_indices = random_split(range(data.shape[0]), [training_len, data.shape[0] - training_len], torch.Generator())
        training_data = data[torch.as_tensor(training_indices)]
        testing_data = data[torch.as_tensor(testing_indices)]

    return training_data[:, :input_dim], testing_data[:, :input_dim], training_data[:, input_dim:], testing_data[:, input_dim:]


def create_dataloader(X_training, X_test, y_training, y_test, batch_size = 16):
    # training
    training_dataset = TensorDataset(X_training, y_training)
    generator = torch.Generator()
    training_loader = DataLoader(training_dataset, batch_size, shuffle=True, generator=generator)

    # testing
    testing_dataset = TensorDataset(X_test, y_test)
    testing_loader = DataLoader(testing_dataset, batch_size)

    return training_loader, testing_loader
