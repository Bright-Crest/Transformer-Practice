import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def read_data(filename, length=None):
    data = np.loadtxt(filename, delimiter=',')
    data = torch.as_tensor(data).float()
    if length is not None:
        data = data[:length]
    return data


def create_dataloader(data: torch.Tensor, src_len, tgt_len, batch_size = 16, training_ratio = 0.8, shuffle=True):
    step = src_len + tgt_len
    # (Total length, Features) => (Total length/(src_len+tgt_len), src_len+tgt_len, Features)
    new_data = data.unfold(0, src_len + tgt_len, step).permute(0, 2, 1)
    training_len = round(new_data.shape[0] * training_ratio)
    if not shuffle:
        training_data = new_data[0:training_len]
        testing_data = new_data[training_len:]
    else:
        training_indices, testing_indices = random_split(range(new_data.shape[0]), [training_len, new_data.shape[0] - training_len], torch.Generator())
        training_data = new_data[torch.as_tensor(training_indices)]
        testing_data = new_data[torch.as_tensor(testing_indices)]

    # training
    training_dataset = TensorDataset(training_data, training_data[:, src_len:])
    generator = torch.Generator()
    training_loader = DataLoader(training_dataset, batch_size, shuffle=True, generator=generator)

    # testing
    testing_dataset = TensorDataset(testing_data[:, :src_len], testing_data[:, src_len:])
    testing_loader = DataLoader(testing_dataset, batch_size)

    return training_loader, testing_loader
