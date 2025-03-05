import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LorenzDataset(Dataset):
    def __init__(self, data, history_size=15, horizon_size=3):
        """
        :param data: NumPy array of time series
        :param history_size: Number of past time steps used as input (N)
        :param horizon_size: Number of future time steps to predict (M)
        """
        self.data = data  
        self.history_size = history_size
        self.horizon_size = horizon_size

    def __len__(self):
        return len(self.data) - self.history_size - self.horizon_size

    def __getitem__(self, idx):
        src = self.data[idx : idx + self.history_size]  
        trg_in = self.data[idx + self.history_size - 1 : idx + self.history_size + self.horizon_size - 1]  
        trg_out = self.data[idx + self.history_size : idx + self.history_size + self.horizon_size]

        return (
            torch.tensor(src, dtype=torch.float32),
            torch.tensor(trg_in, dtype=torch.float32),
            torch.tensor(trg_out, dtype=torch.float32),
        )

def load_lorenz_data(npy_path, history_size=15, horizon_size=5, batch_size=32, shuffle=False, train_split=0.8):
    """
    Loads Lorenz data and splits it into training and validation sets.

    :param npy_path: Path to the `.npy` dataset
    :param history_size: Encoder input size (N)
    :param horizon_size: Decoder input size (M)
    :param batch_size: Batch size for training
    :param shuffle: Whether to shuffle the dataset
    :param train_split: Ratio of data used for training (default 80% train, 20% validation)
    
    :return: train_loader, val_loader
    """
    data = np.load(npy_path)

    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = LorenzDataset(train_data, history_size, horizon_size)
    val_dataset = LorenzDataset(val_data, history_size, horizon_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # Validation should not shuffle

    return train_loader, val_loader

def load_lorenz_data_pred(npy_path, history_size=15, horizon_size=3, batch_size=1, shuffle=False):
    """
    Loads Lorenz test data without splitting into training/validation.

    :param npy_path: Path to the `.npy` test dataset
    :param history_size: Encoder input size (N)
    :param horizon_size: Decoder input size (M)
    :param batch_size: Batch size
    :param shuffle: Whether to shuffle the test set

    :return: DataLoader containing the test dataset
    """
    data = np.load(npy_path)
    test_dataset = LorenzDataset(data, history_size, horizon_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return test_loader
