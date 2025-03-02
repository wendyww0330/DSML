import numpy as np
import torch
from torch.utils.data import Dataset

class LorenzDataset(Dataset):
    def __init__(self, npy_path, history_size=15, horizon_size=3):
        """
        :param history_size: encoder input
        :param horizon_size: decoder input
        """
        self.data = np.load(npy_path)  
        self.history_size = history_size
        self.horizon_size = horizon_size

    def __len__(self):
        return len(self.data) - self.history_size - self.horizon_size

    def __getitem__(self, idx):
        src = self.data[idx : idx + self.history_size]  
        trg_in = self.data[idx + self.history_size - 1 : idx + self.history_size + self.horizon_size - 1]  # Decoder 输入
        trg_out = self.data[idx + self.history_size + 1 : idx + self.history_size + self.horizon_size + 1]

        return (
            torch.tensor(src, dtype=torch.float32),
            torch.tensor(trg_in, dtype=torch.float32),
            torch.tensor(trg_out, dtype=torch.float32),
        )

def load_lorenz_data(npy_path, history_size=15, horizon_size=5, batch_size=32, shuffle=False):
    dataset = LorenzDataset(npy_path, history_size, horizon_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
