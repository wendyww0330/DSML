import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def split_np_array(data: np.ndarray, split: str, history_size: int = 120, horizon_size: int = 30):
    """
    Create multiple training / validation samples from NumPy array using sliding window.

    :param data: NumPy array with shape (time_steps, features).
    :param split: "train" or "val".
    :param history_size: Number of past time steps used as input (sliding window size).
    :param horizon_size: Number of future time steps used as output.
    :return: Tuple (histories, targets) with multiple samples.
    """
    if split == "train":
        total_samples = data.shape[0] - history_size - horizon_size
        start_indices = np.arange(total_samples)  # Generate all valid starting indices
    elif split in ["val", "test"]:
        start_indices = [data.shape[0] - history_size - horizon_size]  # Only one sample for validation
    else:
        raise ValueError("Invalid split type. Use 'train', 'val' or 'test'.")

    histories = []
    targets = []

    for start_index in start_indices:
        label_index = start_index + history_size
        end_index = label_index + horizon_size

        history = data[start_index:label_index]  # Past data (encoder input)
        target = data[label_index:end_index]  # Future data (decoder target)

        histories.append(history)
        targets.append(target)

    return np.array(histories), np.array(targets)


class LorenzDataset(Dataset):
    def __init__(self, input_data, target_data, device="cpu"):
        """
        Dataset class for Lorenz-63 and Lorenz-96 using processed NumPy data.

        :param input_data: Preprocessed history data (encoder input).
        :param target_data: Preprocessed target data (decoder output).
        :param device: The device where tensors should be loaded ("cpu" or "cuda").
        """
        self.input_data = input_data
        self.target_data = target_data
        self.device = device

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        """
        Generates input-output pairs for the Transformer model.

        - `src`: Encoder input.
        - `trg_in`: Decoder input (with last `src` step as first step).
        - `trg_out`: Decoder output (future M steps).
        """
        src = self.input_data[idx]
        trg_out = self.target_data[idx]

        trg_in = np.zeros_like(trg_out)
        trg_in[1:] = trg_out[:-1]
        trg_in[0] = src[-1]  # Use last step of `src` as the first step of `trg_in`

        return (
            torch.tensor(src, dtype=torch.float, device=self.device, requires_grad=False),
            torch.tensor(trg_in, dtype=torch.float, device=self.device, requires_grad=False),
            torch.tensor(trg_out, dtype=torch.float, device=self.device, requires_grad=False),
        )

def get_dataloader(input_data, target_data, batch_size=32, shuffle=True, device="cpu"):
    """
    Creates a DataLoader for the Lorenz dataset.

    :param input_data: Preprocessed history data.
    :param target_data: Preprocessed target data.
    :param batch_size: Number of samples per batch.
    :param shuffle: Whether to shuffle the dataset.
    :param device: The device where tensors should be loaded ("cpu" or "cuda").
    :return: DataLoader instance.
    """
    dataset = LorenzDataset(input_data, target_data, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
