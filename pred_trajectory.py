import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import TimeSeriesForcasting
from model import gen_trg_mask

from data_utils import load_lorenz_data
from scipy.interpolate import interp1d

if torch.backends.mps.is_available():
    device = "mps"  
elif torch.cuda.is_available():
    device = "cuda"  
else:
    device = "cpu"

def T_step_forecast(model: nn.Module, enc_len: int, dec_len: int, initial_enc_input: torch.Tensor, initial_dec_input: torch.Tensor, T: int, device) -> torch.Tensor:
    """
    Generate a trajectory using autoregressive decoding.

    Args:
        model: Trained transformer model
        enc_len: Number of encoder time steps
        dec_len: Number of decoder time steps
        initial_enc_input: Tensor of shape (batch_size, enc_len, input_dim)
        initial_dec_input: Tensor of shape (batch_size, 1, input_dim)
        T: Total number of steps to predict
        device: Computation device (cuda/mps/cpu)

    Returns:
        Generated trajectory of shape (batch_size, T, output_dim)
    """
    
    if len(initial_enc_input.shape) < 3:
        initial_enc_input = initial_enc_input.unsqueeze(0)
    if len(initial_dec_input.shape) < 3:
        initial_dec_input = initial_dec_input.unsqueeze(0)

    traj = initial_dec_input  # Start with the first decoder input

    enc_input = initial_enc_input
    dec_input = initial_dec_input

    for t in tqdm(range(T), desc="Generating trajectory", unit="step"):

        if dec_input.shape[1] > dec_len:
            dec_input = dec_input[:, -dec_len:, :]
        if enc_input.shape[1] > enc_len:
            enc_input = enc_input[:, -enc_len:, :]

        # Generate target mask for Transformer
        trg_mask = gen_trg_mask(dec_input.shape[1], device)

        with torch.no_grad():
            dec_output = model((enc_input, dec_input))  # Transformer inference

        pred = dec_output[:, -1, :].unsqueeze(1)  # Get last predicted time step

        # Update encoder & decoder input with new prediction
        enc_input = torch.cat((enc_input, pred), dim=1)
        dec_input = torch.cat((dec_input, pred), dim=1)
        traj = torch.cat((traj, pred), dim=1)

    return traj[:, 1:, :]
    
def generate_and_save_trajectory(npy_path, model_checkpoint, save_path, history_size=15, forecast_window=100):
    """
    Generates and saves Transformer-predicted Lorenz trajectory.

    :param npy_path: Path to the test `.npy` dataset.
    :param model_checkpoint: Path to the trained model checkpoint.
    :param save_path: Path to save the generated trajectory.
    :param history_size: Encoder input size.
    :param forecast_window: Number of future time steps to predict.
    """

    test_loader = load_lorenz_data(npy_path, history_size, forecast_window, batch_size=1, shuffle=False)

    model = TimeSeriesForcasting()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["state_dict"])
    model.to(device)
    model.eval()

    for i, (src, _, _) in enumerate(test_loader):
        src = src.to(device)
        dec_start = src[:, -1, :].unsqueeze(1)  # Last timestep of src as first decoder input

        # Generate trajectory using autoregressive forecasting
        pred_trajectory = T_step_forecast(model, history_size, forecast_window, src, dec_start, forecast_window, device)

        # Convert to NumPy array and save
        pred_trajectory_np = pred_trajectory.squeeze().cpu().numpy()
        np.save(save_path, pred_trajectory_np)

        print(f"Saved generated trajectory to {save_path}")

        break  # Only process the first sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", required=True, help="Path to the test dataset (.npy)")
    parser.add_argument("--model_checkpoint", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--save_path", required=True, help="Path to save the generated trajectory (.npy)")
    args = parser.parse_args()

    generate_and_save_trajectory(args.npy_path, args.model_checkpoint, args.save_path)