import torch
from tqdm import tqdm
import argparse
import numpy as np
import torch
from model import TimeSeriesForcasting

if torch.backends.mps.is_available():
    device = "mps"  
elif torch.cuda.is_available():
    device = "cuda"  
else:
    device = "cpu"

def generate_and_save_trajectory(npy_path, model_checkpoint, save_path, history_size=15, forecast_window=100000):
    """
    Uses a trained Transformer to generate a trajectory in an autoregressive way and saves it as a .npy file.

    :param npy_path: Path to the test `.npy` dataset.
    :param model_checkpoint: Path to the trained model checkpoint.
    :param save_path: Path to save the generated trajectory.
    :param history_size: Encoder input length (should match training settings).
    :param forecast_window: Number of future time steps to generate.
    """

    # Load test data
    data = np.load(npy_path)
    
    t0_idx = np.random.randint(0, len(data) - 1)
    initial_point = data[t0_idx:t0_idx + history_size]  

    model = TimeSeriesForcasting()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["state_dict"])
    model.to(device)
    model.eval()

    generated_trajectory = list(initial_point)  
    input_seq = torch.tensor(initial_point, dtype=torch.float32).unsqueeze(0).to(device)  # (1, history_size, 3)

    with torch.no_grad():
        for _ in tqdm(range(forecast_window), desc="Generating trajectory", unit="step"):

            pred = model((input_seq, input_seq[:, -1:, :]))  
            next_pred = pred[:, -1, :].squeeze(0)  

            input_seq = torch.cat((input_seq, next_pred.unsqueeze(0).unsqueeze(0)), dim=1)  
            if input_seq.shape[1] > history_size:
                input_seq = input_seq[:, -history_size:, :]  

            generated_trajectory.append(next_pred.cpu().numpy())

    generated_trajectory = np.array(generated_trajectory)

    np.save(save_path, generated_trajectory)
    print(f"Generated trajectory saved at: {save_path}")

    return generated_trajectory



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", required=True, help="Path to the test dataset (.npy)")
    parser.add_argument("--model_checkpoint", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--save_path", required=True, help="Path to save the generated trajectory (.npy)")
    args = parser.parse_args()

    generate_and_save_trajectory(args.npy_path, args.model_checkpoint, args.save_path)