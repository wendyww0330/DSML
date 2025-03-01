import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import TimeSeriesForcasting
from data_utils import load_lorenz_data
from scipy.interpolate import interp1d

if torch.backends.mps.is_available():
    device = "mps"  
elif torch.cuda.is_available():
    device = "cuda"  
else:
    device = "cpu"

def visualize_predictions(npy_path, model_checkpoint, history_size=15, horizon_size=3, num_samples=100):
    """
    Visualizes Transformer predictions vs. Ground Truth for Lorenz trajectory.

    :param npy_path: Path to the test `.npy` dataset.
    :param model_checkpoint: Path to the trained model checkpoint.
    :param history_size: Number of past time steps (N).
    :param horizon_size: Number of future time steps to predict (M).
    :param num_samples: Number of samples to visualize.
    """

    test_loader = load_lorenz_data(npy_path, history_size, horizon_size, batch_size=1, shuffle=False)

    model = TimeSeriesForcasting()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["state_dict"])
    model.to(device)
    model.eval()

    true_values = []
    predictions = []

    with torch.no_grad():
        for i, (src, trg_in, trg_out) in enumerate(test_loader):
            src, trg_in = src.to(device), trg_in.to(device)

            pred = model((src, trg_in[:, :1, :]))  

            for j in range(1, horizon_size):
                # last_pred = pred[0, -1]
                last_pred = pred[:, -1, :]
                last_pred = last_pred.unsqueeze(0)

                # print(f"Iteration {j} - last_pred shape: {last_pred.shape}, trg_in[:, j, :].shape: {trg_in[:, j, :].shape}")

                trg_in[:, j, :] = last_pred  
                pred = model((src, trg_in[:, : (j + 1), :]))

            # print(f"Final predicted shape: {pred.shape}, Ground Truth shape: {trg_out.shape}")

            pred = pred.squeeze().cpu().numpy()
            trg_out = trg_out.squeeze().cpu().numpy()

            predictions.append(pred)
            true_values.append(trg_out)

            if i >= num_samples:
                break  

    true_values = np.array(true_values)
    predictions = np.array(predictions)

    plt.figure(figsize=(12, 6))

    for dim, label in enumerate(["X", "Y", "Z"]):
        plt.subplot(1, 3, dim + 1)
        plt.plot(true_values[:, :, dim].flatten(), label="Ground Truth", color="blue")
        plt.plot(predictions[:, :, dim].flatten(), label="Prediction", color="red", linestyle="dashed")
        plt.title(f"Lorenz {label}-axis Prediction")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()

    plt.tight_layout()
    plt.show()

def smooth_curve(x, y, z, num_points=1000):
    t = np.linspace(0, 1, len(x))
    t_smooth = np.linspace(0, 1, num_points)

    x_smooth = interp1d(t, x, kind='cubic')(t_smooth)
    y_smooth = interp1d(t, y, kind='cubic')(t_smooth)
    z_smooth = interp1d(t, z, kind='cubic')(t_smooth)

    return x_smooth, y_smooth, z_smooth

def visualize_lorenz_attractor(npy_path, model_checkpoint, history_size=15, horizon_size=3, num_samples=2000):
    """
    Visualizes the Lorenz attractor for Transformer predictions vs. Ground Truth.

    :param npy_path: Path to the test `.npy` dataset.
    :param model_checkpoint: Path to the trained model checkpoint.
    :param history_size: Number of past time steps (N).
    :param horizon_size: Number of future time steps to predict (M).
    :param num_samples: Number of samples to visualize.
    """

    test_loader = load_lorenz_data(npy_path, history_size, horizon_size, batch_size=1, shuffle=False)

    model = TimeSeriesForcasting()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["state_dict"])
    model.to(device)
    model.eval()

    true_values = []
    predictions = []

    with torch.no_grad():
        for i, (src, trg_in, trg_out) in enumerate(test_loader):
            src, trg_in = src.to(device), trg_in.to(device)

            pred = model((src, trg_in[:, :1, :])) 

            for j in range(1, horizon_size):
                last_pred = pred[:, -1, :]  
                last_pred = last_pred.unsqueeze(0)  
                trg_in[:, j, :] = last_pred 
                pred = model((src, trg_in[:, : (j + 1), :]))  

            pred = pred.squeeze().cpu().numpy()
            trg_out = trg_out.squeeze().cpu().numpy()

            predictions.append(pred)
            true_values.append(trg_out)

            if i >= num_samples:
                break  

    true_values = np.array(true_values)
    predictions = np.array(predictions)

    x_true, y_true, z_true = true_values[:, :, 0].flatten(), true_values[:, :, 1].flatten(), true_values[:, :, 2].flatten()
    x_pred, y_pred, z_pred = predictions[:, :, 0].flatten(), predictions[:, :, 1].flatten(), predictions[:, :, 2].flatten()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_true, y_true, z_true, label="Ground Truth", color="blue", alpha=0.6)

    x_pred_smooth, y_pred_smooth, z_pred_smooth = smooth_curve(x_pred, y_pred, z_pred)
    ax.plot(x_pred_smooth, y_pred_smooth, z_pred_smooth, label="Prediction", color="red", alpha=0.8)

    # ax.plot(x_pred, y_pred, z_pred, label="Prediction", color="red", alpha=0.8)

    ax.set_title("Lorenz Attractor: Transformer Prediction vs Ground Truth")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()

if __name__ == "__main__":
    npy_path = "lorenz63_test.npy"  
    model_checkpoint = "models/ts.ckpt"  
    # visualize_predictions(npy_path, model_checkpoint)
    visualize_lorenz_attractor(npy_path, model_checkpoint)
