import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    device = "mps"  
elif torch.cuda.is_available():
    device = "cuda"  
else:
    device = "cpu"


def visualize_lorenz_attractor(true_npy_path, pred_npy_path):
    """
    Visualizes Lorenz attractor comparing Ground Truth vs. Transformer Predictions.

    :param true_npy_path: Path to the ground truth `.npy` dataset.
    :param pred_npy_path: Path to the predicted `.npy` dataset.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    predictions = np.load(pred_npy_path)  
    true_values = np.load(true_npy_path)
    # true_values = np.load(true_npy_path)[:predictions.shape[1]]

    x_true, y_true, z_true = true_values[:, 0], true_values[:, 1], true_values[:, 2]
    x_pred, y_pred, z_pred = predictions[:, 0], predictions[:, 1], predictions[:, 2]

    ax.plot(x_true, y_true, z_true, label="Ground Truth", color="blue", alpha=0.6)
    ax.plot(x_pred, y_pred, z_pred, label="Prediction", color="red", linestyle="dashed", alpha=0.8)

    ax.set_title("Lorenz Attractor: Transformer Prediction vs Ground Truth")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_npy_path", required=True, help="Path to the test dataset (.npy)")
    parser.add_argument("--pred_npy_path", required=True, help="Path to the predicted `.npy` dataset")
    args = parser.parse_args()

    visualize_lorenz_attractor(
        true_npy_path=args.true_npy_path, 
        pred_npy_path=args.pred_npy_path
    )
