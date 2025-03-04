import json
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from psd import power_spectrum_error  
import argparse

if torch.backends.mps.is_available():
    device = "mps"  
elif torch.cuda.is_available():
    device = "cuda"  
else:
    device = "cpu"

def smape(true, pred):
    """
    Symmetric mean absolute percentage error
    """
    true, pred = np.array(true), np.array(pred)
    return 100 / len(pred) * np.sum(2 * np.abs(true - pred) / (np.abs(pred) + np.abs(true) + 1e-8))

def evaluate_model(true_npy_path, pred_npy_path, eval_json_path, subset_size=None):
    """
    Evaluates the Transformer model predictions using precomputed trajectory.

    :param true_npy_path: Path to ground truth test dataset.
    :param pred_npy_path: Path to the precomputed Transformer predictions.
    :param eval_json_path: Path to save evaluation results.
    """
    
    true_values = np.load(true_npy_path)  
    predictions = np.load(pred_npy_path)

    # print(f"true_values shape: {true_values.shape}")
    # print(f"predictions shape: {predictions.shape}") 

    if subset_size:
        true_values = true_values[:subset_size]
        predictions = predictions[:subset_size]

    true_values = np.expand_dims(true_values, axis=0)  # (1, time_steps, features)
    predictions = np.expand_dims(predictions, axis=0)  

    # print(f"Expanded true_values shape: {true_values.shape}")
    # print(f"Expanded predictions shape: {predictions.shape}")

    mse = mean_squared_error(true_values.flatten(), predictions.flatten())
    mae = mean_absolute_error(true_values.flatten(), predictions.flatten())
    smape_val = smape(true_values.flatten(), predictions.flatten())
    psd_error = power_spectrum_error(predictions, true_values)

    eval_dict = {
        "MSE": float(mse),
        "MAE": float(mae),
        "sMAPE": float(smape_val),
        "Power Spectrum Error": float(psd_error)
    }


    with open(eval_json_path, "w") as f:
        json.dump(eval_dict, f, indent=4)

    for k, v in eval_dict.items():
        print(f"{k}: {v:.4f}")

    return eval_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_npy_path", required=True, help="Path to the ground truth npy file")
    parser.add_argument("--pred_npy_path", required=True, help="Path to the predicted trajectory npy file")
    parser.add_argument("--eval_json_path", required=True, help="Path to save evaluation results")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of samples to compare (default: all)")
    args = parser.parse_args()

    evaluate_model(
        true_npy_path=args.true_npy_path,
        pred_npy_path=args.pred_npy_path,
        eval_json_path=args.eval_json_path,
        subset_size=args.subset_size
    )
