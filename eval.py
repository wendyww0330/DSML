import json
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from psd import power_spectrum_error  
from model import TimeSeriesForcasting
from data_utils import load_lorenz_data

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

def evaluate_model(npy_path, model_checkpoint, eval_json_path, history_size=15, horizon_size=3):
    """
    Evaluates the trained Transformer model on Lorenz data.
    
    :param npy_path: Path to the test npy dataset
    :param model_checkpoint: Path to the trained model checkpoint
    :param eval_json_path: Path to save evaluation results
    :param history_size: Number of past time steps (N)
    :param horizon_size: Number of future time steps (M)
    """
    test_loader = load_lorenz_data(npy_path, history_size, horizon_size, batch_size=32, shuffle=False)

    subset_size = 5000
    test_loader = list(test_loader)[:subset_size]

    model = TimeSeriesForcasting()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["state_dict"])
    model.to(device)
    model.eval()

    gt, predictions = [], []

    with torch.no_grad():
        for src, trg_in, trg_out in tqdm(test_loader, desc="Evaluating Model"):
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
            gt.append(trg_out)

    gt = np.array(gt)
    predictions = np.array(predictions)

    mse = mean_squared_error(gt.flatten(), predictions.flatten())
    mae = mean_absolute_error(gt.flatten(), predictions.flatten())
    smape_val = smape(gt.flatten(), predictions.flatten())

    psd_error = power_spectrum_error(predictions, gt)

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path", required=True, help="Path to the test npy file")
    parser.add_argument("--model_checkpoint", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--eval_json_path", required=True, help="Path to save evaluation results")
    args = parser.parse_args()

    evaluate_model(
        npy_path=args.npy_path,
        model_checkpoint=args.model_checkpoint,
        eval_json_path=args.eval_json_path
    )
