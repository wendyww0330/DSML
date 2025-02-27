import json
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# from time_series_forecasting.model import TimeSeriesForcasting
from model import TimeSeriesForcasting

from data_utils import split_np_array, get_dataloader  # Import Lorenz Data Processing Functions

if torch.backends.mps.is_available():
    device = "mps"  # Use Metal (Apple M1/M2/M3)
elif torch.cuda.is_available():
    device = "cuda"  # Use CUDA (Linux/Windows)
else:
    device = "cpu"  # Default to CPU

def train(
    data_path: str,
    output_json_path: str = None,
    log_dir: str = "ts_logs",
    model_dir: str = "ts_models",
    batch_size: int = 32,
    epochs: int = 2000,
    input_len: int = 120,
    pred_len: int = 30,
):
    """
    Train the Transformer model on Lorenz-63 or Lorenz-96 data using explicit sliding window preprocessing.
    """

    # Load raw data
    raw_data = np.load(data_path)

    # Use `split_np_array()` to generate input-output pairs
    history, targets = split_np_array(raw_data, split="train", history_size=input_len, horizon_size=pred_len)
    print(f"History Shape: {history.shape}, Targets Shape: {targets.shape}")

    # Create DataLoader
    train_loader = get_dataloader(history, targets, batch_size=batch_size, device=device)
    val_loader = get_dataloader(history, targets, batch_size=batch_size, shuffle=False, device=device)

    # Initialize the Transformer model
    model = TimeSeriesForcasting(
        n_encoder_inputs=3 if "lorenz63" in data_path else 20,  
        n_decoder_inputs=3 if "lorenz63" in data_path else 20,
        lr=1e-5,
        dropout=0.1,
    ).to(device)  

    # Set up logging
    logger = TensorBoardLogger(save_dir=log_dir)

    # Checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="lorenz_transformer",
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=device,  # 自动选择 CPU/MPS/GPU
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to Lorenz dataset (.npy)")
    parser.add_argument("--output_json_path", default=None, help="Path to save training results")
    parser.add_argument("--log_dir", default="ts_logs", help="Directory for TensorBoard logs")
    parser.add_argument("--model_dir", default="ts_models", help="Directory for saving model checkpoints")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--input_len", type=int, default=120, help="Number of encoder input time steps")
    parser.add_argument("--pred_len", type=int, default=30, help="Number of decoder output time steps")
    
    args = parser.parse_args()

    train(
        data_path=args.data_path,
        output_json_path=args.output_json_path,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        input_len=args.input_len,
        pred_len=args.pred_len,
    )
