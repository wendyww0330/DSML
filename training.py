import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import TimeSeriesForcasting
from data_utils import load_lorenz_data

# Determine the computing device
if torch.backends.mps.is_available():
    device = "mps"  
elif torch.cuda.is_available():
    device = "cuda"  
else:
    device = "cpu"

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Add early stopping callback
early_stopping = EarlyStopping(
    monitor="valid_loss",  
    patience=10,  
    mode="min",  
    verbose=True  
)

def train(
    npy_path: str,
    output_json_path: str,
    log_dir: str = "ts_logs",
    model_dir: str = "ts_models",
    batch_size: int = 32,
    epochs: int = 100,
    history_size: int = 15,  # Encoder input: 15 time steps
    horizon_size: int = 3,   # Decoder input: 3 time steps, predicting 3 time steps
):
    """
    Train the Transformer model for Lorenz trajectory forecasting.

    :param npy_path: Path to the `.npy` dataset file.
    :param output_json_path: Path to save the training output JSON file.
    :param log_dir: Directory to store training logs.
    :param model_dir: Directory to save the trained model.
    :param batch_size: Batch size for training.
    :param epochs: Number of training epochs.
    :param history_size: Number of past time steps used as input (N).
    :param horizon_size: Number of future time steps to predict (M).
    """

    train_loader = load_lorenz_data(npy_path, history_size, horizon_size, batch_size=batch_size, shuffle=True)
    val_loader = load_lorenz_data(npy_path, history_size, horizon_size, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    model = TimeSeriesForcasting()

    logger = TensorBoardLogger(save_dir=log_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        devices=1,
        accelerator=device,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Evaluate the model on validation set
    result_val = trainer.test(test_dataloaders=val_loader)

    # Save the best model path and validation loss
    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

    return output_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_path")  
    parser.add_argument("--output_json_path", default=None)
    parser.add_argument("--log_dir")
    parser.add_argument("--model_dir")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    train(
        npy_path=args.npy_path,
        output_json_path=args.output_json_path,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
    )