import torch
from torch.utils.data import DataLoader
from utils.data_loader import fetch_data, preprocess_data, create_sequences, TimeSeriesDataset
from utils.trainer import Trainer
from config import Config
from models.tft_model import TemporalFusionTransformer

def main():
    # Fetch and preprocess data
    data = fetch_data("^BSESN", "2010-01-01", "2025-03-18")  # Example: Train on data up to today
    scaled_data, scaler = preprocess_data(data)
    xs, ys = create_sequences(scaled_data, Config.SEQ_LENGTH)

    # Create dataset and dataloader
    dataset = TimeSeriesDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Initialize model
    input_size = xs.shape[2]
    model = TemporalFusionTransformer(input_size, Config.HIDDEN_SIZE, 1, Config.NUM_LAYERS, Config.NUM_HEADS, Config.DROPOUT)

    # Train model
    trainer = Trainer(model, dataloader, Config.DEVICE, Config.LEARNING_RATE)
    trainer.train(Config.EPOCHS)

    # Save model
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")

if __name__ == "__main__":
    main()