import torch
from torch.utils.data import DataLoader
from utils.data_loader import fetch_data, preprocess_data, create_sequences, TimeSeriesDataset
from utils.trainer import Trainer
from config import Config
from models.tft_model import TemporalFusionTransformer

def main():
    # Load existing model
    input_size = len(Config.FEATURES)
    model = TemporalFusionTransformer(input_size, Config.HIDDEN_SIZE, 1, Config.NUM_LAYERS, Config.NUM_HEADS, Config.DROPOUT)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.to(Config.DEVICE)

    # Fetch new data (e.g., tomorrow's data)
    new_data = fetch_data("^BSESN", "2020-01-01", "2025-03-03")  # Example: Fetch new data
    scaled_data, _ = preprocess_data(new_data)
    xs, ys = create_sequences(scaled_data, Config.SEQ_LENGTH)

    # Create dataset and dataloader
    dataset = TimeSeriesDataset(xs, ys)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Train model with new data
    trainer = Trainer(model, dataloader, Config.DEVICE, Config.LEARNING_RATE)
    trainer.train(Config.EPOCHS // 10)  # Train for fewer epochs

    # Save updated model
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Model updated and saved to {Config.MODEL_PATH}")

if __name__ == "__main__":
    main()