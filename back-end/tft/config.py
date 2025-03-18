import os
import torch  # Import the torch library

class Config:
    # Data paths
    RAW_DATA_PATH = os.path.join("data", "raw", "stock_data.csv")
    PROCESSED_DATA_PATH = os.path.join("data", "processed", "processed_data.csv")
    MODEL_PATH = os.path.join("models", "tft_model.pth")

    # Model hyperparameters
    SEQ_LENGTH = 60
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    DROPOUT = 0.3
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.0005

    # Training settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Now torch is defined

    # Features used in the model
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'BB_upper', 'BB_lower']