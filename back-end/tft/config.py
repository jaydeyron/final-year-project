import os
import torch  # Import the torch library

class Config:
    # Model parameters
    INPUT_SIZE = 10  # Number of features in your data
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    NUM_HEADS = 4
    DROPOUT = 0.1
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 60  # 60 days of historical data
    
    # Features for the model
    FEATURES = [
        'Open', 'High', 'Low', 'Close', 
        'Volume', 'MA_10', 'RSI', 'MACD',
        'Upper_Band', 'Lower_Band'
    ]
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    @classmethod
    def get_model_path(cls, symbol):
        """Get model path for a specific symbol"""
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        safe_symbol = symbol.replace('^', 'SENSEX').replace('.BO', '').replace(':', '_')
        return os.path.join(cls.MODEL_DIR, f"{safe_symbol}.pth")