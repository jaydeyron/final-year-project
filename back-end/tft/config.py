import os
import torch  # Import the torch library

class Config:
    # Model parameters
    INPUT_SIZE = 10  # Number of features in your data
    HIDDEN_SIZE = 64
    NUM_LAYERS = 3
    NUM_HEADS = 4
    DROPOUT = 0.1
    
    # Price constraint parameters
    MAX_CHANGE_PERCENT = 0.05  # 5% max change from last closing price
    
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
    def get_symbol_dir(cls, symbol):
        """Get directory for a specific symbol"""
        # Sanitize symbol for filesystem
        safe_symbol = symbol
        if symbol.endswith('.BO'):
            safe_symbol = symbol[:-3]
        elif symbol.startswith('^'):
            safe_symbol = symbol[1:]
        
        # Create directory if it doesn't exist
        symbol_dir = os.path.join(cls.MODEL_DIR, safe_symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        return symbol_dir
    
    @classmethod
    def get_model_path(cls, symbol):
        """Get paths for model and scaler files"""
        # This method name is used in model_utils.py
        base_dir = os.path.join(cls.MODEL_DIR, symbol.replace('^', 'SENSEX').replace('.BO', ''))
        os.makedirs(base_dir, exist_ok=True)
        return {
            'model': os.path.join(base_dir, 'model.pth'),
            'scaler': os.path.join(base_dir, 'scaler.json')
        }

    # Also add the method expected by newer code (with same functionality)
    @classmethod
    def get_model_paths(cls, symbol):
        """Alias for get_model_path with additional metadata path"""
        paths = cls.get_model_path(symbol)
        paths['metadata'] = os.path.join(os.path.dirname(paths['model']), 'metadata.json')
        return paths