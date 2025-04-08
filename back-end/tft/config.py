import os
import torch

class Config:
    # Model parameters
    INPUT_SIZE = 10  # This will be overridden by actual feature count
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    NUM_HEADS = 8
    DROPOUT = 0.2  # Increased to prevent overfitting
    
    # Price constraint parameters
    MAX_CHANGE_PERCENT = 0.15  # Increased to 15% - more flexibility
    
    # Training parameters
    BATCH_SIZE = 64  # Increased for better gradient estimates
    EPOCHS = 120
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 60  # 60 days of historical data
    
    # Features for the model - basic list, will be expanded by preprocessing
    FEATURES = [
        'Open', 'High', 'Low', 'Close', 
        'Volume', 'EMA10', 'EMA30', 'RSI', 'MACD', 'MACD_Signal',
        'ATR', 'ROC', 'BB_width', 'BB_B', 'Price_to_MA50', 'Price_to_MA200',
        'Volatility_Ratio', 'ADX'
    ]
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # File paths - Using absolute paths to backend/models
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
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
        clean_symbol = symbol.replace('^', 'SENSEX').replace('.BO', '')
        base_dir = os.path.join(cls.MODEL_DIR, clean_symbol)
        os.makedirs(base_dir, exist_ok=True)
        
        return {
            'model': os.path.join(base_dir, 'model.pth'),
            'scaler': os.path.join(base_dir, 'scaler.json')
        }

    @classmethod
    def get_model_paths(cls, symbol):
        """Alias for get_model_path with additional metadata path"""
        paths = cls.get_model_path(symbol)
        paths['metadata'] = os.path.join(os.path.dirname(paths['model']), 'metadata.json')
        return paths