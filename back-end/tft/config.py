import os
import torch

class Config:
    # Enhanced model parameters
    INPUT_SIZE = 10  # This will be overridden by actual feature count
    HIDDEN_SIZE = 192  # Increased from 128 for more capacity
    NUM_LAYERS = 4  # Increased from 3 for more depth
    NUM_HEADS = 8  # Maintained at 8 heads
    DROPOUT = 0.25  # Increased from 0.2 for better regularization
    
    # Price constraint parameters - allow more flexibility
    MAX_CHANGE_PERCENT = 0.12  # Increased from 0.05 to 0.12 (12%)
    
    # Training parameters
    BATCH_SIZE = 32  # Reduced from 64 for better gradient estimates
    EPOCHS = 150  # Increased from 120 for more learning
    LEARNING_RATE = 0.0005  # Reduced from 0.001 for more stable training
    SEQ_LENGTH = 60  # Maintained at 60 days of historical data
    
    # Features for the model
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