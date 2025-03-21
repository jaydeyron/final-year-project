import os
import torch  # Import the torch library

class Config:
    # Model parameters
    INPUT_SIZE = 10  # Number of features in your data
    HIDDEN_SIZE = 128  # Changed from 64 to 128 to match the saved model
    NUM_LAYERS = 3
    NUM_HEADS = 8  # Using 8 heads (multiple of 64)
    DROPOUT = 0.1  # Reduced from previous value to allow more flexibility
    
    # Price constraint parameters
    MAX_CHANGE_PERCENT = 0.10  # Increased to 10% to allow more varied predictions
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100  # Increased to ensure better learning
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 60  # 60 days of historical data
    
    # Features for the model
    FEATURES = [
        'Open', 'High', 'Low', 'Close', 
        'Volume', 'EMA10', 'EMA30', 'RSI', 'MACD', 'MACD_Signal'
    ]
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # File paths - Using absolute paths to backend/models
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /Users/jay/Documents/GitHub/final-year-project/back-end
    MODEL_DIR = os.path.join(BASE_DIR, 'models')  # /Users/jay/Documents/GitHub/final-year-project/back-end/models
    
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
        # This method name is used in model_utils.py
        clean_symbol = symbol.replace('^', 'SENSEX').replace('.BO', '')
        base_dir = os.path.join(cls.MODEL_DIR, clean_symbol)
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