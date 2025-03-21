import os
import torch
import json
from datetime import datetime, timedelta
from models.tft_model import TemporalFusionTransformer
from config import Config

def save_model(symbol, model, scaler, input_size, epochs, additional_info=None):
    """
    Save model, scaler and metadata for a specific stock symbol
    
    Args:
        symbol: Stock symbol (for directory naming)
        model: Trained model
        scaler: Data scaler
        input_size: Model input size
        epochs: Number of epochs trained
        additional_info: Dictionary of additional metadata
    
    Returns:
        Dictionary with paths to saved files
    """
    # Create clean symbol name for directory
    clean_symbol = symbol
    if symbol.endswith('.BO'):
        clean_symbol = symbol[:-3]
    elif symbol.startswith('^'):
        clean_symbol = symbol[1:]
        
    # Create directory structure using absolute path to backend/models
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(backend_dir, 'models', clean_symbol)
    os.makedirs(model_dir, exist_ok=True)
    
    # File paths
    model_path = os.path.join(model_dir, 'model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.json')
    metadata_path = os.path.join(model_dir, 'metadata.json')
    
    print(f"Saving model to directory: {model_dir}")
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # Save scaler parameters
    with open(scaler_path, 'w') as f:
        json.dump({
            'mean_': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [0],
            'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [1],
            'var_': scaler.var_.tolist() if hasattr(scaler, 'var_') else [1]
        }, f)
    
    # Get the hidden size from the model
    hidden_size = model.hidden_size if hasattr(model, 'hidden_size') else 128
    
    # Save metadata
    metadata = {
        'input_size': input_size,
        'hidden_size': hidden_size,  # Save the hidden size in metadata
        'epochs': epochs,
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'clean_symbol': clean_symbol
    }
    
    # Add additional info if provided
    if additional_info and isinstance(additional_info, dict):
        metadata.update(additional_info)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return {
        'model': model_path,
        'scaler': scaler_path,
        'metadata': metadata_path
    }

def load_model(symbol, model_class=None):
    """
    Load model, scaler and metadata for a specific stock symbol
    
    Args:
        symbol: Stock symbol (for directory lookup)
        model_class: Optional model class to instantiate
    
    Returns:
        Tuple of (model, scaler_params, metadata)
    """
    from models.tft_model import TemporalFusionTransformer
    from config import Config
    
    # Create clean symbol name for directory
    clean_symbol = symbol
    if symbol.endswith('.BO'):
        clean_symbol = symbol[:-3]
    elif symbol.startswith('^'):
        clean_symbol = symbol[1:]
        
    # Use absolute path to backend/models directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(backend_dir, 'models', clean_symbol)
    
    # File paths
    model_path = os.path.join(model_dir, 'model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.json')
    metadata_path = os.path.join(model_dir, 'metadata.json')
    
    print(f"Looking for model files in: {model_dir}")
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model files not found for symbol {symbol}")
        return None, None, None
    
    # Load metadata
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Determine input size and hidden size from metadata or use defaults
    input_size = metadata.get('input_size', len(Config.FEATURES)) if metadata else len(Config.FEATURES)
    hidden_size = metadata.get('hidden_size', Config.HIDDEN_SIZE) if metadata else Config.HIDDEN_SIZE
    
    print(f"Loading model with input_size={input_size}, hidden_size={hidden_size}")
    
    # Create model instance with the correct hidden size
    if model_class:
        model = model_class(input_size, hidden_size, 1, Config.NUM_LAYERS, Config.NUM_HEADS, Config.DROPOUT)
    else:
        model = TemporalFusionTransformer(input_size, hidden_size, 1, Config.NUM_LAYERS, Config.NUM_HEADS, Config.DROPOUT)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None
    
    # Load scaler parameters
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    return model, scaler_params, metadata

def make_prediction(symbol, model=None, scaler_params=None, start_date=None, end_date=None):
    """Make prediction using saved or provided model"""
    from utils.data_loader import fetch_data, preprocess_data  # Import here to avoid circular imports
    
    try:
        # Load model and scaler if not provided
        if model is None or scaler_params is None:
            model, scaler_params, _ = load_model(symbol)
            
        if model is None:
            raise Exception(f"No model found for {symbol}")
            
        # Prepare model for inference
        model.to(Config.DEVICE)
        model.eval()
            
        # Get recent data for prediction
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=Config.SEQ_LENGTH + 10)  # Extra days for safety
        
        # Format dates
        
        end_date_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
        start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        
        # Fetch and preprocess data
        data = fetch_data(symbol, start_date_str, end_date_str)
        if data is None or data.empty:
            raise Exception(f"No data available for {symbol}")
            
        scaled_data, _ = preprocess_data(data, scaler_params)
        
        # Create sequence for prediction - get last sequence of SEQ_LENGTH
        if len(scaled_data) < Config.SEQ_LENGTH:
            raise Exception(f"Insufficient data for prediction: need {Config.SEQ_LENGTH}, got {len(scaled_data)}")
            
        sequence = torch.FloatTensor(scaled_data[-Config.SEQ_LENGTH:]).unsqueeze(0).to(Config.DEVICE)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(sequence)
            
        # Assuming the target is the Close price (index 3)
        predicted_value = prediction.item() * scaler_params['scale_'][3] + scaler_params['mean_'][3]
        
        return float(predicted_value)
        
    except Exception as e:
        raise Exception(f"Prediction error for {symbol}: {str(e)}")

        # Fetch and preprocess data
        data = fetch_data(symbol, start_date_str, end_date_str)
        if data is None or data.empty:
            raise Exception(f"No data available for {symbol}")
            
        scaled_data, _ = preprocess_data(data, scaler_params)
        
        # Create sequence for prediction - get last sequence of SEQ_LENGTH
        if len(scaled_data) < Config.SEQ_LENGTH:
            raise Exception(f"Insufficient data for prediction: need {Config.SEQ_LENGTH}, got {len(scaled_data)}")
            
        sequence = torch.FloatTensor(scaled_data[-Config.SEQ_LENGTH:]).unsqueeze(0).to(Config.DEVICE)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(sequence)
            
        # Assuming the target is the Close price (index 3)
        predicted_value = prediction.item() * scaler_params['scale_'][3] + scaler_params['mean_'][3]
        
        return float(predicted_value)
        
    except Exception as e:
        raise Exception(f"Prediction error for {symbol}: {str(e)}")
