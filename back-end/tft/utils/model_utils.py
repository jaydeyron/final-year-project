import torch
import json
import os
from datetime import datetime, timedelta
from models.tft_model import TemporalFusionTransformer
from config import Config

def save_model(symbol, model, scaler, input_size, epochs=None, additional_info=None):
    """Save model, scaler and metadata for a specific symbol"""
    paths = Config.get_model_paths(symbol)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(paths['model']), exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), paths['model'])
    
    # Save the scaler parameters
    scaler_params = {
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist()
    }
    with open(paths['scaler'], 'w') as f:
        json.dump(scaler_params, f)
    
    # Save metadata
    metadata = {
        'symbol': symbol,
        'input_size': input_size,
        'last_trained': datetime.now().isoformat(),
        'epochs': epochs or Config.EPOCHS
    }
    
    if additional_info:
        metadata.update(additional_info)
    
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f)
    
    return paths

def load_model(symbol):
    """Load model, scaler and metadata for a specific symbol"""
    paths = Config.get_model_paths(symbol)
    
    # Check if files exist
    if not os.path.exists(paths['model']) or not os.path.exists(paths['scaler']):
        return None, None, None
    
    # Load metadata if available
    metadata = None
    if os.path.exists(paths['metadata']):
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)
    
    # Determine input size
    input_size = metadata.get('input_size', len(Config.FEATURES)) if metadata else len(Config.FEATURES)
    
    # Load model
    model = TemporalFusionTransformer(
        input_size, 
        Config.HIDDEN_SIZE,
        1,
        Config.NUM_LAYERS,
        Config.NUM_HEADS,
        Config.DROPOUT
    )
    model.load_state_dict(torch.load(paths['model']))
    model.eval()
    
    # Load scaler
    with open(paths['scaler'], 'r') as f:
        scaler_params = json.load(f)
    
    return model, scaler_params, metadata

def list_available_models():
    """List all available trained models"""
    if not os.path.exists(Config.MODEL_DIR):
        return []
    
    models = []
    for symbol_dir in os.listdir(Config.MODEL_DIR):
        model_path = os.path.join(Config.MODEL_DIR, symbol_dir, 'model.pth')
        if os.path.exists(model_path):
            models.append(symbol_dir)
    
    return models

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
