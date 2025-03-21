import os
import torch
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from models.tft_model import TemporalFusionTransformer

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

def load_model(symbol):
    """
    Load a trained model, scaler parameters, and metadata
    Returns: model, scaler_params, metadata
    """
    try:
        paths = Config.get_model_paths(symbol)
        model_path = paths['model']
        scaler_path = paths['scaler']
        metadata_path = paths['metadata']
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            return None, None, None
        
        # Initialize scaler params with the minimum required features
        default_scaler_params = {
            'mean_': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 10 features
            'scale_': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # 10 features
        }
        
        # Try to load saved scaler parameters
        scaler_params = default_scaler_params.copy()
        
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'r') as f:
                    file_content = f.read().strip()
                    if file_content:
                        loaded_params = json.loads(file_content)
                        
                        # Process mean values
                        mean_vals = []
                        if 'mean_' in loaded_params:
                            if isinstance(loaded_params['mean_'], dict):
                                # Convert dict to list
                                mean_vals = [float(loaded_params['mean_'].get(str(i), 0.0)) 
                                         for i in range(len(loaded_params['mean_']))]
                            elif isinstance(loaded_params['mean_'], list):
                                mean_vals = [float(x) for x in loaded_params['mean_']]
                            else:
                                # Single value, make it a list
                                mean_vals = [float(loaded_params['mean_'])]
                                
                        # Process scale values
                        scale_vals = []
                        if 'scale_' in loaded_params:
                            if isinstance(loaded_params['scale_'], dict):
                                scale_vals = [float(loaded_params['scale_'].get(str(i), 1.0)) 
                                          for i in range(len(loaded_params['scale_']))]
                            elif isinstance(loaded_params['scale_'], list):
                                scale_vals = [float(x) for x in loaded_params['scale_']]
                            else:
                                # Single value, make it a list
                                scale_vals = [float(loaded_params['scale_'])]
                        
                        # Ensure we have at least 4 values to cover the Close price index
                        if len(mean_vals) < 4:
                            print(f"Warning: Only {len(mean_vals)} features in scaler mean, extending to 10")
                            mean_vals.extend([0.0] * (10 - len(mean_vals)))
                            
                        if len(scale_vals) < 4:
                            print(f"Warning: Only {len(scale_vals)} features in scaler scale, extending to 10")
                            scale_vals.extend([1.0] * (10 - len(scale_vals)))
                        
                        # Avoid any zero scale values
                        scale_vals = [max(x, 0.0001) for x in scale_vals]
                        
                        # Update the scaler parameters
                        scaler_params['mean_'] = mean_vals
                        scaler_params['scale_'] = scale_vals
                        
                        print(f"Loaded scaler parameters with {len(mean_vals)} features")
            except Exception as e:
                print(f"Error processing scaler file: {str(e)}")
                # Keep the default values if there's an error
        
        # Read metadata
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error reading metadata: {e}")
        
        # Get input size from metadata or default
        input_size = metadata.get('input_size', 10)
        hidden_size = metadata.get('hidden_size', Config.HIDDEN_SIZE)
        
        # Create model
        model = TemporalFusionTransformer(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            num_layers=Config.NUM_LAYERS,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT,
            max_change_percent=Config.MAX_CHANGE_PERCENT
        )
        
        # Load model weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return None, None, None
        
        return model, scaler_params, metadata
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

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
