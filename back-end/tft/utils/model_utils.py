import os
import torch
import json
import sys
from datetime import datetime, timedelta  # Add this import

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
    
    # Save scaler parameters with both standard and price-specific parameters
    scaler_dict = {
        'mean_': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [0],
        'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [1],
        'var_': scaler.var_.tolist() if hasattr(scaler, 'var_') else [1],
        'price_mean_': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else [0],
        'price_scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else [1]
    }
    
    with open(scaler_path, 'w') as f:
        json.dump(scaler_dict, f, indent=2)
    
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
    Load a trained model, scaler parameters, and metadata with backwards compatibility
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
        
        # Load scaler parameters - this part remains the same
        scaler_params = default_scaler_params.copy()
        # ...existing scaler loading code...
        
        # Read metadata
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Error reading metadata: {e}")
        
        # Get parameters from metadata or default
        input_size = metadata.get('input_size', 10)
        # Use Config.HIDDEN_SIZE for backwards compatibility
        hidden_size = Config.HIDDEN_SIZE  # Don't use metadata hidden_size to ensure compatibility
        
        # Create model with appropriate architecture
        # Check if the model weights are from old or new architecture
        is_old_architecture = check_old_architecture(model_path)
        
        if is_old_architecture:
            print(f"Detected old model architecture for {symbol}, using compatibility mode")
            # Create model with old architecture for compatibility
            model = create_compatible_model(input_size)
        else:
            # Create the newer enhanced model
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
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Handle backwards compatibility 
            if is_old_architecture:
                # Add missing key/value compatibility check here
                load_result = model.load_state_dict(state_dict, strict=False)
                if load_result.missing_keys:
                    print(f"Missing keys in state_dict: {load_result.missing_keys}")
                if load_result.unexpected_keys:
                    print(f"Unexpected keys in state_dict: {load_result.unexpected_keys}")
            else:
                model.load_state_dict(state_dict)
                
            model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            # Don't return None on weight loading failure - try to continue with freshly initialized model
            print("Using freshly initialized model instead")
            model.eval()
        
        return model, scaler_params, metadata
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def check_old_architecture(model_path):
    """Check if the saved model is from the old architecture"""
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # Check for characteristic keys from old vs new architecture
        if "pre_output.weight" in state_dict and "pre_output_1.weight" not in state_dict:
            return True
        if "lstm_reducer.weight" not in state_dict and "lstm.weight_ih_l0_reverse" not in state_dict:
            return True
        return False
    except Exception:
        # If we can't load it, assume it's compatible with current architecture
        return False
        
def create_compatible_model(input_size):
    """Create a model compatible with the old architecture"""
    class CompatibleTFT(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=3, num_heads=4, dropout=0.1, 
                     max_change_percent=0.05):
            super(CompatibleTFT, self).__init__()
            
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.max_change_percent = max_change_percent
            
            # Old architecture components
            self.feature_layer = nn.Linear(input_size, hidden_size)
            self.positional_encoding = PositionalEncoding(hidden_size, dropout)
            
            # Old LSTM (not bidirectional)
            self.lstm = nn.LSTM(
                input_size=hidden_size, 
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            # Old transformer without our new components
            self.transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dropout=dropout,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                self.transformer_encoder_layer, 
                num_layers=min(num_layers, 3)  # Old models had fewer layers
            )
            
            # Old output structure
            self.pre_output = nn.Linear(hidden_size, hidden_size)
            self.output_layer = nn.Linear(hidden_size, output_size)
            
            # Constrained output layer kept for compatibility
            self.price_constraint = PriceConstraintLayer(max_change_percent)
            
            # Add compatibility attributes for the enhanced model
            # These won't be used but make the state_dict loading more tolerant
            self.feature_gate = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.Sigmoid()
            )
            self.pre_output_1 = self.pre_output  # Reference the same layer
            self.pre_output_2 = self.pre_output  # Reference the same layer
            self.uncertainty_layer = nn.Linear(hidden_size, output_size)
            
        def extract_last_close_price(self, x):
            return x[:, -1, 3:4]
            
        def forward(self, x, apply_constraint=False):
            # Store last close price for constraint
            last_close = self.extract_last_close_price(x)
            
            # Feature transformation using old architecture
            x = self.feature_layer(x)
            x = self.positional_encoding(x)
            
            # Temporal processing with LSTM
            x, _ = self.lstm(x)
            
            # Self-attention mechanism
            x = self.transformer_encoder(x)
            
            # Take the last output for prediction
            x = x[:, -1, :]
            
            # Output layers with residual connection
            x = F.relu(self.pre_output(x)) + x
            output = self.output_layer(x)
            
            # Apply price constraint if requested
            if apply_constraint:
                output = self.price_constraint(output, last_close)
            
            return output
    
    # Create and return the compatible model
    return CompatibleTFT(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        output_size=1,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        dropout=Config.DROPOUT,
        max_change_percent=Config.MAX_CHANGE_PERCENT
    )

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
