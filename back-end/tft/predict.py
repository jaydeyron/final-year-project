import torch
import argparse
import sys
import os
from datetime import datetime
from utils.data_loader import fetch_data, preprocess_data
from config import Config

# Import shared stock data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from shared.stock_data import get_symbol_info
except ImportError:
    def get_symbol_info(symbol, field_name='symbol'):
        return None

def load_model_for_symbol(symbol):
    """
    Load a model for the specified symbol from models/symbol/model.pth
    Falls back to TCS model if symbol's model doesn't exist
    """
    from models.tft_model import TemporalFusionTransformer
    
    # Create clean symbol name for directory
    clean_symbol = symbol
    if symbol.endswith('.BO'):
        clean_symbol = symbol[:-3]
    elif symbol.startswith('^'):
        clean_symbol = symbol[1:]
    
    # Define model paths
    model_dir = os.path.join(Config.MODEL_DIR, clean_symbol)
    model_path = os.path.join(model_dir, 'model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.json')
    metadata_path = os.path.join(model_dir, 'metadata.json')
    
    # Check if model exists
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"No model found for {symbol} at {model_path}")
        if symbol != "TCS" and clean_symbol != "TCS":
            print("Falling back to TCS model")
            return load_model_for_symbol("TCS")
        else:
            return None, None, None
    
    # Load metadata if available
    metadata = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                import json
                metadata = json.load(f)
                print(f"Loaded metadata for {symbol}: last trained on {metadata.get('training_completed', 'unknown')}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    # Determine input size from metadata or config
    input_size = metadata.get('input_size', len(Config.FEATURES)) if metadata else len(Config.FEATURES)
    
    # Initialize model with the price constraint
    model = TemporalFusionTransformer(
        input_size, 
        Config.HIDDEN_SIZE,
        1,
        Config.NUM_LAYERS,
        Config.NUM_HEADS,
        Config.DROPOUT,
        max_change_percent=Config.MAX_CHANGE_PERCENT
    )
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None, None, None
    
    # Load scaler parameters
    try:
        with open(scaler_path, 'r') as f:
            import json
            scaler_params = json.load(f)
            print(f"Successfully loaded scaler from {scaler_path}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None, None, None
    
    return model, scaler_params, metadata

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained TFT model')
    parser.add_argument('--symbol', type=str, required=True, help='BSE symbol for data fetching')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for model loading')
    parser.add_argument('--model-symbol', type=str, help='Override which model to use (e.g., TCS)')
    parser.add_argument('--days', type=int, default=120, help='Number of days of data to fetch')
    args = parser.parse_args()
    
    # Determine which model to load
    model_symbol = args.model_symbol or args.display_symbol or args.symbol
    
    print(f"Attempting to load model for symbol: {model_symbol}")
    
    try:
        # Load model, scaler and metadata
        model, scaler_params, metadata = load_model_for_symbol(model_symbol)
        
        if model is None:
            print("Failed to load any model, including TCS fallback.")
            sys.exit(1)
        
        # Convert model to evaluation mode
        model.to(Config.DEVICE)
        model.eval()
        
        # Fetch recent data for prediction
        print(f"Loading recent data for {args.symbol}...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get enough data for the sequence length + some padding
        days_to_fetch = args.days
        
        # Get stock info for start date if available
        stock_info = get_symbol_info(args.symbol, field_name='bseSymbol')
        start_date = None
        
        print(f"Fetching {days_to_fetch} days of data ending on {end_date}")
        recent_data = fetch_data(args.symbol, start_date, end_date, days=days_to_fetch)
        
        if recent_data is None or len(recent_data) < Config.SEQ_LENGTH:
            raise ValueError(f"Insufficient data for {args.symbol}. Need at least {Config.SEQ_LENGTH} data points.")
            
        print(f"Processing {len(recent_data)} data points...")
        # Preprocess data using saved scaler
        features, _ = preprocess_data(recent_data, scaler_params)
        
        # Get the last sequence for prediction
        last_sequence = torch.FloatTensor(features[-Config.SEQ_LENGTH:]).unsqueeze(0).to(Config.DEVICE)
        
        print(f"Making prediction using sequence of shape {last_sequence.shape}...")
        # Make prediction with constraint applied
        with torch.no_grad():
            prediction = model(last_sequence, apply_constraint=True)
            
        # Convert prediction to price
        # Using index 3 for Close price (standard in Config.FEATURES)
        close_idx = 3
        predicted_value = prediction.item() * scaler_params['scale_'][close_idx] + scaler_params['mean_'][close_idx]
        
        # Return the prediction to the terminal
        print(f"Predicted value for {args.symbol}: {predicted_value}")
        print(predicted_value)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()