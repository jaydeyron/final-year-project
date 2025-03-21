import torch
import argparse
import sys
import os
from datetime import datetime
from utils.data_loader import fetch_data, preprocess_data
from utils.model_utils import load_model
from config import Config

# Fix the import path for the TFT model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.tft_model import TemporalFusionTransformer

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained TFT model')
    parser.add_argument('--symbol', type=str, required=True, help='BSE symbol for data fetching')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for model loading')
    parser.add_argument('--model-symbol', type=str, help='Override which model to use (e.g., TCS)')
    parser.add_argument('--days', type=int, default=120, help='Number of days of data to fetch')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    args = parser.parse_args()
    
    # Suppress all warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        # Use debug printing to separate from the actual output
        def debug_print(msg):
            if args.debug:
                print(f"DEBUG: {msg}", file=sys.stderr)
        
        # Determine which model to load
        model_symbol = args.model_symbol or args.display_symbol or args.symbol
        # Clean up symbol for model loading
        if model_symbol.endswith('.BO'):
            model_symbol = model_symbol[:-3]
        elif model_symbol.startswith('^'):
            model_symbol = model_symbol[1:]
            
        debug_print(f"Loading model for symbol: {model_symbol}")
        
        # Ensure models directory exists
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(backend_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Load model, scaler and metadata
        model, scaler_params, metadata = load_model(model_symbol)
        
        if model is None:
            debug_print(f"No model found for {model_symbol}, trying TCS as fallback")
            model, scaler_params, metadata = load_model("TCS")
            if model is None:
                raise ValueError(f"Failed to load model for {model_symbol} and TCS fallback")
        
        # Convert model to evaluation mode
        model.to(Config.DEVICE)
        model.eval()
        
        # Fetch recent data for prediction
        debug_print(f"Loading recent data for {args.symbol}")
        end_date = datetime.now().strftime('%Y-%m-%d')
        days_to_fetch = args.days
        
        # Silence the yfinance warning outputs
        with open(os.devnull, 'w') as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = devnull
            
            try:
                # Make sure to include the days parameter
                recent_data = fetch_data(args.symbol, start_date=None, end_date=end_date, days=days_to_fetch)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        
        if recent_data is None or len(recent_data) < Config.SEQ_LENGTH:
            raise ValueError(f"Insufficient data for {args.symbol}. Need at least {Config.SEQ_LENGTH} data points.")
            
        debug_print(f"Processing {len(recent_data)} data points")
        
        # Preprocess data using saved scaler
        features, _ = preprocess_data(recent_data, scaler_params)
        
        # Get the last sequence for prediction
        last_sequence = torch.FloatTensor(features[-Config.SEQ_LENGTH:]).unsqueeze(0).to(Config.DEVICE)
        
        debug_print(f"Making prediction with sequence shape {last_sequence.shape}")
        
        # Make prediction with model - disable constraints to see raw predictions
        with torch.no_grad():
            prediction = model(last_sequence, apply_constraint=False)
            
        # Convert prediction to price (assuming Close is at index 3)
        close_idx = 3  # Index for Close price
        predicted_value = prediction.item() * scaler_params['scale_'][close_idx] + scaler_params['mean_'][close_idx]
        
        # Calculate percent change for debugging
        last_close = recent_data['Close'].iloc[-1]
        percent_change = ((predicted_value - last_close) / last_close) * 100
        
        # Add debug information
        debug_print(f"Last close: {last_close:.2f}")
        debug_print(f"Raw prediction: {predicted_value:.2f}")
        debug_print(f"Percent change: {percent_change:.2f}%")
        
        # Print only the predicted value for API
        print(f"{predicted_value:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()