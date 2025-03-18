import argparse
import sys
import torch
from datetime import datetime
from utils.data_loader import fetch_data, preprocess_data, create_sequences
from utils.model_utils import load_model
from config import Config
from train import get_stock_info

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained TFT model')
    parser.add_argument('--symbol', type=str, required=True, help='BSE symbol for data fetching')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for model loading')
    args = parser.parse_args()
    
    # Symbol to use for loading the model
    model_symbol = args.display_symbol if args.display_symbol else args.symbol
    
    try:
        # Get stock info
        stock_info = get_stock_info(args.symbol, args.display_symbol)
        
        # Load model, scaler and metadata
        model, scaler_params, metadata = load_model(model_symbol)
        if model is None:
            raise ValueError(f"No model found for {model_symbol}. Please train the model first.")
        
        # Convert model to evaluation mode
        model.to(Config.DEVICE)
        model.eval()
        
        # Fetch recent data for prediction
        print(f"Loading recent data for {args.symbol}...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        # Get enough data for the sequence length + some padding
        days_to_fetch = Config.SEQ_LENGTH * 2
        recent_data = fetch_data(args.symbol, None, end_date, days=days_to_fetch)
        
        if recent_data is None or recent_data.empty:
            raise ValueError(f"Failed to fetch data for {args.symbol}")
            
        # Preprocess data using saved scaler
        features, _ = preprocess_data(recent_data, scaler_params)
        
        # Get the last sequence
        last_sequence = torch.FloatTensor(features[-Config.SEQ_LENGTH:]).unsqueeze(0).to(Config.DEVICE)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(last_sequence)
            
        # Convert prediction to price
        # Assuming the target (Close) is at index 3 in the features
        close_idx = 3  
        predicted_value = prediction.item() * scaler_params['scale_'][close_idx] + scaler_params['mean_'][close_idx]
        
        # Print prediction (will be captured by subprocess)
        print(predicted_value)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()