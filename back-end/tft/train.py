import torch
from torch.utils.data import DataLoader
import argparse
import sys
import os
import json
from datetime import datetime
from utils.data_loader import fetch_data, preprocess_data, create_sequences, TimeSeriesDataset
from utils.trainer import Trainer
from utils.model_utils import save_model, load_model
from config import Config
from models.tft_model import TemporalFusionTransformer

# Define niftyStocks data directly in Python
NIFTY_STOCKS = [
    { 
        "name": "Bombay Stock Exchange SENSEX",
        "symbol": "SENSEX",
        "tradingViewSymbol": "SENSEX",
        "bseSymbol": "^BSESN",
        "startDate": "1997-07-01"
    },
    { 
        "name": "Asian Paints Limited",
        "symbol": "ASIANPAINT",
        "tradingViewSymbol": "BSE:ASIANPAINT",
        "bseSymbol": "ASIANPAINT.BO",
        "startDate": "2000-01-03"
    },
    # ...more stocks...
    { 
        "name": "Tata Consultancy Services Limited",
        "symbol": "TCS",
        "tradingViewSymbol": "BSE:TCS",
        "bseSymbol": "TCS.BO",
        "startDate": "2004-08-25"
    }
]

def get_stock_info(symbol, display_symbol=None):
    """Get stock information including start date"""
    # Try to find by BSE symbol first
    for stock in NIFTY_STOCKS:
        if stock.get('bseSymbol') == symbol:
            return stock

    # Try to find by display symbol if provided
    if display_symbol:
        for stock in NIFTY_STOCKS:
            if stock.get('symbol') == display_symbol:
                return stock
    
    # If not found, use defaults
    return {
        "startDate": "2000-01-01",  # Safe default
        "symbol": display_symbol or symbol,
        "bseSymbol": symbol
    }

def main():
    parser = argparse.ArgumentParser(description='Train TFT model for a specific stock')
    parser.add_argument('--symbol', type=str, required=True, help='BSE symbol for data fetching')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for model saving')
    parser.add_argument('--update', action='store_true', help='Update existing model instead of training from scratch')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of training epochs')
    args = parser.parse_args()
    
    # Symbol to use for saving the model
    save_symbol = args.display_symbol if args.display_symbol else args.symbol
    
    # Get stock info for correct dates and symbol handling
    stock_info = get_stock_info(args.symbol, args.display_symbol)
    print(f"Training model for {stock_info.get('name', args.symbol)} (symbol: {args.symbol})")
    
    # Use the start date from stock info
    start_date = stock_info.get('startDate', "2000-01-01")
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Fetch and preprocess data
        print(f"Fetching data from {start_date} to {end_date}...")
        data = fetch_data(args.symbol, start_date, end_date)
        scaled_data, scaler = preprocess_data(data)
        xs, ys = create_sequences(scaled_data, Config.SEQ_LENGTH)

        # Create dataset and dataloader
        dataset = TimeSeriesDataset(xs, ys)
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        input_size = xs.shape[2]
        
        # Initialize or load the model
        if args.update:
            print("Loading existing model for updating...")
            model, _, metadata = load_model(save_symbol)
            
            if model is None:
                print("No existing model found. Training from scratch.")
                model = TemporalFusionTransformer(
                    input_size, 
                    Config.HIDDEN_SIZE,
                    1, 
                    Config.NUM_LAYERS, 
                    Config.NUM_HEADS, 
                    Config.DROPOUT
                )
                total_epochs = args.epochs
            else:
                print("Existing model loaded successfully.")
                total_epochs = metadata.get('epochs', 0) + args.epochs
        else:
            print("Training new model from scratch...")
            model = TemporalFusionTransformer(
                input_size, 
                Config.HIDDEN_SIZE,
                1, 
                Config.NUM_LAYERS, 
                Config.NUM_HEADS, 
                Config.DROPOUT
            )
            total_epochs = args.epochs

        # Train model
        trainer = Trainer(model, dataloader, Config.DEVICE, Config.LEARNING_RATE)
        trainer.train(args.epochs)

        # Save model, scaler, and metadata
        additional_info = {
            'data_start_date': start_date,
            'data_end_date': end_date,
            'training_completed': datetime.now().isoformat(),
            'epochs': total_epochs
        }
        
        saved_paths = save_model(
            save_symbol,
            model, 
            scaler, 
            input_size, 
            total_epochs, 
            additional_info
        )
        
        print(f"Model saved to {saved_paths['model']}")
        print(f"Training completed successfully for {save_symbol}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()