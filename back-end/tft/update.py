import argparse
import sys
import os
from datetime import datetime, timedelta  # Ensure datetime is imported
import torch
from utils.data_loader import fetch_data, preprocess_data, create_sequences, TimeSeriesDataset
from utils.trainer import Trainer
from utils.model_utils import save_model, load_model
from models.tft_model import TemporalFusionTransformer
from torch.utils.data import DataLoader
from config import Config

"""
Script to update an existing model with new data.
This is useful for incremental learning where you want to
keep the existing model but train it on recent data.
"""

def main():
    parser = argparse.ArgumentParser(description='Update existing TFT model with recent data')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol (e.g., "TCS.BO")')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for saving the model')
    parser.add_argument('--days', type=int, default=60, help='Number of days of recent data to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()
    
    # Determine symbols
    symbol = args.symbol
    save_symbol = args.display_symbol or (symbol[:-3] if symbol.endswith('.BO') else symbol)
    
    try:
        # Load existing model
        print(f"Loading existing model for {save_symbol}...")
        model, scaler_params, metadata = load_model(save_symbol)
        
        if model is None:
            print(f"No existing model found for {save_symbol}. Please train a model first.")
            return
            
        print(f"Model loaded successfully. Original training: {metadata.get('training_completed', 'unknown')}")
        
        # Calculate date range for recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        print(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        data = fetch_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if data is None or len(data) < Config.SEQ_LENGTH:
            print(f"Insufficient recent data for {symbol}. Need at least {Config.SEQ_LENGTH} points.")
            return
            
        print(f"Fetched {len(data)} recent data points")
        
        # Preprocess using existing scaler parameters
        scaled_data, _ = preprocess_data(data, scaler_params)
        xs, ys = create_sequences(scaled_data, Config.SEQ_LENGTH)
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(xs, ys)
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        # Update the model with new data
        print(f"Updating model with {len(xs)} new sequences, {args.epochs} epochs...")
        trainer = Trainer(model, dataloader, Config.DEVICE, learning_rate=0.0005)  # Lower learning rate for updating
        trainer.train(args.epochs)
        
        # Save updated model
        total_epochs = metadata.get('epochs', 0) + args.epochs
        updated_info = {
            'data_start_date': metadata.get('data_start_date'),
            'data_end_date': end_date.strftime('%Y-%m-%d'),
            'symbol_name': metadata.get('symbol_name', save_symbol),
            'bse_symbol': symbol,
            'training_completed': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'epochs': total_epochs,
            'data_points': metadata.get('data_points', 0) + len(data)
        }
        
        # Save model with updated info
        input_size = xs.shape[2] if hasattr(xs, 'shape') else metadata.get('input_size', len(Config.FEATURES))
        saved_paths = save_model(
            save_symbol,
            model,
            scaler_params,  # Use same scaler
            input_size,
            total_epochs,
            updated_info
        )
        
        print(f"Updated model saved to {saved_paths['model']}")
        print(f"Model update completed successfully for {save_symbol}")
        
    except Exception as e:
        print(f"Error during model update: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()