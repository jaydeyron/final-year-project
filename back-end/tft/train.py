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

# Import NIFTY_STOCKS from a shared module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from shared.stock_data import NIFTY_STOCKS, get_symbol_info
except ImportError:
    # Fallback if shared module not available
    NIFTY_STOCKS = []
    
    def get_symbol_info(symbol, field_name='symbol'):
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TFT model for a specific stock')
    parser.add_argument('--symbol', type=str, required=True, help='BSE symbol for data fetching (e.g., "TCS.BO")')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for model saving (e.g., "TCS")')
    parser.add_argument('--start-date', type=str, help='Start date for training data (format: YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for training data (format: YYYY-MM-DD)')
    parser.add_argument('--update', action='store_true', help='Update existing model instead of training from scratch')
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of training epochs')
    args = parser.parse_args()
    
    # Symbol to use for saving the model
    save_symbol = args.display_symbol if args.display_symbol else args.symbol
    
    # Get stock info for correct dates and symbol handling
    stock_info = get_symbol_info(args.symbol, 'bseSymbol') or {'startDate': '2000-01-01', 'symbol': save_symbol}
    print(f"Training model for {stock_info.get('name', save_symbol)} (symbol: {args.symbol})")
    
    # Use provided start date or fall back to stock info
    start_date = args.start_date or stock_info.get('startDate', "2000-01-01")
    
    # Use provided end date or default to today
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Ensure models directory exists
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(backend_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Create progress.json file to track training progress
        progress_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'progress.json')
        with open(progress_file, 'w') as f:
            json.dump({"progress": 0}, f)
        
        # Fetch and preprocess data
        print(f"Fetching data from {start_date} to {end_date}...")
        data = fetch_data(args.symbol, start_date, end_date)
        if data is None or len(data) < Config.SEQ_LENGTH * 2:  # Need enough data for sequences
            raise ValueError(f"Insufficient data for symbol {args.symbol}. Need at least {Config.SEQ_LENGTH*2} data points.")
            
        print(f"Fetched {len(data)} data points")
        
        # Preprocess data
        scaled_data, scaler = preprocess_data(data)
        xs, ys = create_sequences(scaled_data, Config.SEQ_LENGTH)
        
        # Log summary of sequence creation
        print(f"Created {len(xs)} sequences for training")
        
        # Create dataset and dataloader
        dataset = TimeSeriesDataset(xs, ys)
        dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        
        input_size = xs.shape[2]
        
        # Initialize or load the model with price constraint
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
                    Config.DROPOUT,
                    max_change_percent=Config.MAX_CHANGE_PERCENT
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
                Config.DROPOUT,
                max_change_percent=Config.MAX_CHANGE_PERCENT
            )
            total_epochs = args.epochs

        # Train model
        print(f"Starting training for {args.epochs} epochs...")
        trainer = Trainer(model, dataloader, Config.DEVICE, Config.LEARNING_RATE)
        trainer.train(args.epochs)

        # Save model, scaler, and metadata
        additional_info = {
            'data_start_date': start_date,
            'data_end_date': end_date,
            'symbol_name': stock_info.get('name', save_symbol),
            'bse_symbol': args.symbol,
            'training_completed': datetime.now().isoformat(),
            'epochs': total_epochs,
            'data_points': len(data)
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
        
        # Update progress to 100% when done
        with open(progress_file, 'w') as f:
            json.dump({"progress": 100}, f)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Update progress file to show error
        try:
            with open(progress_file, 'w') as f:
                json.dump({"progress": -1, "error": str(e)}, f)
        except:
            pass
        raise e

if __name__ == "__main__":
    main()