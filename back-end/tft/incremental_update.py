import argparse
import os
import sys
import logging
import torch
import json
from datetime import datetime, timedelta

# Use absolute imports with the full path to ensure modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Add both directories to path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import the required modules
from tft.utils.data_loader import fetch_data, preprocess_data, create_sequences, TimeSeriesDataset
from tft.utils.model_utils import save_model, load_model
from tft.config import Config
from tft.models.tft_model import TemporalFusionTransformer
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Incrementally update TFT model with latest data')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol to update (e.g., "TCS.BO")')
    parser.add_argument('--display-symbol', type=str, help='Display symbol for model (e.g., "TCS")')
    parser.add_argument('--days', type=int, default=5, help='Number of days of new data to fetch')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs for update')
    args = parser.parse_args()
    
    # Symbol to use for saving/loading the model
    save_symbol = args.display_symbol if args.display_symbol else args.symbol.replace('.BO', '')
    
    try:
        logger.info(f"Starting incremental update for {args.symbol}")
        
        # Step 1: Load existing model and metadata
        model, scaler_params, metadata = load_model(save_symbol)
        
        if model is None:
            logger.error(f"No existing model found for {save_symbol}. Please train a model first.")
            return
            
        # Get the date of last training from metadata
        last_training_date = metadata.get('data_end_date')
        if not last_training_date:
            # If no end date in metadata, use last week as default
            last_training_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            logger.warning(f"No previous training end date found. Using {last_training_date} as default.")
        
        # Step 2: Fetch only the newest data (last N days)
        today = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Fetching new data from {last_training_date} to {today}")
        
        new_data = fetch_data(args.symbol, start_date=last_training_date, end_date=today)
        
        if new_data is None or len(new_data) < 2:  # Need at least some new data
            logger.warning(f"No new data available for {args.symbol} since {last_training_date}")
            return
            
        logger.info(f"Fetched {len(new_data)} new data points")
        
        # Step 3: Process the new data using the existing scaler params
        scaled_data, _ = preprocess_data(new_data, scaler_params)
        
        # Create training sequences from the new data
        xs, ys = create_sequences(scaled_data, Config.SEQ_LENGTH)
        
        if len(xs) < 1:
            logger.warning("Not enough new data to create training sequences")
            return
            
        logger.info(f"Created {len(xs)} sequences from new data")
        
        # Step 4: Create dataset and dataloader
        train_dataset = TimeSeriesDataset(xs, ys)
        train_dataloader = DataLoader(train_dataset, batch_size=min(8, len(xs)), shuffle=True)
        
        # Step 5: Fine-tune the model with new data
        model.train()  # Set model to training mode
        model.to(Config.DEVICE)
        
        # Create optimizer with lower learning rate for fine-tuning
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE * 0.1)
        criterion = torch.nn.MSELoss()
        
        logger.info(f"Fine-tuning model for {args.epochs} epochs on new data")
        for epoch in range(args.epochs):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs.squeeze(), targets)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Add batch loss
                total_loss += loss.item()
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}')
        
        # Step 6: Save updated model
        # Update metadata
        total_epochs = metadata.get('epochs', 0) + args.epochs
        
        additional_info = {
            'data_start_date': metadata.get('data_start_date'),
            'data_end_date': today,
            'symbol_name': metadata.get('symbol_name', save_symbol),
            'bse_symbol': args.symbol,
            'training_completed': datetime.now().isoformat(),
            'epochs': total_epochs,
            'data_points': metadata.get('data_points', 0) + len(new_data),
            'last_update': datetime.now().isoformat(),
            'update_type': 'incremental'
        }
        
        # Save updated model
        input_size = xs.shape[2]
        saved_paths = save_model(
            save_symbol,
            model, 
            scaler_params, 
            input_size, 
            total_epochs, 
            additional_info
        )
        
        logger.info(f"Updated model saved to {saved_paths['model']}")
        logger.info(f"Incremental update completed successfully for {save_symbol}")
        
    except Exception as e:
        logger.error(f"Error during incremental update: {str(e)}")
        raise

if __name__ == "__main__":
    main()
